# Design

Clockwork serves one model per process. A FastAPI layer accepts OpenAI-compatible
requests, an async engine feeds a continuous-batching scheduler, and a model runner
executes flat token batches against a paged KV cache. The reference model
implementation matches Hugging Face greedy decoding token for token; that equivalence
is a pytest gate, not an aspiration.

## Scheduler

The scheduler runs one iteration per engine step and never plans further ahead.

Decode first: every running sequence, in arrival order, gets its next-token slot
before any new work is admitted. If a sequence cannot append (a new block is needed,
or its tail block is shared and copy-on-write needs one), the victim is the
latest-arrival running sequence not yet granted a slot this step; when no such victim
remains, the sequence preempts itself. Preemption is recompute-only, there is no swap
path: the victim's blocks are freed, its computed-token and cached-token counts are
reset, and it re-enters the waiting queue ahead of every never-admitted sequence, FCFS
among the preempted. Recompute discards work proportional to the victim's progress,
which is why the victim is the latest arrival, but it avoids host-device KV transfers
and any allocator swap state, and the victim's tokens are never lost.

Admission second, strict FCFS with head-of-line blocking: only the queue head is
considered, and a head that does not fit admits nothing behind it, so a request can
wait but never starve. Each candidate is matched against the radix cache before
costing: cached prompt blocks are attached to the sequence up front, and only
uncomputed tokens count against the per-step token budget (max_num_batched_tokens). A
candidate is admitted only if the token budget, the sequence cap (max_num_seqs), and
the watermark all hold: free_blocks - blocks_needed >= int(watermark * num_blocks). A
failed admission releases the speculative radix match and leaves no reference behind.
The watermark trades a slice of capacity for headroom: without it, greedy admission
drains the pool and the very next decode step preempts what was just admitted. Head
blocking can idle capacity behind one large request; the alternative, skipping the
head, reorders arrivals under memory pressure and starves long prompts.

Invariant: a sequence is in exactly one of waiting, running, preempted, or finished,
and holds blocks only while running. Same inputs produce the same schedule; there is
no randomness in the scheduler.

## Block allocator and paged KV cache

KV memory is a fixed pool of blocks, block_size tokens each, one K and one V tensor
per layer shaped [num_blocks, block_size, num_kv_heads, head_dim]. A sequence maps
logical block index to physical block id through its block table; slot id is
block_id * block_size + offset. Writes scatter through slot mappings; attention reads
gather through block tables, so blocks never need to be physically contiguous.

The allocator is a LIFO free list with per-block reference counts. LIFO reuse keeps
the hot set of physical blocks small and recently touched; FIFO would cycle writes
through the whole pool for no benefit. Fixed-size blocks waste up to block_size - 1
slots in a sequence's tail block; smaller blocks cut that waste but grow every block
table, slot mapping, and radix node, so block_size 16 is the shipped default. Sharing
is refcounting: a radix hit or a fork increfs the shared blocks. Writing into a shared
block triggers copy-on-write: allocate a private block, copy the physical block, swap
the block table entry, decref the shared block. CoW allocates before it releases, so
an out-of-memory failure leaves the shared block intact. allocate_many is
all-or-nothing for the same reason.

Prefill runs one forward per sequence with no cross-request padding: prompt lengths
vary by orders of magnitude, so a padded prefill batch spends most of its FLOPs on pad
tokens, and the per-step token budget already bounds the work per iteration. Decode
runs as one batched forward across all running sequences, because every decode query
is a single token and batches with no padding. The end-aligned causal offset in paged
prefill (query positions start at the number of already-cached tokens) is what makes
prefix reuse exact rather than approximate.

## Radix prefix cache

The radix tree maps token-id sequences to KV block ids at block granularity: only
full blocks are inserted or matched, so a cached span is always bit-identical KV. On
admission the scheduler asks for the longest block-aligned cached prefix. The match
is capped strictly below the prompt length: the model always prefills at least one
token, because attention needs a query.

The tree holds one reference to every stored block; running sequences hold their own.
Insertion adopts and increfs only blocks the tree did not already have, so on a
duplicate span the caller keeps sole ownership of its copies and frees them with the
sequence. A matched span is increfed before it is returned and its deepest block is
locked; an ancestor can never become a leaf while a descendant exists, so one lock
pins the whole matched path. Eviction is LRU over unlocked leaves, ordered by a
monotonic access counter rather than wall-clock time, and removes whole nodes, so it
may reclaim more blocks than requested; the coarse unit keeps the eviction path from
ever splitting nodes.

Tradeoff: block-granular matching wastes up to block_size - 1 cached tokens per
prompt, but keeps insertion and copy-on-write trivially aligned and avoids partial
block bookkeeping in the allocator. With agent workloads (kilotoken shared system
prompts and tool schemas, short per-turn suffixes) the alignment loss is noise while
the reuse is most of the prompt.

## Attention backends

The torch backend implements paged prefill and decode with gathers plus dense
attention; it is the correctness reference and the CPU path. The Triton backend is a
single decode kernel: one program per (sequence, head), online softmax over block
tiles walked through the block table, GQA by head mapping, float32 accumulation. The
kernel's loop structure is transliterated line for line to a torch test that must
match the torch backend on CPU, and gpu-marked tests compare the kernel itself on CUDA
hosts. Routing is explicit: auto resolves to Triton only when CUDA and a Triton
install are both present, requesting triton without both is an error, and the decode
entry point dispatches per call, so CPU tensors always take the torch path.

## Server

The HTTP layer is one process with one AsyncLLMEngine. Streaming responses are
server-sent events cut at token boundaries by incremental detokenization; a consumer
that leaves the stream early, client disconnects included, aborts the request, which
frees its blocks and releases its radix references. Usage accounting
surfaces radix hits per request (prompt_tokens_details.cached_tokens), which is how
the benchmark harness measures hit rates without instrumenting the engine.
