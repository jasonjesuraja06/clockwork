"""Prometheus text-format metrics for the server, no client library."""

from __future__ import annotations

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class Histogram:
    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
                break
        else:
            self.counts[-1] += 1
        self.total += value
        self.count += 1

    def render(self, name: str) -> list[str]:
        lines = [f"# TYPE {name} histogram"]
        cumulative = 0
        for bound, count in zip(self.buckets, self.counts[:-1], strict=True):
            cumulative += count
            lines.append(f'{name}_bucket{{le="{_fmt(bound)}"}} {cumulative}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum {repr(self.total)}")
        lines.append(f"{name}_count {self.count}")
        return lines


class ServerMetrics:
    """Request counters, token counters, and latency histograms.

    The app runs on a single event loop, so plain integer updates need no lock.
    """

    def __init__(self):
        self.requests: dict[tuple[str, int], int] = {}
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.cached_tokens = 0
        self.ttft = Histogram(TTFT_BUCKETS)
        self.duration = Histogram(DURATION_BUCKETS)

    def observe_request(self, endpoint: str, status: int) -> None:
        key = (endpoint, status)
        self.requests[key] = self.requests.get(key, 0) + 1

    def observe_output(self, out, seconds: float) -> None:
        self.prompt_tokens += out.num_prompt_tokens
        self.generated_tokens += out.num_generated_tokens
        self.cached_tokens += out.num_cached_tokens
        self.duration.observe(seconds)

    def render(self, gauges: dict[str, float], info: dict[str, str]) -> str:
        lines = ["# TYPE clockwork_requests_total counter"]
        for (endpoint, status), count in sorted(self.requests.items()):
            lines.append(
                f'clockwork_requests_total{{endpoint="{_escape(endpoint)}",'
                f'status="{status}"}} {count}'
            )
        for name, value in (
            ("clockwork_prompt_tokens_total", self.prompt_tokens),
            ("clockwork_generated_tokens_total", self.generated_tokens),
            ("clockwork_prefix_cached_tokens_total", self.cached_tokens),
        ):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        lines += self.ttft.render("clockwork_ttft_seconds")
        lines += self.duration.render("clockwork_request_duration_seconds")
        for name, value in gauges.items():
            metric = f"clockwork_engine_{name}"
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {_fmt(value)}")
        labels = ",".join(f'{key}="{_escape(value)}"' for key, value in sorted(info.items()))
        lines.append("# TYPE clockwork_build_info gauge")
        lines.append(f"clockwork_build_info{{{labels}}} 1")
        return "\n".join(lines) + "\n"
