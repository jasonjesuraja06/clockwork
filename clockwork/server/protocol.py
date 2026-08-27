"""Pydantic request and response models for the OpenAI compatible HTTP API."""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field


def random_id(prefix: str) -> str:
    """Return a fresh request id with the given OpenAI-style prefix."""
    return f"{prefix}-{uuid.uuid4().hex}"


def now() -> int:
    """Return the current unix timestamp for response created fields."""
    return int(time.time())


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    ignore_eos: bool = False


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails = Field(default_factory=PromptTokensDetails)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=now)
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: UsageInfo | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=now)
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=now)
    owned_by: str = "clockwork"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: int


class ErrorResponse(BaseModel):
    error: ErrorDetail
