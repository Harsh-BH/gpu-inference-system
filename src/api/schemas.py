"""Request and response shapes (Phase 15).

Pydantic models rather than bare dicts because the response is a contract. A
typo in a dict key is a silent client break; a typo here fails at import.

Every response carries the model name, version, backend and precision that
produced it. A prediction without that context is unauditable -- when accuracy
changes after a deploy, the first question is always "which engine answered
this", and the answer has to be in the response rather than inferred from
whatever was configured at the time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """One ranked class."""

    label: str = Field(description="Human-readable class name")
    class_index: int = Field(description="Index into the model's output layer")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability")


class LatencyBreakdown(BaseModel):
    """Where this request's time went, in milliseconds.

    Returned to the client on purpose. It costs nothing, and it makes "the API
    is slow" a question anyone can answer without server access: a large
    queue_wait_ms means the server is saturated, a large preprocess_ms means
    the image was large.
    """

    preprocess_ms: float
    queue_wait_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    backend: str
    precision: str
    prediction: str = Field(description="Top-1 label")
    confidence: float
    predictions: list[PredictionItem] = Field(description="Top-k, ranked")
    latency: LatencyBreakdown

    # `model_` is reserved by pydantic; this domain genuinely is models.
    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """What a client gets when something fails.

    Deliberately free of internal detail: no stack traces, no file paths, no
    exception types. Those go to the server log with the request_id, which is
    the only thing linking the two. Leaking internals to a client is both an
    information disclosure and a support burden.
    """

    error: str = Field(description="Short machine-readable code")
    detail: str = Field(description="Safe human-readable explanation")
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Liveness. True whenever the process can answer at all."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Readiness: can this process serve a request right now?

    Distinct from health on purpose. A process that is alive but still loading
    a model is healthy and not ready, and a load balancer needs to tell those
    apart -- restarting it would be wrong, routing traffic to it would also be
    wrong.
    """

    ready: bool
    model_name: str
    model_version: str
    backend: str
    precision: str
    math_mode: str
    device: str
    max_batch_size: int
    queue_depth: int
    detail: str | None = None

    model_config = {"protected_namespaces": ()}
