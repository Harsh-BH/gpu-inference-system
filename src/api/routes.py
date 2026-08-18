"""HTTP endpoints (Phases 15, 17; moved onto the pipeline in Phase 18).

WHAT THIS FILE DOES, AND WHAT IT NO LONGER DOES

    read bytes (capped)  -> pipeline.submit(bytes) -> serialise the response

    That is the whole handler. It does not decode images, does not know that a
    queue exists, does not stack a batch, and does not know what softmax is.
    Every transformation lives in `src/stages/`, composed in `src/main.py`.

    Before, this file dispatched preprocessing to a thread pool, built an
    `InferenceRequest`, submitted it to a queue, awaited a Future and ran
    top-k. Five responsibilities in an HTTP handler, and the pipeline's shape
    was something you reconstructed by reading three modules. Now the handler
    has one responsibility -- translate between HTTP and the pipeline -- and
    the pipeline's shape is a list you can read in one screen.

FAILURE MAPPING

    Every failure becomes a status code chosen for what the client should do:

      400  bad image            do not retry, the input is wrong
      413  too large            do not retry, the input is too big
      503  pipeline full        retry after a moment, with Retry-After
      503  engine unavailable   retry, the server is not serving right now
      504  timed out            retry, the deadline passed
      500  anything else        a bug; the client learns nothing else

    500 responses carry a request_id and nothing more. The detail goes to the
    log against that id. A stack trace in an HTTP body is an information leak
    and tells the client nothing they can act on.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import (
    HealthResponse,
    LatencyBreakdown,
    PredictionItem,
    PredictResponse,
    ReadyResponse,
)
from src.inference.base import EngineError, EngineNotAvailableError
from src.pipeline import PipelineError, PipelineFull
from src.preprocessing import PreprocessingError

logger = logging.getLogger(__name__)
router = APIRouter()

_CHUNK = 64 * 1024


async def _read_capped(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing to buffer more than `limit` bytes.

    Content-Length is a claim, not a fact -- it can be absent under chunked
    encoding and can simply lie. Counting what actually arrives is the only
    limit that holds, and stopping at the first byte past the cap means a
    hostile client cannot make the server allocate what it refuses to accept.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"image exceeds the {limit} byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify one image",
)
async def predict(request: Request, file: Annotated[UploadFile, File()]) -> PredictResponse:
    state = request.app.state
    settings = state.settings
    metrics = state.metrics
    request_id = str(uuid.uuid4())
    started = perf_counter()

    if not state.ready:
        metrics.errors_total.labels(reason="not_ready").inc()
        raise HTTPException(status_code=503, detail="model is not loaded yet")

    data = await _read_capped(file, settings.max_upload_bytes)

    try:
        completion = await asyncio.wait_for(
            state.pipeline.submit(data, job_id=request_id),
            timeout=settings.request_timeout_ms / 1000.0,
        )
    except PipelineFull as exc:
        metrics.errors_total.labels(reason="queue_full").inc()
        metrics.requests_total.labels(status="rejected").inc()
        # 503 with Retry-After rather than 429: this is server capacity, not a
        # per-client rate limit, and the distinction matters to a load balancer.
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    except PreprocessingError as exc:
        # Client input problem: say what is wrong, it is safe and actionable.
        # Raised by the decode stage and delivered here through the job.
        metrics.errors_total.labels(reason="bad_image").inc()
        metrics.requests_total.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError:
        metrics.errors_total.labels(reason="timeout").inc()
        metrics.requests_total.labels(status="error").inc()
        raise HTTPException(
            status_code=504,
            detail=f"inference did not complete within {settings.request_timeout_ms:.0f} ms",
        ) from None
    except EngineNotAvailableError as exc:
        metrics.errors_total.labels(reason="engine_unavailable").inc()
        metrics.requests_total.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (EngineError, PipelineError) as exc:
        # Includes CUDA OOM, which is a capacity problem, not a client problem,
        # and a shutdown that stranded this request.
        logger.warning("request %s failed: %s", request_id, exc)
        metrics.errors_total.labels(reason="inference_failed").inc()
        metrics.requests_total.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="inference failed") from exc

    total_ms = (perf_counter() - started) * 1000.0
    metrics.total_latency.observe(total_ms / 1000.0)
    metrics.requests_total.labels(status="ok").inc()

    ranked = completion.result
    meta = state.inference_stage.metadata
    return PredictResponse(
        request_id=request_id,
        model_name=meta.model_name,
        model_version=meta.model_version,
        backend=meta.backend,
        precision=meta.precision,
        prediction=ranked[0].label,
        confidence=ranked[0].confidence,
        predictions=[
            PredictionItem(label=p.label, class_index=p.class_index, confidence=p.confidence)
            for p in ranked
        ],
        latency=LatencyBreakdown(
            stages=completion.stage_ms,
            waits=completion.wait_ms,
            queued_ms=completion.queued_ms,
            pipeline_ms=completion.total_ms,
            total_ms=total_ms,
        ),
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness")
async def health() -> HealthResponse:
    """Is the process alive?

    Answers without touching the model on purpose. A liveness probe that
    depends on the GPU will report failure during a slow model load and get the
    container killed mid-startup, producing a restart loop that looks exactly
    like a crash.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, summary="Readiness")
async def ready(request: Request):
    """Can this process serve a request right now?

    Returns 503 when not, so a load balancer removes it from rotation instead
    of restarting it.
    """
    state = request.app.state
    settings = state.settings
    pipeline = getattr(state, "pipeline", None)
    is_ready = bool(state.ready and pipeline is not None and pipeline.is_running)

    if is_ready:
        meta = state.inference_stage.metadata
        return ReadyResponse(
            ready=True,
            model_name=meta.model_name,
            model_version=meta.model_version,
            backend=meta.backend,
            precision=meta.precision,
            math_mode=meta.math_mode,
            device=meta.device,
            max_batch_size=meta.max_batch_size,
            queue_depth=pipeline.depth,
            stage_depths=pipeline.depths,
        )

    body = ReadyResponse(
        ready=False,
        model_name=settings.model_name,
        model_version=settings.model_version,
        backend=settings.backend.value,
        precision=settings.precision.value,
        math_mode="unknown",
        device=settings.device,
        max_batch_size=settings.max_batch_size,
        queue_depth=0,
        stage_depths={},
        detail=getattr(state, "startup_error", None) or "engine is still loading",
    )
    return JSONResponse(status_code=503, content=body.model_dump())


@router.get("/metrics", summary="Prometheus metrics")
async def metrics_endpoint(request: Request) -> PlainTextResponse:
    state = request.app.state
    # Sampled per scrape, not per request: NVML costs 0.1-1 ms, which is
    # irrelevant every 15 seconds and ruinous at 3000 requests per second.
    state.metrics.observe_gpu(state.settings.device)
    pipeline = getattr(state, "pipeline", None)
    if pipeline is not None and pipeline.is_running:
        state.metrics.observe_depths(pipeline.depths)
    return PlainTextResponse(
        generate_latest(state.metrics.registry), media_type=CONTENT_TYPE_LATEST
    )
