"""Application assembly and lifecycle (Phases 13, 14, 15, 17, 18).

    uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

THE PIPELINE IS THE ARCHITECTURE

    `build_pipeline()` below is the whole serving design, in one readable list.
    Reading it tells you what the system does, in what order, with how much
    concurrency, and where it will batch. Nothing else has to be consulted.

    That is the point of the rewrite. The previous version spread the same
    behaviour across an HTTP handler, a thread pool, a request queue and a
    batch manager, and the only way to know the shape was to read all four and
    hold them in your head at once.

    To adapt this project to a different problem: replace the stages in
    `src/stages/`, edit that list, and leave `src/pipeline/` alone.

STARTUP ORDER, AND WHY IT IS THIS ORDER

    `pipeline.start()` sets every stage up in order, then starts workers. For
    the inference stage, setup is load() + warmup(), so by the time any worker
    exists the engine is warm. Only then is readiness reported.

    Warmup before ready is the entire point of Phase 13. A cold engine's first
    forward pass pays lazy CUDA context creation, kernel module loading, cuDNN
    algorithm selection and allocator growth. Somebody has to pay that; it must
    not be the first user. Reporting ready before warmup hands the whole bill
    to whoever arrives first.

SHUTDOWN ORDER IS THE REVERSE, AND ALSO DELIBERATE

    Stop accepting, drain what is in flight, fail whatever remains with a clean
    error, then release the GPU. A shutdown that simply exits leaves every
    waiting client hanging until their own timeout expires.

MODEL VERSIONING (PHASE 14)

    Nothing here names a model. MODEL_NAME, MODEL_VERSION, BACKEND and
    PRECISION resolve to models/<name>/<version>/ and the engine that reads it.
    Serving v2 instead of v1 is an environment change and a restart. The active
    version is reported on /ready and in every prediction response, so the
    answer to "which model produced this" is never inferred.

A FAILED LOAD IS NOT A CRASH

    If a stage cannot set up -- missing artifact, no GPU, incompatible
    TensorRT plan -- the process still starts and serves /health and /metrics,
    with /ready returning 503 and the reason. That is what an operator needs:
    a container that stays up long enough to be inspected, and a readiness
    probe that keeps it out of rotation. Exiting instead produces a restart
    loop that hides the message.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router
from src.api.schemas import ErrorResponse
from src.config import Settings, get_settings
from src.inference import create_engine
from src.inference.base import EngineError
from src.model.pytorch_model import ModelArtifactError, load_labels
from src.monitoring import Metrics, get_metrics
from src.pipeline import Pipeline, StageSpec
from src.preprocessing import ImagePreprocessor
from src.stages import ClassifyStage, ImageDecodeStage, InferenceStage

logger = logging.getLogger("gpu-inference")

#: Ranked predictions are ~0.1 ms of work, so the classify stage batches
#: generously to vectorise softmax without ever being worth waiting for.
_CLASSIFY_MAX_BATCH = 64


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        stream=sys.stdout,
    )


def build_pipeline(
    settings: Settings, metrics: Metrics, labels: list[str]
) -> tuple[Pipeline, InferenceStage]:
    """Assemble the serving pipeline. Constructs; does not start or load.

    Returned alongside the inference stage because /ready and every prediction
    response need the engine's identity, and that is the one stage the HTTP
    layer legitimately has to know about.
    """
    decode = ImageDecodeStage(
        ImagePreprocessor(
            image_size=settings.image_size,
            max_pixels=settings.max_image_pixels,
            fast_decode=settings.fast_decode,
        )
    )
    infer = InferenceStage(
        create_engine(settings),
        warmup_requests=settings.warmup_requests,
        on_timings=metrics.record_engine_timings,
    )
    classify = ClassifyStage(labels)

    pipeline = Pipeline(
        [
            # CPU-bound, parallel, one image per call. Decoding is the most
            # expensive thing this system does on the CPU and the cheapest to
            # parallelise; PIL releases the GIL during decode and resize, which
            # is the only reason threads help here at all.
            StageSpec(
                decode,
                workers=settings.preprocess_workers,
                max_batch=1,
                capacity=settings.queue_max_size,
            ),
            # GPU-bound, serialised, batched. One CUDA context means a second
            # worker would serialise anyway; the batch is where throughput is
            # bought and tail latency is paid.
            StageSpec(
                infer,
                workers=1,
                max_batch=settings.max_batch_size,
                max_batch_wait_ms=settings.max_batch_wait_ms,
                capacity=settings.queue_max_size,
            ),
            # Microseconds of numpy. A thread hop would cost more than the work.
            StageSpec(
                classify,
                workers=0,
                max_batch=_CLASSIFY_MAX_BATCH,
                capacity=settings.queue_max_size,
            ),
        ],
        on_stage=metrics.record_stage,
    )
    return pipeline, infer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    metrics: Metrics = app.state.metrics
    configure_logging(settings.log_level)

    app.state.ready = False
    app.state.startup_error = None
    app.state.pipeline = None
    app.state.inference_stage = None

    logger.info(
        "starting: model=%s version=%s backend=%s precision=%s device=%s fast_decode=%s",
        settings.model_name,
        settings.model_version,
        settings.backend.value,
        settings.precision.value,
        settings.device,
        settings.fast_decode,
    )

    try:
        labels = load_labels(settings.model_dir)
        pipeline, inference_stage = build_pipeline(settings, metrics, labels)

        started = perf_counter()
        # Loads weights, warms the engine, then starts workers. Anything that
        # fails here fails before a single request can arrive.
        pipeline.start()
        logger.info("pipeline ready in %.0f ms", (perf_counter() - started) * 1000)

        app.state.pipeline = pipeline
        app.state.inference_stage = inference_stage
        app.state.ready = True
        metrics.model_loaded.set(1)

    except (EngineError, ModelArtifactError) as exc:
        # Stay up so /ready can explain why. See the module docstring.
        app.state.startup_error = str(exc)
        metrics.model_loaded.set(0)
        logger.error("startup failed, serving /health and /ready only: %s", exc)

    try:
        yield
    finally:
        logger.info("shutting down")
        app.state.ready = False
        metrics.model_loaded.set(0)
        if app.state.pipeline is not None:
            await app.state.pipeline.stop()
        logger.info("stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Settings are injectable so tests do not need env vars."""
    settings = settings or get_settings()

    app = FastAPI(
        title="gpu-inference-system",
        description="Image classification over HTTP, with the GPU path made visible.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = get_metrics()
    app.include_router(router)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=_code_for(exc.status_code), detail=str(exc.detail)
            ).model_dump(),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        # A missing or misnamed file field lands here. Say which, without
        # echoing the body back.
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="invalid_request",
                detail="expected a multipart upload with a 'file' field",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled(request, exc: Exception):
        """Last line of defence.

        The client gets a code and nothing else. Everything useful goes to the
        log, because a stack trace in an HTTP response is an information leak
        and is not actionable by whoever receives it.
        """
        logger.exception("unhandled error on %s", request.url.path)
        request.app.state.metrics.errors_total.labels(reason="internal").inc()
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error", detail="the server failed to process this request"
            ).model_dump(),
        )

    return app


def _code_for(status: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        422: "invalid_request",
        503: "unavailable",
        504: "timeout",
    }.get(status, "error")


app = create_app()
