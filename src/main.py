"""Application assembly and lifecycle (Phases 13, 14, 15, 17).

    uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

STARTUP ORDER, AND WHY IT IS THIS ORDER

    1. build the engine named by BACKEND        no GPU work yet
    2. load()                                    weights to VRAM, engine plan
    3. warmup()                                  pay initialisation costs here
    4. start the batch manager
    5. only now report ready

    Warmup before ready is the entire point of Phase 13. A cold engine's first
    forward pass pays lazy CUDA context creation, kernel module loading, cuDNN
    algorithm selection and allocator growth. Somebody has to pay that; it must
    not be the first user. Reporting ready before warmup would hand the whole
    bill to whoever arrives first.

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

    If the engine cannot load -- missing artifact, no GPU, incompatible
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
from concurrent.futures import ThreadPoolExecutor
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
from src.monitoring import get_metrics
from src.preprocessing import ImagePreprocessor
from src.queue import BatchManager, RequestQueue

logger = logging.getLogger("gpu-inference")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        stream=sys.stdout,
    )


def _record_batch(metrics, outcome) -> None:
    metrics.batch_size.observe(outcome.batch_size)
    metrics.batch_formation.observe(outcome.formation_ms / 1000.0)
    metrics.h2d.observe(outcome.h2d_ms / 1000.0)
    metrics.compute.observe(outcome.compute_ms / 1000.0)
    metrics.d2h.observe(outcome.d2h_ms / 1000.0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    metrics = app.state.metrics
    configure_logging(settings.log_level)

    app.state.ready = False
    app.state.startup_error = None
    app.state.engine = None
    app.state.queue = None
    app.state.batch_manager = None

    app.state.preprocessor = ImagePreprocessor(
        image_size=settings.image_size, max_pixels=settings.max_image_pixels
    )
    # Preprocessing is the measured bottleneck (Phase 4: ~46 img/s per thread
    # against an engine that absorbs 700-3200), so it gets a pool sized to
    # actually feed the GPU rather than a single worker.
    app.state.preprocess_pool = ThreadPoolExecutor(
        max_workers=settings.preprocess_workers, thread_name_prefix="preprocess"
    )

    logger.info(
        "starting: model=%s version=%s backend=%s precision=%s device=%s",
        settings.model_name,
        settings.model_version,
        settings.backend.value,
        settings.precision.value,
        settings.device,
    )

    try:
        app.state.labels = load_labels(settings.model_dir)
        engine = create_engine(settings)

        started = perf_counter()
        engine.load()
        logger.info("engine loaded in %.0f ms", (perf_counter() - started) * 1000)

        # Phase 13. The cost is paid here, before anyone is told we are ready.
        started = perf_counter()
        engine.warmup(settings.warmup_requests)
        logger.info(
            "warmed up with %d iterations in %.0f ms",
            settings.warmup_requests,
            (perf_counter() - started) * 1000,
        )

        app.state.engine = engine
        app.state.queue = RequestQueue(max_size=settings.queue_max_size)
        app.state.batch_manager = BatchManager(
            engine,
            app.state.queue,
            max_batch_size=settings.max_batch_size,
            max_batch_wait_ms=settings.max_batch_wait_ms,
            on_batch=lambda outcome: _record_batch(metrics, outcome),
        )
        app.state.batch_manager.start()

        app.state.ready = True
        metrics.model_loaded.set(1)
        logger.info(
            "ready: max_batch=%d wait=%.1fms queue=%d",
            settings.max_batch_size,
            settings.max_batch_wait_ms,
            settings.queue_max_size,
        )

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
        if app.state.batch_manager is not None:
            await app.state.batch_manager.stop()
        app.state.preprocess_pool.shutdown(wait=False, cancel_futures=True)
        if app.state.engine is not None:
            app.state.engine.unload()
        logger.info("stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Settings are injectable so tests do not need env vars."""
    settings = settings or get_settings()

    app = FastAPI(
        title="gpu-inference-system",
        description="Image classification over HTTP, with the GPU path made visible.",
        version="0.1.0",
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
