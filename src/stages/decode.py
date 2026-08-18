"""Stage 1: upload bytes -> a normalised tensor.

WHY THIS IS A PER-ITEM STAGE WITH MANY WORKERS

    Decoding is the most expensive thing this system does on the CPU and the
    cheapest to parallelise. Phase 4 measured one thread sustaining ~46 img/s
    against an engine that absorbs 700-3200, and raising the pool from 4 to 16
    workers bought +33% end-to-end throughput with no GPU change at all.

    So: `StageSpec(ImageDecodeStage(...), workers=N, max_batch=1)`. Each worker
    handles one image. Batching would buy nothing here -- there is no shared
    setup cost across images to amortise -- and it would couple the fate of
    sixteen requests to one corrupt JPEG.

WHY IT OWNS NO DECODING CODE

    `ImagePreprocessor` already does this, is already pinned against
    torchvision's exact geometry by `tests/test_preprocessing.py`, and already
    refuses to import torch. This stage is an adapter: it maps the pipeline's
    batch-in/batch-out contract onto that object and converts a client-input
    failure into a per-item result rather than a batch-wide one.

    That is the whole job. If it grew a resize, it would be a second
    implementation of a thing this repo already got exactly right once.
"""

from __future__ import annotations

import numpy as np

from src.pipeline import Stage, StageOutput
from src.preprocessing import ImagePreprocessor, PreprocessingError


class ImageDecodeStage(Stage[bytes, np.ndarray]):
    """Raw upload bytes -> (C, H, W) float32, normalised.

    Thread-safe: `ImagePreprocessor` holds only precomputed constants, so N
    workers may share one instance.
    """

    name = "decode"

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self._preprocessor.output_shape

    def process(self, items: list[bytes]) -> StageOutput[np.ndarray]:
        """One bad image fails one request.

        `PreprocessingError` is always caused by client input -- corrupt bytes,
        an unsupported format, absurd declared dimensions -- so it is returned
        in that item's slot rather than raised. Raising would fail every other
        image in the same call for a fault none of them committed.
        """
        out: StageOutput[np.ndarray] = []
        for data in items:
            try:
                out.append(self._preprocessor.from_bytes(data))
            except PreprocessingError as exc:
                out.append(exc)
        return out
