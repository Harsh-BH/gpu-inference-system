"""A small, reusable staged-pipeline runtime.

Nothing in this package knows about images, models, GPUs or HTTP. That is the
point: it is the part of this project you copy into the next one. The concrete
transformations live in `src/stages/`, and swapping one is a one-line change to
the list in `src/main.py`.

    from src.pipeline import Pipeline, Stage, StageSpec

    class Double(Stage[int, int]):
        name = "double"
        def process(self, items): return [i * 2 for i in items]

    pipeline = Pipeline([StageSpec(Double())])
    pipeline.start()
    completion = await pipeline.submit(21)   # -> Completion(result=42, ...)
    await pipeline.stop()

See `docs/pipeline.md` for the full guide.
"""

from src.pipeline.pipeline import Pipeline
from src.pipeline.stage import (
    Completion,
    Job,
    PipelineError,
    PipelineFull,
    PipelineNotRunning,
    Stage,
    StageContractError,
    StageOutput,
    StageReport,
    StageSpec,
)

__all__ = [
    "Completion",
    "Job",
    "Pipeline",
    "PipelineError",
    "PipelineFull",
    "PipelineNotRunning",
    "Stage",
    "StageContractError",
    "StageOutput",
    "StageReport",
    "StageSpec",
]
