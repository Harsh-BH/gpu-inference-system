"""The concrete stages of *this* application's pipeline.

`src/pipeline/` is the reusable machinery and knows nothing about images.
Everything domain-specific lives here, which makes the boundary the one you cut
along when reusing this project as a template: keep `src/pipeline/`, replace
`src/stages/`, edit the list in `src/main.py`.

    bytes -> ImageDecodeStage -> (3,224,224) -> InferenceStage
          -> (1000,) logits    -> ClassifyStage -> [Prediction, ...]
"""

from src.stages.classify import ClassifyStage
from src.stages.decode import ImageDecodeStage
from src.stages.infer import InferenceStage

__all__ = ["ClassifyStage", "ImageDecodeStage", "InferenceStage"]
