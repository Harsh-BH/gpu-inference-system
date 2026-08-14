"""Model *definition* and artifact loading — separate from model *execution*.

WHY this is not inside pytorch_engine.py

    Three different things need to know what "resnet18 v1" means:

      - scripts/fetch_model.py, to download and lay down the artifacts
      - PyTorchEngine, to reconstruct the graph and load weights
      - scripts/export_onnx.py (Phase 6), to trace the module

    If the architecture table lived in the engine, the ONNX exporter would have
    to import a serving component to get a plain nn.Module. Keeping definition
    separate from execution is what lets the conversion pipeline
    (PyTorch -> ONNX -> TensorRT) be a straight line rather than a circle.

WHY weights are read from models/<name>/<version>/ and not downloaded on demand

    A server that reaches out to the internet during startup is a server that
    fails to start when the network hiccups, and whose "v1" silently becomes
    whatever torchvision ships this month. Provisioning is an explicit,
    separate step (scripts/fetch_model.py). load() reads local files or fails
    with an actionable message. That is also what makes Phase 14 versioning
    real rather than decorative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models

WEIGHTS_FILE = "weights.pth"
LABELS_FILE = "labels.json"
ONNX_FILE = "model.onnx"
ONNX_FP16_FILE = "model.fp16.onnx"


def onnx_filename(precision: str) -> str:
    """Which ONNX graph corresponds to a precision.

    FP16 is a *different file*, not a runtime flag. TensorRT 11 only builds
    strongly-typed networks -- it takes precision from the graph's own dtypes
    and no longer accepts BuilderFlag.FP16 -- and ONNX Runtime never converted
    precision at load time either. One graph per precision is therefore the
    only honest model, and naming it in one place stops the exporter and the
    two runtimes from disagreeing about where it lives.
    """
    return ONNX_FP16_FILE if precision == "fp16" else ONNX_FILE


@dataclass(frozen=True, slots=True)
class ArchSpec:
    """Everything needed to rebuild an architecture and fetch its weights."""

    builder: object  # callable(weights=..., num_classes=...) -> nn.Module
    pretrained_weights: object  # torchvision Weights enum member
    num_classes: int


# A lookup table, not a plugin system. Adding MobileNetV3-Small later is one
# line here; anything more elaborate would be a factory for one product.
ARCHITECTURES: dict[str, ArchSpec] = {
    "resnet18": ArchSpec(
        builder=models.resnet18,
        pretrained_weights=models.ResNet18_Weights.IMAGENET1K_V1,
        num_classes=1000,
    ),
}


class ModelArtifactError(FileNotFoundError):
    """A required artifact is missing or unreadable. Message says how to fix it."""


def arch_spec(name: str) -> ArchSpec:
    try:
        return ARCHITECTURES[name]
    except KeyError:
        raise ModelArtifactError(
            f"unknown architecture {name!r}; known: {sorted(ARCHITECTURES)}"
        ) from None


def build_architecture(name: str) -> nn.Module:
    """Construct the graph with *random* weights. No download, no network."""
    spec = arch_spec(name)
    return spec.builder(weights=None, num_classes=spec.num_classes)


def download_pretrained(name: str) -> tuple[nn.Module, list[str]]:
    """Fetch pretrained weights and category names from torchvision.

    Only scripts/fetch_model.py calls this. The serving path never does.
    """
    spec = arch_spec(name)
    model = spec.builder(weights=spec.pretrained_weights)
    categories = list(spec.pretrained_weights.meta["categories"])
    return model, categories


def load_from_repository(model_dir: Path, name: str) -> nn.Module:
    """Rebuild the architecture and load local weights onto it (CPU).

    weights_only=True is not optional. torch.load() deserialises with pickle by
    default, which executes arbitrary code in the file — a remote-code-execution
    hole if a model artifact is ever fetched from anywhere untrusted. Loading
    tensors only closes it.
    """
    path = model_dir / WEIGHTS_FILE
    if not path.is_file():
        raise ModelArtifactError(
            f"no weights at {path}\n  run: uv run python scripts/fetch_model.py --name {name}"
        )
    model = build_architecture(name)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except Exception as exc:
        raise ModelArtifactError(f"could not load weights from {path}: {exc}") from exc
    return model


def load_labels(model_dir: Path) -> list[str]:
    """Human-readable class names, index-aligned with the output logits."""
    path = model_dir / LABELS_FILE
    if not path.is_file():
        raise ModelArtifactError(
            f"no labels at {path}\n  run: uv run python scripts/fetch_model.py"
        )
    try:
        labels = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ModelArtifactError(f"{path} must contain a JSON array of strings")
    return labels
