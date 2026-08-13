"""Provision a model version into the local model repository.

    uv run python scripts/fetch_model.py

Writes:

    models/resnet18/v1/
        weights.pth     state_dict only (no pickled module, no code)
        labels.json     1000 ImageNet category names, index-aligned with logits

This is a *separate step* from serving on purpose. Downloading during load()
would mean a server that cannot start without the internet, and a "v1" that
silently changes whenever torchvision updates its default weights. Provisioning
once, explicitly, is what makes a version number mean something.

Later phases add model.onnx and fp16/model.engine to the same directory.
"""

from __future__ import annotations

import argparse
import json

import torch

from src.config import get_settings
from src.model.pytorch_model import (
    ARCHITECTURES,
    LABELS_FILE,
    WEIGHTS_FILE,
    download_pretrained,
)


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=settings.model_name, choices=sorted(ARCHITECTURES))
    ap.add_argument("--version", default=settings.model_version)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    model_dir = settings.model_repository / args.name / args.version
    weights_path = model_dir / WEIGHTS_FILE
    labels_path = model_dir / LABELS_FILE

    if weights_path.is_file() and labels_path.is_file() and not args.force:
        print(f"already provisioned: {model_dir}  (use --force to refetch)")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading pretrained {args.name} ...")
    model, categories = download_pretrained(args.name)

    # state_dict, not the module. Saving the module pickles the class and makes
    # loading it arbitrary code execution; a state_dict is just tensors.
    torch.save(model.state_dict(), weights_path)
    labels_path.write_text(json.dumps(categories, indent=0))

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {args.name} {args.version} -> {model_dir}")
    print(f"  parameters      {params:,}")
    print(f"  weights.pth     {weights_path.stat().st_size / 2**20:.1f} MiB on disk")
    print(f"  labels.json     {len(categories)} classes")
    # FP32 weights in VRAM will be ~4 bytes per parameter, which is the first
    # term of every memory estimate in Phase 3.
    print(f"  expect ~{params * 4 / 2**20:.1f} MiB of VRAM for FP32 weights alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
