"""INT8 post-training quantisation (Phase 21).

    uv run python scripts/quantize_int8.py
    uv run python scripts/build_engine.py --precision int8     # if supported

Produces models/<name>/<version>/model.int8.onnx in QDQ form.

WHAT QUANTISATION IS

    Store and compute in 8-bit integers instead of 32-bit floats. A tensor of
    floats is approximated by integers plus two parameters per tensor (or per
    channel):

        real  ~=  scale * (quantised - zero_point)

    `scale` maps the float range onto [-128, 127]; `zero_point` is the integer
    that represents real zero. Choosing them is the whole problem: too wide and
    you waste resolution on values that never occur, too narrow and real values
    clip.

PTQ vs QAT

    Post-training quantisation (PTQ, what this does) takes a trained FP32 model
    and picks scales by observing activations on a few hundred representative
    inputs. Cheap -- minutes, no labels, no training loop.

    Quantisation-aware training (QAT) inserts fake-quantise operations during
    training so the weights learn to be robust to the rounding. More accurate,
    and it costs a training run. For a 4% accuracy recovery on a model someone
    else trained, PTQ is usually where you stop.

WHY CALIBRATION DATA MUST BE REAL

    Scales are chosen from the activation ranges the calibration set produces.
    Calibrate on gaussian noise and you get scales fitted to activations that
    no real image produces, then clip on everything that matters. This script
    calibrates on crops of the sample photo and says so when it cannot.

QDQ FORMAT

    The output graph has QuantizeLinear/DequantizeLinear pairs around the
    tensors that should be 8-bit. That is a *description* of intended
    precision, not an implementation -- ONNX Runtime and TensorRT each decide
    how to honour it, and TensorRT in particular fuses QDQ pairs into its
    kernels rather than executing them literally. This is the same separation
    ONNX has everywhere: the graph says what, the runtime decides how.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.model.pytorch_model import ONNX_FILE
from src.preprocessing import ImagePreprocessor
from src.utils.tensor_info import human_bytes

INT8_FILE = "model.int8.onnx"
SAMPLE_IMAGE = Path("data/dog.jpg")
INPUT_NAME = "input"


def calibration_samples(image_size: int, count: int) -> tuple[list[np.ndarray], str]:
    """Representative inputs for choosing scales.

    Random crops at varying scale from the sample photo: distinct inputs with
    real texture and real activation statistics.
    """
    pre = ImagePreprocessor(image_size=image_size)
    rng = np.random.default_rng(0)

    if SAMPLE_IMAGE.is_file():
        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(SAMPLE_IMAGE)).convert("RGB")
        w, h = img.size
        out = []
        for _ in range(count):
            scale = float(rng.uniform(0.3, 1.0))
            cw, ch = max(int(w * scale), image_size), max(int(h * scale), image_size)
            x = int(rng.integers(0, w - cw + 1))
            y = int(rng.integers(0, h - ch + 1))
            out.append(pre._transform(img.crop((x, y, x + cw, y + ch)))[None, ...])
        return out, f"{count} crops of {SAMPLE_IMAGE}"

    return (
        [
            rng.standard_normal((1, 3, image_size, image_size), dtype=np.float32)
            for _ in range(count)
        ],
        f"{count} gaussian tensors -- NOT representative; scales will be wrong",
    )


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=128, help="calibration images")
    ap.add_argument(
        "--quantize-first-conv",
        action="store_true",
        help="also quantise the stem convolution (TensorRT has no INT8 kernel for it)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.preprocess import quant_pre_process
    except ImportError:
        print(
            "error: onnxruntime quantization tools unavailable. Run: uv sync --extra onnx",
            file=sys.stderr,
        )
        return 1

    source = settings.model_dir / ONNX_FILE
    target = settings.model_dir / INT8_FILE
    if not source.is_file():
        print(f"error: no {source}; run scripts/export_onnx.py first", file=sys.stderr)
        return 1
    if target.is_file() and not args.force:
        print(f"{target} already exists (use --force)")
        return 0

    # The stem convolution takes 3 input channels. INT8 Tensor Core kernels
    # want channel counts that are multiples of 4 (often 16), so TensorRT has
    # no INT8 implementation for a 64x3x7x7 convolution and the build fails
    # with "Could not find any implementation for node conv1...". Leaving it in
    # float is the standard remedy and costs very little: it is one layer of
    # 20, and the first layer is also where quantisation error propagates
    # furthest, so it is the layer you would keep in float anyway.
    exclude: list[str] = []
    if not args.quantize_first_conv:
        import onnx as _onnx

        graph = _onnx.load(str(source))
        convs = [n.name for n in graph.graph.node if n.op_type == "Conv"]
        if convs:
            exclude = [convs[0]]

    samples, description = calibration_samples(settings.image_size, args.samples)
    print(f"\n\033[1mINT8 quantisation\033[0m\n  calibrating on {description}")
    if exclude:
        print(f"  leaving in float: {', '.join(exclude)}  (stem conv, 3 input channels)")

    class Reader(CalibrationDataReader):
        """Feeds calibration inputs one at a time.

        The quantiser runs the graph on each and records the min/max seen at
        every tensor; those ranges become the scales.
        """

        def __init__(self, data: list[np.ndarray]) -> None:
            self._iter = iter({INPUT_NAME: d} for d in data)

        def get_next(self):
            return next(self._iter, None)

    # Shape inference and constant folding first. quantize_static needs static
    # shapes to place QDQ nodes, and our graph has a symbolic batch dimension.
    prepared = settings.model_dir / "model.prepared.onnx"
    try:
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  preprocessing failed ({exc}); quantising the raw graph instead")
        prepared = source

    try:
        quantize_static(
            model_input=str(prepared),
            model_output=str(target),
            calibration_data_reader=Reader(samples),
            # QDQ rather than QOperator: TensorRT understands QDQ and fuses the
            # pairs into its kernels. QOperator bakes in ORT-specific ops that
            # TensorRT cannot consume.
            quant_format=QuantFormat.QDQ,
            per_channel=True,  # per-output-channel scales for conv weights
            # Int8 symmetric, not ORT's default UInt8 asymmetric. TensorRT
            # requires zero_point == 0 outside DLA and rejects the graph with
            # "Non-zero zero point is not supported" otherwise. Symmetric
            # quantisation gives up the ability to centre the range on a
            # skewed distribution -- fine after ReLU, which is already
            # one-sided -- in exchange for dropping the zero-point subtraction
            # from every kernel.
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            # MinMax: scales span the full observed range, so nothing clips.
            # Entropy/percentile methods trade a little clipping for better
            # resolution on the bulk of the distribution, which matters more
            # for models with long activation tails than for a small convnet.
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
            extra_options={
                # ORT quantises biases to Int32 by default. TensorRT 11 rejects
                # a DequantizeLinear whose input is Int32 -- it accepts only
                # Int8/Int4/FP8/FP4 -- so the graph parses in ONNX Runtime and
                # fails in TensorRT with "input has type Int32". Leaving biases
                # in float costs almost nothing (they are one value per output
                # channel) and keeps the graph portable to both runtimes.
                "QuantizeBias": False,
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nerror: quantisation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "  Documented as a limitation rather than faked. See docs/experiments.md.",
            file=sys.stderr,
        )
        return 1
    finally:
        if prepared != source and prepared.is_file():
            prepared.unlink()

    import onnx

    graph = onnx.load(str(target))
    ops = {}
    for node in graph.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    print(f"\n  wrote {target} ({human_bytes(target.stat().st_size)})")
    print(f"  fp32 graph was {human_bytes(source.stat().st_size)}")
    print(f"  QuantizeLinear nodes   {ops.get('QuantizeLinear', 0)}")
    print(f"  DequantizeLinear nodes {ops.get('DequantizeLinear', 0)}")
    print(
        "\n  The QDQ pairs describe intended precision; they are not themselves the\n"
        "  implementation. TensorRT fuses them into its kernels rather than executing\n"
        "  them literally, which is why the file is not simply 4x smaller."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
