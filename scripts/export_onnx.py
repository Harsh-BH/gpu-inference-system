"""Export the PyTorch model to ONNX (Phase 6).

    uv run python scripts/export_onnx.py
    uv run python scripts/export_onnx.py --max-batch 64 --force

Writes models/<name>/<version>/model.onnx.

WHAT ONNX IS

    A serialised *computational graph*, plus the weights it needs. Not a
    program, not a runtime, not a file format for PyTorch — a description of
    the maths, in a form other runtimes can read.

    A .onnx file is a protobuf containing:

      inputs        named tensors the graph expects, with dtype and shape.
                    A dimension can be a fixed integer or a named symbol
                    ("batch"), which is how dynamic shapes are expressed.
      initializers  the constant tensors: every weight and bias. These are the
                    45 MB. They are separate from inputs because they never
                    change between calls.
      nodes         the operators, in topological order. Each names an op type
                    (Conv, Relu, Add, Gemm), its input tensor names and its
                    output tensor names. Nodes are wired together purely by
                    those names — that string matching *is* the graph.
      outputs       named tensors the caller gets back.

    Critically, an operator like `Conv` is a *specification*, not code. It says
    what the output must be given the inputs, padding and stride. It does not
    say which CUDA kernel computes it. That separation is the entire point:
    it is what lets ONNX Runtime and TensorRT each pick their own kernels.

ONNX IS NOT ONNX RUNTIME

    ONNX is the noun; ONNX Runtime is one program that can execute it. TensorRT
    can execute the same file by compiling it into something else entirely.
    Confusing the two is the most common misconception in this area, and Phase 7
    exists partly to make the distinction concrete.

WHY THE BATCH DIMENSION IS DYNAMIC

    Exported with a fixed batch of 1, this graph would serve exactly one image
    per call and dynamic batching (Phase 11) would be impossible — the batch
    manager assembles batches of whatever size happens to be waiting, which is
    rarely a constant.

    So dimension 0 is exported as the symbol "batch" rather than the integer 1.
    The cost is real: a runtime that knows a dimension exactly can specialise
    kernels and preallocate for it, and one that does not must either
    re-plan per shape or optimise for a range. TensorRT makes that trade
    explicit through optimisation profiles (Phase 8); ONNX Runtime handles it
    transparently but not for free.

    Height and width stay fixed at 224. They genuinely never vary here, and
    pretending otherwise would cost optimisation opportunity for nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.config import Settings
from src.model.pytorch_model import ModelArtifactError, load_from_repository
from src.utils.tensor_info import human_bytes

ONNX_FILE = "model.onnx"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def export(model: torch.nn.Module, path: Path, image_size: int, opset: int, max_batch: int):
    """Trace the module and write the graph.

    Exported on CPU in FP32 deliberately. The ONNX file is the *portable*
    artifact — one graph that ONNX Runtime and TensorRT then each specialise,
    including down to FP16. Baking a device or a precision into it here would
    mean re-exporting for every combination, and would make Phase 9's
    comparison test different graphs rather than different runtimes.
    """
    model = model.eval().cpu()
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    # torch 2.13 defaults to the dynamo exporter, which traces via torch.export
    # rather than TorchScript. Dim() declares which axis may vary and over what
    # range; the range matters because it is what a downstream compiler uses to
    # decide how much specialisation is safe.
    batch = torch.export.Dim("batch", min=1, max=max_batch)

    return torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: batch},),
        opset_version=opset,
        dynamo=True,
        # Default is True, which writes the weights to a sibling model.onnx.data
        # and leaves model.onnx a 95 KiB file that is useless on its own. At
        # 45 MB we are nowhere near protobuf's 2 GB ceiling, so a single
        # self-contained artifact is strictly better: nothing to lose track of
        # when copying it to a build machine, and one fewer way to ship a model
        # that loads but cannot run.
        external_data=False,
    )


def describe_graph(path: Path, requested_opset: int, torch_model: torch.nn.Module) -> bool:
    """Print what is actually in the file. This is the teaching payload.

    Returns False if the produced opset differs from the requested one.
    """
    import onnx

    model = onnx.load(str(path))

    section("Graph")
    print(f"  producer        {model.producer_name} {model.producer_version}")
    print(f"  IR version      {model.ir_version}")
    actual_opset = 0
    for imp in model.opset_import:
        if not imp.domain or imp.domain == "ai.onnx":
            actual_opset = imp.version
        print(f"  opset           {imp.version}  (domain {imp.domain or 'ai.onnx'})")

    opset_matches = actual_opset == requested_opset
    if not opset_matches:
        # Worth shouting about. The dynamo exporter emits opset 18 natively and
        # then tries to down-convert; when that fails it keeps 18 and carries on
        # without raising. Asking for 17 to satisfy a downstream compiler and
        # silently getting 18 is precisely the kind of mismatch that surfaces
        # later as an unparseable graph.
        print(
            f"\n  \033[33mNOTE: opset {requested_opset} was requested, opset "
            f"{actual_opset} was produced.\033[0m\n"
            "  The dynamo exporter emits opset 18 and down-converts only if it can;\n"
            "  a failed conversion is not an error. Verify downstream compatibility\n"
            f"  against {actual_opset}, not {requested_opset}."
        )

    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}

    def fmt_shape(value) -> str:
        dims = []
        for d in value.type.tensor_type.shape.dim:
            # dim_param is a *symbol*: the dimension is decided at run time.
            dims.append(d.dim_param if d.dim_param else str(d.dim_value))
        dtype = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        return f"[{', '.join(dims)}] {dtype.lower()}"

    section("Inputs and outputs")
    for value in graph.input:
        if value.name in initializer_names:
            continue  # weights are initializers, not caller-supplied inputs
        print(f"  input   {value.name:<10} {fmt_shape(value)}")
    for value in graph.output:
        print(f"  output  {value.name:<10} {fmt_shape(value)}")
    print(
        "\n  A symbolic dimension (a name rather than a number) is what makes this graph\n"
        "  usable by a dynamic batcher. A fixed 1 there would pin it to one image."
    )

    section("Nodes")
    ops = Counter(node.op_type for node in graph.node)
    print(f"  {len(graph.node)} nodes across {len(ops)} operator types\n")
    for op, count in ops.most_common():
        print(f"    {count:>4}  {op}")
    print(
        "\n  Each node names an op type and the tensor names it reads and writes. The\n"
        "  graph is those names lining up -- there are no pointers and no code here.\n"
        "  'Conv' specifies what the output must be, not which kernel produces it,\n"
        "  which is exactly what lets ONNX Runtime and TensorRT choose differently."
    )

    section("Initializers (the weights)")
    total = sum(init.raw_data.__len__() or 0 for init in graph.initializer)
    if total == 0:  # some producers use typed fields rather than raw_data
        import onnx.numpy_helper

        total = sum(onnx.numpy_helper.to_array(i).nbytes for i in graph.initializer)
    print(f"  {len(graph.initializer)} constant tensors, {human_bytes(total)}")
    print(f"  file on disk: {human_bytes(path.stat().st_size)}")
    print(
        "\n  Initializers are constants baked into the graph, kept separate from inputs\n"
        "  because they never change between calls. This is the bulk of the file."
    )

    # Graph optimisation has already happened, before any runtime sees this.
    section("Optimisation already applied at export")
    torch_bn = sum(
        1 for m in torch_model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    )
    onnx_bn = ops.get("BatchNormalization", 0)
    print(f"  BatchNorm layers in the PyTorch module   {torch_bn}")
    print(f"  BatchNormalization nodes in the graph    {onnx_bn}")
    if torch_bn and not onnx_bn:
        print(
            "\n  All of them are gone. In eval() mode BatchNorm is a fixed affine transform,\n"
            "  y = gamma*(x-mean)/sqrt(var+eps) + beta, and an affine transform applied to\n"
            "  a convolution's output can be folded into that convolution's weights and\n"
            "  bias. Same arithmetic, 20 fewer kernel launches and 20 fewer round trips\n"
            "  through VRAM for intermediate tensors.\n"
            "\n  This is operator fusion, and it has already happened here -- before ONNX\n"
            "  Runtime or TensorRT are involved at all. Worth knowing when attributing a\n"
            "  later speedup: some of what looks like the runtime's work was done at export."
        )
    return opset_matches


def validate(path: Path) -> bool:
    """Structural validation with ONNX's own checker."""
    import onnx

    section("Validation")
    try:
        onnx.checker.check_model(str(path), full_check=True)
        print("  onnx.checker.check_model  passed (full_check)")
    except onnx.checker.ValidationError as exc:
        print(f"  onnx.checker.check_model  FAILED: {exc}", file=sys.stderr)
        return False

    # Structural validity is necessary and not sufficient: a graph can be
    # perfectly well-formed and still compute the wrong thing, if the trace
    # captured something the eager model does not do.
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed; skipping numerical check")
        print("  install with: uv sync --extra onnx")
        return True

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {INPUT_NAME: np.zeros((1, 3, 224, 224), dtype=np.float32)})[0]
    print(f"  onnxruntime CPU run       ok, output {got.shape} {got.dtype}")
    return True


def check_numerical_equivalence(path: Path, model: torch.nn.Module, image_size: int) -> bool:
    """Does the exported graph compute what the PyTorch model computes?

    The check that actually matters. `check_model` verifies the file is
    well-formed; this verifies the export did not silently change the maths --
    a traced graph that captured training-mode BatchNorm, or dropped an
    operation the tracer could not see, passes the checker happily.

    Run on CPU with a fixed provider so this measures the *export*, not
    cuDNN's choice of convolution algorithm.
    """
    import onnxruntime as ort

    from src.benchmark import compare_logits

    section("Numerical equivalence vs PyTorch")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 3, image_size, image_size), dtype=np.float32)

    with torch.inference_mode():
        expected = model.eval().cpu()(torch.from_numpy(data)).numpy()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {INPUT_NAME: data})[0]

    agreement = compare_logits(expected, got)
    print(f"  {agreement}")
    ok = agreement.top1_agreement == 1.0 and agreement.max_abs_logit_diff < 1e-4
    print(
        "  export preserves the model's semantics"
        if ok
        else "  WARNING: exported graph disagrees with PyTorch"
    )
    return ok


def check_dynamic_batch(path: Path, image_size: int, sizes=(1, 3, 8)) -> bool:
    """Prove the symbolic dimension actually accepts varying batch sizes.

    The export declaring a dynamic axis and the graph accepting one are two
    different claims. Without this, a batch-size mismatch would first surface
    under load in Phase 11.
    """
    import onnxruntime as ort

    section("Dynamic batch")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ok = True
    for n in sizes:
        data = np.zeros((n, 3, image_size, image_size), dtype=np.float32)
        try:
            out = session.run(None, {INPUT_NAME: data})[0]
            print(f"  batch {n:>2}  ->  logits {out.shape}")
            ok &= out.shape[0] == n
        except Exception as exc:
            print(f"  batch {n:>2}  ->  FAILED: {exc}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    settings = get_settings_safe()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=settings.model_name)
    ap.add_argument("--version", default=settings.model_version)
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX operator set. 18 is what the dynamo exporter actually emits; "
        "asking for less is silently ignored when down-conversion fails.",
    )
    ap.add_argument(
        "--max-batch",
        type=int,
        default=64,
        help="upper bound declared for the dynamic batch dimension",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing model.onnx")
    args = ap.parse_args()

    model_dir = settings.model_repository / args.name / args.version
    path = model_dir / ONNX_FILE

    if path.is_file() and not args.force:
        print(f"{path} already exists (use --force to re-export)")
        return 0

    try:
        model = load_from_repository(model_dir, args.name)
    except ModelArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n\033[1mExporting\033[0m {args.name} {args.version} -> {path}")
    print(f"  opset {args.opset}, batch dimension dynamic in [1, {args.max_batch}]")

    try:
        export(model, path, settings.image_size, args.opset, args.max_batch)
    except Exception as exc:
        # PRD failure case: export failure must be a clear message, not a
        # traceback ending in the middle of a tracer.
        print(f"\nerror: ONNX export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    ok = describe_graph(path, args.opset, model)

    ok &= validate(path)
    ok &= check_numerical_equivalence(path, model, settings.image_size)
    ok &= check_dynamic_batch(path, settings.image_size)

    if ok:
        print(f"\n\033[32mExported and verified: {path}\033[0m\n")
        return 0
    print("\n\033[31mExport completed but verification failed.\033[0m\n", file=sys.stderr)
    return 1


def get_settings_safe() -> Settings:
    from src.config import get_settings

    return get_settings()


if __name__ == "__main__":
    raise SystemExit(main())
