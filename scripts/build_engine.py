"""Compile an ONNX graph into a TensorRT engine (Phase 8).

    uv run python scripts/build_engine.py                    # fp32
    uv run python scripts/build_engine.py --precision fp16
    uv run python scripts/build_engine.py --max-batch 32 --opt-batch 8

Writes models/<name>/<version>/<precision>/model.engine.

WHY BUILDING IS A SEPARATE STEP FROM SERVING

    This takes minutes. It benchmarks candidate kernels against each other on
    *this* GPU and picks winners. Doing that during server startup would mean
    a multi-minute cold start and a server that cannot boot without a GPU
    identical to the one it will run on. So: build once, ship the plan, load
    it in milliseconds.

THE PIECES, IN THE ORDER THEY APPEAR

    Logger      TensorRT talks back through this. Warnings here are worth
                reading -- unsupported ops and precision fallbacks appear as
                warnings, not errors.
    Builder     the compiler. Owns the tactic search.
    Network     the in-memory graph being compiled. Built by the parser.
    Parser      reads model.onnx and populates the Network. Where an
                unsupported operator shows up as a parse error naming the node.
    Config      how to compile: workspace budget, precision flags, profiles.
    Profile     the shape ranges to optimise for (min/opt/max). Required
                because our batch dimension is dynamic.
    Engine      the compiled result: chosen kernels, fused layers, a memory
                plan. Serialised to disk as a "plan".
    Runtime     deserialises a plan back into an engine at load time.
    Context     one engine can have several; each holds the activation memory
                for one in-flight inference. Concurrency lives here.

TACTICS

    For each layer TensorRT has multiple candidate implementations -- different
    tiling, different algorithms (direct, Winograd, implicit GEMM), different
    layouts. It runs them and times them. That is why the build is slow and why
    the result is specific to this GPU: the winner on an RTX 3050 is not
    necessarily the winner on an A100.

WHY TENSORRT 11 HAS NO FP16 FLAG

    Older TensorRT took a weakly-typed network plus BuilderFlag.FP16 and chose
    per-layer precisions itself. TensorRT 11 removed that: networks are
    STRONGLY_TYPED and precision comes from the graph's own dtypes. So FP16
    means compiling model.fp16.onnx, not passing a flag -- which is arguably
    more honest, since the precision of every tensor is now visible in the
    artifact rather than decided inside the compiler.

PORTABILITY, OR THE LACK OF IT

    An engine is tied to the GPU architecture, the TensorRT version and
    (largely) the driver it was built against. It is a build artifact, not a
    model. models/**/*.engine is gitignored for exactly this reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from src.config import Precision, get_settings
from src.model.pytorch_model import onnx_filename
from src.utils.tensor_info import human_bytes

ENGINE_FILE = "model.engine"
INPUT_NAME = "input"


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def build(
    onnx_path: Path,
    engine_path: Path,
    *,
    image_size: int,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_bytes: int,
    allow_tf32: bool,
    verbose: bool,
) -> bool:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # STRONGLY_TYPED is the only network kind TensorRT 11 builds. Precision is
    # whatever the ONNX graph says it is.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)

    section("Parsing")
    if not parser.parse(onnx_path.read_bytes()):
        # PRD failure case: unsupported ONNX operator. The parser names the
        # offending node, which is the only useful thing to print here.
        print(f"  failed to parse {onnx_path}", file=sys.stderr)
        for i in range(parser.num_errors):
            print(f"    {parser.get_error(i)}", file=sys.stderr)
        return False
    print(
        f"  {network.num_layers} layers, {network.num_inputs} input(s), "
        f"{network.num_outputs} output(s)"
    )
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input  {t.name:<8} {tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name:<8} {tuple(t.shape)} {t.dtype}")

    config = builder.create_builder_config()

    # Workspace is scratch space the tactic search may use. Too small and
    # faster tactics are silently skipped for lack of room; it is a budget,
    # not an allocation, so a generous number costs nothing if unused.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    # TF32 is on by default here too, exactly as in PyTorch and ONNX Runtime.
    # Third runtime, same silent default -- driven from ALLOW_TF32 so a
    # cross-runtime accuracy comparison is not secretly measuring this.
    if allow_tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)

    # An optimisation profile is the price of a dynamic batch dimension.
    # TensorRT specialises kernels for `opt` and guarantees correctness across
    # [min, max]. Shapes near opt run best; a wide range costs peak performance.
    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        (min_batch, 3, image_size, image_size),
        (opt_batch, 3, image_size, image_size),
        (max_batch, 3, image_size, image_size),
    )
    config.add_optimization_profile(profile)

    section("Building")
    print(f"  batch min={min_batch} opt={opt_batch} max={max_batch}")
    print(f"  workspace {human_bytes(workspace_bytes)}, TF32 {'on' if allow_tf32 else 'off'}")
    print("  timing candidate kernels on this GPU; this takes a few minutes ...")

    started = perf_counter()
    plan = builder.build_serialized_network(network, config)
    elapsed = perf_counter() - started

    if plan is None:
        print("  build failed; see TensorRT log above", file=sys.stderr)
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    print(f"  built in {elapsed:.1f}s -> {engine_path} ({human_bytes(engine_path.stat().st_size)})")
    return True


def inspect_engine(engine_path: Path) -> None:
    """Deserialise the plan and report what the compiler decided."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        print("  could not deserialise the engine", file=sys.stderr)
        return

    section("Engine")
    print(f"  I/O tensors        {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = "input " if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
        print(
            f"  {mode} {name:<8} {tuple(engine.get_tensor_shape(name))} "
            f"{engine.get_tensor_dtype(name)}"
        )
    print(f"  optimisation profiles {engine.num_optimization_profiles}")
    print(f"  device memory needed  {human_bytes(engine.device_memory_size_v2)}")
    print(
        "\n  That device memory is the activation working set the execution context\n"
        "  will allocate -- planned once at build time rather than negotiated with an\n"
        "  allocator on every call, which is one of the things TensorRT buys."
    )


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=settings.model_name)
    ap.add_argument("--version", default=settings.model_version)
    ap.add_argument("--precision", choices=[p.value for p in Precision], default="fp32")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument(
        "--opt-batch",
        type=int,
        default=8,
        help="the shape TensorRT specialises hardest for",
    )
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--workspace-mib", type=int, default=1024)
    ap.add_argument("--allow-tf32", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="full TensorRT build log")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        import tensorrt as trt
    except ImportError:
        print("error: tensorrt is not installed. Run: uv sync --extra trt", file=sys.stderr)
        return 1

    model_dir = settings.model_repository / args.name / args.version
    onnx_path = model_dir / onnx_filename(args.precision)
    engine_path = model_dir / args.precision / ENGINE_FILE

    if not onnx_path.is_file():
        print(
            f"error: no ONNX graph at {onnx_path}\n"
            f"  run: uv run python scripts/export_onnx.py --precision {args.precision}",
            file=sys.stderr,
        )
        return 1
    if engine_path.is_file() and not args.force:
        print(f"{engine_path} already exists (use --force to rebuild)")
        return 0

    print(f"\n\033[1mTensorRT {trt.__version__}\033[0m")
    print(f"  {onnx_path}  ->  {engine_path}")

    ok = build(
        onnx_path,
        engine_path,
        image_size=settings.image_size,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_bytes=args.workspace_mib * 1024 * 1024,
        allow_tf32=args.allow_tf32,
        verbose=args.verbose,
    )
    if not ok:
        return 1

    inspect_engine(engine_path)
    print(f"\n\033[32mEngine ready: {engine_path}\033[0m\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
