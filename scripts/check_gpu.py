"""Phase 0 gate: prove the GPU stack is real before writing inference code.

Run this first. If it fails, nothing downstream is worth debugging.

Deliberately standalone — it imports nothing from src/. The environment check
must not depend on application config, because "is my config broken" and "is my
CUDA broken" are different questions and you want to answer them separately.

    uv run python scripts/check_gpu.py

Exits non-zero when CUDA is unavailable, so it doubles as a boot/CI gate.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time

W = 26  # label column width


def row(label: str, value: object, note: str = "") -> None:
    if note:
        print(f"  {label:<{W}} {str(value):<22} {note}")
    else:
        print(f"  {label:<{W}} {value}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def gib(n: int | float) -> str:
    return f"{n / 2**30:.2f} GiB"


# Compute capability -> (architecture, Tensor Core generation, what those cores accept).
# Matters because it decides which of the later phases are even possible: no
# Tensor Cores means FP16 buys you memory bandwidth but not much math throughput.
_ARCH: dict[tuple[int, int], tuple[str, str, str]] = {
    (6, 0): ("Pascal", "none", "-"),
    (6, 1): ("Pascal", "none", "-"),
    (7, 0): ("Volta", "1st gen", "FP16"),
    (7, 2): ("Volta", "1st gen", "FP16"),
    (7, 5): ("Turing", "2nd gen", "FP16, INT8, INT4"),
    (8, 0): ("Ampere", "3rd gen", "FP16, BF16, TF32, INT8"),
    (8, 6): ("Ampere", "3rd gen", "FP16, BF16, TF32, INT8"),
    (8, 7): ("Ampere", "3rd gen", "FP16, BF16, TF32, INT8"),
    (8, 9): ("Ada Lovelace", "4th gen", "FP8, FP16, BF16, TF32, INT8"),
    (9, 0): ("Hopper", "4th gen", "FP8, FP16, BF16, TF32, INT8"),
    (10, 0): ("Blackwell", "5th gen", "FP4, FP8, FP16, BF16, INT8"),
    (12, 0): ("Blackwell", "5th gen", "FP4, FP8, FP16, BF16, INT8"),
}


def driver_version() -> str:
    """Driver version via nvidia-smi. PyTorch exposes the CUDA *build* version
    it was compiled against, not the driver actually loaded — and TensorRT
    compatibility is decided by the driver, so it is worth printing."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return "unknown (nvidia-smi not on PATH)"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, OSError, IndexError):
        return "unknown"


def report_software(torch) -> None:
    section("Software")
    row("PyTorch", torch.__version__)
    row("Built against CUDA", torch.version.cuda or "CPU-only build")
    row("cuDNN", torch.backends.cudnn.version() or "-")
    row("NVIDIA driver", driver_version())
    row("Python", sys.version.split()[0])


def report_device(torch, idx: int) -> None:
    p = torch.cuda.get_device_properties(idx)
    cc = (p.major, p.minor)
    arch, tc_gen, tc_types = _ARCH.get(cc, ("unknown", "unknown", "unknown"))

    section(f"Device {idx}")
    row("Name", p.name)
    row("Compute capability", f"{p.major}.{p.minor}", f"({arch})")
    row("Streaming multiprocessors", p.multi_processor_count, "SMs — the units that run blocks")
    row("Tensor Cores", tc_gen, f"accepts {tc_types}")
    row("BF16 supported", torch.cuda.is_bf16_supported())
    row("TF32 available", cc >= (8, 0), "SM 8.0+ only")

    free, total = torch.cuda.mem_get_info(idx)
    section("VRAM")
    row("Total", gib(total))
    row("Free", gib(free))
    row("In use (other processes)", gib(total - free), "desktop compositor, browsers, ...")
    row("Torch allocated", gib(torch.cuda.memory_allocated(idx)), "tensors we own")
    row("Torch reserved", gib(torch.cuda.memory_reserved(idx)), "held by the caching allocator")


def report_backends() -> None:
    """Which inference runtimes are installed. Missing ones are not errors —
    they are optional extras, and the server is designed to run without them."""
    section("Inference backends")
    row("pytorch", "available", "core dependency")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        cuda_ok = "CUDAExecutionProvider" in providers
        row("onnxruntime", ort.__version__, f"CUDA EP: {'yes' if cuda_ok else 'NO — CPU only'}")
    except ImportError:
        row("onnxruntime", "not installed", "uv sync --extra onnx")

    try:
        import tensorrt as trt

        row("tensorrt", trt.__version__)
    except ImportError:
        row("tensorrt", "not installed", "uv sync --extra trt")


def report_kernel_execution(torch, idx: int) -> None:
    """Run a real kernel — and demonstrate the single most important GPU
    benchmarking trap while we are here.

    CUDA kernel launches are ASYNCHRONOUS. `torch.matmul` returns to Python as
    soon as the work is *queued* on the stream, not when the GPU has finished
    it. So a naive time.perf_counter() around a GPU op measures how long it took
    to enqueue the work — often microseconds — and reports a speedup that does
    not exist.

    torch.cuda.synchronize() blocks the CPU until the stream drains. Every
    timing number in this project is taken with an explicit synchronize for
    exactly this reason.
    """
    section("Kernel execution")
    a = torch.randn(2048, 2048, device=f"cuda:{idx}")
    torch.matmul(a, a)
    torch.cuda.synchronize(idx)  # discard warmup: first launch pays module load

    t0 = time.perf_counter()
    c = torch.matmul(a, a)
    lie = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize(idx)
    truth = (time.perf_counter() - t0) * 1000

    row("Result finite", bool(c.sum().isfinite()), "kernels really ran")
    row("Timed without sync", f"{lie:.3f} ms", "<- WRONG: only measures the launch")
    row("Timed with sync", f"{truth:.3f} ms", "<- actual GPU execution")
    if lie > 0:
        row("Apparent 'speedup'", f"{truth / lie:.0f}x", "the bug this project avoids")


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run: uv sync", file=sys.stderr)
        return 1

    report_software(torch)

    if not torch.cuda.is_available():
        section("CUDA")
        row("Available", False)
        print(
            "\n  CUDA is unavailable. Common causes:\n"
            "    - a CPU-only PyTorch wheel was installed\n"
            "    - the NVIDIA kernel module is not loaded (check `nvidia-smi`)\n"
            "    - the driver is older than the CUDA runtime PyTorch was built against\n"
            "\n  The system can still run with DEVICE=cpu, but every GPU phase is a no-op.",
            file=sys.stderr,
        )
        return 1

    section("CUDA")
    row("Available", True)
    row("Visible devices", torch.cuda.device_count())

    for idx in range(torch.cuda.device_count()):
        report_device(torch, idx)

    report_backends()
    report_kernel_execution(torch, 0)
    print("\n\033[32mGPU stack OK.\033[0m\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
