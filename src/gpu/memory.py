"""VRAM introspection (Phase 3).

WHAT
    A snapshot of GPU memory from both points of view that matter, plus a
    context manager for attributing memory growth to a specific block of code.

WHY THERE ARE TWO POINTS OF VIEW

    Ask "how much GPU memory am I using" and there are three different correct
    answers. Conflating them is the single most common reason people cannot
    explain their own OOM.

    allocated   Bytes in tensors that are alive right now. What your code owns.
                Drops the instant a tensor is garbage collected.

    reserved    Bytes PyTorch's caching allocator holds from the CUDA driver.
                Always >= allocated. When a tensor is freed, PyTorch does NOT
                call cudaFree -- cudaMalloc/cudaFree are expensive and they
                synchronise the device, so freeing per-tensor would serialise
                the whole pipeline. It keeps the block in a pool instead.

    free/total  The driver's view of the physical device, across every process
                on it. This is what nvidia-smi shows, and it includes your CUDA
                context, your reserved pool, and other people's memory --
                the desktop compositor, someone else's training job.

    So: `nvidia-smi` reporting 1.2 GB while `memory_allocated()` reports 45 MB
    is not a bug. It is context + pool + everyone else.

WHY THIS DISTINCTION CAUSES REAL OUTAGES

    You can get CUDA OOM while `free` shows gigabytes available. The allocator
    needs a *contiguous* block of the requested size; a pool fragmented by a
    long tail of differently-sized allocations may have no single block big
    enough. Dynamic batching makes this worse, because batch 3, 7 and 8 all
    request different sizes. Phase 11 is where that stops being theoretical.

WHY INFERENCE USES SO MUCH LESS MEMORY THAN TRAINING

    Under torch.inference_mode() there is no autograd graph, so an intermediate
    activation is freed as soon as the next layer has consumed it. Peak memory
    is roughly the largest few tensors alive at once, not the sum of all of
    them. Training must retain every activation for the backward pass, which is
    why the same model and batch size can need an order of magnitude more VRAM
    to train than to serve.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.utils.tensor_info import human_bytes


def _index(device: str | int | torch.device | None) -> int | None:
    """Normalise a device spec to a CUDA ordinal, or None if this is not CUDA.

    Returning None rather than raising keeps every function here safe to call
    on a CPU-only box; the memory report degrades to zeros instead of the
    monitoring code becoming the thing that crashes the server.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """GPU memory at one instant, from both the allocator's and driver's view."""

    available: bool  # False on CPU-only; every field below is then 0
    allocated: int  # live tensors
    reserved: int  # allocator pool held from the driver
    peak_allocated: int  # high-water mark since the last reset
    peak_reserved: int
    free: int  # driver: unused on the physical device
    total: int  # driver: device capacity

    @property
    def used_on_device(self) -> int:
        """Everything occupying VRAM, including other processes and our context."""
        return self.total - self.free

    @property
    def pool_overhead(self) -> int:
        """Reserved but not currently holding a live tensor.

        Healthy in steady state -- it is the allocator avoiding cudaMalloc. A
        large and *growing* value is the signature of fragmentation.
        """
        return self.reserved - self.allocated

    @property
    def context_and_others(self) -> int:
        """VRAM used on the device that our allocator does not account for.

        Dominated by the CUDA context: the driver's per-process working set of
        kernel code, constant memory and internal buffers. It is charged the
        moment a process first touches CUDA, before a single tensor exists,
        and it is invisible to memory_allocated(). On a small card it is a
        substantial fraction of capacity, which is why this project reports it
        rather than pretending VRAM starts empty.
        """
        return max(0, self.used_on_device - self.reserved)

    def __str__(self) -> str:
        if not self.available:
            return "cuda unavailable"
        return (
            f"allocated {human_bytes(self.allocated):>10} | "
            f"reserved {human_bytes(self.reserved):>10} | "
            f"device {human_bytes(self.used_on_device):>10} / {human_bytes(self.total)}"
        )


_EMPTY = MemorySnapshot(False, 0, 0, 0, 0, 0, 0)


def get_gpu_memory(device: str | int | torch.device | None = None) -> MemorySnapshot:
    """Current memory state. Safe to call when CUDA is absent."""
    idx = _index(device)
    if idx is None:
        return _EMPTY
    free, total = torch.cuda.mem_get_info(idx)
    return MemorySnapshot(
        available=True,
        allocated=torch.cuda.memory_allocated(idx),
        reserved=torch.cuda.memory_reserved(idx),
        peak_allocated=torch.cuda.max_memory_allocated(idx),
        peak_reserved=torch.cuda.max_memory_reserved(idx),
        free=free,
        total=total,
    )


def get_peak_memory(device: str | int | torch.device | None = None) -> int:
    """High-water mark of live tensors since the last reset_peak_memory().

    This, not the current allocation, is the number that decides whether a
    batch size fits: OOM happens at the peak, and the peak is transient.
    """
    idx = _index(device)
    return 0 if idx is None else torch.cuda.max_memory_allocated(idx)


def reset_peak_memory(device: str | int | torch.device | None = None) -> None:
    """Zero the high-water marks so the next block is measured in isolation.

    Without this, every peak reading after the first is really the peak of the
    largest batch you have ever run, and a batch-size sweep reports a flat line.
    """
    idx = _index(device)
    if idx is not None:
        torch.cuda.reset_peak_memory_stats(idx)


def empty_cache(device: str | int | torch.device | None = None) -> None:
    """Return unused pooled blocks to the driver.

    Costs a device synchronise and makes the next allocation slower, so it does
    not belong on the request path. It belongs in exactly two places: after
    unloading a model, and while recovering from OOM.
    """
    if _index(device) is not None:
        torch.cuda.empty_cache()


@dataclass(slots=True)
class MemoryScope:
    """Result of a `memory_scope` block."""

    label: str
    before: MemorySnapshot
    after: MemorySnapshot
    peak_allocated: int

    @property
    def allocated_delta(self) -> int:
        return self.after.allocated - self.before.allocated

    @property
    def reserved_delta(self) -> int:
        return self.after.reserved - self.before.reserved

    @property
    def peak_above_entry(self) -> int:
        """Transient headroom the block needed beyond what it kept.

        For an inference call this is the activation memory: allocated at the
        start of the forward pass, gone by the end, and the reason a batch size
        that "fits" by weight size still OOMs.
        """
        return max(0, self.peak_allocated - self.before.allocated)

    def __str__(self) -> str:
        if not self.after.available:
            return f"{self.label:<26} cuda unavailable"
        return (
            f"{self.label:<26} "
            f"allocated {_signed(self.allocated_delta):>12}  "
            f"reserved {_signed(self.reserved_delta):>12}  "
            f"peak +{human_bytes(self.peak_above_entry):>10}"
        )


def _signed(n: int) -> str:
    return f"{'+' if n >= 0 else '-'}{human_bytes(abs(n))}"


@contextmanager
def memory_scope(
    label: str, device: str | int | torch.device | None = None
) -> Iterator[MemoryScope]:
    """Attribute memory growth and transient peak to one block of code.

        with memory_scope("model load") as m:
            engine.load()
        print(m)   # model load   allocated +44.59 MiB  reserved +46.00 MiB ...

    Synchronises on entry and exit. Memory is only meaningful at a point where
    the GPU has actually finished: reading it mid-stream measures whichever
    kernels happen to have completed, which is not reproducible.
    """
    idx = _index(device)
    if idx is None:
        empty = _EMPTY
        yield MemoryScope(label, empty, empty, 0)
        return

    torch.cuda.synchronize(idx)
    reset_peak_memory(idx)
    before = get_gpu_memory(idx)
    scope = MemoryScope(label, before, before, before.allocated)
    try:
        yield scope
    finally:
        torch.cuda.synchronize(idx)
        scope.after = get_gpu_memory(idx)
        scope.peak_allocated = torch.cuda.max_memory_allocated(idx)


def limit_process_memory(fraction: float, device: str | int | torch.device | None = None) -> None:
    """Cap this process's allocator at `fraction` of total VRAM.

    Used by the OOM experiment so that deliberately exhausting memory hits our
    own ceiling instead of the physical device. On a machine whose display is
    driven by the same GPU, starving the driver of VRAM can hang the compositor
    -- the experiment is supposed to demonstrate OOM handling, not to take the
    desktop down with it.

    It is also a real production technique: it keeps one greedy model from
    denying memory to everything else sharing the card.
    """
    idx = _index(device)
    if idx is not None:
        torch.cuda.set_per_process_memory_fraction(fraction, idx)
