"""Memory introspection tests.

The CPU-only paths are tested unconditionally: this module is imported by the
monitoring layer, and monitoring code that crashes on a machine without a GPU
is worse than no monitoring at all.
"""

import numpy as np
import pytest
import torch

from src.config import Settings
from src.gpu.memory import (
    MemorySnapshot,
    get_gpu_memory,
    get_peak_memory,
    memory_scope,
    reset_peak_memory,
)
from src.inference.pytorch_engine import PyTorchEngine
from src.model.pytorch_model import WEIGHTS_FILE

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --- degrades safely without a GPU ---------------------------------------


def test_cpu_device_returns_an_empty_snapshot():
    snap = get_gpu_memory("cpu")
    assert snap.available is False
    assert snap.allocated == 0
    assert str(snap) == "cuda unavailable"


def test_peak_helpers_are_noops_on_cpu():
    reset_peak_memory("cpu")
    assert get_peak_memory("cpu") == 0


def test_memory_scope_still_yields_on_cpu():
    with memory_scope("noop", "cpu") as scope:
        pass
    assert scope.allocated_delta == 0
    assert "cuda unavailable" in str(scope)


# --- snapshot arithmetic --------------------------------------------------


def test_derived_quantities():
    snap = MemorySnapshot(
        available=True,
        allocated=100,
        reserved=250,
        peak_allocated=180,
        peak_reserved=250,
        free=700,
        total=1000,
    )
    assert snap.used_on_device == 300
    assert snap.pool_overhead == 150  # reserved but not holding a live tensor
    assert snap.context_and_others == 50  # on-device usage the allocator cannot see


# --- real GPU behaviour ---------------------------------------------------


@cuda_only
def test_allocation_is_visible_and_released():
    before = get_gpu_memory().allocated
    x = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")  # exactly 4 MiB
    assert get_gpu_memory().allocated - before == 4 * 1024 * 1024
    del x
    assert get_gpu_memory().allocated == before


@cuda_only
def test_reserved_is_never_below_allocated():
    # The invariant that makes 'nvidia-smi disagrees with my code' explicable.
    x = torch.zeros(256, 1024, device="cuda")
    snap = get_gpu_memory()
    assert snap.reserved >= snap.allocated
    assert snap.pool_overhead >= 0
    del x


@cuda_only
def test_reserved_survives_freeing_the_tensor():
    """PyTorch does not cudaFree on delete -- it pools the block.

    This is why memory_allocated() dropping to zero does not make nvidia-smi
    drop to zero, and why empty_cache() exists at all.
    """
    x = torch.zeros(8 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    reserved_with = get_gpu_memory().reserved
    del x
    after = get_gpu_memory()
    assert after.allocated < reserved_with
    assert after.reserved == reserved_with  # still held from the driver


@cuda_only
def test_memory_scope_measures_transient_peak_not_just_the_delta():
    """The number that decides whether a batch fits is the peak, not the end state."""
    with memory_scope("transient") as scope:
        big = torch.zeros(16 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        del big  # gone before the scope exits

    assert scope.allocated_delta == 0  # kept nothing
    assert scope.peak_above_entry >= 16 * 1024 * 1024  # but needed 16 MiB to get there


@cuda_only
def test_reset_peak_isolates_consecutive_measurements():
    # Without a reset between blocks, every later peak is really the largest
    # peak ever seen, and a batch-size sweep reports a flat line.
    big = torch.zeros(32 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    del big

    reset_peak_memory()
    with memory_scope("small") as scope:
        small = torch.zeros(1024, dtype=torch.float32, device="cuda")
        del small
    assert scope.peak_above_entry < 32 * 1024 * 1024


@cuda_only
def test_model_weights_cost_what_the_arithmetic_says():
    s = Settings(_env_file=None)
    if not (s.model_dir / WEIGHTS_FILE).is_file():
        pytest.skip("model not provisioned")

    engine = PyTorchEngine(s)
    with memory_scope("load") as scope:
        engine.load()
    try:
        expected = 11_689_512 * 4  # resnet18 params x FP32
        # Within 2%: PyTorch pads allocations to allocator block granularity.
        assert scope.allocated_delta == pytest.approx(expected, rel=0.02)
    finally:
        engine.unload()


@cuda_only
def test_activation_memory_grows_with_batch_size():
    s = Settings(_env_file=None, max_batch_size=32)
    if not (s.model_dir / WEIGHTS_FILE).is_file():
        pytest.skip("model not provisioned")

    peaks = {}
    with PyTorchEngine(s) as engine:
        engine.warmup(3)
        for bs in (4, 32):
            with memory_scope(f"batch {bs}") as scope:
                engine.predict(np.zeros((bs, 3, 224, 224), dtype=np.float32))
            peaks[bs] = scope.peak_above_entry

    # Weights are fixed; activations are not. This is the entire reason a batch
    # size that passes testing can OOM under a traffic spike.
    assert peaks[32] > peaks[4] * 3
