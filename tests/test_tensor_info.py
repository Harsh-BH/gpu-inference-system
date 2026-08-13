"""describe() has a numpy branch and a torch branch. Both need to be right,
because these numbers are what the VRAM estimates in later phases are built on.
"""

import numpy as np
import torch

from src.utils.tensor_info import describe, human_bytes


def test_numpy_array():
    info = describe(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert info.shape == (1, 3, 224, 224)
    assert info.dtype == "float32"
    assert info.device == "cpu"
    assert info.numel == 150_528
    assert info.nbytes == 602_112
    assert info.contiguous


def test_torch_cpu_tensor_agrees_with_numpy():
    info = describe(torch.zeros(1, 3, 224, 224, dtype=torch.float32))
    assert info.shape == (1, 3, 224, 224)
    assert info.dtype == "float32"  # 'torch.' prefix stripped
    assert info.device == "cpu"
    assert info.nbytes == 602_112


def test_fp16_is_exactly_half_the_bytes():
    # The entire memory argument for FP16 in Phase 5, in one assertion.
    fp32 = describe(torch.zeros(8, 3, 224, 224, dtype=torch.float32))
    fp16 = describe(torch.zeros(8, 3, 224, 224, dtype=torch.float16))
    assert fp16.nbytes * 2 == fp32.nbytes


def test_non_contiguous_is_detected():
    # A transposed view. Flagged because it forces cudaMemcpy to stage through
    # a temporary instead of doing a straight DMA.
    assert not describe(np.zeros((4, 8)).T).contiguous
    assert not describe(torch.zeros(4, 8).t()).contiguous


@torch.no_grad()
def test_device_is_reported_for_cuda():
    if not torch.cuda.is_available():
        return
    info = describe(torch.zeros(1, 3, 224, 224, device="cuda:0"))
    assert info.device == "cuda:0"


def test_explain_shows_the_arithmetic():
    text = describe(np.zeros((1, 3, 224, 224), dtype=np.float32)).explain()
    assert "150,528 elements" in text
    assert "4 bytes/element" in text


def test_human_bytes():
    assert human_bytes(512) == "512 B"
    assert human_bytes(602_112) == "588.00 KiB"
    assert human_bytes(6 * 1024**3) == "6.00 GiB"
