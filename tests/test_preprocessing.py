"""Preprocessing tests.

The headline test is `test_matches_torchvision_exactly`. Everything downstream
in this project -- PyTorch vs ONNX Runtime vs TensorRT accuracy, FP16 drift,
INT8 calibration error -- is measured as a difference between runtimes. If the
input tensor is already subtly wrong, every one of those differences is
contaminated and the conclusions are noise.

So we pin against the reference implementation the weights were evaluated with,
across several aspect ratios including odd-numbered ones (which is where the
round()-vs-floor() crop offset actually diverges).
"""

import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImagePreprocessor,
    PreprocessingError,
)


def make_image(w: int, h: int, mode: str = "RGB", seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]
    arr = rng.integers(0, 256, size=(h, w, channels), dtype=np.uint8)
    return Image.fromarray(arr.squeeze() if channels == 1 else arr, mode=mode)


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def pre():
    return ImagePreprocessor(image_size=224)


# --- the one that matters -------------------------------------------------


@pytest.mark.parametrize(
    "size",
    [
        (500, 300),  # landscape
        (300, 500),  # portrait
        (400, 400),  # square
        (301, 499),  # odd dimensions: round() vs // diverge on the crop offset
        (225, 226),  # barely larger than the crop
        (1024, 768),
    ],
)
def test_matches_torchvision_exactly(pre, size):
    from torchvision import transforms

    reference = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    img = make_image(*size)

    expected = reference(img).numpy()
    got = pre._transform(img)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=0)


# --- output contract ------------------------------------------------------


def test_output_shape_dtype_and_contiguity(pre):
    out = pre.from_bytes(encode(make_image(640, 480)))
    assert out.shape == (3, 224, 224) == pre.output_shape
    assert out.dtype == np.float32
    # Contiguity is not cosmetic: a non-contiguous array forces cudaMemcpy to
    # stage through a temporary buffer instead of a straight DMA.
    assert out.flags.c_contiguous


def test_normalisation_is_actually_applied(pre):
    # Uniform mid-grey (128/255 ~= 0.502) has a known post-normalisation value
    # per channel. This catches mean/std being dropped or swapped.
    grey = Image.new("RGB", (300, 300), (128, 128, 128))
    out = pre._transform(grey)
    for c in range(3):
        expected = (128 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert out[c].mean() == pytest.approx(expected, abs=1e-5)


def test_memory_footprint_is_what_the_docs_claim(pre):
    out = pre.from_bytes(encode(make_image(300, 300)))
    assert out.nbytes == 3 * 224 * 224 * 4 == 602_112


# --- format handling ------------------------------------------------------


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB"])
def test_non_rgb_modes_are_converted(pre, mode):
    out = pre.from_bytes(encode(make_image(300, 300, mode=mode)))
    assert out.shape == (3, 224, 224)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "WEBP"])
def test_common_formats_decode(pre, fmt):
    out = pre.from_bytes(encode(make_image(300, 300), fmt=fmt))
    assert out.shape == (3, 224, 224)


# --- rejection paths ------------------------------------------------------


def test_empty_payload_rejected(pre):
    with pytest.raises(PreprocessingError, match="empty"):
        pre.from_bytes(b"")


def test_garbage_bytes_rejected(pre):
    with pytest.raises(PreprocessingError, match="unrecognised|unsupported"):
        pre.from_bytes(b"this is not an image, it is a sentence")


def test_truncated_image_rejected(pre):
    data = encode(make_image(300, 300))
    with pytest.raises(PreprocessingError):
        pre.from_bytes(data[: len(data) // 3])


def test_decompression_bomb_rejected_before_decode():
    # The guard reads dimensions from the header, so an oversized image is
    # refused without ever allocating its pixels.
    pre = ImagePreprocessor(image_size=224, max_pixels=1_000)
    with pytest.raises(PreprocessingError, match="pixels"):
        pre.from_bytes(encode(make_image(300, 300)))


def test_missing_file_rejected(pre, tmp_path):
    with pytest.raises(PreprocessingError, match="no such image"):
        pre.from_path(tmp_path / "nope.jpg")


# --- batching -------------------------------------------------------------


def test_stack_builds_a_batch(pre):
    samples = [pre.from_bytes(encode(make_image(300, 300, seed=i))) for i in range(4)]
    batch = pre.stack(samples)
    assert batch.shape == (4, 3, 224, 224)
    assert batch.dtype == np.float32
    assert batch.flags.c_contiguous


def test_stack_preserves_per_sample_identity(pre):
    # The batch manager depends on this: row i must still be request i, or
    # every client gets someone else's prediction.
    samples = [pre.from_bytes(encode(make_image(300, 300, seed=i))) for i in range(3)]
    batch = pre.stack(samples)
    for i, s in enumerate(samples):
        np.testing.assert_array_equal(batch[i], s)


def test_stack_rejects_empty(pre):
    with pytest.raises(PreprocessingError):
        pre.stack([])
