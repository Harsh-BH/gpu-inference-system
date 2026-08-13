"""Image bytes -> the exact float32 tensor a ResNet expects.

WHAT
    JPEG/PNG bytes in, a contiguous (3, H, W) float32 numpy array out, plus a
    stacking helper that turns a list of those into an (N, 3, H, W) batch.

WHY this is written by hand instead of using torchvision.transforms
    Two reasons, both structural.

    1. This module must not import torch. Preprocessing runs on every request
       regardless of which backend serves it; if it pulled in torch, the
       TensorRT-only deployment would carry a 3 GB dependency it never
       executes. The whole engine abstraction leaks if its input stage does not
       respect the same boundary.

    2. "How does an image become a tensor" is one of the things this project
       exists to explain. `transforms.Compose([...])` explains nothing.

    The cost of hand-writing it is that it must match torchvision *exactly* --
    resize-then-crop geometry, interpolation filter, normalisation constants.
    A 2-pixel difference in crop geometry is worth ~0.5% top-1 accuracy, which
    is enough to poison every cross-runtime accuracy comparison in this repo.
    tests/test_preprocessing.py asserts the match against torchvision directly.

THE PIPELINE, AND WHY EACH STEP EXISTS

    bytes
      -> Image.open          lazy: reads the header only, does not decode yet
      -> pixel-count guard   a 10 KB PNG can declare 60000x60000 and allocate
                             ~10 GB on decode. Checked BEFORE decoding.
      -> exif_transpose      phone photos carry rotation in metadata, not in
                             pixels. Skipping this classifies sideways images.
      -> convert("RGB")      normalises grayscale/RGBA/palette/CMYK to 3 channels
      -> resize shorter side aspect-preserving. Squashing straight to 224x224
                             distorts objects and measurably costs accuracy.
      -> center crop         224x224, matching how the model was evaluated
      -> HWC uint8 -> CHW    layout change, see below
      -> /255, normalise     into the distribution the weights were trained on

NCHW, AND WHY THE AXES ARE IN THAT ORDER

    PIL hands you (Height, Width, Channels) — interleaved RGB, the layout image
    formats use. Conv kernels want (Channels, Height, Width): all of the red
    plane contiguous, then green, then blue, because a convolution sweeps one
    channel's spatial neighbourhood at a time and that layout gives it
    coalesced reads. Batching prepends N, giving NCHW.

    (NHWC / "channels last" is what Tensor Cores actually prefer, which is why
     TensorRT often transposes internally. Phase 19 measures that.)

MEMORY, CONCRETELY

    One 224x224 RGB sample as float32:
        3 * 224 * 224 = 150,528 elements
        150,528 * 4 bytes = 602,112 bytes = 0.574 MiB

    At batch 8 that is 4.6 MiB crossing PCIe per request — small, which is
    itself the point: for this model the transfer is not the bottleneck, and
    Phase 19 has the numbers to prove it rather than assume it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# The distribution ImageNet-pretrained torchvision weights were trained on.
# Per-channel RGB, on pixels already scaled to [0, 1]. Using different values
# here does not raise an error -- it just quietly degrades every prediction,
# which is why they live in one named place.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# torchvision's standard eval recipe resizes the shorter side to 256 before
# cropping 224. Kept as a ratio so a different image_size stays consistent.
_RESIZE_RATIO = 256 / 224


class PreprocessingError(ValueError):
    """Input could not be turned into a tensor.

    A ValueError subclass on purpose: this is always caused by client input
    (corrupt bytes, unsupported format, absurd dimensions), never by a server
    fault, so the API layer maps it to 4xx and never to 500.
    """


class ImagePreprocessor:
    """Stateless per-request, but holds precomputed constants.

    A class rather than a bare function because mean/std are reshaped to
    (3, 1, 1) once at construction instead of on every request. At a few
    hundred requests per second that is free throughput, and it gives the
    batch manager one object to hold.
    """

    def __init__(
        self,
        image_size: int = 224,
        *,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        max_pixels: int = 50_000_000,
    ) -> None:
        self.image_size = image_size
        self.resize_size = int(round(image_size * _RESIZE_RATIO))
        self.max_pixels = max_pixels
        # (3, 1, 1) so they broadcast across H and W of a CHW array.
        self._mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """(C, H, W) of a single preprocessed sample."""
        return (3, self.image_size, self.image_size)

    # --- entry points ---------------------------------------------------

    def from_bytes(self, data: bytes) -> np.ndarray:
        """Decode and preprocess raw upload bytes. Returns (3, H, W) float32."""
        if not data:
            raise PreprocessingError("empty image payload")
        try:
            img = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as exc:
            raise PreprocessingError("unrecognised or unsupported image format") from exc
        except OSError as exc:
            raise PreprocessingError(f"could not read image: {exc}") from exc
        return self._transform(img)

    def from_path(self, path: str | Path) -> np.ndarray:
        """Load from disk. For CLI and tests only.

        Never call this with a client-supplied path: that is arbitrary file
        read. The HTTP layer uses from_bytes exclusively.
        """
        p = Path(path)
        if not p.is_file():
            raise PreprocessingError(f"no such image file: {p}")
        return self.from_bytes(p.read_bytes())

    def stack(self, samples: list[np.ndarray]) -> np.ndarray:
        """Combine per-sample (3, H, W) arrays into one (N, 3, H, W) batch.

        This is the moment N independent requests become a single GPU
        operation. np.stack copies into fresh contiguous memory, which is
        required -- the samples arrived from different requests at different
        times and are scattered across the heap, while the H2D copy needs one
        contiguous block to DMA.
        """
        if not samples:
            raise PreprocessingError("cannot stack an empty batch")
        return np.stack(samples, axis=0)

    # --- the actual work ------------------------------------------------

    def _transform(self, img: Image.Image) -> np.ndarray:
        # Guard before decoding. img.size comes from the header, so this
        # rejects decompression bombs without ever allocating their pixels.
        w, h = img.size
        if w * h > self.max_pixels:
            raise PreprocessingError(
                f"image is {w}x{h} = {w * h:,} pixels, limit is {self.max_pixels:,}"
            )
        if w == 0 or h == 0:
            raise PreprocessingError("image has zero width or height")

        try:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")  # forces the decode
            img = self._resize_shorter_side(img)
            img = self._center_crop(img)
        except OSError as exc:
            # Truncated/corrupt files usually fail here, mid-decode, not at open().
            raise PreprocessingError(f"corrupt or truncated image data: {exc}") from exc

        # HWC uint8 -> CHW float32. transpose() returns a view; ascontiguousarray
        # materialises it, and that single copy is the only one in this function.
        arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        chw = np.ascontiguousarray(arr, dtype=np.float32)
        chw /= 255.0  # in-place from here on -- no further allocations
        chw -= self._mean
        chw /= self._std
        return chw

    def _resize_shorter_side(self, img: Image.Image) -> Image.Image:
        """Scale so the shorter side is resize_size, preserving aspect ratio.

        Geometry copied from torchvision.transforms.Resize with an int size.
        BILINEAR in Pillow is a true convolution-based resampling filter, i.e.
        antialiased, which is what torchvision's PIL path uses.
        """
        w, h = img.size
        target = self.resize_size
        if w <= h:
            ow, oh = target, int(target * h / w)
        else:
            oh, ow = target, int(target * w / h)
        if (ow, oh) == (w, h):
            return img
        return img.resize((ow, oh), Image.BILINEAR)

    def _center_crop(self, img: Image.Image) -> Image.Image:
        """Crop image_size x image_size from the centre.

        round() rather than floor division, matching torchvision exactly. The
        two differ by one pixel on odd offsets, and a systematic one-pixel
        shift is enough to move top-1 accuracy.
        """
        w, h = img.size
        size = self.image_size
        if w < size or h < size:
            # Cannot happen after resize_shorter_side (256 > 224), but a caller
            # could construct a preprocessor with a pathological ratio.
            raise PreprocessingError(f"image {w}x{h} is smaller than crop {size}x{size}")
        left = int(round((w - size) / 2.0))
        top = int(round((h - size) / 2.0))
        return img.crop((left, top, left + size, top + size))
