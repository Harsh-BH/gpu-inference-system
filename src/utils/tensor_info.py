"""Introspection for whatever array is currently in flight (Phase 2).

WHY a whole module for printing a shape

    Nearly every bug in an inference pipeline is a shape, dtype, device or
    layout bug, and all four are invisible until you print them:

      - shape:   forgot to unsqueeze, so (3,224,224) reaches a model wanting (1,3,224,224)
      - dtype:   float64 from numpy's default, silently reinterpreted by TensorRT
      - device:  a CPU tensor meeting CUDA weights, or a silent implicit copy
      - layout:  a transposed view that is not contiguous, so the DMA path is
                 slower than it looks and cudaMemcpy has to stage it

    Printing "shape: [1, 3, 224, 224] dtype: float32 device: cuda:0 | 0.57 MiB"
    at each hop turns all four into things you can see.

Works on numpy arrays and torch tensors by duck typing -- deliberately no torch
import, so this stays usable in the ONNX/TensorRT paths.
"""

from __future__ import annotations

from dataclasses import dataclass


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} GiB"


@dataclass(frozen=True, slots=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    nbytes: int
    contiguous: bool

    def __str__(self) -> str:
        flag = "" if self.contiguous else "  [NON-CONTIGUOUS]"
        return (
            f"shape={list(self.shape)}  dtype={self.dtype}  device={self.device}  "
            f"numel={self.numel:,}  mem={human_bytes(self.nbytes)}{flag}"
        )

    def explain(self) -> str:
        """The memory arithmetic, spelled out.

        Worth showing because it is the basis of every VRAM estimate later:
        a tensor is just numel * bytes-per-element, and everything about batch
        size and precision follows from that one product.
        """
        per_elem = self.nbytes // self.numel if self.numel else 0
        dims = " x ".join(str(d) for d in self.shape)
        return (
            f"{dims} = {self.numel:,} elements x {per_elem} bytes/element "
            f"({self.dtype}) = {human_bytes(self.nbytes)}"
        )


def describe(obj) -> TensorInfo:
    """Describe a numpy array or torch tensor without importing torch."""
    shape = tuple(int(d) for d in obj.shape)

    if hasattr(obj, "element_size"):  # torch.Tensor
        numel = int(obj.numel())
        nbytes = numel * obj.element_size()
        device = str(obj.device)
        dtype = str(obj.dtype).removeprefix("torch.")
        contiguous = bool(obj.is_contiguous())
    else:  # numpy.ndarray
        numel = int(obj.size)
        nbytes = int(obj.nbytes)
        device = "cpu"  # numpy is host memory by definition
        dtype = str(obj.dtype)
        contiguous = bool(obj.flags.c_contiguous)

    return TensorInfo(
        shape=shape,
        dtype=dtype,
        device=device,
        numel=numel,
        nbytes=nbytes,
        contiguous=contiguous,
    )


def trace(label: str, obj) -> None:
    """Print one hop of the tensor's journey. Debug aid for the CLI."""
    print(f"  {label:<22} {describe(obj)}")
