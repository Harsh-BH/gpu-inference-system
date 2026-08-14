# One request, from HTTP to CUDA kernel and back

The final report the project set out to produce. This traces a single
`POST /predict` through every layer, with the measured cost of each. Numbers are
from an RTX 3050 Laptop (Ampere SM 8.6, 20 SMs, 6 GB, 60 W cap) serving
TensorRT FP16 at `MAX_BATCH_SIZE=16` with 8 preprocessing workers.

---

## 1. HTTP arrives

A multipart POST reaches uvicorn, which parses headers and hands a
`starlette.UploadFile` to the handler. Nothing is buffered yet.

The handler reads the body in 64 KiB chunks, **counting as it goes**, and aborts
with 413 the moment the count passes `MAX_UPLOAD_BYTES`. Content-Length is not
trusted: it can be absent under chunked encoding and it can lie. A hostile
client cannot make the server allocate what it refuses to accept.

## 2. Preprocessing — 21.9 ms, the largest single cost

Dispatched to a `ThreadPoolExecutor`, not run inline, so the event loop stays
free to accept other requests.

```
JPEG bytes
  -> Image.open           header only, no decode yet
  -> pixel-count guard    a 10 KB PNG can declare 60000x60000 and ask for ~10 GB
  -> exif_transpose       phone photos carry rotation in metadata, not pixels
  -> convert("RGB")       forces the decode; normalises grayscale/RGBA/CMYK
  -> resize shorter->256  aspect-preserving; squashing costs ~0.5% top-1
  -> center crop 224      matching how the weights were evaluated
  -> HWC uint8 -> CHW     one copy; conv kernels want channel-planar
  -> /255, -mean, /std    in place, no further allocations
```

Out comes `(3, 224, 224)` float32 — 150,528 elements, **588 KiB**.

This stage is the system's bottleneck and the reason the pool exists. One
thread sustains ~46 img/s; the GPU absorbs 3300.

## 3. The queue — a Future is created

The tensor, a `request_id` and an `asyncio.Future` become an
`InferenceRequest`, submitted with `put_nowait`. Full queue → immediate 503 with
`Retry-After`, never a wait. The handler then awaits its Future and yields the
event loop.

This is where HTTP's request/response shape and the GPU's batch shape are
reconciled: the request will be reunited with its own row of an output tensor
later, on a different task.

## 4. Batch formation — the manager collects

A single background task pulls from the queue until **either** 16 requests are
in hand **or** the oldest has waited `MAX_BATCH_WAIT_MS`. The timer starts with
the first request and is never reset, so a steady trickle cannot postpone
dispatch forever.

`np.stack` copies the samples into one contiguous `(N, 3, 224, 224)` block.
The copy is required: those samples arrived at different times and are scattered
across the heap, while the DMA engine needs one contiguous region.

Under load at 25 concurrent clients this stage costs 21 ms of queue wait.

## 5. Host to device — 2.2 ms at batch 32

`engine.predict` runs via `asyncio.to_thread`; calling it inline would stall the
event loop for the entire forward pass.

`torch.from_numpy` wraps the numpy buffer with no copy. `.to(device, dtype)`
crosses PCIe into VRAM, casting to FP16 on the way. This is the only moment
"CPU memory" becomes "GPU memory".

The memory is pageable, so the driver stages it through an internal pinned
buffer. Phase 19 measured pinned memory as 1.16–1.54× faster here and not worth
the complexity: 0.22 ms against a 21.9 ms preprocessing stage.

## 6. CUDA — the kernels run

`context.set_input_shape` fixes the batch dimension, selecting kernels
specialised for it inside the engine's optimisation profile.
`execute_async_v3` enqueues the whole plan onto a stream.

What happens on the device:

- Each fused layer is a **kernel launch**. TensorRT has already fused conv +
  bias + ReLU into single kernels and planned the layout once for the whole
  graph, which is why it avoids the `nchwToNhwc`/`nhwcToNchw` transposes that
  cost PyTorch FP16 **22.4% of its GPU time**.
- A launch creates a **grid** of **blocks**. Each block is assigned to one of
  the 20 **SMs** and never migrates.
- Within a block, threads are grouped into **warps** of 32 that execute in
  lockstep. The warp is what the SM actually schedules; having many resident is
  how memory latency gets hidden.
- Convolutions are lowered to matrix multiplies and issued to **Tensor Cores** —
  `cutlass_tensorop_f16_s16816`, where `s16816` is the 16×8×16 MMA fragment
  shape. Each instruction multiplies small tiles and accumulates in FP32, which
  is why FP16 storage does not mean FP16 accumulation.
- Data moves registers → shared memory/L1 → L2 → VRAM, spanning roughly 1 to
  400 cycles.

`stream.synchronize()` blocks the host until the GPU is genuinely finished.
Without it every timing in this project would measure enqueue cost — Phase 0
shows that gap as 0.065 ms versus 5.723 ms on the same matmul.

**6.98 ms of compute at batch 32.** The equivalent PyTorch FP16 number is
23.14 ms.

## 7. Device to host — 0.13 ms

`(N, 1000)` logits come back: 4 KB per sample. Three orders of magnitude
smaller than the input, which is why the return trip never matters for
classification and would matter enormously for segmentation.

## 8. Demultiplexing — the correctness-critical step

The batch manager walks the output tensor and resolves each Future with **its
own row**:

```python
for i, request in enumerate(batch):
    request.resolve(logits[i])
```

Get this wrong and every client receives a confident, plausible prediction
belonging to somebody else. No exception, no log line, no failing test unless
someone wrote one — `tests/test_batching.py` writes per-request markers into the
input and checks they come back in the right place.

## 9. Postprocessing — 0.10 ms

The handler's `await` returns with a `(1000,)` array.

Softmax subtracts the row max before exponentiating — not an optimisation, a
correctness requirement: `exp(89)` overflows float32 to `inf`, and `inf/inf` is
`nan`, so a confident prediction would return `nan` confidence for every class.

`argpartition` finds the top 5 in O(n) rather than sorting 1000.

Note that `argmax(logits) == argmax(softmax(logits))` — softmax is monotonic, so
the *predicted class* never depends on it. Only the confidence number does,
which is why the cross-runtime comparisons report class agreement and confidence
drift separately.

## 10. Response

```json
{
  "request_id": "…", "model_name": "resnet18", "model_version": "v1",
  "backend": "tensorrt", "precision": "fp16",
  "prediction": "Samoyed", "confidence": 0.8847,
  "predictions": [ … top 5 … ],
  "latency": { "preprocess_ms": 21.9, "queue_wait_ms": 21.0,
               "inference_ms": 2.4, "postprocess_ms": 0.1, "total_ms": 45.4 }
}
```

The engine identity travels with every prediction. After a deploy changes
accuracy, "which engine produced this" must be answerable from the response
rather than inferred from what happened to be configured.

---

## Where the time actually goes

At 25 concurrent clients, 223 req/s, p50 107 ms:

| stage | cost | who is doing the work |
|---|---|---|
| read + validate | <1 ms | CPU |
| **preprocess** | **21.9 ms** | **CPU — the bottleneck** |
| queue wait | 21.0 ms | waiting |
| batch formation | ~5 ms | waiting |
| H2D | 2.2 ms | PCIe |
| **inference** | **7.0 ms** | **GPU** |
| D2H | 0.1 ms | PCIe |
| postprocess | 0.1 ms | CPU |

The GPU accounts for about 7 ms of a 107 ms request. Its raw capability is
3300 img/s and the system delivers 223 req/s — roughly 7% of the accelerator.

Everything else follows from that one number. The optimisation that matters
next is not a faster kernel; it is decoding JPEGs somewhere other than a Python
thread pool — GPU JPEG decode via nvJPEG, or accepting pre-decoded tensors from
clients. TensorRT already made the model 5× faster and moved end-to-end
throughput by far less, exactly as Amdahl's law says it must.
