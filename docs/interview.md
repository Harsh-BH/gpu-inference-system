# Interview preparation

Answers grounded in measurements from this repository rather than definitions.
Where a number appears, it was measured on an RTX 3050 Laptop (Ampere SM 8.6,
20 SMs, 6 GB, 60 W) — see [experiments.md](experiments.md).

---

### 1. Why use GPU inference?

Because convolution is embarrassingly parallel and a GPU has thousands of ALUs
to exploit that. But the honest answer is conditional: measured here, the GPU
was **5.4× faster at the model** and only **1.85× faster at the request**,
because preprocessing is a serial CPU stage the GPU cannot help with. A GPU
pays off when the model is the bottleneck. Establish that it is before buying
one.

### 2. What is CUDA?

NVIDIA's programming model and runtime for general-purpose GPU work: a C++
dialect for writing kernels, a driver API for launching them, and libraries
(cuBLAS, cuDNN) of pre-tuned kernels. PyTorch, ONNX Runtime and TensorRT all
bottom out in the same CUDA calls — which is why they can differ 5× in speed
while agreeing on every prediction.

### 3. What is a CUDA kernel?

A function that runs on the GPU, executed by many threads at once. You launch
it with a grid/block configuration and each thread computes its own index from
built-in variables. A ResNet-18 forward pass in eager PyTorch is roughly 50
kernel launches; TensorRT fuses that down considerably.

### 4. What is a thread?

The smallest unit of execution, with its own registers and program counter.
Threads are cheap — thousands are resident per SM — and individually slow.

### 5. What is a warp?

32 threads scheduled together in lockstep. The warp, not the thread, is what
an SM actually schedules. Two consequences: a data-dependent branch that
diverges within a warp executes *both* paths with threads masked off, and
memory accesses are coalesced per warp, so 32 threads reading 32 consecutive
addresses is one transaction while 32 scattered reads are 32.

### 6. What is a block?

A group of threads (up to 1024) assigned to a single SM, able to share that
SM's shared memory and synchronise with `__syncthreads()`. A block never
migrates between SMs. Block size is an occupancy decision: too large and fewer
blocks fit per SM, too small and you waste scheduling slots.

### 7. What is an SM?

A Streaming Multiprocessor: the independent core of a GPU, with its own
schedulers, registers, shared memory/L1, CUDA cores and Tensor Cores. This GPU
has 20; an A100 has 108. Blocks are distributed across SMs, which is where the
parallelism actually comes from.

### 8. What is a Tensor Core?

A dedicated unit that computes a small matrix multiply-accumulate in one
instruction, rather than element-by-element on CUDA cores. Verified by kernel
name in this repo: FP32 runs `scudnn_winograd` on CUDA cores and never touches
one; FP16 runs `cutlass_tensorop_f16_s16816`, where `s16816` is the 16×8×16
fragment shape. Note it accumulates in FP32 even with FP16 inputs.

### 9. What is VRAM?

The GPU's own memory. High bandwidth, and physically separate from host RAM —
everything must be copied across PCIe to be computed on. 6 GB here, of which
**104.81 MiB is gone before a single tensor exists**, spent on the CUDA context.

### 10. Why does inference consume more memory than the model weights?

Activations. ResNet-18's weights are 44.69 MiB and never grow, but at batch 32
the transient peak is 214.38 MiB. Measured scaling:

| batch | peak | per image |
|---|---|---|
| 1 | 27.05 MiB | 27.05 MiB |
| 32 | 214.38 MiB | 6.70 MiB |

Memory is **affine, not proportional** — a fixed part plus ~6.7 MiB per image.
Inference is still far cheaper than training, because without an autograd graph
each activation is freed as soon as the next layer consumes it, so the peak is
the largest few tensors alive at once rather than their sum.

### 11. What is a tensor?

A multi-dimensional array with a dtype, a shape, a device, and a memory layout
(strides). The last two are what make it a *tensor* rather than a list of
numbers: the same bytes reinterpreted with different strides is a different
tensor, and a "transposed" tensor is usually a view, not a copy.

### 12. Why does tensor shape matter?

It determines memory (`numel × bytes-per-element`), which kernel is selected,
and whether the call runs at all. A missing batch dimension is the most common
inference bug. This project validates shape and dtype in the base class so no
backend can skip it — because wrong dtype does not crash TensorRT, it
reinterprets the bytes and returns confident garbage.

### 13. What is ONNX?

A serialised computational graph plus its weights: inputs, outputs, nodes
(operators), and initializers (constants), wired together by tensor *names*.
Critically, an operator like `Conv` is a **specification, not code** — it says
what the output must be, not which kernel produces it.

### 14. ONNX vs ONNX Runtime?

ONNX is the file; ONNX Runtime is one program that executes it. TensorRT
executes the same file by compiling it into something else entirely. This repo
feeds one `model.onnx` to both.

### 15. Why use ONNX?

Portability and as a compilation target. It decouples "what the model computes"
from "which framework trained it" and from "what executes it". In this project
it is the artifact TensorRT compiles.

Worth knowing: the export is **not** a neutral serialisation. It folded all 20
BatchNorm layers into their convolutions before any runtime saw the file.

### 16. What is TensorRT?

NVIDIA's inference compiler and runtime. It parses a graph, fuses layers, picks
per-layer kernels by *timing candidates on the actual GPU*, plans memory, and
emits a serialised engine. Measured here: **5.20× PyTorch FP32 at batch 8**
while using less VRAM.

### 17. ONNX Runtime vs TensorRT?

ORT is portable and general; TensorRT is NVIDIA-only and compiles for your
specific GPU. Measured: ORT was **indistinguishable from eager PyTorch**
(0.99–1.03×) while TensorRT FP16 was 4.9–5.2×. ORT's whole-graph optimisation
had little left to do on a 51-node convnet whose BatchNorms were already folded;
TensorRT's win came from kernel selection and layout planning.

### 18. What is a TensorRT engine?

The compiled artifact: chosen kernels, fused layers, a memory plan, serialised
to disk. It is **not portable** — tied to GPU architecture, TensorRT version and
driver. Ours are gitignored for that reason, and the engine loader raises a
message saying exactly this when deserialisation fails.

### 19. What is FP16?

Half precision: 1 sign, 5 exponent, 10 mantissa bits. Half the memory and
bandwidth of FP32, and Tensor Core eligible. Measured: weights 44.69 → 22.96
MiB, throughput 1.8×, top-1 agreement 100%, max logit drift 3.4e-02.

### 20. FP16 vs BF16?

Same 16 bits, split differently. FP16 has 5 exponent / 10 mantissa; BF16 has
**8 exponent** (same range as FP32) / 7 mantissa. BF16 trades precision for
range, so it does not overflow where FP32 would not — which is why training
prefers it. For inference on a model whose activations sit well inside FP16's
range, FP16's extra mantissa bits are worth more.

### 21. What is INT8 quantisation?

Representing tensors as 8-bit integers via `real ≈ scale × (q - zero_point)`,
with scales chosen by calibration. Implemented here, and **found not worth it**:
18% faster than FP16 at batch 8, 3.5% at batch 32, while top-1 agreement fell
from 100% to 90.6%.

### 22. What is dynamic batching?

Grouping independently-arriving requests into one GPU call, dispatching when
either a size or a time threshold trips. The timer must start with the first
request in the batch and never reset, or a steady trickle postpones dispatch
indefinitely.

### 23. Why does batching improve throughput?

It amortises fixed per-call cost — kernel launch overhead, weight reads that
happen once regardless — across more images. Measured: `ms/img` falls from 2.505
at batch 1 to 1.438 at batch 16.

### 24. Why can batching increase latency?

Because every image waits for the whole batch. Measured in isolation: batch 1 is
2.50 ms p50 and 332 img/s; batch 16 is 23.01 ms p50 and 703 img/s — **2.1× the
throughput for 9.2× the latency.** Plus every request pays up to
`MAX_BATCH_WAIT_MS` even when there is nobody to batch with.

**But that is only true of an idle system, and it is worth saying so.** Measured
on a loaded server at 25 concurrent clients, dynamic batching *improved* both:
165.6 → 197.1 req/s **and** p99 295.0 → 177.9 ms. Once requests are queueing,
the dominant term is waiting for the GPU to be free, not the GPU call itself.
Batching eight requests into one call drains the queue roughly eight times
faster — queue wait fell 96.1 → 19.2 ms. The batch call is slower and every
request still finishes sooner.

### 25. What causes CUDA OOM?

Activations scaling with batch size while weights stay fixed, plus allocator
fragmentation: you can OOM with gigabytes "free" because no single *contiguous*
block is large enough. Demonstrated here — batch 512 peaked at 3.35 GiB, batch
1024 failed against a 4.25 GiB cap, and the process kept serving afterwards.

### 26. How would you optimise GPU memory?

Measure first — `allocated` (live tensors), `reserved` (allocator pool) and the
driver's `total - free` are three different numbers. Then: lower precision
(FP16 halved weights), cap batch size, `inference_mode` so activations are freed
immediately, and `set_per_process_memory_fraction` to stop one model starving
others. Note `empty_cache()` belongs on the OOM-recovery and unload paths only —
it synchronises the device.

### 27. How would you reduce inference latency?

Find the actual stage first. Here the breakdown at 25 concurrent clients was
preprocess 21.9 ms, queue 21.0 ms, inference 7.0 ms — so a faster kernel was
the wrong lever. In order: reduce `MAX_BATCH_WAIT_MS`, reduce batch size, use a
compiled runtime, lower precision, and move preprocessing off the critical path.

### 28. How would you increase throughput?

Batch harder, use a faster runtime (5× measured), and fix whatever is actually
saturating. Here that was CPU preprocessing: raising the pool 4 → 16 workers
gave +33% throughput with no GPU change at all.

### 29. What if one GPU cannot handle the load?

Replicate: one process per GPU, each with its own engine and queue, behind a
load balancer. Shard only if the model does not fit on one GPU, since sharding
adds inter-GPU synchronisation to every forward pass. ResNet-18 at 45 MB is
nowhere near needing it.

### 30. Model replication vs sharding?

Replication copies the whole model to each GPU and splits *requests* — simple,
linear, no cross-GPU traffic. Sharding splits the *model* and is required only
when weights or activations exceed one device; it makes every forward pass a
distributed operation.

### 31. What is a CUDA stream?

An ordered queue of GPU work. Operations within a stream run in issue order;
operations in different streams *may* overlap. Measured here: two streams gave
**1.01×**, i.e. nothing, because one ResNet-18 pass at batch 8 already occupies
all 20 SMs. Streams permit overlap; they do not manufacture it.

### 32. What is asynchronous GPU execution?

Kernel launches return to the host as soon as work is *enqueued*, not when it
completes, so the CPU can keep issuing. It is also why timing is easy to get
wrong.

### 33. Why can Python timing be wrong for GPU operations?

Because of exactly that. Measured on the same 2048² matmul: **0.065 ms without
`synchronize()`, 5.723 ms with it — 89× optimistic.** Every timing in this repo
synchronises explicitly.

### 34. How do you benchmark inference correctly?

Warm up (per shape — cuDNN caches an algorithm per input shape); synchronise
around every measurement; report percentiles, not averages; use enough
iterations for the percentile you claim (p99 from 100 samples is the
second-worst sample); and **record the hardware's physical state**. This project
reported a false 2.3% regression before it started logging clocks: a 60 W
power-capped card measures whatever runs last in a sweep on a hotter, slower
GPU. The fix was interleaving, order rotation, and a measured 5% noise floor.

### 35. How would you debug poor GPU utilisation?

Check `nvidia-smi` utilisation and power first — if utilisation is low, the GPU
is starved and the problem is upstream. Here, server-side queue wait stayed at
7–27 ms even at 100 concurrent clients, proving the backlog was in
preprocessing, not inference. Then profile kernels: `torch.profiler` showed
FP16 spending **22.4% of GPU time** on layout transposes that compute nothing.

### 36. How would you handle 1000 concurrent users?

Not by accepting all of them. A bounded queue that returns 503 keeps the server
serving what it can; an unbounded one converts a spike into an OOM kill that
drops everything in flight. Measured: throughput peaked at 25 clients and *fell*
by 100 while p99 went 158 → 683 ms. Little's Law — concurrency = throughput ×
latency — means once throughput saturates, extra concurrency is pure waiting.
Scale horizontally and shed the rest.

### 37. How would you handle model versioning?

Artifacts under `models/<name>/<version>/`, selected by config, never hardcoded.
The active version is reported on `/ready` and in every prediction response, so
"which model produced this" is answerable rather than inferred. Rollout is
config plus restart, behind a load balancer using readiness.

### 38. How would you handle TensorRT engine incompatibility?

Expect it — engines are tied to architecture, TensorRT version and driver.
Build on the target host as a deploy step, never bake engines into an image, and
fail loudly on deserialisation with a message naming the cause and the rebuild
command. Keep the ONNX graph as the portable artifact.

### 39. What happens when an ONNX operator is unsupported?

The parser fails and names the node. Options: change opset, replace the op,
write a plugin, or let the runtime fall back per-node (ORT does this to CPU —
correct but slow, and **silent**). Watch for the version trap: this project
asked for opset 17 and got 18, because the exporter emits 18 and down-converts
only if it can, without raising.

### 40. How would you design this system for production?

Roughly as built: config-driven, one interface with swappable backends, bounded
queue, dynamic batching, warmup before ready, health separate from readiness,
per-stage metrics as histograms, errors mapped to what the client should do, and
no internals in responses.

What I would add: GPU JPEG decode (nvJPEG) since preprocessing is the measured
bottleneck, request deadlines propagated so the batcher can drop work already
past its timeout, a canary path for version rollout, and a labelled eval set —
this project can prove runtimes *agree* but never that they are *accurate*.

---

## The three questions worth volunteering

**"What surprised you?"** That `PRECISION=fp32` was not FP32. PyTorch enables
TF32 convolutions by default on Ampere — 10 mantissa bits instead of 23 —
while tensors still report `float32`. Against a CPU reference the default
disagreed by 3.56e-03 versus 5.25e-06 with it off: **680× worse, silently, with
top-1 unaffected.** It would have contaminated every accuracy comparison
downstream.

**"What did you get wrong?"** I reported that throughput dropped at batch 32.
Re-running the two sizes in isolation gave 662.9 and 665.2 img/s —
indistinguishable. The card is power-capped and whatever runs last in a sweep is
measured on a slower GPU. The fix was structural, not editorial: telemetry on
every row, order rotation, and a noise floor derived from measured variance.

**"What would you do differently?"** Build the load generator first. I spent
Phases 4–9 optimising a model that turned out to be 7% of the request, and only
Phase 12 showed the system was CPU-bound end to end. The bench numbers were all
correct and all beside the point.
