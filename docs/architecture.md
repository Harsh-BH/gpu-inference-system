# Architecture

Every diagram here describes code that exists. Nothing is aspirational.

## System

```mermaid
flowchart TB
    C[Client] -->|POST /predict<br/>multipart| API[FastAPI handler]

    subgraph proc["Server process"]
        API -->|read, capped at<br/>MAX_UPLOAD_BYTES| Q0{{"queue 0<br/>bounded — rejects"}}
        Q0 -->|503 + Retry-After<br/>when full| API

        subgraph pipe["Pipeline"]
            Q0 --> S1["decode<br/>16 workers, batch 1<br/>bytes → 3,224,224"]
            S1 --> Q1{{"queue 1<br/>bounded — blocks"}}
            Q1 --> S2["infer<br/>1 worker, batch 16<br/>size OR deadline"]
            S2 --> Q2{{"queue 2<br/>bounded — blocks"}}
            Q2 --> S3["classify<br/>inline, batch 64<br/>softmax + top-k"]
        end

        S3 -->|"Completion(result, stage_ms, wait_ms)"| API
        API --> C
        S2 -->|"N,3,224,224"| ENG[InferenceEngine]
        ENG -->|"N,1000 logits"| S2
    end

    ENG -.selected by BACKEND.-> PT[PyTorchEngine]
    ENG -.-> ORT[ONNXRuntimeEngine]
    ENG -.-> TRT[TensorRTEngine]

    PT & ORT & TRT --> CUDA[CUDA]
    CUDA --> GPU[(GPU: 20 SMs<br/>Tensor Cores<br/>6 GB VRAM)]
```

Each stage has its own bounded queue. **Queue 0 rejects; the rest block** — a
full downstream queue stalls the upstream worker, and the stall propagates back
to queue 0 where it becomes a 503. Dropping a request that has already been
decoded would throw away the most expensive work the system does.

The three stages differ only in how they are *run*, and each setting is a
measurement:

| stage | workers | batch | why |
|---|---|---|---|
| decode | 16 | 1 | CPU-bound and PIL releases the GIL, so threads scale. No per-call fixed cost, so batching would buy nothing. |
| infer | 1 | 16 | One CUDA context serialises anyway. Per-*call* fixed costs (launches, H2D, the Python/CUDA boundary) are what a batch amortises. |
| classify | 0 (inline) | 64 | 0.09 ms of numpy. A thread hop would cost more than the work. |

See [pipeline.md](pipeline.md) for the runtime itself.

## Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant H as Handler
    participant PL as Pipeline
    participant D as decode x16
    participant I as infer x1
    participant K as classify
    participant E as Engine
    participant G as GPU

    C->>H: POST /predict
    H->>H: read bytes (capped)
    H->>PL: submit(bytes)
    alt queue 0 full
        PL-->>C: 503 + Retry-After
    end
    H->>H: await Completion

    PL->>D: process([bytes])
    D->>D: draft decode, resize, crop, normalise
    D-->>PL: (3,224,224) float32

    PL->>I: process([...]) collected until<br/>MAX_BATCH_SIZE or MAX_BATCH_WAIT_MS
    I->>I: np.stack -> (N,3,224,224)
    I->>E: predict(batch)
    E->>G: H2D copy
    G->>G: ~50 kernels across the SMs
    G->>E: D2H copy (N x 1000 logits)
    E-->>I: logits + h2d/compute/d2h timings
    I-->>PL: row i -> item i

    PL->>K: process([logits])
    K-->>PL: top-5 predictions
    PL->>H: Completion(result, stage_ms, wait_ms)
    H-->>C: 200 JSON
```

`predict()` runs in the infer stage's thread, not on the event loop: it
synchronises on CUDA, and running it inline would stall every other request —
health checks, new arrivals, metrics — for the whole forward pass.

## Tensor lifecycle

```mermaid
flowchart LR
    A["JPEG bytes<br/>~100 KB"] -->|"draft: scaled inverse DCT<br/>when the input is oversized"| B["PIL image<br/>387x304 RGB<br/>(1546x1213 if FAST_DECODE=false)"]
    B --> C["resized<br/>shorter side 256"]
    C --> D["cropped<br/>224x224x3 uint8"]
    D --> E["CHW float32<br/>normalised<br/>588 KiB"]
    E --> F["batched<br/>N,3,224,224"]
    F -->|PCIe| G["VRAM<br/>N x 588 KiB"]
    G --> H["logits<br/>N,1000 float32"]
    H -->|PCIe| I["host<br/>4 KB per sample"]
    I --> J["softmax<br/>top-5 labels"]
```

One sample is `3 × 224 × 224 = 150,528` elements × 4 bytes = **588 KiB**. At
batch 32 that is 18 MiB crossing PCIe — small enough that Phase 4 found the
transfer irrelevant next to a 214 MiB activation peak.

## Model conversion pipeline

```mermaid
flowchart LR
    W["weights.pth<br/>state_dict"] -->|fetch_model.py| R[Model repository]
    R -->|export_onnx.py| O1["model.onnx<br/>fp32, 44.66 MiB"]
    R -->|export_onnx.py<br/>--precision fp16| O2["model.fp16.onnx"]
    O1 -->|quantize_int8.py<br/>PTQ + calibration| O3["model.int8.onnx<br/>QDQ, 11.33 MiB"]

    O1 -->|build_engine.py| E1["fp32/model.engine<br/>51.55 MiB"]
    O2 -->|build_engine.py| E2["fp16/model.engine"]
    O3 -->|build_engine.py| E3["int8/model.engine<br/>11.67 MiB"]

    O1 -.direct.-> ORT[ONNX Runtime]
    E1 & E2 & E3 --> TRT[TensorRT runtime]
```

Building is separate from serving because it takes 10–100 s of tactic timing on
*this* GPU. Engines are gitignored: they are build artifacts tied to the
architecture, TensorRT version and driver, not models.

## GPU execution hierarchy

```mermaid
flowchart TB
    K["CUDA kernel<br/>e.g. one convolution"] --> GR["Grid<br/>all blocks for this launch"]
    GR --> BL["Block<br/>up to 1024 threads<br/>shares SMEM, can sync"]
    BL --> WP["Warp<br/>32 threads, lockstep<br/>the real scheduling unit"]
    WP --> TH["Thread<br/>own registers"]

    BL -.scheduled onto.-> SM["SM (20 on this GPU)"]
    SM --> CC["CUDA cores<br/>FP32/INT32"]
    SM --> TC["Tensor Cores<br/>matrix-multiply-accumulate"]
    SM --> SMEM["Shared memory / L1"]
```

A block is assigned to one SM and never migrates. Warps are what the SM
actually schedules; a branch that diverges within a warp costs both paths.
Occupancy — resident warps per SM — is what hides memory latency.

## Memory hierarchy

```mermaid
flowchart TB
    REG["Registers<br/>per thread, ~1 cycle"] --> SMEM["Shared memory / L1<br/>per block, ~30 cycles"]
    SMEM --> L2["L2 cache<br/>device-wide"]
    L2 --> VRAM["Global memory / VRAM<br/>6 GB, ~400 cycles"]
    VRAM -->|PCIe, ~GB/s| HOST["Host RAM"]

    style REG fill:#2d6a4f,color:#fff
    style VRAM fill:#9d4edd,color:#fff
```

The cliff between VRAM and host is why the whole design keeps data on the
device once it arrives, and why the CUDA context costs 104.81 MiB before a
single tensor exists (Experiment 4).

## Dynamic batching

```mermaid
flowchart TB
    subgraph arrivals["Independent arrivals"]
        A[Request A<br/>t=0ms]
        B[Request B<br/>t=1ms]
        D[Request C<br/>t=3ms]
    end
    A & B & D --> Q[Queue]
    Q --> W{"MAX_BATCH_SIZE reached?<br/>OR oldest waited<br/>MAX_BATCH_WAIT_MS?"}
    W -->|no| Q
    W -->|yes| S["stack -> (3,3,224,224)"]
    S --> G[One GPU call]
    G --> M["logits (3,1000)"]
    M -->|row 0| A2[Response A]
    M -->|row 1| B2[Response B]
    M -->|row 2| D2[Response C]
```

The deadline is started by the **first** request in the batch and never reset —
otherwise a steady trickle postpones dispatch indefinitely.

Row-to-request mapping is the one place a bug produces confident wrong answers
for everyone with nothing raised. It is now a single `zip(..., strict=True)`
inside the pipeline runner rather than hand-written per batching implementation;
`tests/test_pipeline.py::test_result_i_goes_to_job_i` checks it with per-request
markers, and `tests/test_stages.py` checks the engine's own row split.

In practice this system rarely fills a batch: a 900-request run averaged **2.3
images per GPU call** against a `MAX_BATCH_SIZE` of 16, because decode cannot
feed the GPU fast enough to reach the size trigger before the deadline.

## Multi-GPU scaling

```mermaid
flowchart TB
    LB[Load balancer] --> P0["Process 0<br/>CUDA_VISIBLE_DEVICES=0<br/>own queue + batcher"]
    LB --> P1["Process 1<br/>CUDA_VISIBLE_DEVICES=1"]
    LB --> PN["Process N"]
    P0 --> G0[(GPU 0)]
    P1 --> G1[(GPU 1)]
    PN --> GN[(GPU N)]
```

**Not implemented** — this machine has one GPU. The shape is replication, not
sharding: ResNet-18 is 45 MB and fits many times over, so every GPU gets a full
copy and requests are load-balanced across processes. Sharding a model across
GPUs is for models that do not fit on one, and it costs inter-GPU
synchronisation on every forward pass.

One process per GPU rather than threads: a single CUDA context serialises
anyway, and separate processes sidestep the GIL entirely.
