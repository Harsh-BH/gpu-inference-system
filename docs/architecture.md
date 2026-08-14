# Architecture

Every diagram here describes code that exists. Nothing is aspirational.

## System

```mermaid
flowchart TB
    C[Client] -->|POST /predict<br/>multipart| API[FastAPI handler]

    subgraph proc["Server process"]
        API -->|read, capped at<br/>MAX_UPLOAD_BYTES| PP[Preprocess pool<br/>N threads]
        PP -->|3,224,224 float32| Q{{"Request queue<br/>bounded, FIFO"}}
        Q -->|503 when full| API
        Q --> BM[Batch manager<br/>size OR deadline]
        BM -->|N,3,224,224| ENG[InferenceEngine]
        ENG --> BM
        BM -->|row i -> request i| API
        API -->|softmax + top-k| C
    end

    ENG -.selected by BACKEND.-> PT[PyTorchEngine]
    ENG -.-> ORT[ONNXRuntimeEngine]
    ENG -.-> TRT[TensorRTEngine]

    PT & ORT & TRT --> CUDA[CUDA]
    CUDA --> GPU[(GPU: 20 SMs<br/>Tensor Cores<br/>6 GB VRAM)]
```

Preprocessing sits **before** the queue on purpose. Phase 4 measured one
preprocessing thread sustaining ~46 img/s against an engine that absorbs
700–3200; putting it inside the batch worker would serialise the slowest stage.

## Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant H as Handler
    participant P as Preprocess pool
    participant Q as Queue
    participant B as Batch manager
    participant E as Engine
    participant G as GPU

    C->>H: POST /predict
    H->>H: read bytes (capped)
    H->>P: from_bytes(data)
    P-->>H: (3,224,224) float32
    H->>Q: submit(request + Future)
    alt queue full
        Q-->>C: 503 + Retry-After
    end
    H->>H: await Future

    B->>Q: collect until MAX_BATCH_SIZE<br/>or MAX_BATCH_WAIT_MS
    B->>B: np.stack -> (N,3,224,224)
    B->>E: predict(batch)  [in a thread]
    E->>G: H2D copy
    G->>G: ~50 kernels across the SMs
    G->>E: D2H copy (N x 1000 logits)
    E-->>B: logits + stage timings
    B->>H: resolve Future with row i
    H->>H: softmax + top-k
    H-->>C: 200 JSON
```

## Tensor lifecycle

```mermaid
flowchart LR
    A["JPEG bytes<br/>~100 KB"] --> B["PIL image<br/>1546x1213 RGB"]
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
otherwise a steady trickle postpones dispatch indefinitely. Row-to-request
mapping is the one place a bug produces confident wrong answers for everyone
with nothing raised; `tests/test_batching.py` checks it with per-request markers.

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
