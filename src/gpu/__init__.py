"""GPU mechanics: memory, streams, profiling.

Distinct from src/monitoring/, which exports metrics to Prometheus. This
package is about *what the GPU is doing*; that one is about *telling someone
else about it*.
"""

from src.gpu.memory import (
    MemoryScope,
    MemorySnapshot,
    empty_cache,
    get_gpu_memory,
    get_peak_memory,
    limit_process_memory,
    memory_scope,
    reset_peak_memory,
)

__all__ = [
    "MemoryScope",
    "MemorySnapshot",
    "empty_cache",
    "get_gpu_memory",
    "get_peak_memory",
    "limit_process_memory",
    "memory_scope",
    "reset_peak_memory",
]
