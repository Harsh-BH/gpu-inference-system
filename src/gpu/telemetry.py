"""GPU clock, power, temperature and utilisation.

WHY A BENCHMARK NEEDS THIS

    This project measured "throughput peaks at batch 16 and drops at batch 32"
    and it was wrong. Re-running the two sizes in isolation gave 662.9 and
    665.2 img/s -- identical. The apparent drop was thermal drift: batch 32 ran
    last, after ~40 seconds of sustained load, on a laptop GPU with a 60 W cap.

    Sampling clocks made the cause visible. Under load this card sits at
    59.3-60.1 W against its 60 W limit with SM clocks at 1627-1725 MHz, down
    from 1912 MHz idle. It is power-limited, so any two measurements taken at
    different points in a long sweep are not directly comparable.

    Recording this alongside every result is what stops the next 2% difference
    from becoming a false finding. A benchmark that cannot see throttling will
    confidently attribute it to whatever it happened to be varying.

Sampling costs roughly 0.1-1 ms via NVML, so this belongs at measurement
boundaries, never inside a timed loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Telemetry:
    """A point sample of the GPU's physical state."""

    sm_clock_mhz: int
    power_w: float
    temperature_c: int
    utilization_pct: int

    def __str__(self) -> str:
        return (
            f"{self.sm_clock_mhz} MHz, {self.power_w:.1f} W, "
            f"{self.temperature_c} C, {self.utilization_pct}% util"
        )


def sample_telemetry(device: int | None = None) -> Telemetry | None:
    """Current GPU state, or None if unavailable.

    Returns None rather than raising: NVML is missing on some systems, absent
    inside many containers, and restricted for non-root users on others.
    Telemetry is diagnostic, so it must never be the reason a benchmark or a
    metrics endpoint fails.
    """
    if not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device() if device is None else device
    try:
        return Telemetry(
            # torch reports clock in MHz and power in milliwatts.
            sm_clock_mhz=int(torch.cuda.clock_rate(idx)),
            power_w=torch.cuda.power_draw(idx) / 1000.0,
            temperature_c=int(torch.cuda.temperature(idx)),
            utilization_pct=int(torch.cuda.utilization(idx)),
        )
    except (ModuleNotFoundError, RuntimeError, AttributeError, OSError):
        return None
