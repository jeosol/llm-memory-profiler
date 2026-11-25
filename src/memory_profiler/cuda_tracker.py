# The python script PyTorch CUDA hooks to capture exact phase-by-phase VRAM peaks on a
# RTX PRO 5000 72GB, separating active tensor allocations from CUDA caching allocator fragmentation.

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List
import torch


@dataclass
class PhaseMetrics:
    phase_name: str
    allocated_gb: float
    peak_allocated_gb: float
    reserved_gb: float
    elapsed_ms: float


@dataclass
class StepTelemetry:
    step: int
    phases: List[PhaseMetrics] = field(default_factory=list)

    @property
    def max_step_vram_gb(self) -> float:
        return max((p.peak_allocated_gb for p in self.phases), default=0.0)


class CudaMemoryTracker:
    """Tracks phase-level CUDA memory peaks and allocator overhead."""

    BYTES_PER_GB = 1024**3

    def __init__(self, device: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available.")
        self.device = torch.device(f"cuda:{device}")
        self.device_name = torch.cuda.get_device_name(self.device)
        self.total_vram_gb = (
            torch.cuda.get_device_properties(self.device).total_memory / self.BYTES_PER_GB
        )
        self.history: List[StepTelemetry] = []
        self._current_step: StepTelemetry | None = None

    def start_step(self, step: int) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._current_step = StepTelemetry(step=step)

    @contextmanager
    def track_phase(self, phase_name: str):
        if self._current_step is None:
            raise RuntimeError("Call start_step() before track_phase().")

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()

        yield

        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        alloc_gb = torch.cuda.memory_allocated(self.device) / self.BYTES_PER_GB
        peak_gb = torch.cuda.max_memory_allocated(self.device) / self.BYTES_PER_GB
        reserved_gb = torch.cuda.memory_reserved(self.device) / self.BYTES_PER_GB

        self._current_step.phases.append(
            PhaseMetrics(
                phase_name=phase_name,
                allocated_gb=round(alloc_gb, 3),
                peak_allocated_gb=round(peak_gb, 3),
                reserved_gb=round(reserved_gb, 3),
                elapsed_ms=round(elapsed_ms, 2),
            )
        )

    def end_step(self) -> StepTelemetry:
        if self._current_step is None:
            raise RuntimeError("No active step to end.")
        completed = self._current_step
        self.history.append(completed)
        self._current_step = None
        return completed

    def summary_report(self) -> Dict[str, float]:
        return {
            "gpu": self.device_name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "peak_observed_gb": max((s.max_step_vram_gb for s in self.history), default=0.0),
        }
