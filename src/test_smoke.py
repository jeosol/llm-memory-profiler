# Python script for quick smoke test on local workstation to confirm the GPU, and RTX PRO 5000 72GB reports
# clean phase telemetry by simulating a 1B parameter block (bf16 weights)

import torch
from memory_profiler.cuda_tracker import CudaMemoryTracker

tracker = CudaMemoryTracker(device=0)
print(f"Detected Hardware: {tracker.device_name} ({tracker.total_vram_gb:.1f} GB)")

# Simulate a 1B parameter block in BF16 (~2 GB weights)
model = torch.nn.Linear(16384, 32768, bias=False, dtype=torch.bfloat16, device="cuda:0")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(2):
    tracker.start_step(step)
    x = torch.randn(8, 2048, 16384, dtype=torch.bfloat16, device="cuda:0")

    with tracker.track_phase("forward"):
        out = model(x)
        loss = out.sum()

    with tracker.track_phase("backward"):
        loss.backward()

    with tracker.track_phase("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    telemetry = tracker.end_step()
    print(f"\nStep {step} Peak VRAM: {telemetry.max_step_vram_gb} GB")
    for p in telemetry.phases:
        print(f"  {p.phase_name:<15} | Peak: {p.peak_allocated_gb:>6} GB | Reserved: {p.reserved_gb:>6} GB | {p.elapsed_ms:>6} ms")
