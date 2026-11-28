# This Python script instantiates an HuggingFace causal language model from it's
# config (avoiding slow weight downloads when benchmarking memory), runs a real forward,
# backward, and optimizer loop wrapped in DeepSpeed ZeRO-3 Offload, compares observed
# CUDA peaks on the RTX PRO 5000 72GB against AnalyticalMemoryEstimator, and exports both
# JSON telemetry and a clean Markdown variance table.

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import deepspeed
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from memory_profiler.cuda_tracker import CudaMemoryTracker
from memory_profiler.estimator import AnalyticalMemoryEstimator, ModelSpec


def build_deepspeed_config(
    micro_batch_size: int,
    zero_stage: int,
    offload_optimizer: bool,
    offload_params: bool,
) -> dict:
    zero_cfg: dict = {
        "stage": zero_stage,
        "contiguous_gradients": True,
        "overlap_comm": False,
        "sub_group_size": int(1e9),
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": int(1e9),
        "stage3_max_reuse_distance": int(1e9),
    }
    if offload_optimizer:
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if offload_params:
        zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}

    return {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": 1,
        "zero_optimization": zero_cfg,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 1000,
    }


def extract_model_spec(model_id: str, model: torch.nn.Module) -> ModelSpec:
    cfg = model.config
    total_params = sum(p.numel() for p in model.parameters())
    return ModelSpec(
        name=model_id,
        num_params_billions=round(total_params / 1e9, 4),
        num_layers=getattr(cfg, "num_hidden_layers", 32),
        hidden_size=getattr(cfg, "hidden_size", 4096),
        num_attention_heads=getattr(cfg, "num_attention_heads", 32),
    )


def export_markdown_table(report: dict, md_path: Path) -> None:
    est = report["analytical_estimate"]
    obs = report["observed_telemetry"]
    delta_gb = round(obs["peak_vram_gb"] - est["total_vram_gb"], 3)
    delta_pct = round((delta_gb / max(est["total_vram_gb"], 0.001)) * 100.0, 1)

    lines = [
        f"### VRAM Benchmark: `{report['model_id']}` on {obs['gpu']}",
        "",
        f"* **Sequence Length:** `{report['seq_len']}` | **Micro-Batch:** `{report['micro_batch_size']}`",
        f"* **Strategy:** `ZeRO-{report['zero_stage']}` | **CPU Opt Offload:** `{report['offload_optimizer']}` | **Act Checkpointing:** `{report['activation_checkpointing']}`",
        "",
        "| Metric | Analytical Estimate (GB) | Observed Peak (GB) | Variance (Delta) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Static Weights (BF16)** | {est['weights_gb']} | — | — |",
        f"| **Static Gradients (BF16)** | {est['gradients_gb']} | — | — |",
        f"| **Optimizer States (FP32 AdamW)** | {est['optimizer_gb']} | — | — |",
        f"| **Dynamic Activations** | {est['activations_gb']} | — | — |",
        f"| **Total Peak Allocated VRAM** | **{est['total_vram_gb']} GB** | **{obs['peak_vram_gb']} GB** | **{delta_gb:+.3f} GB ({delta_pct:+.1f}%)** |",
        f"| **Peak Reserved (CUDA Caching Allocator)** | — | {obs['peak_reserved_gb']} GB | — |",
        f"| **Mean Step Latency** | — | {obs['mean_step_ms']} ms | — |",
    ]
    md_path.write_text("\n".join(lines) + "\n")


def run_benchmark(args: argparse.Namespace) -> None:
    tracker = CudaMemoryTracker(device=0)
    estimator = AnalyticalMemoryEstimator(target_vram_gb=tracker.total_vram_gb)

    attn_impl = "sdpa" if args.flash_attention else "eager"
    config = AutoConfig.from_pretrained(args.model_id, attn_implementation=attn_impl)

    # Instantiate architecture directly in BF16 without downloading multi-GB weights
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    spec = extract_model_spec(args.model_id, model)
    estimate = estimator.estimate(
        spec=spec,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        sharding_degree=1,
        zero_stage=args.zero_stage,
        offload_optimizer=args.offload_optimizer,
        offload_params=args.offload_params,
        activation_checkpointing=args.activation_checkpointing,
        use_flash_attention=args.flash_attention,
    )

    ds_config = build_deepspeed_config(
        micro_batch_size=args.micro_batch_size,
        zero_stage=args.zero_stage,
        offload_optimizer=args.offload_optimizer,
        offload_params=args.offload_params,
    )

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    vocab_size = getattr(config, "vocab_size", 32000)
    for step in range(args.steps):
        tracker.start_step(step)
        input_ids = torch.randint(
            0, vocab_size, (args.micro_batch_size, args.seq_len), device=tracker.device
        )

        with tracker.track_phase("forward"):
            outputs = engine(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        with tracker.track_phase("backward"):
            engine.backward(loss)

        with tracker.track_phase("optimizer_step"):
            engine.step()

        tracker.end_step()

    # Exclude step 0 warmup allocations when reporting steady-state latency/reserved peaks
    steady_steps = tracker.history[1:] if len(tracker.history) > 1 else tracker.history
    peak_vram = max(s.max_step_vram_gb for s in steady_steps)
    peak_reserved = max(p.reserved_gb for s in steady_steps for p in s.phases)
    mean_step_ms = sum(sum(p.elapsed_ms for p in s.phases) for s in steady_steps) / len(steady_steps)

    report = {
        "model_id": args.model_id,
        "model_spec": spec.model_dump(),
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "zero_stage": args.zero_stage,
        "offload_optimizer": args.offload_optimizer,
        "offload_params": args.offload_params,
        "activation_checkpointing": args.activation_checkpointing,
        "flash_attention": args.flash_attention,
        "analytical_estimate": asdict(estimate),
        "observed_telemetry": {
            "gpu": tracker.device_name,
            "total_vram_gb": round(tracker.total_vram_gb, 2),
            "peak_vram_gb": round(peak_vram, 3),
            "peak_reserved_gb": round(peak_reserved, 3),
            "mean_step_ms": round(mean_step_ms, 2),
        },
        "step_traces": [asdict(s) for s in tracker.history],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{args.model_id.split('/')[-1]}_z{args.zero_stage}_seq{args.seq_len}_bs{args.micro_batch_size}"

    json_path = out_dir / f"{slug}.json"
    md_path = out_dir / f"{slug}.md"

    json_path.write_text(json.dumps(report, indent=2))
    export_markdown_table(report, md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark analytical vs actual LLM training VRAM.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--zero-stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--offload-optimizer", action="store_true")
    parser.add_argument("--offload-params", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="benchmarks/rtx5000_blackwell_72gb")
    run_benchmark(parser.parse_args())
