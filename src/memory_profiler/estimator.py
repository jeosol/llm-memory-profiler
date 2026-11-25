# This Python script implements the exact static (2N + 2N + 12N) and dynamic
# activation formulas from DeepMind's Scaling Book to predict VRAM requirements
# before execution

from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    num_params_billions: float = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)

# @dataclass used to make classes that primary store data much cleaner and less repetitive to write
@dataclass
class MemoryBreakdown:
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    total_vram_gb: float
    fits_in_target_vram: bool


class AnalyticalMemoryEstimator:
    """Models static and dynamic VRAM footprints for mixed-precision LLM training."""

    BYTES_PER_GB = 1024**3

    def __init__(self, target_vram_gb: float = 72.0):
        self.target_vram_gb = target_vram_gb

    def estimate(
        self,
        spec: ModelSpec,
        seq_len: int,
        micro_batch_size: int,
        sharding_degree: int = 1,
        zero_stage: Literal[0, 1, 2, 3] = 0,
        offload_optimizer: bool = False,
        offload_params: bool = False,
        activation_checkpointing: bool = False,
        use_flash_attention: bool = True,
    ) -> MemoryBreakdown:
        total_params = spec.num_params_billions * 1e9

        # Static Footprints (BF16/FP16 = 2 bytes, FP32 AdamW = 12 bytes)
        raw_weights = 2 * total_params
        raw_grads = 2 * total_params
        raw_opt = 12 * total_params

        if offload_params:
            weights_bytes = 0.0
        elif zero_stage == 3:
            weights_bytes = raw_weights / sharding_degree
        else:
            weights_bytes = raw_weights

        if zero_stage in (2, 3):
            grads_bytes = raw_grads / sharding_degree
        else:
            grads_bytes = raw_grads

        if offload_optimizer:
            opt_bytes = 0.0
        elif zero_stage in (1, 2, 3):
            opt_bytes = raw_opt / sharding_degree
        else:
            opt_bytes = raw_opt

        # Dynamic Activation Footprint (Korthikanti et al. / Scaling Book formulation)
        s, b, h, a = seq_len, micro_batch_size, spec.hidden_size, spec.num_attention_heads
        if use_flash_attention:
            # FlashAttention eliminates the quadratic (5 * s * a / h) attention matrix term
            per_layer_act = s * b * h * 34
        else:
            per_layer_act = s * b * h * (34 + (5 * s * a) / h)

        if activation_checkpointing:
            # Stores only layer boundary inputs (~2 * s * b * h bytes per layer) + 1 active layer
            act_bytes = (spec.num_layers * 2 * s * b * h) + per_layer_act
        else:
            act_bytes = spec.num_layers * per_layer_act

        w_gb = weights_bytes / self.BYTES_PER_GB
        g_gb = grads_bytes / self.BYTES_PER_GB
        o_gb = opt_bytes / self.BYTES_PER_GB
        a_gb = act_bytes / self.BYTES_PER_GB
        total_gb = w_gb + g_gb + o_gb + a_gb

        return MemoryBreakdown(
            weights_gb=round(w_gb, 2),
            gradients_gb=round(g_gb, 2),
            optimizer_gb=round(o_gb, 2),
            activations_gb=round(a_gb, 2),
            total_vram_gb=round(total_gb, 2),
            fits_in_target_vram=total_gb < (self.target_vram_gb * 0.90),  # 10% CUDA allocator buffer
        )


if __name__ == "__main__":
    llama3_8b = ModelSpec(
        name="Llama-3-8B",
        num_params_billions=8.03,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
    )
    estimator = AnalyticalMemoryEstimator(target_vram_gb=72.0)

    print("--- Baseline BF16 (No Sharding) ---")
    print(estimator.estimate(llama3_8b, seq_len=4096, micro_batch_size=2))

    print("\n--- ZeRO-3 + CPU Optimizer Offload + Activation Checkpoint - Use Flash Attention  ---")
    print(
        estimator.estimate(
            llama3_8b,
            seq_len=4096,
            micro_batch_size=2,
            zero_stage=3,
            offload_optimizer=True,
            activation_checkpointing=True,
            use_flash_attention=False,
        )
    )
        
    print("\n--- ZeRO-3 + CPU Optimizer Offload + Activation Checkpointing ---")
    print(
        estimator.estimate(
            llama3_8b,
            seq_len=4096,
            micro_batch_size=2,
            zero_stage=3,
            offload_optimizer=True,
            activation_checkpointing=True,
            use_flash_attention=True,
        )
    )
