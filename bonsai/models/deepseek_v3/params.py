# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.deepseek_v3 import modeling as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        SCALE = None
        MOE_GATE_UP = None # [E, 2I, D] -> same
        MOE_DOWN = None    # [E, D, I] -> same
        ROUTER = None      # [E, D] -> same

    # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    mapping = {
        r"model.embed_tokens.weight": ("embedder.embedding", Transform.EMBED),
        r"model.norm.weight": ("final_norm.scale", Transform.SCALE),
        r"lm_head.weight": ("lm_head.w", Transform.LINEAR),
        
        # Layers
        r"model.layers.([0-9]+).input_layernorm.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        r"model.layers.([0-9]+).post_attention_layernorm.weight": (r"layers.\1.post_attention_layernorm.scale", Transform.SCALE),
        
        # Attention
        # Standard Q proj (if no lora)
        r"model.layers.([0-9]+).self_attn.q_proj.weight": (r"layers.\1.attn.q_proj.kernel", Transform.LINEAR),
        
        # MLA Q
        r"model.layers.([0-9]+).self_attn.q_a_proj.weight": (r"layers.\1.attn.q_a_proj.kernel", Transform.LINEAR),
        r"model.layers.([0-9]+).self_attn.q_a_layernorm.weight": (r"layers.\1.attn.q_a_layernorm.scale", Transform.SCALE),
        r"model.layers.([0-9]+).self_attn.q_b_proj.weight": (r"layers.\1.attn.q_b_proj.kernel", Transform.LINEAR),
        
        # MLA KV
        r"model.layers.([0-9]+).self_attn.kv_a_proj_with_mqa.weight": (r"layers.\1.attn.kv_a_proj_with_mqa.kernel", Transform.LINEAR),
        r"model.layers.([0-9]+).self_attn.kv_a_layernorm.weight": (r"layers.\1.attn.kv_a_layernorm.scale", Transform.SCALE),
        r"model.layers.([0-9]+).self_attn.kv_b_proj.weight": (r"layers.\1.attn.kv_b_proj.kernel", Transform.LINEAR),
        
        # Output
        r"model.layers.([0-9]+).self_attn.o_proj.weight": (r"layers.\1.attn.o_proj.kernel", Transform.LINEAR),
        
        # MLP (Standard)
        r"model.layers.([0-9]+).mlp.gate_proj.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"model.layers.([0-9]+).mlp.up_proj.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"model.layers.([0-9]+).mlp.down_proj.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        
        # MoE Shared Experts
        r"model.layers.([0-9]+).mlp.shared_experts.gate_proj.weight": (r"layers.\1.mlp.shared_experts.gate_proj.kernel", Transform.LINEAR),
        r"model.layers.([0-9]+).mlp.shared_experts.up_proj.weight": (r"layers.\1.mlp.shared_experts.up_proj.kernel", Transform.LINEAR),
        r"model.layers.([0-9]+).mlp.shared_experts.down_proj.weight": (r"layers.\1.mlp.shared_experts.down_proj.kernel", Transform.LINEAR),
        
        # MoE Router
        r"model.layers.([0-9]+).mlp.gate.weight": (r"layers.\1.mlp.router.weight", Transform.ROUTER),
        r"model.layers.([0-9]+).mlp.gate.e_score_correction_bias": (r"layers.\1.mlp.router.e_score_correction_bias", Transform.BIAS),
        
        # MoE Experts
        r"model.layers.([0-9]+).mlp.experts.gate_up_proj": (r"layers.\1.mlp.gate_up_proj", Transform.MOE_GATE_UP),
        r"model.layers.([0-9]+).mlp.experts.down_proj": (r"layers.\1.mlp.down_proj", Transform.MOE_DOWN),
    }
    return mapping


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        # Check if it matches MLP but in a MoE layer or vice versa.
        # Actually, the regex should separate them.
        # But wait, MLP keys like `mlp.gate_proj` might match `mlp.experts...` if not careful?
        # No, `mlp.gate_proj` vs `mlp.experts`.
        # However, `mlp` in `layers.X.mlp` could be `DeepseekV3MLP` or `DeepseekV3MoE`.
        # In `DeepseekV3MoE`, there is no `gate_proj` directly, it has `shared_experts.gate_proj` and `experts...`.
        # So the keys are distinct.
        return None, None
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape, reshape_first = transform
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        
        # Special handling for parameter vs value wrapping if needed, but nnx.to_pure_dict unwraps Params.
        # So state_dict[key] should be an Array.
        
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key} -> {key}: {tensor.shape} vs {state_dict[key].shape}")
        # Only apply sharding if sharding_dict is provided
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        if key not in state_dict:
             raise ValueError(f"Key {key} not found in state dict path for {st_key}")
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> model_lib.DeepseekV3:
    """Load tensors from the safetensors file and create a DeepseekV3 model."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    model = nnx.eval_shape(lambda: model_lib.DeepseekV3(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)
    # Only use sharding if mesh is provided
    sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh)) if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    
    # Track used keys to ensure we don't miss important ones or overwrite
    
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    # Optional logging for ignored keys
                    continue
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value if transform else None, sharding)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    # Tie embeddings if needed (DeepSeek usually doesn't, but config has the flag)
    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    gc.collect()
    return nnx.merge(graph_def, state_dict)
