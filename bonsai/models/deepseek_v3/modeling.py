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

import dataclasses
import math
from functools import partial
from typing import TypeAlias

import jax
from flax import nnx
from jax import P
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

_K_MASK = jnp.finfo(jnp.bfloat16).min
ShardingSpec = PartitionSpec


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    
    # Attention weights
    q_a_weight: ShardingSpec
    q_b_weight: ShardingSpec
    kv_a_weight: ShardingSpec
    kv_b_weight: ShardingSpec
    o_weight: ShardingSpec
    
    # MLP/MoE weights
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    
    # MoE specific sharding
    expert_gate_up: ShardingSpec # [E, 2*H_inter, H]
    expert_down: ShardingSpec    # [E, H, H_inter]
    router_weight: ShardingSpec  # [E, H]

    rms_norm: ShardingSpec
    
    # Activations
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardingCfg(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            q_a_weight=P(None, None),
            q_b_weight=P(None, None),
            kv_a_weight=P(None, None),
            kv_b_weight=P(None, None),
            o_weight=P(None, None),
            ffw_weight_df=P(None, None),
            ffw_weight_fd=P(None, None),
            expert_gate_up=P(None, None, None),
            expert_down=P(None, None, None),
            router_weight=P(None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
        )

    @staticmethod
    def default():
        # TODO: Define appropriate sharding for DeepSeekV3
        return ShardingCfg(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            q_a_weight=P("tp", "fsdp"),
            q_b_weight=P("tp", "fsdp"),
            kv_a_weight=P("tp", "fsdp"),
            kv_b_weight=P("tp", "fsdp"),
            o_weight=P("tp", "fsdp"),
            ffw_weight_df=P("fsdp", "tp"),
            ffw_weight_fd=P("tp", "fsdp"),
            expert_gate_up=P("expert", "fsdp", "tp"), 
            expert_down=P("expert", "tp", "fsdp"),
            router_weight=P("fsdp", "tp"),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btf=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
        )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    moe_intermediate_size: int
    num_heads: int
    head_dim: int # This is v_head_dim in deepseek config
    
    # DeepSeek Specific
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    norm_topk_prob: bool
    rope_interleave: bool
    
    rope_theta: int
    rope_scaling_factor: float
    local_rope_theta: float
    norm_eps: float
    tie_word_embeddings: bool
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    @classmethod
    def _from_param(cls, use_sharding: bool, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardingCfg.default()
        return cls(**kwargs)

    @classmethod
    def deepseek_v3(cls, use_sharding: bool = False):
        return cls._from_param(
            use_sharding,
            num_layers=61,
            vocab_size=129280,
            emb_dim=7168,
            mlp_dim=18432,
            moe_intermediate_size=2048,
            num_heads=128,
            head_dim=128, # v_head_dim
            n_shared_experts=1,
            n_routed_experts=256,
            routed_scaling_factor=2.5,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            n_group=8,
            topk_group=4,
            num_experts_per_tok=8,
            first_k_dense_replace=3,
            norm_topk_prob=True,
            rope_interleave=True,
            rope_theta=10000, # default from config seems to be implied standard if not set, but deepseek usually 10000
            rope_scaling_factor=1.0,
            local_rope_theta=10000.0,
            norm_eps=1e-6,
            tie_word_embeddings=False,
        )


def shard(x: jnp.ndarray, s: ShardingSpec):
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x


class LayerCache(nnx.Module):
    def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        # Cache stores key_states and value_states
        # key_states: [B, S, H, D_qk_nope + D_qk_rope] -> actually [B, S, num_heads, qk_head_dim]
        # value_states: [B, S, num_heads, v_head_dim]
        # DeepSeek uses MHA-like shape for cache effectively (num_heads) because of the projection up from latent
        
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        
        # NOTE: DeepSeek MLA projects to num_heads heads. 
        # Unlike GQA, here num_kv_heads is technically equal to num_heads effectively after projection for the attention op.
        # But for storage optimization, one might store the compressed latent. 
        # However, as per PyTorch reference 'past_key_values.update(key_states, value_states, ...)', 
        # it caches the projected states.
        
        k_shape = (batch_size, cache_size, cfg.num_heads, self.qk_head_dim)
        v_shape = (batch_size, cache_size, cfg.num_heads, cfg.head_dim)
        
        self.k_cache = shard(nnx.Cache(jnp.zeros(k_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
        self.v_cache = shard(nnx.Cache(jnp.zeros(v_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
        self.size = self.k_cache.shape[1]
        batch_sharding = P(cfg.shd_cfg.act_btnh[0]) if cfg.shd_cfg.act_btnh else P(None)
        self.start_ind = shard(nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32)), batch_sharding)
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))  # scalar for compute efficiency.


Cache: TypeAlias = list[LayerCache]


class Einsum(nnx.Module):
    def __init__(self, einsum_str: str, shape: tuple[int, ...], *, shd: ShardingSpec, rngs: nnx.Rngs):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = shard(nnx.Param(nnx.initializers.normal(dtype=jnp.bfloat16)(rngs.params(), shape)), shd)

    @jax.named_scope("einsum")
    def __call__(self, x: ArrayLike) -> Array:
        return jnp.einsum(self.einsum_str, x, self.w[...])


def _generate_pos_embeddings(
    positions: jax.Array, head_dim: int, rope_theta: int = 10000
) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, interleave=False):
    # q, k: [B, T, H, D]
    # cos, sin: [B, T, D] -> [B, T, 1, D]
    dtype = q.dtype
    cos = cos[:, :, None, :].astype(dtype)
    sin = sin[:, :, None, :].astype(dtype)
    
    if interleave:
        b, t, h, d = q.shape
        q = q.reshape(b, t, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, t, h, d)
        k = k.reshape(b, t, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, t, h, d)
        
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nnx.Module):
    def __init__(self, dim: int, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.scale = shard(nnx.Param(nnx.initializers.ones_init()(rngs.params(), dim, dtype=jnp.bfloat16)), cfg.shd_cfg.rms_norm)
        self.norm_eps = cfg.norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + self.norm_eps)
        return jnp.astype(self.scale[...] * x / rms, dtype)


def count_left_pads(x: jax.Array) -> int:
    """Count left padding tokens."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def count_right_pads(x: jax.Array, pad_id) -> int:
    result = jnp.where(
        jnp.all(x == pad_id, axis=1), x.shape[1], jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


class DeepseekV3Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.cfg = cfg
        
        self.q_lora_rank = cfg.q_lora_rank
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.v_head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        
        linear = partial(nnx.Linear, use_bias=False, dtype=jnp.bfloat16, rngs=rngs)
        
        if self.q_lora_rank is None:
            self.q_proj = shard(linear(cfg.emb_dim, self.num_heads * self.qk_head_dim), self.shd_cfg.q_a_weight)
        else:
            self.q_a_proj = shard(linear(cfg.emb_dim, self.q_lora_rank), self.shd_cfg.q_a_weight)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, cfg, rngs=rngs)
            self.q_b_proj = shard(linear(self.q_lora_rank, self.num_heads * self.qk_head_dim), self.shd_cfg.q_b_weight)
            
        self.kv_a_proj_with_mqa = shard(linear(cfg.emb_dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim), self.shd_cfg.kv_a_weight)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg, rngs=rngs)
        self.kv_b_proj = shard(linear(cfg.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), self.shd_cfg.kv_b_weight)
        
        self.o_proj = shard(linear(self.num_heads * self.v_head_dim, cfg.emb_dim), self.shd_cfg.o_weight)
        self.scale = self.qk_head_dim**-0.5

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        b, t, d = x.shape
        
        # Query Projection
        if self.q_lora_rank is None:
            q_states = self.q_proj(x)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        
        q_states = q_states.reshape(b, t, self.num_heads, self.qk_head_dim)
        # Split q into nope and rope parts
        q_nope, q_rope = jnp.split(q_states, [self.qk_nope_head_dim], axis=-1)
        
        # KV Projection (Compressed)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_input, k_rope = jnp.split(compressed_kv, [self.cfg.kv_lora_rank], axis=-1)
        
        # Project compressed latent to full KV heads
        k_input = self.kv_a_layernorm(k_input)
        kv_states = self.kv_b_proj(k_input)
        kv_states = kv_states.reshape(b, t, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = jnp.split(kv_states, [self.qk_nope_head_dim], axis=-1)
        
        # Prepare RoPE for K
        # k_rope is [B, T, qk_rope_head_dim] -> needs to be broadcast to heads or tiled?
        # PyTorch: k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        # and later expand to heads.
        k_rope = k_rope[:, :, None, :] # [B, T, 1, D_rope]
        k_rope = jnp.broadcast_to(k_rope, (b, t, self.num_heads, self.qk_rope_head_dim))
        
        # RoPE Logic
        # Calculate position ids
        left_pads = count_left_pads(segment_ids)
        if cache.start_ind[...].ndim > 0:
            cache.start_ind[...] = jnp.where(cache.start_ind[...] < 0, left_pads, cache.start_ind[...])
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind[...]
        sin, cos = _generate_pos_embeddings(position_ids, self.qk_rope_head_dim)
        
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, interleave=self.cfg.rope_interleave)
        
        # Concatenate parts
        query_states = jnp.concatenate([q_nope, q_rope], axis=-1) # [B, T, H, D]
        key_states = jnp.concatenate([k_nope, k_rope], axis=-1)   # [B, T, H, D]
        
        # Update Cache
        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], value_states, slice_indices)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], key_states, slice_indices)
        
        # Attention
        # query_states: [B, T, H, D]
        # k_cache: [B, S, H, D]
        attn_logits = jnp.einsum("bthd,bshd->btsh", query_states, cache.k_cache[...]) * self.scale
        
        # Masking
        q_pos = cache.cur_ind[...] + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind[...][:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)
        kv_segment_ids = (ts[None, :] >= cache.start_ind[...][:, None]) & (ts[None, :] < cache.cur_ind[...] + t)
        k_pos = ts[None, :] - cache.start_ind[...][:, None]
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        final_mask = causal_mask & segment_mask # [B, T, S]
        
        attn_mask = final_mask[:, :, :, None] # [B, T, S, 1] broadcast over heads
        attn_logits = jnp.where(attn_mask, attn_logits, _K_MASK)
        
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=2).astype(attn_logits.dtype)
        
        # context: [B, T, H, V]
        context = jnp.einsum("btsh,bshv->bthv", attn_weights, cache.v_cache[...])
        context = context.reshape(b, t, -1)
        
        cache.cur_ind[...] = cache.cur_ind[...] + t
        
        return self.o_proj(context)

class MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, hidden_dim: int, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        linear = partial(nnx.Linear, use_bias=False, dtype=jnp.bfloat16, rngs=rngs)
        self.gate_proj = shard(linear(cfg.emb_dim, hidden_dim), self.shd_cfg.ffw_weight_df)
        self.up_proj = shard(linear(cfg.emb_dim, hidden_dim), self.shd_cfg.ffw_weight_df)
        self.down_proj = shard(linear(hidden_dim, cfg.emb_dim), self.shd_cfg.ffw_weight_fd)

    @jax.named_scope("feed_forward")
    def __call__(self, x: ArrayLike) -> Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        outputs = self.down_proj(activations)
        return outputs


class TopkRouter(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.n_routed_experts = cfg.n_routed_experts
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.top_k = cfg.num_experts_per_tok
        self.routed_scaling_factor = cfg.routed_scaling_factor
        
        self.weight = shard(nnx.Param(nnx.initializers.normal(dtype=jnp.bfloat16)(rngs.params(), (self.n_routed_experts, cfg.emb_dim))), cfg.shd_cfg.router_weight)
        self.e_score_correction_bias = nnx.Param(jnp.zeros((self.n_routed_experts,), dtype=jnp.bfloat16))

    def __call__(self, x: Array) -> tuple[Array, Array]:
        # x: [B, T, D]
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)
        
        # router_logits: [N, E]
        router_logits = jnp.dot(x_flat, self.weight[...].T)
        
        # Helper for routing logic
        router_logits_sigmoid = jax.nn.sigmoid(router_logits)
        router_logits_for_choice = router_logits_sigmoid + self.e_score_correction_bias[...]
        
        # Group logic
        group_scores = router_logits_for_choice.reshape(-1, self.n_group, self.n_routed_experts // self.n_group)
        group_scores = jnp.sum(jnp.sort(group_scores, axis=-1)[..., -2:], axis=-1) # Top 2 in each group
        
        # Select top groups
        # [N, n_group]
        top_groups = jnp.argsort(group_scores, axis=-1)[..., -self.topk_group:] 
        
        group_mask = jnp.zeros_like(group_scores).at[jnp.arange(x_flat.shape[0])[:, None], top_groups].set(1.0)
        
        # Expand mask
        score_mask = group_mask[:, :, None] # [N, G, 1]
        score_mask = jnp.broadcast_to(score_mask, (x_flat.shape[0], self.n_group, self.n_routed_experts // self.n_group))
        score_mask = score_mask.reshape(-1, self.n_routed_experts)
        
        scores_for_choice = jnp.where(score_mask > 0, router_logits_for_choice, 0.0)
        
        # Select top k experts
        topk_indices = jnp.argsort(scores_for_choice, axis=-1)[..., -self.top_k:] # [N, k]
        # Gather weights
        topk_weights = jnp.take_along_axis(router_logits_sigmoid, topk_indices, axis=-1)
        
        if self.cfg.norm_topk_prob:
            topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20)
        
        topk_weights = topk_weights * self.routed_scaling_factor
        
        return topk_indices, topk_weights


class MoE(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.router = TopkRouter(cfg, rngs=rngs)
        
        # Shared Experts
        self.shared_experts = MLP(cfg, cfg.moe_intermediate_size * cfg.n_shared_experts, rngs=rngs)
        
        # Routed Experts (Naive implementation for now, mirroring PyTorch loop)
        # Weights: [E, 2*H_inter, H] for gate_up, [E, H, H_inter] for down
        self.num_experts = cfg.n_routed_experts
        self.inter_dim = cfg.moe_intermediate_size
        self.hidden_dim = cfg.emb_dim
        
        self.gate_up_proj = shard(nnx.Param(nnx.initializers.normal(dtype=jnp.bfloat16)(rngs.params(), (self.num_experts, 2 * self.inter_dim, self.hidden_dim))), cfg.shd_cfg.expert_gate_up)
        self.down_proj = shard(nnx.Param(nnx.initializers.normal(dtype=jnp.bfloat16)(rngs.params(), (self.num_experts, self.hidden_dim, self.inter_dim))), cfg.shd_cfg.expert_down)

    def experts_forward(self, hidden_states, top_k_indices, top_k_weights):
        # hidden_states: [N, D]
        # top_k_indices: [N, k]
        # top_k_weights: [N, k]
        
        final_hidden_states = jnp.zeros_like(hidden_states)
        
        # Parallel approach:
        # 1. Expand input to [N, k, D]
        # 2. Gather weights for the k experts: [N, k, ...weight_shape...]
        # 3. Apply MLP
        # 4. Weighted sum
        
        # Gather expert weights
        # top_k_indices: [N, k]
        
        # gate_up_proj: [E, 2*I, D]
        # selected_gate_up: [N, k, 2*I, D]
        selected_gate_up = jnp.take(self.gate_up_proj[...], top_k_indices, axis=0)
        
        # down_proj: [E, D, I]
        # selected_down: [N, k, D, I]
        selected_down = jnp.take(self.down_proj[...], top_k_indices, axis=0)
        
        # Linear 1: x @ W_gate_up.T
        # [N, 1, D] @ [N, k, D, 2*I] (transpose of selected_gate_up)
        # Einsum: "nkd, nkid -> nki" where i=2*I
        # selected_gate_up is [N, k, 2*I, D]
        gate_up = jnp.einsum("nd, nkid -> nki", hidden_states, selected_gate_up)
        
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = nnx.silu(gate) * up
        
        # Linear 2: hidden @ W_down.T
        # [N, k, I] @ [N, k, I, D] (transpose of selected_down)
        # selected_down is [N, k, D, I]
        out = jnp.einsum("nki, nkdi -> nkd", hidden, selected_down)
        
        # Weighted sum
        # top_k_weights: [N, k] -> [N, k, 1]
        out = out * top_k_weights[:, :, None]
        
        return jnp.sum(out, axis=1)

    def __call__(self, x: Array) -> Array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Shared
        shared_out = self.shared_experts(x_flat)
        
        # Routed
        topk_indices, topk_weights = self.router(x)
        routed_out = self.experts_forward(x_flat, topk_indices, topk_weights)
        
        final = shared_out + routed_out
        return final.reshape(orig_shape)


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.attn = DeepseekV3Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        
        if layer_idx >= cfg.first_k_dense_replace:
            self.mlp = MoE(cfg=cfg, rngs=rngs)
        else:
            self.mlp = MLP(cfg, cfg.mlp_dim, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        outputs = attn_output + self.mlp(self.post_attention_layernorm(attn_output))
        return outputs


class DeepseekV3(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.bfloat16, rngs=rngs),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None if get_abstract_mesh().empty else cfg.shd_cfg.act_btd
        self.layers = nnx.List([DecoderLayer(cfg=cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.lm_head = Einsum(
            einsum_str="BTD,DV->BTV", shape=(cfg.emb_dim, cfg.vocab_size), shd=cfg.shd_cfg.emb_dv, rngs=rngs
        )

    def init_cache(
        self, cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
    ) -> Cache:
        cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
        return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        x = self.embedder.embedding[...].at[(tokens,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], segment_ids)
        logits = self.lm_head(self.final_norm(x))
        return logits


@jax.jit
def forward(model: nnx.Module, cache: Cache, tokens: Array, pad_id: int) -> tuple[Array, nnx.Cache]:
    segment_ids = 1 * (tokens != pad_id)
    num_right_pads = count_right_pads(tokens, pad_id)
    logits = model(tokens, segment_ids, cache, num_right_pads)
    target_ind = tokens.shape[-1] - num_right_pads - 1
    return logits[:, target_ind], cache