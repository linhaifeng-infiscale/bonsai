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


import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoTokenizer

from bonsai.models.deepseek_v3 import modeling
from bonsai.utils import Sampler


def tokenize(tokenizer, input: list[str]):
    pad_idx = tokenizer.pad_token_id
    # DeepSeek-V3 doesn't have a public chat template in all versions, 
    # we use a simple prompt for the tiny demo.
    lines = [f"User: {l}\nAssistant:" for l in input]
    lines = [tokenizer.encode(line) for line in lines]
    max_len = max(len(line) for line in lines)
    return jnp.array([np.pad(l, (max_len - len(l), 0), constant_values=pad_idx) for l in lines])


def run_tiny_model():
    print("🚀 Initializing Tiny DeepSeek-V3 (MLA + MoE) for demonstration...")
    
    # Tiny configuration that fits in memory
    config = modeling.ModelConfig._from_param(
        use_sharding=False,
        num_layers=4,
        vocab_size=10000,
        emb_dim=256,
        mlp_dim=512,
        moe_intermediate_size=128,
        num_heads=8,
        head_dim=32, # v_head_dim
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.0,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        rope_interleave=True,
        rope_theta=10000,
        rope_scaling_factor=1.0,
        local_rope_theta=10000.0,
        norm_eps=1e-6,
        tie_word_embeddings=False,
    )

    query = [
        "JAX is a powerful library for",
        "The secret of Mixture-of-Experts is",
    ]

    # Use a standard tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenize(tokenizer, query)
    batch_size, token_len = tokens.shape

    generate_steps = 20
    
    # Initialize model with random weights
    rngs = nnx.Rngs(params=0)
    model = modeling.DeepseekV3(config, rngs=rngs)
    
    # Initialize MLA cache
    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    key = jax.random.key(42)
    sampler = Sampler(temperature=0.7, top_p=0.9, top_k=50)
    jit_sampler = jax.jit(sampler)

    print("🏗️  Running Prefill...")
    logits, cache = modeling.forward(model, cache, tokens, tokenizer.pad_token_id)
    next_tokens = jit_sampler(logits, key=key)

    print("✍️  Generating...")
    tokens_list = [next_tokens]
    for i in range(generate_steps):
        logits, cache = modeling.forward(model, cache, next_tokens, tokenizer.pad_token_id)
        next_tokens = jit_sampler(logits, key=key)
        tokens_list.append(next_tokens)

    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    
    print("\n" + "="*30)
    for i, q in enumerate(query):
        seq_tokens = all_output_tokens[i]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Prompt: {q}")
        print(f"Output: {decoded}")
        print("-" * 30)

if __name__ == "__main__":
    run_tiny_model()