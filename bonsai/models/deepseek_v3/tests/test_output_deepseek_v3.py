import os
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from safetensors.torch import save_file
from transformers import AutoConfig
from jax import P

# Import local implementations
from bonsai.models.deepseek_v3 import modeling, params

# Try to import DeepSeek V3 from transformers
try:
    from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM, DeepseekV3Config
except ImportError:
    # Fallback if not registered in AutoConfig yet or path issues, try direct file import if needed
    # But usually assuming the environment is set up with the transformers repo
    import sys
    # Add src to path if needed, but assuming installed or available
    pass

class TestDeepseekV3Outputs(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        
        # Define a tiny config for testing
        self.tiny_config_args = {
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": 1.0,
            "kv_lora_rank": 16,
            "q_lora_rank": 32,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "qk_nope_head_dim": 8,
            "n_group": 1,
            "topk_group": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0, # Test MoE in all layers for coverage
            "norm_topk_prob": True,
            "rope_theta": 10000,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
        }
        
        # Torch Config
        self.torch_config = DeepseekV3Config(**self.tiny_config_args)
        self.torch_model = DeepseekV3ForCausalLM(self.torch_config).eval()
        
        # Bonsai Config
        self.bonsai_config = modeling.ModelConfig._from_param(
            use_sharding=False,
            num_layers=self.tiny_config_args["num_hidden_layers"],
            vocab_size=self.tiny_config_args["vocab_size"],
            emb_dim=self.tiny_config_args["hidden_size"],
            mlp_dim=self.tiny_config_args["intermediate_size"],
            moe_intermediate_size=self.tiny_config_args["moe_intermediate_size"],
            num_heads=self.tiny_config_args["num_attention_heads"],
            head_dim=self.tiny_config_args["v_head_dim"],
            n_shared_experts=self.tiny_config_args["n_shared_experts"],
            n_routed_experts=self.tiny_config_args["n_routed_experts"],
            routed_scaling_factor=self.tiny_config_args["routed_scaling_factor"],
            kv_lora_rank=self.tiny_config_args["kv_lora_rank"],
            q_lora_rank=self.tiny_config_args["q_lora_rank"],
            qk_rope_head_dim=self.tiny_config_args["qk_rope_head_dim"],
            qk_nope_head_dim=self.tiny_config_args["qk_nope_head_dim"],
            n_group=self.tiny_config_args["n_group"],
            topk_group=self.tiny_config_args["topk_group"],
            num_experts_per_tok=self.tiny_config_args["num_experts_per_tok"],
            first_k_dense_replace=self.tiny_config_args["first_k_dense_replace"],
            norm_topk_prob=self.tiny_config_args["norm_topk_prob"],
            rope_interleave=True,
            rope_theta=self.tiny_config_args["rope_theta"],
            rope_scaling_factor=1.0,
            local_rope_theta=10000.0,
            norm_eps=1e-6,
            tie_word_embeddings=False,
        )

        # Save torch model to temp file
        self.temp_dir = tempfile.TemporaryDirectory()
        state_dict = self.torch_model.state_dict()
        save_file(state_dict, os.path.join(self.temp_dir.name, "model.safetensors"))
        
        # Load Bonsai model
        self.nnx_model = params.create_model_from_safe_tensors(self.temp_dir.name, self.bonsai_config)
        
        # Precision matching
        graph_def, state = nnx.split(self.nnx_model)
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
        self.nnx_model = nnx.merge(graph_def, state)

        self.batch_size = 2
        self.seq_len = 10
        self.relaxed_tol = 1e-3

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_nnx_cache(self, batch_size: int):
        return self.nnx_model.init_cache(
            cfg=self.bonsai_config, batch_size=batch_size, token_len=self.seq_len, generate_steps=5, dtype=jnp.float32
        )

    def test_full_forward(self):
        # Create random inputs
        input_ids = torch.randint(0, self.tiny_config_args["vocab_size"], (self.batch_size, self.seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Torch forward
        with torch.no_grad():
            torch_out = self.torch_model(input_ids, attention_mask=attention_mask)
            torch_logits = torch_out.logits
        
        # Jax forward
        jax_input_ids = jnp.array(input_ids.numpy())
        jax_segment_ids = jnp.array(attention_mask.numpy())
        cache = self._init_nnx_cache(self.batch_size)
        
        # We need to run step by step or modify call for full sequence?
        # The modeling.py `__call__` does full sequence processing (loop over layers with x).
        # It takes `tokens` and `segment_ids`.
        
        # num_right_pads for last token logits extraction in forward wrapper, 
        # but here we call model directly for full sequence logits?
        # model.__call__ returns logits for all tokens?
        # Yes: `logits = self.lm_head(self.final_norm(x))` where x is sequence.
        
        # Need to handle cache update? The `Attention` module updates cache in-place.
        # But for full sequence (prefill), we usually provide the full sequence.
        # The `Attention` implementation:
        # `cache.v_cache.value = jax.lax.dynamic_update_slice(...)`
        # This updates based on `cur_ind`.
        # So we should be fine calling it once for the whole sequence.
        
        num_right_pads = 0 # assuming no padding for this test
        jax_logits = self.nnx_model(jax_input_ids, jax_segment_ids, cache, num_right_pads)
        
        # Compare
        torch_logits_np = torch_logits.numpy()
        jax_logits_np = np.array(jax_logits)
        
        # Tolerances might need adjustment due to BF16/FP32 differences if original model was BF16, 
        # but here we initialized random weights in FP32 (default torch) and cast jax to FP32.
        
        np.testing.assert_allclose(jax_logits_np, torch_logits_np, rtol=self.relaxed_tol, atol=self.relaxed_tol)

if __name__ == "__main__":
    absltest.main()
