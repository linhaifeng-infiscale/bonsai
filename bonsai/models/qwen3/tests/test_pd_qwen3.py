import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from bonsai.models.qwen3 import modeling
import dataclasses

class TestPDConsistency(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        # 使用完整的 Qwen3-0.6B 配置进行验证
        self.config = modeling.ModelConfig.qwen3_0_6b(use_sharding=False)
        self.batch_size = 1
        self.tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        self.pad_id = 0

    def test_prefill_decode_consistency(self):
        """验证全量 Prefill 和增量 Decode 的输出一致性。"""
        rngs = nnx.Rngs(42)
        model = modeling.Qwen3(self.config, rngs=rngs)
        
        # 缩小权重以增强数值稳定性
        graph, state = nnx.split(model)
        state = jax.tree.map(lambda x: x * 0.01 if isinstance(x, jax.Array) else x, state)
        nnx.update(model, state)

        @nnx.jit
        def forward_step(m, c, t):
            return modeling.forward(m, c, t, self.pad_id)

        # --- 方式 A: 全量 Prefill ---
        cache_prefill = model.init_cache(self.config, self.batch_size, self.tokens.shape[1], 0, dtype=jnp.float32)
        logits_prefill, _ = forward_step(model, cache_prefill, self.tokens)
        
        # --- 方式 B: 增量 Decode ---
        cache_decode = model.init_cache(self.config, self.batch_size, 1, self.tokens.shape[1] - 1, dtype=jnp.float32)
        
        logits_decode = None
        for i in range(self.tokens.shape[1]):
            token_step = self.tokens[:, i : i + 1]
            logits_step, _ = forward_step(model, cache_decode, token_step)
            logits_decode = logits_step
            
        # 对比
        diff = np.abs(logits_prefill - logits_decode)
        print(f"\nMax difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")
        
        np.testing.assert_allclose(
            logits_prefill, 
            logits_decode, 
            rtol=1e-5, 
            atol=1e-5, 
            err_msg="Prefill logits and incremental Decode logits mismatch!"
        )

if __name__ == "__main__":
    absltest.main()
