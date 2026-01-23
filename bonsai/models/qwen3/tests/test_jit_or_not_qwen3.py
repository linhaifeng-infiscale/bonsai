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
from absl.testing import absltest
from flax import nnx
from bonsai.models.qwen3 import modeling

class TestJitConsistency(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # 统一精度
        jax.config.update("jax_default_matmul_precision", "float32")
        # 使用 Qwen3-0.6B 配置
        self.config = modeling.ModelConfig.qwen3_0_6b(use_sharding=False)
        self.batch_size = 1
        self.seq_len = 10
        self.tokens = jax.random.randint(jax.random.key(0), (self.batch_size, self.seq_len), 0, self.config.vocab_size)
        self.pad_id = 0

    def test_jit_vs_no_jit(self):
        """对比开启 JIT 和禁用 JIT 的输出结果。"""
        rngs = nnx.Rngs(42)
        model = modeling.Qwen3(self.config, rngs=rngs)
        
        # 缩小权重以增强数值稳定性
        graph, state = nnx.split(model)
        state = jax.tree.map(lambda x: x * 0.01 if isinstance(x, jax.Array) else x, state)
        nnx.update(model, state)

        # 1. 开启 JIT 运行
        # modeling.forward 已经被 @jax.jit 装饰
        cache_jit = model.init_cache(self.config, self.batch_size, self.seq_len, 0, dtype=jnp.float32)
        logits_jit, _ = modeling.forward(model, cache_jit, self.tokens, self.pad_id)
        
        # 2. 禁用 JIT 运行
        # 使用 jax.disable_jit() 上下文管理器强行跳过编译，直接执行 Python 逻辑
        cache_no_jit = model.init_cache(self.config, self.batch_size, self.seq_len, 0, dtype=jnp.float32)
        with jax.disable_jit():
            logits_no_jit, _ = modeling.forward(model, cache_no_jit, self.tokens, self.pad_id)
            
        # 3. 对比结果
        diff = np.abs(logits_jit - logits_no_jit)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\nJIT vs No-JIT Comparison:")
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        # JIT 优化（如算子重排）可能会引入极微小的浮点数差异，通常在 1e-6 级别
        np.testing.assert_allclose(
            logits_jit, 
            logits_no_jit, 
            rtol=1e-5, 
            atol=1e-5, 
            err_msg="JIT outputs and No-JIT outputs mismatch!"
        )
        print("[Success] JIT consistency check passed.")

if __name__ == "__main__":
    absltest.main()
