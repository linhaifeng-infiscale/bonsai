import os
import warnings
from absl.testing import absltest
import logging

# Aggressively filter warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

# Suppress logging
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Set environment variable to simulate devices BEFORE importing JAX
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from bonsai.models.qwen3 import modeling

class TestShardingConsistency(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # Ensure JAX sees the devices
        self.devices = jax.devices()
        if len(self.devices) < 4:
            self.skipTest(f"Skipping sharding test: found {len(self.devices)} devices, need at least 4. "
                          "Try running with XLA_FLAGS='--xla_force_host_platform_device_count=4'")
        
        # Create a mesh: 2x2 for FSDP and TP
        self.device_mesh = mesh_utils.create_device_mesh((2, 2), self.devices[:4])
        self.mesh = Mesh(self.device_mesh, axis_names=("fsdp", "tp"))

        # Create config for Qwen3-0.6B
        self.config_noshard = modeling.ModelConfig.qwen3_0_6b(use_sharding=False)
        self.config_shard = modeling.ModelConfig.qwen3_0_6b(use_sharding=True)

    def test_sharding_consistency(self):
        # 1. Initialize Reference Model (No Sharding)
        rngs = nnx.Rngs(42)
        model_ref = modeling.Qwen3(self.config_noshard, rngs=rngs)
        
        # 2. Initialize Sharded Model (With Sharding)
        # We initialize inside the mesh to ensure weights are sharded at creation if applicable
        # (Though we will overwrite them to ensure exact match)
        with self.mesh:
            model_shard = modeling.Qwen3(self.config_shard, rngs=rngs)

        # 3. Copy weights from Reference to Sharded to ensure identical starting point
        graph_ref, state_ref = nnx.split(model_ref)
        graph_shard, state_shard = nnx.split(model_shard)
        
        def copy_to_sharded(ref_leaf, shard_leaf):
            # If the destination leaf has sharding info (is a jax.Array with sharding), 
            # put the reference value on that sharding layout.
            if isinstance(shard_leaf, jax.Array) and hasattr(shard_leaf, 'sharding'):
                 return jax.device_put(ref_leaf, shard_leaf.sharding)
            return ref_leaf

        new_state_shard = jax.tree.map(copy_to_sharded, state_ref, state_shard)
        nnx.update(model_shard, new_state_shard)

        # 4. Prepare inputs
        batch_size = 4
        seq_len = 10
        pad_id = 0
        tokens = jax.random.randint(jax.random.key(1), (batch_size, seq_len), 0, self.config_noshard.vocab_size)

        # 5. Run Reference Model
        cache_ref = model_ref.init_cache(self.config_noshard, batch_size, seq_len, 0, dtype=jnp.float32)
        logits_ref, _ = modeling.forward(model_ref, cache_ref, tokens, pad_id)

        # 6. Run Sharded Model
        # We must run inside the mesh context so that `shard()` calls in `forward` work correctly
        with self.mesh:
            cache_shard = model_shard.init_cache(self.config_shard, batch_size, seq_len, 0, dtype=jnp.float32)
            # cache_shard buffers should also be initialized/sharded correctly inside init_cache
            logits_shard, _ = modeling.forward(model_shard, cache_shard, tokens, pad_id)

        # 7. Compare outputs
        # We relax tolerance slightly for distributed math variations
        np.testing.assert_allclose(logits_ref, logits_shard, atol=1e-5, rtol=1e-5)
        print("\nConsistency test passed: Single-device and Multi-device (sharded) outputs match.")

if __name__ == "__main__":
    absltest.main()