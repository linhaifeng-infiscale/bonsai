import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl import logging
from flax import nnx

from bonsai.models.qwen3 import modeling

class TestHardwareQwen3(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        self.devices = jax.devices()
        self.platform = self.devices[0].platform
        logging.info(f"--- Hardware Test Context ---")
        logging.info(f"Platform: {self.platform}")
        logging.info(f"Devices: {self.devices}")
        logging.info(f"Device Count: {jax.device_count()}")
        logging.info(f"-----------------------------")
        
        # Use a smaller config for faster hardware verification if on CPU
        # but keep it representative.
        self.config = modeling.ModelConfig.qwen3_0_6b(use_sharding=False)
        self.batch_size = 2
        self.seq_len = 8
        self.gen_steps = 4

    def test_cpu_vs_tpu_consistency(self):
        """Compares model output on CPU vs TPU to ensure numerical consistency."""
        try:
            cpu_devices = jax.devices("cpu")
        except RuntimeError:
            cpu_devices = []

        try:
            tpu_devices = jax.devices("tpu")
        except RuntimeError:
            tpu_devices = []

        if not cpu_devices or not tpu_devices:
            self.skipTest("Skipping CPU vs TPU consistency test: Both CPU and TPU devices are required.")

        cpu_dev = cpu_devices[0]
        tpu_dev = tpu_devices[0]
        logging.info(f"Comparing CPU ({cpu_dev}) vs TPU ({tpu_dev})")

        # 1. Setup shared initial state
        rngs = nnx.Rngs(42)
        # Create model structure (values don't matter much yet as we will sync them)
        model_ref = modeling.Qwen3(self.config, rngs=rngs)
        graph, state = nnx.split(model_ref)

        # 2. Helper to run on specific device
        def run_on_device(device, state_data):
            # Move state to device
            state_dev = jax.device_put(state_data, device)
            model_dev = nnx.merge(graph, state_dev)
            
            # Create inputs on device
            with jax.default_device(device):
                tokens = jax.random.randint(jax.random.key(0), (self.batch_size, self.seq_len), 0, self.config.vocab_size)
                # Ensure cache is created on the correct device
                cache = model_dev.init_cache(self.config, self.batch_size, self.seq_len, 1, dtype=jnp.float32)
            
            # JIT compile for specific device backend
            @jax.jit
            def step(m, c, t):
                return modeling.forward(m, c, t, 0)
            
            # Run
            logits, _ = step(model_dev, cache, tokens)
            return logits

        # 3. Execution
        logging.info("Running on CPU...")
        logits_cpu = run_on_device(cpu_dev, state)
        
        logging.info("Running on TPU...")
        logits_tpu = run_on_device(tpu_dev, state)

        # 4. Verification
        # Convert to numpy for comparison
        res_cpu = np.array(logits_cpu)
        res_tpu = np.array(logits_tpu)

        # Check tolerance (float32 operations might have slight divergence across backends)
        # 1e-4 is standard for cross-backend float32
        np.testing.assert_allclose(res_cpu, res_tpu, rtol=1e-4, atol=1e-4, err_msg="Mismatch between CPU and TPU outputs")
        logging.info("CPU and TPU outputs match within tolerance.")

    def test_compilation_and_execution(self):
        """Tests that the model compiles and runs on the available hardware."""
        rngs = nnx.Rngs(42)
        
        logging.info("Initializing model...")
        t0 = time.time()
        model = modeling.Qwen3(self.config, rngs=rngs)
        init_time = time.time() - t0
        logging.info(f"Initialization took {init_time:.4f}s")

        tokens = jax.random.randint(jax.random.key(0), (self.batch_size, self.seq_len), 0, self.config.vocab_size)
        pad_id = 0

        # Use float32 for wider hardware compatibility and precision testing
        cache = model.init_cache(self.config, self.batch_size, self.seq_len, self.gen_steps, dtype=jnp.float32)

        @nnx.jit
        def step(m, c, t):
            return modeling.forward(m, c, t, pad_id)

        # 1. Prefill
        logging.info("Running prefill (first run includes JIT compilation)...")
        t0 = time.time()
        logits, cache = step(model, cache, tokens)
        jax.block_until_ready(logits)
        prefill_time = time.time() - t0
        logging.info(f"Prefill time: {prefill_time:.4f}s")

        # 2. Decode steps
        logging.info(f"Running {self.gen_steps} decode steps...")
        next_token = jnp.argmax(logits, axis=-1)[:, None]
        
        latencies = []
        for i in range(self.gen_steps):
            t_step = time.time()
            logits, cache = step(model, cache, next_token)
            jax.block_until_ready(logits)
            step_time = time.time() - t_step
            latencies.append(step_time)
            next_token = jnp.argmax(logits, axis=-1)[:, None]
            logging.info(f"  Step {i} latency: {step_time:.4f}s")
        
        avg_latency = sum(latencies) / len(latencies)
        logging.info(f"Average decode latency: {avg_latency:.4f}s")

        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (self.batch_size, self.config.vocab_size))

    def test_sharded_hardware_run(self):
        """Tests that sharded model can run if multiple devices are available."""
        if jax.device_count() < 2:
            self.skipTest("Need at least 2 devices for sharded hardware test.")
        
        from jax.sharding import Mesh
        from jax.experimental import mesh_utils
        
        # Determine mesh shape based on available devices
        num_devices = jax.device_count()
        if num_devices >= 4:
            mesh_shape = (2, 2)
            axis_names = ("fsdp", "tp")
        else:
            mesh_shape = (num_devices,)
            axis_names = ("fsdp",)
            
        devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = Mesh(devices, axis_names=axis_names)
        
        config_shard = modeling.ModelConfig.qwen3_0_6b(use_sharding=True)
        rngs = nnx.Rngs(42)
        
        with mesh:
            model = modeling.Qwen3(config_shard, rngs=rngs)
            tokens = jax.random.randint(jax.random.key(0), (self.batch_size, self.seq_len), 0, config_shard.vocab_size)
            cache = model.init_cache(config_shard, self.batch_size, self.seq_len, 1, dtype=jnp.float32)
            
            @nnx.jit
            def step(m, c, t):
                return modeling.forward(m, c, t, 0)
            
            logging.info(f"Running sharded model on {num_devices} devices...")
            logits, _ = step(model, cache, tokens)
            jax.block_until_ready(logits)
            logging.info("Sharded run successful.")
            
        self.assertIsNotNone(logits)

if __name__ == "__main__":
    absltest.main()
