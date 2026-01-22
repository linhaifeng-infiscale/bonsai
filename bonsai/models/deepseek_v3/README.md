# DeepSeek-V3

DeepSeek-V3 is a large Mixture-of-Experts (MoE) language model with Multi-Head Latent Attention (MLA).

## Usage

```python
import jax
from flax import nnx
from bonsai.models.deepseek_v3 import DeepseekV3, ModelConfig, create_model_from_safe_tensors

# Configuration
config = ModelConfig.deepseek_v3()

# Initialize model
model = DeepseekV3(config, rngs=nnx.Rngs(params=0))

# Or load from checkpoints
# model = create_model_from_safe_tensors("/path/to/checkpoints", config)

# Inference
input_ids = jax.numpy.array([[1, 2, 3]])
cache = model.init_cache(config, batch_size=1, token_len=3, generate_steps=10)
segment_ids = jax.numpy.ones_like(input_ids)
num_right_pads = 0

logits = model(input_ids, segment_ids, cache, num_right_pads)
```
