# Qwen3 硬件测试开发问题记录

在开发 `test_hardware_qwen3.py` 过程中，遇到了以下两个主要问题：

## 1. 属性错误：`jax.available_devices()` 不存在

### 问题描述
在尝试检测环境中是否存在 TPU 设备时，调用了 `jax.available_devices()`，导致程序抛出 `AttributeError: module 'jax' has no attribute 'available_devices'`。

### 原因分析
在当前版本的 JAX 中，没有 `available_devices` 这个直接属性。

### 解决方案
改用更稳健的 `try-except` 方式来获取特定后端的设备列表：
```python
try:
    tpu_devices = jax.devices("tpu")
except RuntimeError:
    tpu_devices = []
```

---

## 2. 数值一致性错误：CPU 与 TPU 输出不匹配

### 问题描述
在执行 `test_cpu_vs_tpu_consistency` 时，CPU 和 TPU 的输出结果差异巨大（Mismatched elements > 99%），远超 `float32` 的正常误差范围。

### 原因分析
*   **默认精度差异：** 在 TPU 上，JAX 默认的矩阵乘法（matmul）精度通常为了性能被设置为 `bfloat16` 或 `tensorfloat32` 模式。
*   **CPU 行为：** 在 CPU 上，JAX 默认使用全精度 `float32`。
*   这种精度不对等导致了计算图在不同后端执行时结果迅速分叉。

### 解决方案
在测试脚本的 `setUp` 中强制统一 JAX 的全局矩阵乘法精度为 `float32`：
```python
jax.config.update("jax_default_matmul_precision", "float32")
```
设置后，CPU 与 TPU 的输出在 `1e-4` 的公差下顺利通过一致性检查。
