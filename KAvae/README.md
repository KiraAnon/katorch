# KAvae

Wan2.1 VAE decoder。

## 规格

| 项 | 值 |
|---|---|
| 结构 | Wan2.1（**不是 Flux**！） |
| latent 通道 | 16 |
| 压缩比 | 8× |
| 通道序列 | 384 → 192 → 96 → 3 |

## 文件

| 文件 | 作用 |
|---|---|
| `vae_block.py` | `RMSNorm2D` / `Conv2DNHWC` / `ResBlock` / `VaeAttention` |
| `vae_decoder.py` | `WanVaeDecoder`（4 段） |

## 4 段划分

| 段 | 内容 | 输入 | 输出 |
|---|---|---|---|
| 1 | conv1 + 2×ResBlock + Attn + 3×ResBlock + Resample | (1,16,16,16) | (1,32,32,192) |
| 2 | 3×ResBlock + Resample | (1,32,32,192) | (1,64,64,192) |
| 3 | 3×ResBlock + Resample | (1,64,64,192) | (1,128,128,96) |
| 4 | 3×ResBlock + head | (1,128,128,96) | (1,128,128,3) |

## 关键细节（踩过的坑）

1. **模型文件**：`qwen_image_vae.safetensors`（Wan2.1 结构），**不是 `flux2-vae`**（那是 Flux，512 通道）
2. **Resample 输出通道是 `dim//2`**，不是 dim
3. **nearest ×2 用 CONCAT 实现**（`unsqueeze`/`repeat` 不支持）
4. **Resample 是交织**，不是拼接（顺序不同）
5. **有 conv2(1×1)**，别漏

## latent 归一化（必须做）

```python
# VAE → DiT
"latent = (latent - latents_mean) / latents_std

# DiT → VAE
z = x * latents_std + latents_mean
```

Wan21 的 mean/std：
```
mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
std  = [ 2.8184,  1.4541,  2.3275, 2.6558,  1.2196, 1.7708, 2.6052, 2.0743,
         3.2687,  2.1526, 2.8652, 1.5579,  1.6382, 1.1253, 2.8251, 1.9160]
```

## 精度

**0.078%**（完整 4 段）。
