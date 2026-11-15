# KAdit

Anima DiT（图像生成主干）。

## 规格

| 项 | 值 |
|---|---|
| hidden | 2048 |
| 层数 | 28 |
| heads | 16 × 128 |
| cross-attn context | 1024 |
| patch | 2×2 |
| 输入通道 | 16（latent）+ 1（mask）= 17，patch 后 68 |

## 文件

| 文件 | 作用 |
|---|---|
| `anima_block.py` | 原始 DiTBlock（用旧 LN，**已弃用**） |
| `bal_block.py` | **balanced DiTBlock**（当前方案，用 BalancedNorm） |
| `anima_head.py` | PatchEmbed / TimestepEmbedding / FinalLayer |
| `rope3d.py` | **3D RoPE**（t/h/w 三段，和 ComfyUI 对拍 0.0 误差） |
| `chain.py` | 28 层分块构造 |

## RoPE 关键参数（全踩过）

| 项 | 值 |
|---|---|
| head_dim | 128 |
| dim_t | 44（22 对） |
| dim_h | 42（21 对） |
| dim_w | 42（21 对） |
| **NTK 底数** | **42870.94**（不是 10000！） |
| **cos/sin 排布** | **前后半配对**（不是同半重复） |

## Timestep 编码（容易错）

```python
# 正确
half = 1024
expo = -log(10000) * arange(half) / half
e = t * exp(expo)
emb = cat([cos(e), sin(e)])       # cos 在前！

# 然后
adaln_lora = linear_2(silu(linear_1(emb)))   # [1, 6144]
t_embedding = emb                              # 注意是原始 sincos
t_embedding = t_embedding_norm(t_embedding)    # RMSNorm(2048)
```

## 采样配置（来自工作流）

| 项 | 值 |
|---|---|
| shift | **3.0**（ModelSamplingAuraFlow） |
| 步数 | 30 |
| cfg | 5 |
| sampler | er_sde + simple |
| sigma | `time_snr_shift(3, t) = 3t/(1+2t)` |

## patchify 顺序（踩过）

必须 **`(通道, 高, 宽)`**，不是 `(高, 宽, 通道)`。

## 精度

| 项 | 结果 |
|---|---|
| 单层 NPU vs torch | 0.14% |
| 两层 | 0.20% |
| 28 层（torch） | nan=0 inf=0 |
