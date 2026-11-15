"""真实 3D RoPE（t/h/w 分段）：复刻 ComfyUI 的 VideoRopePosition3DEmb。

head_dim=128 -> dim_h=42, dim_w=42, dim_t=44
每段用不同的频率，图像 T=1 时 t 段角度恒为 0。
输出 cos/sin 形状 [S, 1, 128]，配合 rotate_half 使用。
"""
import sys, math
import torch
import numpy as np

HEAD_DIM = 128
DIM_H = HEAD_DIM // 6 * 2      # 42
DIM_W = DIM_H                   # 42
DIM_T = HEAD_DIM - 2 * DIM_H    # 44

# ComfyUI 的 VideoRopePosition3DEmb 会给 h/w 段乘 NTK 系数：
#   h_ntk = ratio ** (dim_h / (dim_h - 2))，theta = 10000 * h_ntk
# in_channels=16 的 Anima 用 rope_h_extrapolation_ratio = 4.0（model_detection.py）
NTK_RATIO = 4.0
BASE = 10000.0 * NTK_RATIO ** (DIM_H / (DIM_H - 2))     # 42870.94


def rope3d_cos_sin(T, H, W, head_dim=HEAD_DIM, base=BASE, fps=None):
    """返回 (cos, sin)，形状 [T*H*W, 1, head_dim]。"""
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h

    dim_spatial_range = (torch.arange(0, dim_h, 2)[: dim_h // 2].float() / dim_h)
    dim_temporal_range = (torch.arange(0, dim_t, 2)[: dim_t // 2].float() / dim_t)

    h_spatial_freqs = 1.0 / (base ** dim_spatial_range)
    w_spatial_freqs = 1.0 / (base ** dim_spatial_range)
    temporal_freqs = 1.0 / (base ** dim_temporal_range)

    seq = torch.arange(max(H, W, T), dtype=torch.float)
    half_emb_h = torch.outer(seq[:H], h_spatial_freqs)     # [H, 21]
    half_emb_w = torch.outer(seq[:W], w_spatial_freqs)     # [W, 21]
    half_emb_t = torch.outer(seq[:T], temporal_freqs)      # [T, 22]

    # 组装成 [T,H,W,64] 的角度
    # t 段：每个 (t,h,w) 用 half_emb_t[t]，扩展到 H,W
    ang_t = half_emb_t[:, None, None, :].expand(T, H, W, dim_t // 2)      # [T,H,W,22]
    ang_h = half_emb_h[None, :, None, :].expand(T, H, W, dim_h // 2)      # [T,H,W,21]
    ang_w = half_emb_w[None, None, :, :].expand(T, H, W, dim_w // 2)      # [T,H,W,21]
    ang = torch.cat([ang_t, ang_h, ang_w], dim=-1)                        # [T,H,W,64]

    # 前后半配对（split-half）：第 k 对是 (k, k+64)，所以 cos/sin 要拼成
    # [a0..a63, a0..a63]。交错成 [a0,a0,a1,a1,..] 是错的，和 rotate_half 配不上。
    a = ang.reshape(T * H * W, head_dim // 2)
    cos = torch.cat([a.cos(), a.cos()], dim=-1).unsqueeze(1)
    sin = torch.cat([a.sin(), a.sin()], dim=-1).unsqueeze(1)
    return cos.float(), sin.float()


def apply_rotate_half(x, cos, sin):
    """x: [S, H, D]，cos/sin: [S, 1, D]。
    rotate_half: 前一半和后一半交换，且后半取负。
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


# ---- 自检 ----
print("=== 3D RoPE 参数 ===")
print("  head_dim=%d, dim_t=%d, dim_h=%d, dim_w=%d" % (HEAD_DIM, DIM_T, DIM_H, DIM_W))
print("  对数: t=%d, h=%d, w=%d, 合计=%d" % (DIM_T//2, DIM_H//2, DIM_W//2, DIM_T//2+DIM_H//2+DIM_W//2))

# 图像 T=1, 8x8
S_H = S_W = 8
cos, sin = rope3d_cos_sin(1, S_H, S_W)
print("\n=== T=1, 8x8 ===")
print("  cos: %s  sin: %s" % (tuple(cos.shape), tuple(sin.shape)))
print("  cos[0,:5] =", cos[0,0,:5].tolist())
print("  sin[0,:5] =", sin[0,0,:5].tolist())
print("  t 段（前44维）是否恒等: cos=%.4f sin=%.6f"
      % (cos[0,0,:DIM_T].mean(), sin[0,0,:DIM_T].abs().max()))
print("  h 段（44..86）:", cos[0,0,DIM_T:DIM_T+4].tolist())

# 旋转自检：角度 0 的维度应该不变
xt = torch.randn(64, 4, HEAD_DIM)
y = apply_rotate_half(xt, cos.expand(-1, 4, -1), sin.expand(-1, 4, -1))
print("\n  旋转后形状:", tuple(y.shape))
print("  t 段（角度0）应不变: %.6f" % (y[..., :DIM_T] - xt[..., :DIM_T]).abs().max())
print("  h 段应改变: %.4f" % (y[..., DIM_T:DIM_T+DIM_H] - xt[..., DIM_T:DIM_T+DIM_H]).abs().max())

# 存下来给 e2e 用
np.save("/data/data/com.termux/files/home/npt/e2e/cos.npy", cos.numpy())
np.save("/data/data/com.termux/files/home/npt/e2e/sin.npy", sin.numpy())
print("\n已存 cos.npy / sin.npy")
