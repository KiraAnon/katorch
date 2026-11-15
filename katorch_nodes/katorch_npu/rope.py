"""3D RoPE（t/h/w 分段），复刻 ComfyUI 的 VideoRopePosition3DEmb。

head_dim=128 -> dim_h=42, dim_w=42, dim_t=44；图像 T=1 时 t 段角度恒为 0。
输出 cos/sin 形状 [S, 1, 128]，配 rotate_half（前后半配对）使用。

h/w 段要乘 NTK 系数：theta = 10000 * ratio ** (dim_h / (dim_h - 2))，
Anima（in_channels=16）的 rope_h_extrapolation_ratio = 4.0。
"""
import torch

HEAD_DIM = 128
DIM_H = HEAD_DIM // 6 * 2      # 42
DIM_W = DIM_H                  # 42
DIM_T = HEAD_DIM - 2 * DIM_H   # 44
NTK_RATIO = 4.0
BASE = 10000.0 * NTK_RATIO ** (DIM_H / (DIM_H - 2))


def rope3d_cos_sin(T, H, W, head_dim=HEAD_DIM, base=BASE):
    """返回 (cos, sin)，形状 [T*H*W, 1, head_dim]，float32。"""
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h

    dim_spatial_range = (torch.arange(0, dim_h, 2)[: dim_h // 2].float() / dim_h)
    dim_temporal_range = (torch.arange(0, dim_t, 2)[: dim_t // 2].float() / dim_t)

    h_spatial_freqs = 1.0 / (base ** dim_spatial_range)
    w_spatial_freqs = 1.0 / (base ** dim_spatial_range)
    temporal_freqs = 1.0 / (base ** dim_temporal_range)

    seq = torch.arange(max(H, W, T), dtype=torch.float)
    half_emb_h = torch.outer(seq[:H], h_spatial_freqs)
    half_emb_w = torch.outer(seq[:W], w_spatial_freqs)
    half_emb_t = torch.outer(seq[:T], temporal_freqs)

    ang_t = half_emb_t[:, None, None, :].expand(T, H, W, dim_t // 2)
    ang_h = half_emb_h[None, :, None, :].expand(T, H, W, dim_h // 2)
    ang_w = half_emb_w[None, None, :, :].expand(T, H, W, dim_w // 2)
    ang = torch.cat([ang_t, ang_h, ang_w], dim=-1).reshape(T * H * W, head_dim // 2)

    # 前后半配对：cos/sin 要拼成 [a0..a63, a0..a63]，交错成 [a0,a0,..] 和 rotate_half 配不上
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).unsqueeze(1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).unsqueeze(1)
    return cos.float(), sin.float()


def apply_rotate_half(x, cos, sin):
    """x: [S,H,D]，cos/sin: [S,1,D]。"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin
