"""Wan2.1 VAE 的完整 decoder（T=1，纯 2D CNN）。

结构（从 ComfyUI 的 comfy/ldm/wan/vae.py 读出）：
  dims = [384, 384, 384, 192, 96]
  conv1:    16 -> 384        (CausalConv3d k3)
  middle:   ResBlock(384,384), Attention(384), ResBlock(384,384)
  upsamples（按 i 分组）:
    i=0 (384->384): 3x ResBlock(384,384) + Resample(384)
    i=1 (384->384): 3x ResBlock(192,384) + Resample(384)   [in_dim 减半]
    i=2 (384->192): 3x ResBlock(192,192) + Resample(192)
    i=3 (192->96):  3x ResBlock(96,96)                     [最后一组不 upsample]
  head:     RMS_norm(96), SiLU, CausalConv3d(96->3)

共 14 个 ResBlock、1 个 Attention、3 个 Resample。

Resample 的 nearest×2 上采样是纯重排，放在 CPU 做（不进图）；
图里只跑 conv。这样每个图块之间由 CPU 做一次上采样，代价很小。
"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys
sys.path.insert(0, _KA_WORK)
import math
import torch
import torch.nn as nn

from vae_block import ResBlock, VaeAttention, RMSNorm2D, Conv2DNHWC, conv2d_nhwc


class Resample(nn.Module):
    """nearest ×2 上采样 + 3x3 conv。H/W 必须构造时给定。"""

    def __init__(self, w, b, dim, H=None, W=None):
        super().__init__()
        self.H, self.W = H, W
        self.conv = Conv2DNHWC(w, b, kh=3, kw=3, pad=1)

    def forward(self, x):
        # 必须是「交织」复制：cat([x,x]) 得到的是 [w0..w15, w0..w15]（拼接），
        # 会把图切成两半。用 reshape+cat 得到 [w0,w0,w1,w1,..]，和 nearest 一致。
        # lowering 没有 repeat_interleave，所以走 reshape + CONCAT。
        n, h, w, c = x.shape
        x = x.reshape(n, h, w, 1, c)
        x = torch.cat([x, x], dim=3).reshape(n, h, 2 * w, c)
        x = x.reshape(n, h, 1, 2 * w, c)
        x = torch.cat([x, x], dim=2).reshape(n, 2 * h, 2 * w, c)
        return self.conv(x)


def load_vae(sd, pfx, dim, C_in, C_out, has_shortcut=None):
    def f(v): return None if v is None else v.float()
    w = dict(
        n1_gamma=f(sd[pfx + "residual.0.gamma"]),
        c1_w=f(sd[pfx + "residual.2.weight"]), c1_b=f(sd[pfx + "residual.2.bias"]),
        n2_gamma=f(sd[pfx + "residual.3.gamma"]),
        c2_w=f(sd[pfx + "residual.6.weight"]), c2_b=f(sd[pfx + "residual.6.bias"]),
        sc_w=f(sd.get(pfx + "shortcut.weight")), sc_b=f(sd.get(pfx + "shortcut.bias")))
    return ResBlock(w, C_in, C_out)


class WanVaeDecoder(nn.Module):
    def __init__(self, sd, H0=8, W0=8):
        super().__init__()
        self.H0, self.W0 = H0, W0
        def f(v): return None if v is None else v.float()
        P = "decoder."

        # WanVAE.decode 的第一步是顶层 conv2（1x1, 16->16），不是 decoder.conv1。
        # 漏掉这一步，后面全错（实测相关系数从 0.31 变 1.00）。
        self.conv2 = Conv2DNHWC(f(sd["conv2.weight"]), f(sd["conv2.bias"]), kh=1, kw=1, pad=0)
        self.conv1 = Conv2DNHWC(f(sd[P + "conv1.weight"]), f(sd[P + "conv1.bias"]), kh=3, kw=3, pad=1)

        self.mid0 = load_vae(sd, P + "middle.0.", 384, 384, 384)
        qkv_w = f(sd[P + "middle.1.to_qkv.weight"])
        qkv_b = f(sd[P + "middle.1.to_qkv.bias"])
        D = 384
        self.mid1 = VaeAttention(D, f(sd[P + "middle.1.norm.gamma"]),
                                 qkv_w[0:D], qkv_w[D:2*D], qkv_w[2*D:3*D],
                                 qkv_b[0:D], qkv_b[D:2*D], qkv_b[2*D:3*D],
                                 f(sd[P + "middle.1.proj.weight"]), f(sd[P + "middle.1.proj.bias"]),
                                 H=H0, W=W0)
        self.mid2 = load_vae(sd, P + "middle.2.", 384, 384, 384)

        # upsamples 按 (层号, 索引) 组织
        self.ups = nn.ModuleList()
        idx = 0
        dims = [384, 384, 384, 192, 96]
        in_dim_last = None
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            cur_in = in_dim
            for _ in range(3):
                self.ups.append(load_vae(sd, "%supsamples.%d." % (P, idx), out_dim, cur_in, out_dim))
                cur_in = out_dim
                idx += 1
            if i != len(dims) - 2:
                self.ups.append(Resample(f(sd["%supsamples.%d.resample.1.weight" % (P, idx)]),
                                         f(sd["%supsamples.%d.resample.1.bias" % (P, idx)]), out_dim,
                                         H=H0 * (2 ** (i + 1)), W=W0 * (2 ** (i + 1))))
                idx += 1
            in_dim_last = out_dim

        self.head_norm = RMSNorm2D(f(sd[P + "head.0.gamma"]), in_dim_last)
        self.head_conv = Conv2DNHWC(f(sd[P + "head.2.weight"]), f(sd[P + "head.2.bias"]), kh=3, kw=3, pad=1)
        self.out_dim = in_dim_last

    def forward(self, z):
        """z: [N,H,W,16] latent -> [N,8H,8W,3] image。"""
        x = self.conv1(self.conv2(z))
        x = self.mid0(x)
        x = self.mid1(x)
        x = self.mid2(x)
        for layer in self.ups:
            x = layer(x)
        x = self.head_norm(x)
        x = torch.nn.functional.silu(x)
        return self.head_conv(x)
