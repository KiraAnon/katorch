"""从 safetensors 权重直接构造 NptDiTBlock，绕开 ComfyUI 的 ops 系统。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import json
import struct
import sys

sys.path.insert(0, _KA_WORK)
import numpy as np
import torch
import torch.nn as nn

from blockbuild import NptLayerNorm, NptRMSNorm, NptGelu

SP = _KA_MODELS + "/diffusion_models/anima_v1.safetensors"

# ---- 权重读取 ----
with open(SP, "rb") as f:
    _n = struct.unpack("<Q", f.read(8))[0]
    _hdr = json.loads(f.read(_n))
_BASE = 8 + _n


def _bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load(name):
    e = _hdr[name]
    o1, o2 = e["data_offsets"]
    with open(SP, "rb") as f:
        f.seek(_BASE + o1)
        raw = f.read(o2 - o1)
    return torch.from_numpy(np.ascontiguousarray(_bf16(raw).reshape(e["shape"]), np.float32))


def has(name):
    return name in _hdr


# ---- 层 ----
class Lin(nn.Module):
    def __init__(self, w, b=None):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.b = nn.Parameter(b.clone()) if b is not None else None

    def forward(self, x):
        y = torch.matmul(x, self.w.T)
        return y if self.b is None else y + self.b


class RMS(nn.Module):
    def __init__(self, w, eps=1e-6, k=1.0):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.eps = eps
        self.K = k

    def forward(self, x):
        xs = x * (1.0 / self.K)
        v = (xs * xs).mean(-1, keepdim=True)
        inv = torch.rsqrt(v + self.eps)
        return xs * inv * self.w


class LN(nn.Module):
    def __init__(self, w=None, b=None, dim=2048, eps=1e-6, k=64.0):
        super().__init__()
        self.K = k
        self.w = nn.Parameter(w.clone() if w is not None else torch.ones(dim))
        self.b = nn.Parameter(b.clone() if b is not None else torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        xs = x * (1.0 / self.K)
        m = xs.mean(-1, keepdim=True)
        d = xs - m
        v = (d * d).mean(-1, keepdim=True)
        inv = torch.rsqrt(v + self.eps)
        return d * inv * self.w + self.b


class GELU(nn.Module):
    def forward(self, x):
        k = 0.7978845608028654
        return 0.5 * x * (1.0 + torch.tanh(k * (x + 0.044715 * x * x * x)))


class AdaLN3(nn.Module):
    """[SiLU, Lin(in,mid), Lin(mid,3D)]，最后一层权重切成 3 份。"""

    def __init__(self, l1w, l2w):
        super().__init__()
        self.l1 = Lin(l1w)
        D = l2w.shape[0] // 3
        self.parts = nn.ModuleList([Lin(l2w[i * D:(i + 1) * D].clone()) for i in range(3)])
        self.D = D

    def forward(self, emb, lora_sh, lora_sc, lora_ga):
        # ComfyUI 是 Sequential(SiLU, Linear(2048->256), Linear(256->3D))，
        # silu 在最前面，不是套在 l1 外面
        h = self.l1(torch.nn.functional.silu(emb))
        return (self.parts[0](h) + lora_sh,
                self.parts[1](h) + lora_sc,
                self.parts[2](h) + lora_ga)


class Attn(nn.Module):
    def __init__(self, qw, kw, vw, ow, qn, kn, n_head, head_dim, is_self, S=4, M=4):
        super().__init__()
        self.S, self.M = S, M
        self.q, self.k, self.v, self.o = Lin(qw), Lin(kw), Lin(vw), Lin(ow)
        self.qn = RMS(qn)
        self.kn = RMS(kn)
        self.H, self.D = n_head, head_dim
        self.is_self = is_self
        self.scale = 1.0 / (head_dim ** 0.5)
        P = torch.zeros(head_dim, head_dim)
        for i in range(head_dim // 2):
            P[i, i + head_dim // 2] = 1.0
            P[i + head_dim // 2, i] = -1.0
        self.register_buffer("rotP", P)

    def forward(self, x, context, cos, sin):
        S, M = self.S, self.M
        kv = x if context is None else context
        H, D = self.H, self.D
        q = self.q(x).reshape(S, H, D)
        k = self.k(kv).reshape(M, H, D)
        v = self.v(kv).reshape(M, H, D)
        q = self.qn(q)
        k = self.kn(k)
        if self.is_self:
            q = q * cos + torch.matmul(q, self.rotP) * sin
            k = k * cos + torch.matmul(k, self.rotP) * sin
        qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        sc = torch.matmul(qt, kt.transpose(-1, -2)) * self.scale
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p, vt).transpose(0, 1).reshape(S, H * D)
        return self.o(o)


class DiTBlock(nn.Module):
    def __init__(self, pfx, n_head=16, head_dim=128, hidden=2048, S=4, M=4):
        super().__init__()
        self.ln_sa = LN(dim=hidden)
        self.ln_ca = LN(dim=hidden)
        self.ln_mlp = LN(dim=hidden)
        self.attn_sa = Attn(load(pfx + "self_attn.q_proj.weight"), load(pfx + "self_attn.k_proj.weight"),
                            load(pfx + "self_attn.v_proj.weight"), load(pfx + "self_attn.output_proj.weight"),
                            load(pfx + "self_attn.q_norm.weight"), load(pfx + "self_attn.k_norm.weight"),
                            n_head, head_dim, True, S, S)
        self.attn_ca = Attn(load(pfx + "cross_attn.q_proj.weight"), load(pfx + "cross_attn.k_proj.weight"),
                            load(pfx + "cross_attn.v_proj.weight"), load(pfx + "cross_attn.output_proj.weight"),
                            load(pfx + "cross_attn.q_norm.weight"), load(pfx + "cross_attn.k_norm.weight"),
                            n_head, head_dim, False, S, M)
        self.mlp_in = Lin(load(pfx + "mlp.layer1.weight"))
        self.mlp_out = Lin(load(pfx + "mlp.layer2.weight"))
        self.mlp_act = GELU()
        self.ad_sa = AdaLN3(load(pfx + "adaln_modulation_self_attn.1.weight"), load(pfx + "adaln_modulation_self_attn.2.weight"))
        self.ad_ca = AdaLN3(load(pfx + "adaln_modulation_cross_attn.1.weight"), load(pfx + "adaln_modulation_cross_attn.2.weight"))
        self.ad_mlp = AdaLN3(load(pfx + "adaln_modulation_mlp.1.weight"), load(pfx + "adaln_modulation_mlp.2.weight"))

    def forward(self, x, emb, lora_sh, lora_sc, lora_ga, context, cos, sin):
        # 单 batch 单帧：emb[B,T,D] 压成 [D]，和 2D 序列 [S,D] 广播
        sh, sc, ga = self.ad_sa(emb, lora_sh, lora_sc, lora_ga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        n = self.ln_sa(x) * (1.0 + sc) + sh
        x = x + ga * self.attn_sa(n, None, cos, sin)
        sh, sc, ga = self.ad_ca(emb, lora_sh, lora_sc, lora_ga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        n = self.ln_ca(x) * (1.0 + sc) + sh
        x = x + ga * self.attn_ca(n, context, cos, sin)
        sh, sc, ga = self.ad_mlp(emb, lora_sh, lora_sc, lora_ga)
        sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
        n = self.ln_mlp(x) * (1.0 + sc) + sh
        x = x + ga * self.mlp_out(self.mlp_act(self.mlp_in(n)))
        return x
