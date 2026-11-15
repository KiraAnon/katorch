"""LLMAdapter 完整实现 + 验证。

结构（6 层）：
  embed(32128,1024) -> [T5 token 嵌入]
  每层: norm_self_attn -> self_attn -> 残差
        norm_cross_attn -> cross_attn(attend 到 Qwen3 输出) -> 残差
        norm_mlp -> mlp(1024->4096->1024) -> 残差
  out_proj(1024->1024) -> norm -> 输出
"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, json, struct, time
sys.path.insert(0, _KA_WORK)
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

H = _KA_HOME
ANIMA = H + "/ComfyUI/models/diffusion_models/anima_v1.safetensors"
MD = 1024
NH = 16
HDD = MD // NH    # 64


def rope_cos_sin(seq_len, head_dim=HDD, base=10000.0):
    freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    ang = torch.outer(pos, freq)          # [S, D/2]
    ang = torch.cat([ang, ang], dim=-1)   # [S, D]  标准 rope：前后半相同
    return ang.cos(), ang.sin()


def rotate_half(x):
    d = x.shape[-1]
    return torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, S, D]，cos/sin: [S, D]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class Attn(nn.Module):
    def __init__(self, wq, wk, wv, wo, qn, kn):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.qn, self.kn = qn, kn

    def forward(self, x, ctx, cos, sin, cos_ctx=None, sin_ctx=None):
        B, S, _ = x.shape
        n_ctx = ctx.shape[1]
        q = F.linear(x, self.wq).view(B, S, NH, HDD).transpose(1, 2)
        k_src = ctx if ctx is not x else x
        k = F.linear(k_src, self.wk).view(B, k_src.shape[1], NH, HDD).transpose(1, 2)
        v = F.linear(k_src, self.wv).view(B, k_src.shape[1], NH, HDD).transpose(1, 2)
        # q/k 的 RMSNorm（per-head）
        q = q / torch.sqrt((q*q).mean(-1, keepdim=True) + 1e-6) * self.qn
        k = k / torch.sqrt((k*k).mean(-1, keepdim=True) + 1e-6) * self.kn
        # rope：q 和 k 长度可能不同（cross-attn 时 k 来自 context）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos_ctx, sin_ctx)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, S, MD)
        return F.linear(o, self.wo)


class Block(nn.Module):
    def __init__(self, sd, n):
        super().__init__()
        p = "net.llm_adapter.blocks.%d." % n
        g = lambda k: sd[p + k].float()
        self.sa = Attn(g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                       g("self_attn.v_proj.weight"), g("self_attn.o_proj.weight"),
                       g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight"))
        self.ca = Attn(g("cross_attn.q_proj.weight"), g("cross_attn.k_proj.weight"),
                       g("cross_attn.v_proj.weight"), g("cross_attn.o_proj.weight"),
                       g("cross_attn.q_norm.weight"), g("cross_attn.k_norm.weight"))
        self.n_sa = g("norm_self_attn.weight")
        self.n_ca = g("norm_cross_attn.weight")
        self.n_mlp = g("norm_mlp.weight")
        self.m0w, self.m0b = g("mlp.0.weight"), g("mlp.0.bias")
        self.m2w, self.m2b = g("mlp.2.weight"), g("mlp.2.bias")
        self.use_self = True

    def fwd(self, x, ctx, cos, sin, cos_ctx, sin_ctx):
        if self.use_self:
            h = x / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-6) * self.n_sa
            x = x + self.sa(h, h, cos, sin, cos, sin)
        h = x / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-6) * self.n_ca
        x = x + self.ca(h, ctx, cos, sin, cos_ctx, sin_ctx)
        h = x / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-6) * self.n_mlp
        h = F.linear(F.gelu(F.linear(h, self.m0w, self.m0b)), self.m2w, self.m2b)
        return x + h


class LLMAdapter(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.embed = sd["net.llm_adapter.embed.weight"].float()
        self.out_proj = sd["net.llm_adapter.out_proj.weight"].float()
        self.out_bias = sd["net.llm_adapter.out_proj.bias"].float()
        self.norm = sd["net.llm_adapter.norm.weight"].float()
        self.blocks = nn.ModuleList([Block(sd, i) for i in range(6)])

    def forward(self, source_hidden, t5_ids):
        S_t = t5_ids.shape[1]
        S_c = source_hidden.shape[1]
        x = F.embedding(t5_ids, self.embed)
        cos, sin = rope_cos_sin(S_t)
        cos_c, sin_c = rope_cos_sin(S_c)
        for b in self.blocks:
            x = b.fwd(x, source_hidden, cos, sin, cos_c, sin_c)
        x = F.linear(x, self.out_proj, self.out_bias)
        return x / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-6) * self.norm


# ---- 加载权重 ----
from safetensors.torch import load_file
print("加载 anima_v1...", flush=True)
sd = load_file(ANIMA, device="cpu")
print("  总 keys: %d" % len(sd), flush=True)

ad = LLMAdapter(sd).eval()
print("  LLMAdapter 参数: %.2f M" % (sum(p.numel() for p in ad.parameters())/1e6))

# ---- 前向 ----
from tokenizers import Tokenizer
tk = Tokenizer.from_file(H + "/ComfyUI/comfy/text_encoders/t5_tokenizer/tokenizer.json")

text = "a cat sitting on a mat, masterpiece, best quality"
ids = torch.tensor([tk.encode(text).ids])
print("\n=== 前向 ===")
print("  T5 ids: %s (len=%d)" % (ids.tolist(), ids.shape[1]))

torch.manual_seed(0)
S_c = 24
src = torch.randn(1, S_c, MD) * 0.5
t0 = time.time()
with torch.no_grad():
    out = ad(src, ids)
print("  输出: %s  std=%.4f  耗时 %.2fs" % (tuple(out.shape), out.std(), time.time()-t0))

# pad 到 512（ComfyUI 的做法）
if out.shape[1] < 512:
    out_p = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
print("  pad 后: %s" % (tuple(out_p.shape),))
out_p.numpy().astype(np.float16).tofile(H + "/npt/e2e/o_ctx_real.bin")
print("  已存 o_ctx_real.bin")
