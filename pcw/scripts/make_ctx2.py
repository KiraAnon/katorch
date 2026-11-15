"""正确的 ctx 生成：Qwen3 编码 -> LLMAdapter(6层) -> 512 零填充。
   这才是 DiT 块真正消费的 context（旧 i_ctx.bin 即此产物）。
"""
import sys, os
sys.path.insert(0, r"D:\comfyui\ComfyUI-master")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import comfy.sd

ANIMA = r"D:\comfyui\ComfyUI-master\models\diffusion_models\anima-base-v1.0.safetensors"
CLIP_P = r"D:\comfyui\ComfyUI-master\models\text_encoders\qwen_3_06b_base.safetensors"
OUT = r"F:\KAtorch\pcw\ctx"
M = 512
MD = 1024
NH = 16
HDD = MD // NH


def rope_cos_sin(seq_len, head_dim=HDD, base=10000.0):
    freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    ang = torch.outer(pos, freq)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def rotate_half(x):
    d = x.shape[-1]
    return torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class Attn(nn.Module):
    def __init__(self, wq, wk, wv, wo, qn, kn):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.qn, self.kn = qn, kn

    def forward(self, x, ctx, cos, sin, cos_ctx=None, sin_ctx=None):
        B, S, _ = x.shape
        q = F.linear(x, self.wq).view(B, S, NH, HDD).transpose(1, 2)
        k_src = ctx if ctx is not x else x
        k = F.linear(k_src, self.wk).view(B, k_src.shape[1], NH, HDD).transpose(1, 2)
        v = F.linear(k_src, self.wv).view(B, k_src.shape[1], NH, HDD).transpose(1, 2)
        q = q / torch.sqrt((q*q).mean(-1, keepdim=True) + 1e-6) * self.qn
        k = k / torch.sqrt((k*k).mean(-1, keepdim=True) + 1e-6) * self.kn
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

    def fwd(self, x, ctx, cos, sin, cos_ctx, sin_ctx):
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


print("加载 anima 权重 ...", flush=True)
sd = load_file(ANIMA, device="cpu")
ad = LLMAdapter(sd).eval()
print("  LLMAdapter 参数量 %.2fM" % (sum(p.numel() for p in ad.parameters())/1e6), flush=True)

print("加载 CLIP ...", flush=True)
clip = comfy.sd.load_clip([CLIP_P], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)


def make(text, name):
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    hidden = cond[0][0]
    meta = cond[0][1]
    if isinstance(hidden, list):
        hidden = hidden[0]
    t5 = meta.get("t5xxl_ids")
    print("  [%s] qwen_hidden=%s  t5_ids len=%d" % (name, tuple(hidden.shape), len(t5)), flush=True)
    with torch.no_grad():
        out = ad(hidden.float(), t5.unsqueeze(0).long())
    arr = out[0].numpy()
    seq = arr.shape[0]
    print("         adapter out %s  std=%.4f  absmax=%.3f"
          % (tuple(arr.shape), arr.std(), np.abs(arr).max()), flush=True)
    buf = np.zeros((M, MD), np.float32)
    buf[:min(seq, M)] = arr[:M]
    p = os.path.join(OUT, name + ".bin")
    buf.astype(np.float16).tofile(p)
    print("         -> %s (%d bytes, %d 行有效)" % (p, os.path.getsize(p), min(seq, M)), flush=True)


os.makedirs(OUT, exist_ok=True)
make("masterpiece, best quality, very aesthetic, 1girl, solo, white dress, "
     "long white hair, standing in shallow water, soft lighting, full body, detailed eyes", "ctx_pos2")
make("", "ctx_neg2")
print("OK")
