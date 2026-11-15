"""提示词编码：Qwen3-0.6B 前向 + net.llm_adapter -> ctx (512,1024) fp16。

纯 torch 实现，不依赖 ComfyUI 或 transformers。
"""
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- safetensors 读取 ----------------

_DT = {"F32": np.float32, "F16": np.float16, "BF16": None, "I32": np.int32, "I64": np.int64}


def load_st(path, prefixes=None):
    """读 safetensors -> {name: torch.float32 张量}。prefixes 给定时只读匹配的键（省内存）。"""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k, e in hdr.items():
            if k == "__metadata__":
                continue
            if prefixes is not None and not any(k.startswith(p) for p in prefixes):
                continue
            o1, o2 = e["data_offsets"]
            dt = e.get("dtype", "F32")
            f.seek(base + o1)
            raw = f.read(o2 - o1)
            if dt == "BF16":
                arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                continue
            out[k] = torch.from_numpy(np.ascontiguousarray(arr.reshape(e["shape"])))
    return out


# ---------------- Qwen3-0.6B ----------------

NH = 16          # query heads
NKV = 8          # kv heads
HD = 128         # head dim
EPS = 1e-6
ROPE_BASE = 1000000.0


def rms(x, w):
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)) * w


def rope_tables(S, device="cpu"):
    inv = 1.0 / (ROPE_BASE ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    t = torch.arange(S, dtype=torch.float32)
    ang = torch.outer(t, inv)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().to(device), ang.sin().to(device)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    rot = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


class Qwen3(nn.Module):
    def __init__(self, sd, n_layers=28):
        super().__init__()
        self.sd = sd
        self.embed = sd["model.embed_tokens.weight"]
        self.n_layers = n_layers
        self.norm = sd["model.norm.weight"]

    def forward(self, ids):
        x = F.embedding(ids, self.embed)
        S = ids.shape[0]
        cos, sin = rope_tables(S)
        for L in range(self.n_layers):
            p = "model.layers.%d." % L
            h = rms(x, self.sd[p + "input_layernorm.weight"])
            q = F.linear(h, self.sd[p + "self_attn.q_proj.weight"]).view(S, NH, HD)
            k = F.linear(h, self.sd[p + "self_attn.k_proj.weight"]).view(S, NKV, HD)
            v = F.linear(h, self.sd[p + "self_attn.v_proj.weight"]).view(S, NKV, HD)
            q = rms(q, self.sd[p + "self_attn.q_norm.weight"])
            k = rms(k, self.sd[p + "self_attn.k_norm.weight"])
            q = apply_rope(q, cos, sin).transpose(0, 1).unsqueeze(0)
            k = apply_rope(k, cos, sin).transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            k = k.repeat_interleave(NH // NKV, dim=1)
            v = v.repeat_interleave(NH // NKV, dim=1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.squeeze(0).transpose(0, 1).reshape(S, NH * HD)
            x = x + F.linear(o, self.sd[p + "self_attn.o_proj.weight"])
            h = rms(x, self.sd[p + "post_attention_layernorm.weight"])
            g = F.silu(F.linear(h, self.sd[p + "mlp.gate_proj.weight"]))
            u = F.linear(h, self.sd[p + "mlp.up_proj.weight"])
            x = x + F.linear(g * u, self.sd[p + "mlp.down_proj.weight"])
        return rms(x, self.norm)


# ---------------- net.llm_adapter ----------------

class AdapterAttn(nn.Module):
    def __init__(self, sd, p):
        super().__init__()
        self.wq = sd[p + "q_proj.weight"]; self.wk = sd[p + "k_proj.weight"]
        self.wv = sd[p + "v_proj.weight"]; self.wo = sd[p + "o_proj.weight"]
        self.qn = sd[p + "q_norm.weight"]; self.kn = sd[p + "k_norm.weight"]
        self.hd = self.qn.shape[0]

    def forward(self, x, ctx, cos, sin, cos_c, sin_c):
        B, S, _ = x.shape
        nh = self.wq.shape[0] // self.hd
        nkv = self.wk.shape[0] // self.hd
        q = F.linear(x, self.wq).view(B, S, nh, self.hd).transpose(1, 2)
        ks = ctx if ctx is not None else x
        k = F.linear(ks, self.wk).view(B, ks.shape[1], nkv, self.hd).transpose(1, 2)
        v = F.linear(ks, self.wv).view(B, ks.shape[1], nkv, self.hd).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        q = q * cos[None, None] + torch.cat([-q[..., self.hd//2:], q[..., :self.hd//2]], -1) * sin[None, None]
        k = k * cos_c[None, None] + torch.cat([-k[..., self.hd//2:], k[..., :self.hd//2]], -1) * sin_c[None, None]
        if nh != nkv:
            k = k.repeat_interleave(nh // nkv, dim=1)
            v = v.repeat_interleave(nh // nkv, dim=1)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, S, nh * self.hd)
        return F.linear(o, self.wo)


class LLMAdapter(nn.Module):
    def __init__(self, sd, n_blocks=6):
        super().__init__()
        self.embed = sd["net.llm_adapter.embed.weight"]
        self.out_w = sd["net.llm_adapter.out_proj.weight"]
        self.out_b = sd["net.llm_adapter.out_proj.bias"]
        self.norm = sd["net.llm_adapter.norm.weight"]
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            p = "net.llm_adapter.blocks.%d." % i
            self.blocks.append(AdapterAttn(sd, p + "self_attn."))
        self.sa = [b for b in self.blocks]
        self.ca = nn.ModuleList()
        self.nms = []
        self.mlps = []
        for i in range(n_blocks):
            p = "net.llm_adapter.blocks.%d." % i
            self.ca.append(AdapterAttn(sd, p + "cross_attn."))
            self.nms.append((sd[p + "norm_self_attn.weight"], sd[p + "norm_cross_attn.weight"],
                             sd[p + "norm_mlp.weight"]))
            self.mlps.append((sd[p + "mlp.0.weight"], sd[p + "mlp.0.bias"],
                              sd[p + "mlp.2.weight"], sd[p + "mlp.2.bias"]))
        self.hd = self.sa[0].hd

    def _rope(self, S):
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        t = torch.arange(S, dtype=torch.float32)
        ang = torch.outer(t, inv)
        ang = torch.cat([ang, ang], -1)
        return ang.cos(), ang.sin()

    def forward(self, hidden, t5_ids):
        S_t = t5_ids.shape[1]
        x = F.embedding(t5_ids, self.embed)
        cos, sin = self._rope(S_t)
        cos_c, sin_c = self._rope(hidden.shape[1])
        for i, b in enumerate(self.sa):
            ns, nc, nm = self.nms[i]
            h = rms(x, ns)
            x = x + b(h, h, cos, sin, cos, sin)
            h = rms(x, nc)
            x = x + self.ca[i](h, hidden, cos, sin, cos_c, sin_c)
            h = rms(x, nm)
            m0w, m0b, m2w, m2b = self.mlps[i]
            x = x + F.linear(F.gelu(F.linear(h, m0w, m0b)), m2w, m2b)
        x = F.linear(x, self.out_w, self.out_b)
        return rms(x, self.norm)


# ---------------- 对外接口 ----------------

class Encoder(object):
    def __init__(self, qwen_st, anima_st, tok_dir, M=512, MD=1024):
        self.M = M
        self.MD = MD
        print("[enc] 载入 Qwen3 ...", flush=True)
        qsd = load_st(qwen_st)
        n_layers = len(set(k.split(".")[2] for k in qsd if k.startswith("model.layers.")))
        self.qwen = Qwen3(qsd, n_layers).eval()
        print("[enc] Qwen3 层数 = %d" % n_layers, flush=True)
        print("[enc] 载入 tokenizer ...", flush=True)
        from .tok import QwenTok, T5Tok
        qdir = os.path.join(tok_dir, "qwen25_tokenizer")
        if not os.path.isdir(qdir):
            qdir = tok_dir
        self.qtok = QwenTok(os.path.join(qdir, "vocab.json"),
                            os.path.join(qdir, "merges.txt"))
        tpath = os.path.join(tok_dir, "t5_tokenizer", "tokenizer.json")
        if not os.path.exists(tpath):
            tpath = os.path.join(tok_dir, "t5_tok.json")
        self.ttok = T5Tok(tpath)
        print("[enc] 载入 anima(llm_adapter) ...", flush=True)
        asd = load_st(anima_st, prefixes=("net.llm_adapter.",))
        if not asd:
            # 有些 checkpoint（如 miaomiaoHarem_anima12）没有 net. 前缀
            raw = load_st(anima_st, prefixes=("llm_adapter.",))
            asd = {"net." + k: v for k, v in raw.items()}
        if not asd:
            raise RuntimeError("%s 里找不到 llm_adapter 权重" % anima_st)
        self.adapter = LLMAdapter(asd).eval()
        self.asd = asd

    @torch.no_grad()
    def encode(self, text):
        qids = self.qtok.encode(text)
        tids = self.ttok.encode(text, add_eos=True)
        ids = torch.tensor(qids, dtype=torch.long)
        hidden = self.qwen.forward(ids).unsqueeze(0)
        t5 = torch.tensor([tids], dtype=torch.long)
        ctx = self.adapter(hidden, t5)[0].numpy()
        buf = np.zeros((self.M, self.MD), np.float32)
        n = min(len(ctx), self.M)
        buf[:n] = ctx[:n]
        return buf.astype(np.float16)
