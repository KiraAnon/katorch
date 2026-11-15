"""按块切分模型 -> 多张 Neuron 图 + 清单

用法：
    from blockbuild import build_blocks, MODEL_CFG
    blocks = build_blocks(layers, block_size=8, outdir="/tmp/npt_blk")
    # -> [{"idx":0,"first":0,"last":8,"graph":"...","ops":608,"npz":...}, ...]

为什么按块：
  - 每层一张图：Anima 448 个模型，撞 APUSys 220 上限
  - 一张巨图：编译时间超线性（16 层 94 秒），且内存吃紧
  - 按块：图数 4~7 张，单块编译 < 25 秒
"""

import json
import math
import os

import torch
import torch.nn as nn

from lowering import Lowering

# Qwen3-0.6B 的真实参数（head_dim=128, 16 q heads, 8 kv heads）
MODEL_CFG = dict(hidden=1024, ffn=3072, n_head=16, n_kv=8, head_dim=128, T=4)


class NptRMSNorm(nn.Module):
    K = 64.0

    """fp16 安全的 RMSNorm。

    直接算 x^2 会溢出：hidden 的 std 能到 17、max 到 454，x^2 = 2e5 >> fp16 max 65504，
    变成 inf 后 rsqrt(inf)=0，归一化直接失效，残差一路放大。
    做法：先乘 1/K 再平方，保持 gamma 是原值。数学上完全等价（var 远大于 eps 时）。
    """

    def __init__(self, orig, eps=1e-6, k=None):
        super().__init__()
        self.K = float(k if k is not None else NptRMSNorm.K)
        # y = x*rsqrt(mean(x^2)+eps)*w
        #   = (x/K) * rsqrt(mean((x/K)^2) + eps/K^2) * w
        # 但 eps/K^2 会在 fp16 下下溢成 0（1e-6/4096=2.4e-10 < 6e-8），
        # 于是 padding 行（x=0, var=0）算出 rsqrt(0)=inf，0*inf=NaN 污染整图。
        # 因为正常数据的 var/K^2 远大于 eps，直接用未缩放的 eps 在数学上等价。
        self.weight = nn.Parameter(orig.weight.detach().float().clone())
        self.eps = eps

    def forward(self, x):
        xs = x * (1.0 / self.K)
        var = (xs * xs).mean(-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps)
        return xs * inv * self.weight


class NptGelu(nn.Module):
    """GELU 的 tanh 近似：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))

    MDLA 没有 GELU 原语，但这个展开只用 MUL/ADD/TANH，实测最大误差 4.1e-4。
    """

    def forward(self, x):
        k = 0.7978845608028654
        return 0.5 * x * (1.0 + torch.tanh(k * (x + 0.044715 * x * x * x)))


class NptLayerNorm(nn.Module):
    """fp16 安全的 LayerNorm。

    和 NptRMSNorm 一样先除以 K 再平方，否则 d^2 在 fp16 溢出。
    eps 不缩放（var/K^2 远大于 eps，数学等价，且避免 eps/K^2 下溢）。
    """

    def __init__(self, orig, eps=1e-6, k=64.0):
        super().__init__()
        self.K = float(k)
        w = orig.weight.detach().float().clone() if orig.weight is not None else torch.ones(orig.normalized_shape)
        b = orig.bias.detach().float().clone() if orig.bias is not None else torch.zeros(orig.normalized_shape)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)
        self.eps = eps

    def forward(self, x):
        xs = x * (1.0 / self.K)
        m = xs.mean(-1, keepdim=True)
        d = xs - m
        var = (d * d).mean(-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps)
        return d * inv * self.weight + self.bias


class RepeatKVLinear(nn.Module):
    """把 k/v 投影的输出通道按 head 重复 rep 份，使 GQA 变成普通 attention。
    重排在构造时完成（repeat_interleave 只在建图时跑一次），所以图里没有 expand。"""

    def __init__(self, linear, n_kv, rep, dim):
        super().__init__()
        w = linear.weight.detach().float()
        out_f, inp = w.shape
        w = w.reshape(n_kv, dim, inp).repeat_interleave(rep, dim=0).reshape(out_f * rep, inp)
        self.weight = nn.Parameter(w)
        if linear.bias is not None:
            b = linear.bias.detach().float().reshape(n_kv, dim).repeat_interleave(rep, 0).reshape(-1)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class NptAttention(nn.Module):
    """Qwen3 attention，扁平实现：无 KV cache、无 control flow、无切片。"""

    def __init__(self, attn, cfg):
        super().__init__()
        self.T, self.H, self.KV, self.D = cfg["T"], cfg["n_head"], cfg["n_kv"], cfg["head_dim"]
        rep = self.H // self.KV
        self.q_proj = attn.q_proj
        self.k_proj = RepeatKVLinear(attn.k_proj, self.KV, rep, self.D)
        self.v_proj = RepeatKVLinear(attn.v_proj, self.KV, rep, self.D)
        self.o_proj = attn.o_proj
        # q/k 的输入 std≈0.04，不需要缩放（缩放反而丢精度）
        self.q_norm = NptRMSNorm(attn.q_norm, k=1.0)
        self.k_norm = NptRMSNorm(attn.k_norm, k=1.0)
        self.scale = 1.0 / math.sqrt(self.D)
        # rotate_half: rot[i] = -x[i+d2], rot[i+d2] = x[i]
        # 右乘矩阵 y = x @ P 要求 P[j][i] = coeff(x[j] -> y[i])
        P = torch.zeros(self.D, self.D)
        for i in range(self.D // 2):
            P[i, i + self.D // 2] = 1.0          # y[i]    += x[i+d2]
            P[i + self.D // 2, i] = -1.0         # y[i+d2] += -x[i]
        self.register_buffer("rotP", P)

    def _rope(self, x, cos, sin):
        return x * cos + torch.matmul(x, self.rotP) * sin

    def forward(self, x, mask, cos, sin):
        T, H, KV, D = self.T, self.H, self.KV, self.D
        q = self.q_proj(x).reshape(T, H, D)
        k = self.k_proj(x).reshape(T, H, D)      # 已经在权重层重复成 H 个头
        v = self.v_proj(x).reshape(T, H, D)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        sc = torch.matmul(qt, kt.transpose(-1, -2)) * self.scale + mask
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p, vt).transpose(0, 1).reshape(T, H * D)
        return self.o_proj(o)


class NptBlock(nn.Module):
    """一层 Qwen3：attention + MLP + 两条残差。"""

    def __init__(self, layer, cfg):
        super().__init__()
        self.ln1 = NptRMSNorm(layer.input_layernorm)
        self.attn = NptAttention(layer.self_attn, cfg)
        self.ln2 = NptRMSNorm(layer.post_attention_layernorm)
        self.mlp = layer.mlp

    def forward(self, x, mask, cos, sin):
        r = x
        x = r + self.attn(self.ln1(x), mask, cos, sin)
        r = x
        return r + self.mlp(self.ln2(x))


class BlockRange(nn.Module):
    """把一个块内的多层串成一条链，fx 会展开成一张图。"""

    def __init__(self, layers, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([NptBlock(l, cfg) for l in layers])

    def forward(self, x, mask, cos, sin):
        for b in self.blocks:
            x = b(x, mask, cos, sin)
        return x


def build_blocks(layers, outdir, cfg=None, block_size=8, verbose=True):
    """把 layers 切成 ceil(n/block_size) 张图。返回块清单。"""
    cfg = dict(cfg or MODEL_CFG)
    T, H, D = cfg["T"], cfg["hidden"], cfg["head_dim"]
    os.makedirs(outdir, exist_ok=True)
    blocks = []
    n = len(layers)
    for start in range(0, n, block_size):
        chunk = layers[start:start + block_size]
        tag = "blk%02d_%02d" % (start, start + len(chunk))
        low = Lowering(os.path.join(outdir, tag))
        x = low.input((T, H))
        mask = low.input((T, T))
        cos = low.input((T, 1, D))
        sin = low.input((T, 1, D))
        mod = BlockRange(chunk, cfg).eval()
        y = low.lower_module(mod, [x, mask, cos, sin])
        low.output(y)
        errs = low.validate()
        if errs:
            raise RuntimeError("块 %s 形状校验失败: %s" % (tag, errs[:3]))
        gp = os.path.join(outdir, tag + ".txt")
        with open(gp, "w") as f:
            f.write(low.graph_text())
        blocks.append(dict(idx=len(blocks), first=start, last=start + len(chunk),
                           graph=gp, ops=len(low.ops)))
        if verbose:
            print("  %-14s 层 %2d~%-2d  %4d 算子" % (tag, start, start + len(chunk) - 1, len(low.ops)))
    manifest = os.path.join(outdir, "manifest.json")
    with open(manifest, "w") as f:
        json.dump(dict(cfg=cfg, block_size=block_size, total_layers=n, blocks=blocks), f, indent=2)
    return blocks
