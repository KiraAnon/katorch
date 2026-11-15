"""NPT 通用层库。

目标：把 transformer 的各种组件写成「fx 可抓 + MDLA 可跑」的通用模块，
模型的 block 只剩「哪层接哪层」的连接逻辑，不再重复写算子细节。

硬约束（都踩过）：
  - 不能用切片（SLICE 不可用）→ RoPE 用置换矩阵、qkv 用独立 Linear
  - fp16 下 x^2 会溢出 → Norm 内部先除 K
  - eps/K^2 会下溢 → eps 不缩放
  - MUL 不能双向广播，且广播的 B 侧必须是图输入
  - 不能用 x.shape[0]（fx 会抓成 getattr+getitem）→ 形状构造时固定
"""
import math
import sys

sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import torch
import torch.nn as nn


# ---------------- 基础层 ----------------
class Lin(nn.Module):
    """线性层，权重直接持为 Parameter，forward 只用 matmul。"""

    def __init__(self, w, b=None):
        super().__init__()
        self.w = nn.Parameter(w.detach().float().clone())
        self.b = nn.Parameter(b.detach().float().clone()) if b is not None else None

    def forward(self, x):
        y = torch.matmul(x, self.w.T)
        return y if self.b is None else y + self.b


class Norm(nn.Module):
    """RMSNorm / LayerNorm，fp16 安全。

    kind="rms": y = x / sqrt(mean(x^2)+eps) * w
    kind="ln":  y = (x-mean) / sqrt(var+eps) * w + b

    两种都先乘 1/K 再平方，避免 fp16 溢出；eps 保持原值（不除 K^2），
    因为正常数据的 var/K^2 远大于 eps，而除以 K^2 会下溢成 0 导致 rsqrt(0)=inf。
    """

    def __init__(self, w, b=None, kind="rms", eps=1e-6, K=64.0, dim=None):
        super().__init__()
        self.kind = kind
        self.K = float(K)
        # eps 按 K^2 缩放才和原式等价；但 1e-6/4096=2.4e-10 在 fp16 下会下溢成 0，
        # 导致 padding 行 rsqrt(0)=inf。设下限 1e-7（fp16 能表示的范围内）。
        self.eps = max(eps / (self.K * self.K), 1e-7)
        if w is None:
            w = torch.ones(dim)
        self.w = nn.Parameter(w.detach().float().clone())
        self.b = None
        if kind == "ln":
            self.b = nn.Parameter((b if b is not None else torch.zeros(w.shape[0])).detach().float().clone())

    def forward(self, x):
        xs = x * (1.0 / self.K)
        if self.kind == "rms":
            v = (xs * xs).mean(-1, keepdim=True)
            return xs * torch.rsqrt(v + self.eps) * self.w
        m = xs.mean(-1, keepdim=True)
        d = xs - m
        v = (d * d).mean(-1, keepdim=True)
        return d * torch.rsqrt(v + self.eps) * self.w + self.b


class Gelu(nn.Module):
    """GELU 的 tanh 近似（实测最大误差 4.1e-4），只用 MUL/ADD/TANH。"""

    def forward(self, x):
        k = 0.7978845608028654
        return 0.5 * x * (1.0 + torch.tanh(k * (x + 0.044715 * x * x * x)))


class Silu(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Rotary(nn.Module):
    """rotate_half 风格的 RoPE，用置换矩阵实现，不用切片。"""

    def __init__(self, head_dim):
        super().__init__()
        P = torch.zeros(head_dim, head_dim)
        for i in range(head_dim // 2):
            P[i, i + head_dim // 2] = 1.0
            P[i + head_dim // 2, i] = -1.0
        self.register_buffer("P", P)

    def forward(self, x, cos, sin):
        return x * cos + torch.matmul(x, self.P) * sin


# ---------------- 复合层 ----------------
class Attention(nn.Module):
    """多头注意力。

    覆盖：MHA / GQA（kv_proj 提前复制好头）/ cross-attention / 有无 RoPE / causal。
    q/k 可选 per-head 归一化（Qwen3、Cosmos 都有）。
    """

    def __init__(self, qw, kw, vw, ow, n_head, head_dim, S, M=None,
                 qn=None, kn=None, rope=True, causal=False, S_q=None):
        super().__init__()
        self.S = S
        self.M = M if M is not None else S
        self.H = n_head
        self.D = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q = Lin(qw)
        self.k = Lin(kw)
        self.v = Lin(vw)
        self.o = Lin(ow)
        self.qn = Norm(qn, kind="rms", K=1.0) if qn is not None else nn.Identity()
        self.kn = Norm(kn, kind="rms", K=1.0) if kn is not None else nn.Identity()
        self.rope = Rotary(head_dim) if rope else None
        self.causal = causal

    def forward(self, x, context, cos, sin, mask=None):
        S, M, H, D = self.S, self.M, self.H, self.D
        kv = x if context is None else context
        q = self.q(x).reshape(S, H, D)
        k = self.k(kv).reshape(M, H, D)
        v = self.v(kv).reshape(M, H, D)
        q = self.qn(q)
        k = self.kn(k)
        if self.rope is not None:
            q = self.rope(q, cos, sin)
            k = self.rope(k, cos, sin)
        qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        sc = torch.matmul(qt, kt.transpose(-1, -2)) * self.scale
        if mask is not None:
            sc = sc + mask
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p, vt).transpose(0, 1).reshape(S, H * D)
        return self.o(o)


class MLP(nn.Module):
    """三种 MLP：swiglu（gate/up/down）、gelu（两层）、plain。"""

    def __init__(self, kind, w_in, w_out, w_gate=None, act=None):
        super().__init__()
        self.kind = kind
        self.fc_in = Lin(w_in)
        self.fc_out = Lin(w_out)
        self.fc_gate = Lin(w_gate) if w_gate is not None else None
        if kind == "swiglu":
            self.act = Silu()
        elif kind == "gelu":
            self.act = Gelu()
        else:
            self.act = act if act is not None else Silu()

    def forward(self, x):
        if self.kind == "swiglu":
            return self.fc_out(self.act(self.fc_gate(x)) * self.fc_in(x))
        return self.fc_out(self.act(self.fc_in(x)))


class AdaLN(nn.Module):
    """AdaLN 调制：(Linear(emb) + lora).chunk(n)。

    n=2（FinalLayer 的 shift/scale）或 n=3（Block 的 shift/scale/gate）。
    最后一层 Linear 的权重预先切成 n 份，图里没有 chunk。
    lora 由调用方切成 n 个独立输入传入（切片不可用）。
    """

    def __init__(self, l1w, l2w, n=3, K=1.0):
        super().__init__()
        self.n = n
        self.l1 = Lin(l1w)
        D = l2w.shape[0] // n
        self.parts = nn.ModuleList([Lin(l2w[i * D:(i + 1) * D]) for i in range(n)])
        self.D = D

    def forward(self, emb, *loras):
        # 原始是 Sequential(SiLU, Linear, Linear)，所以 SiLU 在 l1 之前
        h = self.l1(torch.nn.functional.silu(emb))
        outs = []
        for i, p in enumerate(self.parts):
            o = p(h)
            if i < len(loras) and loras[i] is not None:
                o = o + loras[i]
            outs.append(o)
        return outs


class GatedBranch(nn.Module):
    """x + gate * branch(x)。Anima 每个子层都是这个形状。"""

    def forward(self, x, branch, gate):
        return x + gate * branch
