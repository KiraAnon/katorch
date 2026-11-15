"""BalancedNorm：残差坐标 + mean-abs 预条件。

数学链（严格等价）：
  x_source = d * x_bal,  d = g * r
  y = r * x_bal = x_source / g
  Norm(x_source) == Norm(y)            (g > 0 标量，Norm 尺度不变)
  且 mean(z^2) 被 Cauchy-Schwarz 限制在 [1/2, dim]，与行幅度无关
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT_EPS_DEFAULT = 1e-3     # sqrt(1e-6)


class BalancedNorm(nn.Module):
    """输入 x_bal（已按 d 对齐），内部还原到 y 空间后做 mean-abs 预条件 Norm。"""

    def __init__(self, r, w, b=None, kind="rms", eps=1e-6, K=16.0):
        super().__init__()
        self.kind = kind
        # K：防止 mean() 在 fp16 下累加溢出。
        # 2048 个数求和，只要均值 > 32 就会撞上 65504。K 把 y 和 sqrt_eps 一起缩小：
        #   y'=y/K, eps'=sqrt_eps/K  =>  s'=s/K, a'=K*a, z'=a'c'=a*c=z, e'=e  (完全等价)
        # K 必须让 sqrt_eps/K 仍在 fp16 正常数区间（>= 6e-5），所以取 16。
        self.K = float(K)
        self.register_buffer("r", (r.detach().float() / self.K).clone())
        self.w = nn.Parameter(w.detach().float().clone())
        self.b = None
        if kind == "ln":
            self.b = nn.Parameter((b if b is not None else torch.zeros(w.shape[0])).detach().float().clone())
        self.eps = float(eps)
        # 用 Python float：lowering 会生成 nd=1 的常量；0 维标量 MDLA 不认
        self.sqrt_eps = math.sqrt(eps) / self.K

    # 算 mean(z^2) 之前先把 z 缩 VK 倍：rsqrt 会补回来，数学等价。
    # 不缩的话，长尾输入下单个 z^2 就能冲过 fp16 上限 65504 -> v=inf -> rsqrt(inf)=0 -> 整行归一化成 0。
    # 按 VERIFIED.md 坑#15：K > sqrt(max(z^2) * D / 65504)。
    # 残差是长尾分布，z 的 max 由 (max/g) 决定，浅层和深层能差上千倍，
    # 所以取一个对所有层都够用的值。代价是浅层 zs 变小、精度略降，但不影响收敛。
    VK = 1024.0

    def forward(self, x_bal):
        y = x_bal * self.r
        if self.kind == "rms":
            s = torch.abs(y).mean(-1, keepdim=True) + self.sqrt_eps
            b_ = torch.rsqrt(s)
            a = b_ * b_
            z = a * y
            e = a * self.sqrt_eps
            zs = z * (1.0 / self.VK)
            es = e * (1.0 / self.VK)
            v = (zs * zs).mean(-1, keepdim=True) + es * es
            return zs * torch.rsqrt(v) * self.w
        mu = y.mean(-1, keepdim=True)
        c = y - mu
        s = torch.abs(c).mean(-1, keepdim=True) + self.sqrt_eps
        b_ = torch.rsqrt(s)
        a = b_ * b_
        z = a * c
        e = a * self.sqrt_eps
        zs = z * (1.0 / self.VK)
        es = e * (1.0 / self.VK)
        v = (zs * zs).mean(-1, keepdim=True) + es * es
        return zs * torch.rsqrt(v) * self.w + self.b


def make_d(maxabs):
    """d[c] = 2^ceil(log2(maxabs[c]))；零通道用 1。"""
    m = torch.as_tensor(maxabs, dtype=torch.float64)
    nz = m > 0
    e = torch.zeros_like(m)
    e[nz] = torch.ceil(torch.log2(m[nz]))
    d = torch.pow(torch.tensor(2.0, dtype=torch.float64), e)
    d[~nz] = 1.0
    return d.float()


def make_g(d):
    """g = median(d) 的最近 2 的幂（对数距离，平局向上）。"""
    med = float(d.double().sort().values[d.numel() // 2])
    return math.ldexp(1.0, math.floor(math.log2(med) + 0.5))
