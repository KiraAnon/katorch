"""Wan2.1 VAE 的 ResBlock 移植到 MDLA。

结构（decoder.middle.0 / upsamples.N）：
  residual = [ RMS_norm, SiLU, Conv3x3, RMS_norm, SiLU, Conv3x3 ]  (权重索引 .0 .2 .3 .6)
  shortcut = Conv1x1 (in != out 时) 或 Identity

T=1 时 CausalConv3d 退化成 Conv2d：
  取 weight[:, :, 2, :, :]（时间核第 3 片，因为时间维前面 pad 了 2）
  padding 只剩空间 (1,1)

RMS_norm = F.normalize(x, dim=1) * sqrt(C) * gamma，没有 eps 参与缩放。
  fp16 风险：384 个平方和，|x|>13 就溢出 → 先除 8，再把 8 乘回 RSQRT 结果。
"""
import math
import torch
import torch.nn as nn

from .layers import Lin


def conv2d_nhwc(x, w, b, pad=(1, 1, 1, 1), stride=(1, 1), act=0):
    """NHWC 包装：内部转成 NCHW 调标准 conv2d，lowering 会识别 F.conv2d
    并自动做布局转换（NCHW→NHWC→CONV_2D→NCHW）。

    x: [N,H,W,C]  w: [O,C,KH,KW]（OIHW，常量）  b: [O]
    权重必须是原始 OIHW 布局：这样 lowering 能把它当常量预转置成 OHWC。
    """
    xc = x.permute(0, 3, 1, 2)
    if pad[0] == pad[1] == pad[2] == pad[3]:
        xp = xc
        pp = (pad[2], pad[0])
    else:
        xp = torch.nn.functional.pad(xc, (pad[0], pad[1], pad[2], pad[3]))
        pp = 0
    y = torch.nn.functional.conv2d(xp, w, b, stride=(stride[1], stride[0]), padding=pp)
    return y.permute(0, 2, 3, 1)


class RMSNorm2D(nn.Module):
    """通道维 RMSNorm：y = x / ||x||_2 * sqrt(C) * gamma（gamma 预先乘好 sqrt(C)）。"""

    def __init__(self, gamma, C, K=8.0, eps=1e-6):
        super().__init__()
        self.C = C
        self.K = float(K)
        self.eps = eps
        # gamma * sqrt(C) 预先合成
        g = gamma.reshape(-1).float() * math.sqrt(C)
        self.g = nn.Parameter(g.clone())

    def forward(self, x):
        # x: [N, H, W, C]（NHWC，和设备端一致）
        # 推导：xs = x/K，则 sum(xs^2) = sum(x^2)/K^2
        #   rsqrt(sum(xs^2)+eps) = K / sqrt(sum(x^2)+eps*K^2)
        #   要得到 1/sqrt(sum(x^2)+eps*K^2) 需要再除以 K
        xs = x * (1.0 / self.K)
        s = (xs * xs).sum(dim=-1, keepdim=True)
        r = torch.rsqrt(s + self.eps) * (1.0 / self.K)
        return x * r * self.g


class Conv2DNHWC(nn.Module):
    """把 Conv2d 的 OIHW 权重转成 NHWC 布局（O, KH, KW, C），供设备端直接用。

    这里只做转换和持有，实际计算由 host 的 CONV_2D 完成（当前用 torch 模拟）。
    """

    def __init__(self, w5d, b, kh=3, kw=3, pad=1, stride=1):
        super().__init__()
        # w5d: [Cout, Cin, kt, kh, kw] → 取时间片 → [Cout, Cin, kh, kw]（保持 OIHW）
        # CausalConv3d 时间维前面 pad 2，T=1 时用第 3 片（索引 2）；
        # 时间核为 1 时（如 shortcut）没有 padding，取索引 0。
        if w5d.dim() == 5:
            ti = 2 if w5d.shape[2] == 3 else 0
            w = w5d[:, :, ti, :, :]
        else:
            w = w5d
        self.w = nn.Parameter(w.float().contiguous().clone())
        self.b = nn.Parameter(b.float().clone()) if b is not None else None
        self.pad4 = (pad, pad, pad, pad)
        self.stride2 = (stride, stride)

    def forward(self, x):
        return conv2d_nhwc(x, self.w, self.b, self.pad4, self.stride2)

    def weight_nhwc(self):
        """设备端用的 OHWC 布局。"""
        return self.w.detach().permute(0, 2, 3, 1).contiguous()


class ResBlock(nn.Module):
    """VAE 的 ResBlock（T=1，纯 2D）。"""

    def __init__(self, w, C_in, C_out):
        super().__init__()
        self.n1 = RMSNorm2D(w["n1_gamma"], C_in)
        self.c1 = Conv2DNHWC(w["c1_w"], w["c1_b"])
        self.n2 = RMSNorm2D(w["n2_gamma"], C_out)
        self.c2 = Conv2DNHWC(w["c2_w"], w["c2_b"])
        if C_in != C_out:
            self.sc = Conv2DNHWC(w["sc_w"], w["sc_b"], kh=1, kw=1, pad=0)
        else:
            self.sc = None

    def forward(self, x):
        h = self.n1(x)
        h = torch.nn.functional.silu(h)
        h = self.c1(h)
        h = self.n2(h)
        h = torch.nn.functional.silu(h)
        h = self.c2(h)
        s = x if self.sc is None else self.sc(x)
        return h + s


# ---------- 参考实现（对照 ComfyUI 的语义）----------
def ref_resblock(x_nchw, w, C_in, C_out):
    """x_nchw: [1, C, 1, H, W]（带时间维，T=1）。"""
    def rms_norm(x, gamma):
        # F.normalize(x, dim=1) * sqrt(C) * gamma，gamma 要按输入维数对齐
        C = x.shape[1]
        norm = x / x.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        shape = [1] * x.dim()
        shape[1] = C
        return norm * math.sqrt(C) * gamma.reshape(shape)

    def caus_conv(x5, w5, b):
        # x5: [N, C, 1, H, W] → 去掉时间维做 2D conv
        x = x5.squeeze(2)
        ti = 2 if w5.shape[2] == 3 else 0
        w2 = w5[:, :, ti, :, :]
        pad = 1 if w2.shape[-1] == 3 else 0
        return torch.nn.functional.conv2d(x, w2, b, padding=pad).unsqueeze(2)

    x_nchw = x_nchw if x_nchw.dim() == 5 else x_nchw.unsqueeze(2)
    h = rms_norm(x_nchw, w["n1_gamma"])
    h = torch.nn.functional.silu(h)
    h = caus_conv(h, w["c1_w"], w["c1_b"])
    h = rms_norm(h, w["n2_gamma"])
    h = torch.nn.functional.silu(h)
    h = caus_conv(h, w["c2_w"], w["c2_b"])
    if C_in != C_out:
        s = torch.nn.functional.conv2d(x_nchw, w["sc_w"][:, :, 2, :, :], w["sc_b"])
    else:
        s = x_nchw
    return h + s


class VaeAttention(nn.Module):
    """VAE 的单头空间自注意力。

    原实现 to_qkv 是 Conv2d(dim, dim*3, 1) 再 chunk(3)，
    这里拆成 3 个独立 1x1 conv，图里没有切片。
    """

    def __init__(self, dim, w_norm, w_q, w_k, w_v, b_q, b_k, b_v, w_proj, b_proj, H=None, W=None):
        super().__init__()
        self.dim = dim
        self.H, self.W = H, W
        self.norm = RMSNorm2D(w_norm, dim)
        self.q = Conv2DNHWC(w_q, b_q, kh=1, kw=1, pad=0)
        self.k = Conv2DNHWC(w_k, b_k, kh=1, kw=1, pad=0)
        self.v = Conv2DNHWC(w_v, b_v, kh=1, kw=1, pad=0)
        self.proj = Conv2DNHWC(w_proj, b_proj, kh=1, kw=1, pad=0)

    def forward(self, x):
        # x: [N,H,W,C]。h/w/c 必须来自构造时的常量：fx 抓不了 x.shape 解包
        identity = x
        h, w, c = self.H, self.W, self.dim
        n = 1
        xn = self.norm(x)
        q = self.q(xn)          # [N,H,W,C]
        k = self.k(xn)
        v = self.v(xn)

        # 空间展平成序列：[N,H,W,C] -> [N, H*W, C]
        qs = q.reshape(n, h * w, c)
        ks = k.reshape(n, h * w, c)
        vs = v.reshape(n, h * w, c)

        # 单头：score = q @ k^T / sqrt(c)
        scale = 1.0 / (c ** 0.5)
        s = torch.matmul(qs, ks.transpose(-1, -2)) * scale     # [N, HW, HW]
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, vs)                                 # [N, HW, C]

        o = o.reshape(n, h, w, c)
        return self.proj(o) + identity
