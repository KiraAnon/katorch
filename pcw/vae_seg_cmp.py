"""逐段对比 NPU 与 torch，找出精度崩在哪一段。

NPU 的中间结果在 /data/local/tmp/vae_run/o1..o4.bin（由 vae_npu_run.py 生成）。
这里用同样的分段跑 torch（fp32），逐段比较。
"""
import os, sys
sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import numpy as np
import torch
import torch.nn as nn

from vae_block import Conv2DNHWC, ResBlock, VaeAttention, RMSNorm2D
from safetensors.torch import load_file

H = "/data/data/com.termux/files/home"
TMP = "/data/local/tmp/vae_run"
HW = int(os.environ.get("KA_HW", "64"))

sd = load_file(H + "/ComfyUI/models/vae/qwen_image_vae.safetensors", device="cpu")
Px = "decoder."


def f(v):
    return None if v is None else v.float()


def mk_res(pfx, C_in, C_out):
    w = dict(
        n1_gamma=f(sd[pfx + "residual.0.gamma"]),
        c1_w=f(sd[pfx + "residual.2.weight"]), c1_b=f(sd[pfx + "residual.2.bias"]),
        n2_gamma=f(sd[pfx + "residual.3.gamma"]),
        c2_w=f(sd[pfx + "residual.6.weight"]), c2_b=f(sd[pfx + "residual.6.bias"]),
        sc_w=f(sd.get(pfx + "shortcut.weight")), sc_b=f(sd.get(pfx + "shortcut.bias")))
    return ResBlock(w, C_in, C_out)


def mk_conv5(name):
    return Conv2DNHWC(f(sd[name + ".weight"]), f(sd[name + ".bias"]))


def up2_t(x):
    return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class Seg1(nn.Module):
    def __init__(s, hw):
        super().__init__()
        s.conv1 = mk_conv5(Px + "conv1")
        s.m0 = mk_res(Px + "middle.0.", 384, 384)
        qkv_w = f(sd[Px + "middle.1.to_qkv.weight"])
        qkv_b = f(sd[Px + "middle.1.to_qkv.bias"])
        Dm = 384
        s.m1 = VaeAttention(Dm, f(sd[Px + "middle.1.norm.gamma"]),
                            qkv_w[0:Dm], qkv_w[Dm:2 * Dm], qkv_w[2 * Dm:3 * Dm],
                            qkv_b[0:Dm], qkv_b[Dm:2 * Dm], qkv_b[2 * Dm:3 * Dm],
                            f(sd[Px + "middle.1.proj.weight"]), f(sd[Px + "middle.1.proj.bias"]),
                            H=hw, W=hw)
        s.m2 = mk_res(Px + "middle.2.", 384, 384)
        s.u0 = mk_res(Px + "upsamples.0.", 384, 384)
        s.u1 = mk_res(Px + "upsamples.1.", 384, 384)
        s.u2 = mk_res(Px + "upsamples.2.", 384, 384)

    def forward(s, z):
        z = s.conv1(z)
        z = s.m0(z)
        z = s.m1(z)
        z = s.m2(z)
        z = s.u0(z)
        z = s.u1(z)
        return s.u2(z)


class Seg2(nn.Module):
    def __init__(s):
        super().__init__()
        s.c3 = mk_conv5(Px + "upsamples.3.resample.1")
        s.u4 = mk_res(Px + "upsamples.4.", 192, 384)
        s.u5 = mk_res(Px + "upsamples.5.", 384, 384)
        s.u6 = mk_res(Px + "upsamples.6.", 384, 384)

    def forward(s, x):
        x = s.c3(x)
        x = s.u4(x)
        x = s.u5(x)
        return s.u6(x)


class Seg3(nn.Module):
    def __init__(s):
        super().__init__()
        s.c7 = mk_conv5(Px + "upsamples.7.resample.1")
        s.u8 = mk_res(Px + "upsamples.8.", 192, 192)
        s.u9 = mk_res(Px + "upsamples.9.", 192, 192)
        s.u10 = mk_res(Px + "upsamples.10.", 192, 192)

    def forward(s, x):
        x = s.c7(x)
        x = s.u8(x)
        x = s.u9(x)
        return s.u10(x)


class Seg4(nn.Module):
    def __init__(s):
        super().__init__()
        s.c11 = mk_conv5(Px + "upsamples.11.resample.1")
        s.u12 = mk_res(Px + "upsamples.12.", 96, 96)
        s.u13 = mk_res(Px + "upsamples.13.", 96, 96)
        s.u14 = mk_res(Px + "upsamples.14.", 96, 96)
        s.hn = RMSNorm2D(f(sd[Px + "head.0.gamma"]), 96)
        s.hc = mk_conv5(Px + "head.2")

    def forward(s, x):
        x = s.c11(x)
        x = s.u12(x)
        x = s.u13(x)
        x = s.u14(x)
        x = s.hn(x)
        x = torch.nn.functional.silu(x)
        return s.hc(x)


def cmp(name, a_t, b_n, path):
    a = a_t.detach().numpy().astype(np.float32).ravel()
    if not os.path.exists(path):
        print("  %-4s 无 NPU 结果 %s" % (name, path), flush=True)
        return
    b = np.fromfile(path, dtype=np.float16).astype(np.float32).ravel()
    if a.size != b.size:
        print("  %-4s 尺寸不一致 torch=%d npu=%d" % (name, a.size, b.size), flush=True)
        return
    d = np.abs(a - b)
    print("  %-4s torch std=%.4f  npu std=%.4f  MAX=%.5f  MEAN=%.6f  rel=%.2f%%"
          % (name, a.std(), b.std(), d.max(), d.mean(), 100 * d.max() / (np.abs(a).max() + 1e-9)), flush=True)


torch.manual_seed(1)
z = torch.randn(1, HW, HW, 16) * 0.5

print("=== 逐段对比 ===", flush=True)
s1 = Seg1(HW).eval()
s2 = Seg2().eval()
s3 = Seg3().eval()
s4 = Seg4().eval()

with torch.no_grad():
    a1 = s1(z)
    cmp("s1", a1, None, TMP + "/o1.bin")

    u1 = up2_t(a1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
    a2 = s2(u1)
    cmp("s2", a2, None, TMP + "/o2.bin")

    u2 = up2_t(a2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
    a3 = s3(u2)
    cmp("s3", a3, None, TMP + "/o3.bin")

    u3 = up2_t(a3.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
    a4 = s4(u3)
    cmp("s4", a4, None, TMP + "/o4.bin")

# 顺便验证 up2 的 numpy 版和 torch interpolate 是否一致
xin = np.random.randn(1, 4, 5, 3).astype(np.float16)
xn = np.repeat(np.repeat(xin, 2, axis=1), 2, axis=2)
xt = torch.nn.functional.interpolate(
    torch.from_numpy(xin.astype(np.float32)).permute(0, 3, 1, 2), scale_factor=2, mode="nearest"
).permute(0, 2, 3, 1).numpy()
print("\nup2 一致性: MAX=%.6f （0 表示 numpy 版和 torch 版相同）" % np.abs(xn.astype(np.float32) - xt).max(), flush=True)
