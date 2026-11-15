"""VAE decoder 上 NPU：512×512 输出，上采样放 CPU。

为什么上采样不在 NPU：MDLA 的 CONCAT 只支持「沿最后一维」，而 nearest 2× 需要在
H/W 维复制，构造出来的图会带 5 维张量 + 第 3 维 concat，编译直接 rc=-516
（实测 hw=8/16/24/32/48/64 全部失败，不是尺寸问题，是这个写法 MDLA 就不接受）。

新的分段（上采样在段间用 numpy 做，代价是几 MB 的内存拷贝）：

  段1  conv1 + mid(Res,Attn,Res) + 3×Res                 (1,64,64,16)  → (1,64,64,384)
  CPU  上采样 ×2  → (1,128,128,384)，同时做 384→192 的 conv 不行，conv 留在 NPU
  段2  conv(384→192) + 3×Res                             (1,128,128,384) → (1,128,128,192)
  CPU  上采样 ×2  → (1,256,256,192)
  段3  conv(192→192) + 3×Res                             (1,256,256,192) → (1,256,256,192)
  CPU  上采样 ×2  → (1,512,512,192)
  段4  conv(192→96) + 3×Res + head                       (1,512,512,192) → (1,512,512,3)

环境变量：
  KA_SEGS   要跑哪几段，默认 "1"
  KA_HW     latent 边长，默认 64（= 512 输出）
  KA_OUT    输出目录
"""
import os, sys, time, subprocess

sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import numpy as np
import torch
import torch.nn as nn

from vae_block import Conv2DNHWC, ResBlock, VaeAttention, RMSNorm2D
from lowering import Lowering, Unsupported
from safetensors.torch import load_file

H = "/data/data/com.termux/files/home"
D = H + "/npt"
P = os.environ.get("KA_VAE4", H + "/ComfyUI/models/vae/qwen_image_vae.safetensors")
HW = int(os.environ.get("KA_HW", "64"))
SEGS = [int(x) for x in os.environ.get("KA_SEGS", "1").split(",") if x.strip()]
OUT = os.environ.get("KA_OUT", D + "/vae_npu512")
HOST = D + "/host_new"
os.makedirs(OUT, exist_ok=True)

print("VAE → NPU  latent=%d（输出 %d×%d）  段=%s" % (HW, HW * 8, HW * 8, SEGS), flush=True)

sd = load_file(P, device="cpu")


def f(v):
    return None if v is None else v.float()


Px = "decoder."


def mk_res(pfx, C_in, C_out):
    w = dict(
        n1_gamma=f(sd[pfx + "residual.0.gamma"]),
        c1_w=f(sd[pfx + "residual.2.weight"]), c1_b=f(sd[pfx + "residual.2.bias"]),
        n2_gamma=f(sd[pfx + "residual.3.gamma"]),
        c2_w=f(sd[pfx + "residual.6.weight"]), c2_b=f(sd[pfx + "residual.6.bias"]),
        sc_w=f(sd.get(pfx + "shortcut.weight")), sc_b=f(sd.get(pfx + "shortcut.bias")))
    return ResBlock(w, C_in, C_out)


def mk_conv5(name, kh=3, kw=3, pad=1):
    return Conv2DNHWC(f(sd[name + ".weight"]), f(sd[name + ".bias"]), kh=kh, kw=kw, pad=pad)


class Seg1(nn.Module):
    """conv1 + 3×Res + Attn + 3×Res  （不含上采样）"""

    def __init__(s, hw):
        super().__init__()
        s.conv2 = Conv2DNHWC(f(sd["conv2.weight"]), f(sd["conv2.bias"]), kh=1, kw=1, pad=0)
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
        x = s.conv2(z)
        x = s.conv1(x)
        x = s.m0(x)
        x = s.m1(x)
        x = s.m2(x)
        x = s.u0(x)
        x = s.u1(x)
        return s.u2(x)


class Seg2(nn.Module):
    """upsamples.3 的 conv（384→192）+ upsamples.4..6（192→384→384→384）"""

    def __init__(s, hw):
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
    """upsamples.7 的 conv（384→192）+ upsamples.8..10"""

    def __init__(s, hw):
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
    """upsamples.11 的 conv（192→96）+ upsamples.12..14 + head"""

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


IN_SHAPES = {
    1: (1, HW, HW, 16),
    2: (1, HW * 2, HW * 2, 384),
    3: (1, HW * 4, HW * 4, 384),
    4: (1, HW * 8, HW * 8, 192),
}
OUT_SHAPES = {
    1: (1, HW, HW, 384),
    2: (1, HW * 2, HW * 2, 384),
    3: (1, HW * 4, HW * 4, 384),
    4: (1, HW * 8, HW * 8, 3),
}


def build_seg(i):
    if i == 1:
        return Seg1(HW).eval()
    if i == 2:
        return Seg2(HW * 2).eval()
    if i == 3:
        return Seg3(HW * 4).eval()
    return Seg4().eval()


t_all = time.time()
for i in SEGS:
    seg = build_seg(i)
    d = "%s/s%d" % (OUT, i)
    os.makedirs(d, exist_ok=True)
    inshape = IN_SHAPES[i]
    print("\n=== 段%d  输入 %s ===" % (i, (inshape,)), flush=True)
    t0 = time.time()
    try:
        low = Lowering(d)
        low.lower_module(seg, [low.input(inshape)])
    except Unsupported as e:
        print("  段%d 不支持: %s" % (i, e), flush=True)
        continue
    txt = low.graph_text()
    open(d + "/g.txt", "w").write(txt)
    errs = low.validate()
    print("  lowering %.1fs  %d 算子  校验 %s"
          % (time.time() - t0, len(low.ops), "通过" if not errs else str(errs)[:90]), flush=True)
    if errs:
        continue

    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def cmd(c):
        p.stdin.write(c + "\n")
        p.stdin.flush()
        return p.stdout.readline().strip()

    r = cmd('{"cmd":"build","path":"%s/g.txt"}' % d)
    if '"ok":true' not in r:
        print("  build 失败 %s" % r[:80], flush=True)
        p.kill()
        continue
    t1 = time.time()
    r = cmd('{"cmd":"compile"}')
    dt = time.time() - t1
    if '"ok":true' not in r:
        print("  compile 失败 %s  (%d 算子, %.1fs)" % (r[:60], len(low.ops), dt), flush=True)
        p.kill()
        continue
    blob = "%s/s%d.dla" % (OUT, i)
    cmd('{"cmd":"export","path":"%s"}' % blob)
    sz = os.path.getsize(blob) / 1e6 if os.path.exists(blob) else 0
    print("  compile %.1fs  OK  blob %.0f MB" % (dt, sz), flush=True)
    cmd('{"cmd":"quit"}')
    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()

print("\n总耗时 %.0fs" % (time.time() - t_all))
