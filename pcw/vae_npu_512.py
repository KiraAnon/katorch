"""VAE decoder 上 NPU：512×512 输出（latent 64×64）。

基于 t_vae_4seg.py 改写，两处不同：
  1. 尺寸从 8×8 latent（64×64 图）放大到 64×64 latent（512×512 图）
  2. 可以只跑指定的段（KA_SEGS=1 只做段1），先验证可行性再全做

4 段划分（输出尺寸）：
  段1  (1,64,64,16)  → (1,128,128,192)   conv1+mid+attn+3×Res+Resample
  段2  (1,128,128,192)→ (1,256,256,192)  3×Res+Resample
  段3  (1,256,256,192)→ (1,512,512,96)   3×Res+Resample
  段4  (1,512,512,96) → (1,512,512,3)    3×Res+head

环境变量：
  KA_SEGS   要处理哪些段，逗号分隔，默认 "1"
  KA_VAE4   VAE 权重
  KA_OUT    输出目录（图 + blob + 中间结果）
"""
import os, sys, time, subprocess

sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import numpy as np
import torch
import torch.nn as nn

from vae_block import Conv2DNHWC, ResBlock, VaeAttention, RMSNorm2D
from vae_decoder import Resample, WanVaeDecoder
from lowering import Lowering, Unsupported
from safetensors.torch import load_file

H = "/data/data/com.termux/files/home"
D = H + "/npt"
P = os.environ.get("KA_VAE4", H + "/ComfyUI/models/vae/qwen_image_vae.safetensors")
OUT = os.environ.get("KA_OUT", D + "/vae_npu512")
SEGS = [int(x) for x in os.environ.get("KA_SEGS", "1").split(",") if x.strip()]
HOST = D + "/host_new"
LAT = 64                      # latent 边长（512/8）
os.makedirs(OUT, exist_ok=True)

print("VAE → NPU  输出 512×512（latent %dx%d）" % (LAT, LAT), flush=True)
print("段: %s   输出目录: %s" % (SEGS, OUT), flush=True)

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


def mk_ups(i, C_in, C_out):
    return mk_res("%supsamples.%d." % (Px, i), C_in, C_out)


class Seg1(nn.Module):
    def __init__(s, hw):
        super().__init__()
        s.conv1 = Conv2DNHWC(f(sd[Px + "conv1.weight"]), f(sd[Px + "conv1.bias"]))
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
        s.u0 = mk_ups(0, 384, 384)
        s.u1 = mk_ups(1, 384, 384)
        s.u2 = mk_ups(2, 384, 384)
        s.rs = Resample(f(sd[Px + "upsamples.3.resample.1.weight"]),
                        f(sd[Px + "upsamples.3.resample.1.bias"]), 384, H=hw, W=hw)

    def forward(s, z):
        x = s.conv1(z)
        x = s.m0(x)
        x = s.m1(x)
        x = s.m2(x)
        x = s.u0(x)
        x = s.u1(x)
        x = s.u2(x)
        return s.rs(x)


class Seg2(nn.Module):
    def __init__(s, hw):
        super().__init__()
        s.u4 = mk_res("%supsamples.4." % Px, 192, 384)
        s.u5 = mk_res("%supsamples.5." % Px, 384, 384)
        s.u6 = mk_res("%supsamples.6." % Px, 384, 384)
        s.rs = Resample(f(sd[Px + "upsamples.7.resample.1.weight"]),
                        f(sd[Px + "upsamples.7.resample.1.bias"]), 384, H=hw, W=hw)

    def forward(s, x):
        x = s.u4(x)
        x = s.u5(x)
        x = s.u6(x)
        return s.rs(x)


class Seg3(nn.Module):
    def __init__(s, hw):
        super().__init__()
        s.u8 = mk_res("%supsamples.8." % Px, 192, 192)
        s.u9 = mk_res("%supsamples.9." % Px, 192, 192)
        s.u10 = mk_res("%supsamples.10." % Px, 192, 192)
        s.rs = Resample(f(sd[Px + "upsamples.11.resample.1.weight"]),
                        f(sd[Px + "upsamples.11.resample.1.bias"]), 192, H=hw, W=hw)

    def forward(s, x):
        x = s.u8(x)
        x = s.u9(x)
        x = s.u10(x)
        return s.rs(x)


class Seg4(nn.Module):
    def __init__(s):
        super().__init__()
        s.u12 = mk_res("%supsamples.12." % Px, 96, 96)
        s.u13 = mk_res("%supsamples.13." % Px, 96, 96)
        s.u14 = mk_res("%supsamples.14." % Px, 96, 96)
        s.hn = RMSNorm2D(f(sd[Px + "head.0.gamma"]), 96)
        s.hc = Conv2DNHWC(f(sd[Px + "head.2.weight"]), f(sd[Px + "head.2.bias"]))

    def forward(s, x):
        x = s.u12(x)
        x = s.u13(x)
        x = s.u14(x)
        x = s.hn(x)
        x = torch.nn.functional.silu(x)
        return s.hc(x)


# 每段的输入形状（512 版）
IN_SHAPES = {
    1: (1, LAT, LAT, 16),
    2: (1, LAT * 2, LAT * 2, 192),
    3: (1, LAT * 4, LAT * 4, 192),
    4: (1, LAT * 8, LAT * 8, 96),
}


def build_seg(i):
    if i == 1:
        return Seg1(LAT).eval(), LAT
    if i == 2:
        return Seg2(LAT * 2).eval(), LAT * 2
    if i == 3:
        return Seg3(LAT * 4).eval(), LAT * 4
    return Seg4().eval(), LAT * 8


t_all = time.time()
for i in SEGS:
    seg, hw = build_seg(i)
    d = "%s/s%d" % (OUT, i)
    os.makedirs(d, exist_ok=True)
    inshape = IN_SHAPES[i]
    print("\n=== 段%d  输入 %s ===" % (i, (inshape,)), flush=True)
    t0 = time.time()
    try:
        low = Lowering(d)
        xi = low.input(inshape)
        low.lower_module(seg, [xi])
    except Unsupported as e:
        print("  段%d 不支持: %s" % (i, e), flush=True)
        continue
    txt = low.graph_text()
    open(d + "/g.txt", "w").write(txt)
    errs = low.validate()
    print("  lowering %.1fs  %d 算子  校验 %s"
          % (time.time() - t0, len(low.ops), "通过" if not errs else str(errs)[:80]), flush=True)
    if errs:
        continue

    t0 = time.time()
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def cmd(c):
        p.stdin.write(c + "\n")
        p.stdin.flush()
        return p.stdout.readline().strip()

    r = cmd('{"cmd":"build","path":"%s/g.txt"}' % d)
    print("  build %s" % r[:80], flush=True)
    if '"ok":true' not in r:
        p.kill()
        continue
    t1 = time.time()
    r = cmd('{"cmd":"compile"}')
    print("  compile %.1fs  %s" % (time.time() - t1, r[:80]), flush=True)
    if '"ok":true' not in r:
        p.kill()
        continue
    blob = "%s/s%d.dla" % (OUT, i)
    r = cmd('{"cmd":"export","path":"%s"}' % blob)
    sz = os.path.getsize(blob) / 1e6 if os.path.exists(blob) else 0
    print("  export %s  %.0f MB  段总耗时 %.0fs" % (r[:60], sz, time.time() - t0), flush=True)
    cmd('{"cmd":"quit"}')
    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()

print("\n全部耗时 %.0fs" % (time.time() - t_all))
