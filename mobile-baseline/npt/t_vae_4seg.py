"""VAE decoder 分 4 段，链式跑完，对比完整 torch decoder。

分段（按 ComfyUI 的 upsamples 索引）：
  段1: conv1 + mid0 + mid1 + mid2 + ups[0..2] + ups[3](Resample)  8x8  -> 16x16
  段2: ups[4..6] + ups[7](Resample)                                16x16 -> 32x32
  段3: ups[8..10] + ups[11](Resample)                              32x32 -> 64x64
  段4: ups[12..14] + head                                          64x64 -> 64x64
"""
import sys, os, time, subprocess
sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import numpy as np, torch
import torch.nn as nn
from vae_block import Conv2DNHWC, ResBlock, VaeAttention, RMSNorm2D
from vae_decoder import Resample, WanVaeDecoder
from lowering import Lowering, Unsupported
from safetensors.torch import load_file

D = "/data/data/com.termux/files/home/npt"
OUT = D + "/vae_4seg"
os.makedirs(OUT, exist_ok=True)
P = "/data/data/com.termux/files/home/ComfyUI/models/vae/qwen_image_vae.safetensors"

print("加载权重...", flush=True)
sd = load_file(P, device="cpu")
def f(v): return None if v is None else v.float()
Px = "decoder."

full = WanVaeDecoder(sd, H0=8, W0=8).eval()

def mk_res(pfx, C_in, C_out):
    w = dict(
        n1_gamma=f(sd[pfx + "residual.0.gamma"]),
        c1_w=f(sd[pfx + "residual.2.weight"]), c1_b=f(sd[pfx + "residual.2.bias"]),
        n2_gamma=f(sd[pfx + "residual.3.gamma"]),
        c2_w=f(sd[pfx + "residual.6.weight"]), c2_b=f(sd[pfx + "residual.6.bias"]),
        sc_w=f(sd.get(pfx + "shortcut.weight")), sc_b=f(sd.get(pfx + "shortcut.bias")))
    return ResBlock(w, C_in, C_out)

def mk_ups(i, C_in, C_out, H, W):
    return mk_res("%supsamples.%d." % (Px, i), C_in, C_out)

# ---- 段定义 ----
class Seg1(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1 = Conv2DNHWC(f(sd[Px + "conv1.weight"]), f(sd[Px + "conv1.bias"]))
        s.m0 = mk_res(Px + "middle.0.", 384, 384)
        qkv_w = f(sd[Px + "middle.1.to_qkv.weight"]); qkv_b = f(sd[Px + "middle.1.to_qkv.bias"])
        Dm = 384
        s.m1 = VaeAttention(Dm, f(sd[Px + "middle.1.norm.gamma"]),
                            qkv_w[0:Dm], qkv_w[Dm:2*Dm], qkv_w[2*Dm:3*Dm],
                            qkv_b[0:Dm], qkv_b[Dm:2*Dm], qkv_b[2*Dm:3*Dm],
                            f(sd[Px + "middle.1.proj.weight"]), f(sd[Px + "middle.1.proj.bias"]),
                            H=8, W=8)
        s.m2 = mk_res(Px + "middle.2.", 384, 384)
        s.u0 = mk_ups(0, 384, 384, 8, 8)
        s.u1 = mk_ups(1, 384, 384, 8, 8)
        s.u2 = mk_ups(2, 384, 384, 8, 8)
        s.rs = Resample(f(sd[Px + "upsamples.3.resample.1.weight"]),
                        f(sd[Px + "upsamples.3.resample.1.bias"]), 384, H=8, W=8)
    def forward(s, z):
        x = s.conv1(z); x = s.m0(x); x = s.m1(x); x = s.m2(x)
        x = s.u0(x); x = s.u1(x); x = s.u2(x)
        return s.rs(x)

class Seg2(nn.Module):
    def __init__(s, H=16, W=16):
        super().__init__()
        # upsamples.4: 192 -> 384（in_dim 减半）
        s.u4 = mk_res("%supsamples.4." % Px, 192, 384)
        s.u5 = mk_res("%supsamples.5." % Px, 384, 384)
        s.u6 = mk_res("%supsamples.6." % Px, 384, 384)
        s.rs = Resample(f(sd[Px + "upsamples.7.resample.1.weight"]),
                        f(sd[Px + "upsamples.7.resample.1.bias"]), 384, H=H, W=W)
    def forward(s, x):
        x = s.u4(x); x = s.u5(x); x = s.u6(x)
        return s.rs(x)

class Seg3(nn.Module):
    def __init__(s, H=32, W=32):
        super().__init__()
        s.u8 = mk_res("%supsamples.8." % Px, 192, 192)
        s.u9 = mk_res("%supsamples.9." % Px, 192, 192)
        s.u10 = mk_res("%supsamples.10." % Px, 192, 192)
        s.rs = Resample(f(sd[Px + "upsamples.11.resample.1.weight"]),
                        f(sd[Px + "upsamples.11.resample.1.bias"]), 192, H=H, W=W)
    def forward(s, x):
        x = s.u8(x); x = s.u9(x); x = s.u10(x)
        return s.rs(x)

class Seg4(nn.Module):
    def __init__(s, H=64, W=64):
        super().__init__()
        s.u12 = mk_res("%supsamples.12." % Px, 96, 96)
        s.u13 = mk_res("%supsamples.13." % Px, 96, 96)
        s.u14 = mk_res("%supsamples.14." % Px, 96, 96)
        s.hn = RMSNorm2D(f(sd[Px + "head.0.gamma"]), 96)
        s.hc = Conv2DNHWC(f(sd[Px + "head.2.weight"]), f(sd[Px + "head.2.bias"]))
    def forward(s, x):
        x = s.u12(x); x = s.u13(x); x = s.u14(x)
        x = s.hn(x); x = torch.nn.functional.silu(x)
        return s.hc(x)

segs = [Seg1().eval(), Seg2().eval(), Seg3().eval(), Seg4().eval()]
torch.manual_seed(0)
z = torch.randn(1, 8, 8, 16) * 0.5
with torch.no_grad():
    ref = full(z).numpy()
print("torch 全 decoder: %s std=%.4f" % (tuple(ref.shape), ref.std()), flush=True)

# 逐段 lowering + 编译
print("\n=== 逐段 lowering ===", flush=True)
for i, seg in enumerate(segs):
    d = "%s/s%d" % (OUT, i + 1)
    os.makedirs(d, exist_ok=True)
    in_shape = (1, 8 if i == 0 else 8*(2**i), 8 if i == 0 else 8*(2**i), 16 if i == 0 else [384,384,192][min(i-1,2)])
    if i == 0: in_shape = (1, 8, 8, 16)
    elif i == 1: in_shape = (1, 16, 16, 192)    # 段1 的 Resample 输出 dim//2
    elif i == 2: in_shape = (1, 32, 32, 192)    # 段2 的 Resample 输出 dim//2
    else: in_shape = (1, 64, 64, 96)     # 段3 的 Resample 输出 96
    low = Lowering(d)
    xi = low.input(in_shape)
    try:
        low.lower_module(seg, [xi])
    except Unsupported as e:
        print("  段%d UNSUPPORTED: %s" % (i+1, e)); sys.exit(1)
    txt = low.graph_text(); open(d + "/g.txt", "w").write(txt)
    errs = low.validate()
    print("  段%d: 输入 %s  %d 算子  校验 %s" % (i+1, in_shape, len(low.ops), "通过" if not errs else errs[:1]), flush=True)
    if errs: sys.exit(1)

# 存输入输出，链式跑
z.numpy().astype(np.float16).tofile(OUT + "/i_z.bin")
SHAPES = [(1,8,8,16),(1,16,16,192),(1,32,32,192),(1,64,64,96)]

print("\n=== 链式执行 ===", flush=True)
p = subprocess.Popen([D + "/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush(); return p.stdout.readline().strip()
s('{"cmd":"cachedir","path":"%s/nc"}' % OUT)
for i in range(4):
    d = "%s/s%d" % (OUT, i + 1)
    s('{"cmd":"cachekey","v":"vseg%d"}' % (i+1))
    r = s('{"cmd":"build","path":"%s/g.txt"}' % d)
    t0 = time.time()
    r2 = s('{"cmd":"compile"}')
    print("  段%d build %s  compile %.1fs %s" % (i+1, r[:30], time.time()-t0, r2[:30]), flush=True)
    if '"ok":true' not in r2:
        p.kill(); sys.exit(1)

# 链式：0:i_z.bin,@1; 1:@1,@2; 2:@2,@3; 3:@3,o_img.bin
spec = "0:%s/i_z.bin,@1;1:@1,@2;2:@2,@3;3:@3,%s/o_img.bin" % (OUT, OUT)
t0 = time.time()
r = s('{"cmd":"chain","spec":"%s"}' % spec)
print("\n  链式 4 段: %.2fs  %s" % (time.time()-t0, r[:40]), flush=True)
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=180)
except: p.kill()

pf = OUT + "/o_img.bin"
if os.path.exists(pf):
    got = np.fromfile(pf, dtype=np.float16).astype(np.float32).reshape(ref.shape)
    e = np.abs(got - ref)
    print("\n=== 完整 decoder 对比 ===")
    print("  NPU  : std=%.4f  min=%.3f max=%.3f" % (got.std(), got.min(), got.max()))
    print("  torch: std=%.4f  min=%.3f max=%.3f" % (ref.std(), ref.min(), ref.max()))
    print("  误差 : MAX_ABS=%.5f  rel=%.3f%%" % (e.max(), 100*e.max()/(np.abs(ref).max()+1e-9)))
else:
    print("\n无输出")
