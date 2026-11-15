"""定位 VAE 段1 编译失败 (rc=-516) 的根因：把段1 拆成单个算子逐个编译。

每个测试都单独起一个 host_new 进程，避免状态污染。
输出：哪个组件能编、哪个不能，以及体积和算子数。

环境变量：
  KA_HW   空间边长（默认 64，对应 512×512 输出）
"""
import os, sys, subprocess, time

sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import torch
import torch.nn as nn

from vae_block import Conv2DNHWC, ResBlock, VaeAttention, RMSNorm2D
from vae_decoder import Resample
from lowering import Lowering, Unsupported
from safetensors.torch import load_file

H = "/data/data/com.termux/files/home"
D = H + "/npt"
P = H + "/ComfyUI/models/vae/qwen_image_vae.safetensors"
HOST = D + "/host_new"
HW = int(os.environ.get("KA_HW", "64"))
OUT = "/data/local/tmp/vae_diag"
os.makedirs(OUT, exist_ok=True)

sd = load_file(P, device="cpu")
Px = "decoder."


def f(v):
    return None if v is None else v.float()


def build_res(pfx, C_in, C_out):
    w = dict(
        n1_gamma=f(sd[pfx + "residual.0.gamma"]),
        c1_w=f(sd[pfx + "residual.2.weight"]), c1_b=f(sd[pfx + "residual.2.bias"]),
        n2_gamma=f(sd[pfx + "residual.3.gamma"]),
        c2_w=f(sd[pfx + "residual.6.weight"]), c2_b=f(sd[pfx + "residual.6.bias"]),
        sc_w=f(sd.get(pfx + "shortcut.weight")), sc_b=f(sd.get(pfx + "shortcut.bias")))
    return ResBlock(w, C_in, C_out)


def build_attn(hw):
    Dm = 384
    qkv_w = f(sd[Px + "middle.1.to_qkv.weight"])
    qkv_b = f(sd[Px + "middle.1.to_qkv.bias"])
    return VaeAttention(Dm, f(sd[Px + "middle.1.norm.gamma"]),
                        qkv_w[0:Dm], qkv_w[Dm:2 * Dm], qkv_w[2 * Dm:3 * Dm],
                        qkv_b[0:Dm], qkv_b[Dm:2 * Dm], qkv_b[2 * Dm:3 * Dm],
                        f(sd[Px + "middle.1.proj.weight"]), f(sd[Px + "middle.1.proj.bias"]),
                        H=hw, W=hw)


def run_compile(tag, module, in_shape):
    d = "%s/%s" % (OUT, tag)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    try:
        low = Lowering(d)
        low.lower_module(module, [low.input(in_shape)])
    except Unsupported as e:
        print("  %-22s LOWERING-FAIL %s" % (tag, str(e)[:70]), flush=True)
        return
    except Exception as e:
        print("  %-22s LOWERING-ERR  %s: %s" % (tag, type(e).__name__, str(e)[:70]), flush=True)
        return
    errs = low.validate()
    txt = low.graph_text()
    open(d + "/g.txt", "w").write(txt)
    n_ops = len(low.ops)
    if errs:
        print("  %-22s VALIDATE-FAIL %s" % (tag, str(errs)[:70]), flush=True)
        return

    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def cmd(c):
        p.stdin.write(c + "\n")
        p.stdin.flush()
        return p.stdout.readline().strip()

    r = cmd('{"cmd":"build","path":"%s/g.txt"}' % d)
    if '"ok":true' not in r:
        print("  %-22s BUILD-FAIL   %s" % (tag, r[:70]), flush=True)
        p.kill()
        return
    t1 = time.time()
    r = cmd('{"cmd":"compile"}')
    dt = time.time() - t1
    if '"ok":true' in r:
        blob = "%s/%s.dla" % (d, tag)
        cmd('{"cmd":"export","path":"%s"}' % blob)
        sz = os.path.getsize(blob) / 1e6 if os.path.exists(blob) else 0
        print("  %-22s OK    %3d ops  compile %.1fs  %.0f MB" % (tag, n_ops, dt, sz), flush=True)
    else:
        print("  %-22s COMPILE-FAIL %s  (%d ops, %.1fs)" % (tag, r[:50], n_ops, dt), flush=True)
    cmd('{"cmd":"quit"}')
    try:
        p.wait(timeout=60)
    except Exception:
        p.kill()


print("VAE 段1 拆解诊断  HW=%d  (输出 %d×%d)" % (HW, HW * 8, HW * 8), flush=True)
C = HW * HW * 384 * 4 / 1e6

print("\n[1] conv1   (1,%d,%d,16) -> 384 通道" % (HW, HW), flush=True)
run_compile("conv1", Conv2DNHWC(f(sd[Px + "conv1.weight"]), f(sd[Px + "conv1.bias"])),
            (1, HW, HW, 16))

print("\n[2] 单个 ResBlock 384→384", flush=True)
run_compile("resblock", build_res(Px + "middle.0.", 384, 384), (1, HW, HW, 384))

print("\n[3] attention 384 通道", flush=True)
run_compile("attn", build_attn(HW), (1, HW, HW, 384))

print("\n[4] Resample 384 (上采样 2x)", flush=True)
run_compile("resample", Resample(f(sd[Px + "upsamples.3.resample.1.weight"]),
                                 f(sd[Px + "upsamples.3.resample.1.bias"]), 384, H=HW, W=HW),
            (1, HW, HW, 384))

print("\n(供参考：HW=%d 时单张 384 通道特征图约 %.0f MB fp16)" % (HW, C), flush=True)
