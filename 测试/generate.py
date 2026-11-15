"""S=256 完整生图：balanced 28 层 + 30 步 + VAE。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, json, math, time, subprocess
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from anima_block import Lin, load
from anima_head import PatchEmbed
from bal_norm import BalancedNorm
from bal_block2 import BalDiTBlock
from lowering import Lowering
from vae_decoder import WanVaeDecoder
from safetensors.torch import load_file

H = _KA_HOME
D = H + "/npt"
OUT = D + "/gen5"
E2E = D + "/e2eper256"
os.makedirs(OUT, exist_ok=True)
os.makedirs(E2E, exist_ok=True)
CACHE = E2E + "/nc"
os.makedirs(CACHE, exist_ok=True)

SHIFT, STEPS = 3.0, 30
W = 256
LH = LW = W // 8          # 32
S_H = S_W = LH // 2       # 16
S = S_H * S_W             # 256
M = 512

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
print("=== 配置: %dx%d, latent %d, token %d, %d 步 ===" % (W,W,LH,S,STEPS), flush=True)

LAT_MEAN = torch.tensor([-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,
                         0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]).view(1,16,1,1)
LAT_STD = torch.tensor([2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
                        3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]).view(1,16,1,1)
def ts(a,t): return t if a==1.0 else a*t/(1+(a-1)*t)
sigs = [float(ts(SHIFT, (k+1)/1000)) for k in range(1000)]
sigs = [sigs[-(1+int(x*(1000/STEPS)))] for x in range(STEPS)] + [0.0]

ctx = torch.from_numpy(np.fromfile(D+"/gen/i_ctx.bin", dtype=np.float16).astype(np.float32)).reshape(M,1024)
pe = PatchEmbed(load("net.x_embedder.proj.1.weight")).eval()
W1, W2 = load("net.t_embedder.1.linear_1.weight"), load("net.t_embedder.1.linear_2.weight")
# ComfyUI 在把时间嵌入送进 adaln 之前还会过一道 RMSNorm（final_norm=True 的等价物）
TNORM = load("net.t_embedding_norm.weight")

class BalFinal(nn.Module):
    def __init__(s, d_in, g, hidden=2048):
        super().__init__()
        s.ln = BalancedNorm(torch.full((hidden,), d_in/float(g)), torch.ones(hidden), kind="ln")
        s.ad_l1 = Lin(load("net.final_layer.adaln_modulation.1.weight"))
        w_l2 = load("net.final_layer.adaln_modulation.2.weight")
        Dd = w_l2.shape[0]//2
        s.ad_p0 = Lin(w_l2[0:Dd]); s.ad_p1 = Lin(w_l2[Dd:2*Dd])
        s.lin = Lin(load("net.final_layer.linear.weight"))
    def forward(s, x, emb, lsh, lsc):
        h = F.silu(emb); h = s.ad_l1(h)
        return s.lin(s.ln(x) * (1.0 + s.ad_p1(h) + lsc) + s.ad_p0(h) + lsh)

# ===== 生成 14 块图 =====
print("\n[1] 生成图", flush=True)
class Pair(nn.Module):
    """两层叠加，层间做 post_scale 转换（d_i 空间 -> d_{i+1} 空间）。"""
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        x = s.a(x, e, l1, l2, l3, c, co, si) * s.ps
        return s.b(x, e, l1, l2, l3, c, co, si)

t0 = time.time()
for bi in range(14):
    dd = "%s/b%02d" % (E2E, bi)
    os.makedirs(dd, exist_ok=True)
    if os.path.exists(dd+"/g.txt"): continue
    b0 = 2*bi
    m = Pair(BalDiTBlock("net.blocks.%d." % b0, pb[b0]["d_in"], pb[b0]["gs"], S=S, M=M).eval(),
             BalDiTBlock("net.blocks.%d." % (b0+1), pb[b0+1]["d_in"], pb[b0+1]["gs"], S=S, M=M).eval(),
             pb[b0]["post_scale"]).eval()
    low = Lowering(dd)
    from t_chain import HID, HD, CTX
    args = [low.input((S,HID)), low.input((1,1,HID)), low.input((1,1,HID)), low.input((1,1,HID)),
            low.input((1,1,HID)), low.input((M,CTX)), low.input((S,1,HD)), low.input((S,1,HD))]
    low.lower_module(m, args)
    if low.validate(): print("  块%d 校验失败" % bi); sys.exit(1)
    open(dd+"/g.txt","w").write(low.graph_text())
    print("  块%02d: %d 算子" % (bi, len(low.ops)), flush=True)

d_pe = OUT+"/pe"; os.makedirs(d_pe, exist_ok=True)
low = Lowering(d_pe); low.lower_module(pe, [low.input((S,68))])
open(d_pe+"/g.txt","w").write(low.graph_text())
d_fl = OUT+"/fl"; os.makedirs(d_fl, exist_ok=True)
bf = BalFinal(pb[27]["d_in"], pb[27]["gs"][2]).eval()
low2 = Lowering(d_fl)
low2.lower_module(bf, [low2.input((S,2048)), low2.input((1,2048)), low2.input((1,2048)), low2.input((1,2048))])
open(d_fl+"/g.txt","w").write(low2.graph_text())
print("  PatchEmbed %d, FinalLayer %d  (总 %.0fs)" % (len(low.ops), len(low2.ops), time.time()-t0), flush=True)

# ===== 编译 =====
print("\n[2] 编译", flush=True)
def compile_only(gdir, key):
    p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c+"\n"); p.stdin.flush()
        l = p.stdout.readline(); return l.strip() if l else ""
    s('{"cmd":"cachedir","path":"%s"}' % CACHE)
    s('{"cmd":"cachekey","v":"%s"}' % key)
    r = s('{"cmd":"build","path":"%s"}' % gdir)
    if '"ok":true' not in r: p.kill(); return "build:"+r[:30]
    r = s('{"cmd":"compile"}')
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=400)
    except: p.kill()
    return None if '"ok":true' in r else "compile:"+r[:30]

t0 = time.time()
for bi in range(14):
    e = compile_only("%s/b%02d/g.txt" % (E2E, bi), "b%02d" % bi)
    if e: print("  块%d 失败: %s" % (bi, e)); sys.exit(1)
for nm, g in [("pe", d_pe+"/g.txt"), ("fl", d_fl+"/g.txt")]:
    e = compile_only(g, nm)
    if e: print("  %s 失败: %s" % (nm, e)); sys.exit(1)
print("  16/16 编译完成 %.0fs (%.2fs/块)" % (time.time()-t0, (time.time()-t0)/16), flush=True)

# ===== 采样 =====
print("\n[3] 采样 %d 步" % STEPS, flush=True)
def run(gdir, key, ins, outs, timeout=400):
    p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c+"\n"); p.stdin.flush()
        l = p.stdout.readline(); return l.strip() if l else ""
    s('{"cmd":"cachedir","path":"%s"}' % CACHE)
    s('{"cmd":"cachekey","v":"%s"}' % key)
    s('{"cmd":"build","path":"%s"}' % gdir)
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r: p.kill(); return False
    spec = "%d %s 1 %s" % (len(ins), " ".join("%d %s"%(i,f) for i,f in enumerate(ins)), outs)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=timeout)
    except: p.kill()
    return '"ok":true' in r

def t_embed(sv):
    h=1024
    e=-torch.log(torch.tensor(10000.0))*torch.arange(h,dtype=torch.float32)/h
    v=torch.tensor([float(sv)])[:,None]*torch.exp(e)[None,:]
    return torch.cat([torch.cos(v),torch.sin(v)],dim=-1)

from t_rope3d import rope3d_cos_sin
cos, sin = rope3d_cos_sin(1, S_H, S_W)
cos.numpy().astype(np.float16).tofile(OUT+"/i_cos.bin")
sin.numpy().astype(np.float16).tofile(OUT+"/i_sin.bin")
ctx.numpy().astype(np.float16).tofile(OUT+"/i_ctx.bin")

torch.manual_seed(42)
x = torch.randn(1,16,LH,LW)
TAIL = [OUT+"/i_te.bin", OUT+"/i_lsh.bin", OUT+"/i_lsc.bin", OUT+"/i_lga.bin",
        OUT+"/i_ctx.bin", OUT+"/i_cos.bin", OUT+"/i_sin.bin"]
t_all = time.time()
t_blk_total = 0.0
t_dump = 0.0
for step in range(STEPS):
    t_step = time.time()
    sv = sigs[step]
    te = t_embed(sv)
    lora = F.linear(F.silu(F.linear(te, W1)), W2)      # lora 用未归一化的 te
    te = te * torch.rsqrt((te * te).mean(-1, keepdim=True) + 1e-6) * TNORM
    te.numpy().astype(np.float16).tofile(OUT+"/i_te.bin")
    lora[:,:2048].numpy().astype(np.float16).tofile(OUT+"/i_lsh.bin")
    lora[:,2048:4096].numpy().astype(np.float16).tofile(OUT+"/i_lsc.bin")
    lora[:,4096:6144].numpy().astype(np.float16).tofile(OUT+"/i_lga.bin")

    pp_ = torch.zeros(S,68); ll = x[0].permute(1,2,0)
    for i in range(S_H):
        for j in range(S_W):
            # 和 ComfyUI 的 Rearrange("... -> b t h w (c r m n)") 一致：按 (通道,高,宽) 展平
            pp_[i*S_W+j,:64] = ll[i*2:(i+1)*2, j*2:(j+1)*2, :].permute(2,0,1).reshape(-1)
    pp_.numpy().astype(np.float16).tofile(OUT+"/i_patch.bin")
    if step == 0:
        print("  探针 patch : min=%.3f max=%.3f std=%.3f  (68 维里后 4 维应是 0: %s)"
              % (pp_.min(), pp_.max(), pp_.std(), bool((pp_[:,64:]==0).all())), flush=True)

    if not run(d_pe+"/g.txt", "pe", [OUT+"/i_patch.bin"], OUT+"/o_h.bin"):
        print("  step%d PatchEmbed 失败" % step); sys.exit(1)
    h = torch.from_numpy(np.fromfile(OUT+"/o_h.bin", dtype=np.float16).astype(np.float32)).reshape(S,2048)
    if step == 0:
        print("  探针 h0    : max=%.3f std=%.4f  d_in=%.1f -> x_bal max=%.4f  te max=%.3f"
              % (h.abs().max(), h.std(), pb[0]["d_in"], (h/pb[0]["d_in"]).abs().max(), te.abs().max()), flush=True)
    cur = OUT+"/i_xb.bin"
    (h/pb[0]["d_in"]).numpy().astype(np.float16).tofile(cur)

    t_blk = time.time()
    for bi in range(14):
        nxt = OUT+"/o_xb.bin" if bi==13 else OUT+"/m_xb.bin"
        if not run("%s/b%02d/g.txt" % (E2E, bi), "b%02d" % bi, [cur]+TAIL, nxt):
            print("  step%d 块%d 失败" % (step, bi)); sys.exit(1)
        v = torch.from_numpy(np.fromfile(nxt, dtype=np.float16).astype(np.float32)).reshape(S,2048)
        if step == 0:
            print("    块%02d max=%-12.4g std=%-10.4g inf=%d" % (bi, v.abs().max(), v.std(), torch.isinf(v).sum()), flush=True)
            if torch.isinf(v).any():
                print("    ^^^ 块%d 起爆，停在这里" % bi, flush=True); sys.exit(2)
        v = v * pb[2*bi+1]["post_scale"]
        cur = OUT+"/i_xb.bin"
        v.numpy().astype(np.float16).tofile(cur)
    t_blk_total += time.time() - t_blk

    if not run(d_fl+"/g.txt", "fl", [cur, OUT+"/i_te.bin", OUT+"/i_lsh.bin", OUT+"/i_lsc.bin"], OUT+"/o_tok.bin"):
        print("  step%d FinalLayer 失败" % step); sys.exit(1)
    tok = torch.from_numpy(np.fromfile(OUT+"/o_tok.bin", dtype=np.float16).astype(np.float32)).reshape(S,64)
    if step == 0:
        print("    FinalLayer max=%-12.4g std=%-10.4g inf=%d" % (tok.abs().max(), tok.std(), torch.isinf(tok).sum()), flush=True)
    v_lat = torch.zeros(1,16,LH,LW)
    for i in range(S_H):
        for j in range(S_W):
            v_lat[0,:,i*2:(i+1)*2, j*2:(j+1)*2] = tok[i*S_W+j].reshape(2,2,16).permute(2,0,1)
    if step == 0:
        print("  探针 token : max=%.3f std=%.4f  块总耗时 %.2fs (%.2fs/块)"
              % (tok.abs().max(), tok.std(), time.time()-t_blk, (time.time()-t_blk)/14), flush=True)
    x = x - v_lat * (sigs[step]-sigs[step+1])
    if step % 5 == 0 or step == STEPS-1:
        print("  step%2d sigma=%.4f x.std=%.4f v.std=%.4f inf=%d  %.2fs"
              % (step, sv, x.std(), v_lat.std(), torch.isinf(x).sum(), time.time()-t_step), flush=True)

print("  采样 %.0fs, 其中 14 块 %.0fs (%.3fs/块, %.1fs/步)"
      % (time.time()-t_all, t_blk_total, t_blk_total/(STEPS*14), time.time()-t_all/STEPS), flush=True)

print("\n[4] VAE", flush=True)
z = x * LAT_STD + LAT_MEAN
print("  z min=%.2f max=%.2f std=%.2f" % (z.min(), z.max(), z.std()), flush=True)
vae_sd = load_file(H+"/ComfyUI/models/vae/qwen_image_vae.safetensors", device="cpu")
dec = WanVaeDecoder(vae_sd, H0=LH, W0=LW).eval()
# 先把采样结果存下来，以后单独重跑 VAE 不用再等 30 步
x[0].numpy().astype(np.float16).tofile(OUT+"/final_latent.bin")
with torch.no_grad():
    # conv2d_nhwc 要 4D，这里补一个 batch 维
    img = dec(z[0].permute(1,2,0).unsqueeze(0)).squeeze(0)
print("  图像 %s min=%.3f max=%.3f std=%.3f" % (tuple(img.shape), img.min(), img.max(), img.std()))
img8 = torch.clamp((img+1)*127.5,0,255).to(torch.uint8).numpy()
f = open(OUT+"/final.ppm","wb")
f.write(("P6\n%d %d\n255\n" % (img8.shape[1], img8.shape[0])).encode())
f.write(img8.tobytes()); f.close()
print("  已存 final.ppm (%dx%d)" % (img8.shape[1], img8.shape[0]))