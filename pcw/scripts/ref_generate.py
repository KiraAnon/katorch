"""PC 端纯 PyTorch 参考实现：与手机端 NPU 链路使用完全相同的参数，
   用来判定「发白」是采样循环的问题还是 NPU 路径的问题。
"""
import sys, os, json
sys.path.insert(0, r"F:\KAtorch\pcw\ref")

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock
from anima_head import PatchEmbed
from bal_norm import BalancedNorm
from t_rope3d import rope3d_cos_sin
from vae_decoder import WanVaeDecoder
from safetensors.torch import load_file

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, flush=True)

SHIFT, STEPS = 3.0, 30
W = 256
LH = LW = W // 8
S_H = S_W = LH // 2
S = S_H * S_W
M = 512
CFG = 5.0

cal = json.load(open(r"D:\comfyui\ComfyUI-master\..\..\KAtorch\pcw\calib8.json")) if False else None
cal = json.load(open(r"F:\KAtorch\pcw\calib8.json"))
pb = cal["per_block"]

LAT_MEAN = torch.tensor([-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,
                         0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]).view(1,16,1,1).to(DEV)
LAT_STD = torch.tensor([2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
                        3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]).view(1,16,1,1).to(DEV)

def ts(a, t): return t if a == 1.0 else a * t / (1 + (a - 1) * t)
sigs = [float(ts(SHIFT, (k + 1) / 1000)) for k in range(1000)]
sigs = [sigs[-(1 + int(x * (1000 / STEPS)))] for x in range(STEPS)] + [0.0]

class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)

print("构建 14 个块 ...", flush=True)
blocks = []
for bi in range(14):
    b0, b1 = 2 * bi, 2 * bi + 1
    m = Pair(BalDiTBlock("net.blocks.%d." % b0, pb[b0]["d_in"], pb[b0]["gs"], S=S, M=M),
             BalDiTBlock("net.blocks.%d." % (b1), pb[b1]["d_in"], pb[b1]["gs"], S=S, M=M),
             pb[b0]["post_scale"]).eval().to(DEV)
    blocks.append(m)

pe = PatchEmbed(load("net.x_embedder.proj.1.weight")).eval().to(DEV)
W1, W2 = load("net.t_embedder.1.linear_1.weight").to(DEV), load("net.t_embedder.1.linear_2.weight").to(DEV)
TNORM = load("net.t_embedding_norm.weight").to(DEV)

class BalFinal(nn.Module):
    def __init__(s, d_in, g, hidden=2048):
        super().__init__()
        s.ln = BalancedNorm(torch.full((hidden,), d_in / float(g)), torch.ones(hidden), kind="ln")
        s.ad_l1 = load("net.final_layer.adaln_modulation.1.weight")
        w_l2 = load("net.final_layer.adaln_modulation.2.weight")
        Dd = w_l2.shape[0] // 2
        s.ad_p0 = w_l2[0:Dd]; s.ad_p1 = w_l2[Dd:2*Dd]
        s.lin = load("net.final_layer.linear.weight")
    def forward(s, x, emb, lsh, lsc):
        h = F.silu(emb); h = s.ad_l1(h)
        return s.lin(s.ln(x) * (1.0 + s.ad_p1(h) + lsc) + s.ad_p0(h) + lsh)

fl = BalFinal(pb[27]["d_in"], pb[27]["gs"][2]).eval().to(DEV)

ctx_pos = torch.from_numpy(np.fromfile(r"F:\KAtorch\pcw\ctx\ctx_pos2.bin", dtype=np.float16)
                           .astype(np.float32)).reshape(M, 1024).to(DEV)
ctx_neg = torch.from_numpy(np.fromfile(r"F:\KAtorch\pcw\ctx\ctx_neg2.bin", dtype=np.float16)
                           .astype(np.float32)).reshape(M, 1024).to(DEV)
cos, sin = rope3d_cos_sin(1, S_H, S_W)
cos = cos.reshape(S, 1, 128).to(DEV); sin = sin.reshape(S, 1, 128).to(DEV)

def t_embed(sv):
    h = 1024
    e = -torch.log(torch.tensor(10000.0)) * torch.arange(h, dtype=torch.float32) / h
    v = torch.tensor([float(sv)])[:, None] * torch.exp(e)[None, :]
    return torch.cat([torch.cos(v), torch.sin(v)], dim=-1)

def stack(x, ctx):
    h = pe(torch.stack([
        torch.cat([x[0, :, i*2:(i+1)*2, j*2:(j+1)*2].permute(1, 2, 0).reshape(-1),
                   x.new_zeros(4)]) for i in range(S_H) for j in range(S_W)])) if False else None
    # patchify（与 generate.py 完全一致）
    ll = x[0].permute(1, 2, 0)
    pp = []
    for i in range(S_H):
        for j in range(S_W):
            pp.append(torch.cat([ll[i*2:(i+1)*2, j*2:(j+1)*2, :].permute(2, 0, 1).reshape(-1),
                                 x.new_zeros(4)]))
    pp = torch.stack(pp)
    h = pe(pp).reshape(S, 2048)
    cur = h / pb[0]["d_in"]
    for bi in range(14):
        cur = blocks[bi](cur, te, lsh, lsc, lga, ctx, cos, sin)
        cur = cur * pb[2*bi+1]["post_scale"]
    tok = fl(cur, te, lsh, lsc)
    v = torch.zeros(1, 16, LH, LW, device=DEV)
    for i in range(S_H):
        for j in range(S_W):
            v[0, :, i*2:(i+1)*2, j*2:(j+1)*2] = tok[i*S_W+j].reshape(2, 2, 16).permute(2, 0, 1)
    return v

torch.manual_seed(42)
x = torch.randn(1, 16, LH, LW, device=DEV)
import time
t0 = time.time()
with torch.no_grad():
    for step in range(STEPS):
        sv = sigs[step]
        te_raw = t_embed(sv).to(DEV)
        lora = F.linear(F.silu(F.linear(te_raw, W1)), W2)
        te = te_raw * torch.rsqrt((te_raw * te_raw).mean(-1, keepdim=True) + 1e-6) * TNORM
        lsh, lsc, lga = lora[:, :2048], lora[:, 2048:4096], lora[:, 4096:6144]
        v_pos = stack(x, ctx_pos)
        v_neg = stack(x, ctx_neg)
        v = v_neg + CFG * (v_pos - v_neg)
        x = x - v * (sigs[step] - sigs[step + 1])
        if step % 5 == 0 or step == STEPS - 1:
            print("  step%2d sigma=%.4f x.std=%.4f v.std=%.4f  %.1fs"
                  % (step, sv, x.std(), v.std(), time.time() - t0), flush=True)

print("采样 %.1fs  final x.std=%.4f" % (time.time() - t0, x.std()), flush=True)
x[0].cpu().numpy().astype(np.float16).tofile(r"F:\KAtorch\pcw\ref_latent.bin")

print("VAE 解码 ...", flush=True)
z = x * LAT_STD + LAT_MEAN
vae_sd = load_file(r"D:\comfyui\ComfyUI-master\models\vae\Qwen\qwen_image_vae.safetensors", device="cpu")
dec = WanVaeDecoder(vae_sd, H0=LH, W0=LW).eval()
with torch.no_grad():
    img = dec(z[0].cpu().permute(1, 2, 0).unsqueeze(0)).squeeze(0)
print("图像 %s std=%.3f" % (tuple(img.shape), img.std()))
img8 = torch.clamp((img + 1) * 127.5, 0, 255).to(torch.uint8).numpy()
f = open(r"F:\KAtorch\pcw\ref_img.ppm", "wb")
f.write(("P6\n%d %d\n255\n" % (img8.shape[1], img8.shape[0])).encode())
f.write(img8.tobytes()); f.close()
print("-> F:\\KAtorch\\pcw\\ref_img.ppm")
