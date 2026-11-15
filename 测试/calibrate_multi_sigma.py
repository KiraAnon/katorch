"""多 timestep 校准：覆盖 sigma=1.0/0.7/0.4/0.1，取逐通道 max。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, math, json, time
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from anima_block import DiTBlock, load, Lin, RMS
from anima_head import PatchEmbed
from bal_norm import make_d
from t_rope3d import rope3d_cos_sin

H = _KA_HOME
D = H + "/npt"
S, M, W = 256, 512, 256
LH = LW = W//8
S_H = S_W = LH//2
SHIFT = 3.0

def ts(a,t): return t if a==1.0 else a*t/(1+(a-1)*t)
def t_embed(sv):
    h=1024
    e=-torch.log(torch.tensor(10000.0))*torch.arange(h,dtype=torch.float32)/h
    v=torch.tensor([float(sv)])[:,None]*torch.exp(e)[None,:]
    return torch.cat([torch.cos(v),torch.sin(v)],dim=-1)

ctx = torch.from_numpy(np.fromfile(D+"/gen/i_ctx.bin", dtype=np.float16).astype(np.float32)).reshape(M,1024)
W1, W2 = load("net.t_embedder.1.linear_1.weight"), load("net.t_embedder.1.linear_2.weight")
TNORM = load("net.t_embedding_norm.weight")
cos, sin = rope3d_cos_sin(1, S_H, S_W)
cos = cos.reshape(S,1,128); sin = sin.reshape(S,1,128)
pe = PatchEmbed(load("net.x_embedder.proj.1.weight")).eval()

blocks = []
for bi in range(28):
    b = DiTBlock("net.blocks.%d." % bi, S=S, M=M).eval()
    b.attn_ca.S, b.attn_ca.M = S, M
    b.attn_ca.k = Lin(load("net.blocks.%d.cross_attn.k_proj.weight" % bi))
    b.attn_ca.v = Lin(load("net.blocks.%d.cross_attn.v_proj.weight" % bi))
    b.attn_ca.kn = RMS(load("net.blocks.%d.cross_attn.k_norm.weight" % bi))
    blocks.append(b)

def block_stats(b, x, te, lsh, lsc, lga):
    s = []
    sh, sc, ga = b.ad_sa(te, lsh, lsc, lga); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
    s.append(float(torch.nan_to_num(x.abs()).median()))
    n = b.ln_sa(x) * (1.0+sc) + sh
    x = x + ga * b.attn_sa(n, None, cos, sin)
    sh, sc, ga = b.ad_ca(te, lsh, lsc, lga); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
    s.append(float(torch.nan_to_num(x.abs()).median()))
    n = b.ln_ca(x) * (1.0+sc) + sh
    x = x + ga * b.attn_ca(n, ctx, cos, sin)
    sh, sc, ga = b.ad_mlp(te, lsh, lsc, lga); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
    s.append(float(torch.nan_to_num(x.abs()).median()))
    n = b.ln_mlp(x) * (1.0+sc) + sh
    x = x + ga * b.mlp_out(b.mlp_act(b.mlp_in(n)))
    return x, s

# 多个 sigma 点校准
SIGMAS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0955]
print("=== 多 timestep 校准：sigma = %s ===" % SIGMAS, flush=True)

all_medians = []   # 每个 sigma 一组
all_maxabs = []
t0 = time.time()
for si, sv in enumerate(SIGMAS):
    te = t_embed(sv)
    lora = F.linear(F.silu(F.linear(te, W1)), W2)      # lora 用未归一化的 te
    te = te * torch.rsqrt((te * te).mean(-1, keepdim=True) + 1e-6) * TNORM
    lsh, lsc, lga = lora[:,:2048], lora[:,2048:4096], lora[:,4096:6144]
    torch.manual_seed(42)
    lat = torch.randn(1,16,LH,LW)
    pp_ = torch.zeros(S,68); ll = lat[0].permute(1,2,0)
    for i in range(S_H):
        for j in range(S_W):
            pp_[i*S_W+j,:64] = ll[i*2:(i+1)*2, j*2:(j+1)*2, :].permute(2,0,1).reshape(-1)
    meds = []
    with torch.no_grad():
        h = pe(pp_)
        all_maxabs.append(h.abs().amax(dim=0).double())
        for b in blocks:
            h, s = block_stats(b, h, te, lsh, lsc, lga)
            all_maxabs.append(h.abs().amax(dim=0).double())
            meds.extend(s)
    all_medians.append(meds)
    print("  sigma=%.4f: 边界0 median=%.4f, 边界27 median=%.1f"
          % (sv, meds[0], meds[-1]), flush=True)

# 每个 norm 取所有 sigma 的最大值
n_norm = len(all_medians[0])
norm_med = [max(all_medians[si][i] for si in range(len(SIGMAS))) for i in range(n_norm)]
print("\n校准 %.1fs, %d 个 Norm, 多 sigma 取 max" % (time.time()-t0, n_norm))

def pow2(v): return 1.0 if v <= 0 else math.ldexp(1.0, round(math.log2(v)))

per_block = []
for bi in range(28):
    d_in = pow2(norm_med[3*(bi+1)]) if bi < 27 else pow2(norm_med[3*bi])
    gs = [pow2(norm_med[3*bi+j]) for j in range(3)]
    d_next = pow2(norm_med[3*(bi+2)]) if bi < 26 else d_in
    per_block.append({"d_in": d_in, "gs": gs,
                      "r": [d_in/g for g in gs], "post_scale": d_in/d_next})

print("\n=== 结果 ===")
print("%4s %10s %20s %10s" % ("块", "d_in", "gs", "post_scale"))
for bi in range(0, 28, 4):
    p = per_block[bi]
    print("%4d %10.2f %20s %10.4f" % (bi, p["d_in"], ["%.2f"%g for g in p["gs"]], p["post_scale"]))
print("\n  r 范围: %.3f ~ %.3f" % (min(min(p["r"]) for p in per_block), max(max(p["r"]) for p in per_block)))
print("  估计每块输出 x_bal:")
for bi in range(0, 28, 4):
    out_med = norm_med[3*(bi+1)] if bi < 27 else norm_med[83]
    print("    块%02d: d=%.1f 输出median=%.1f -> x_bal=%.3f" % (bi, per_block[bi]["d_in"], out_med, out_med/per_block[bi]["d_in"]))

json.dump({"per_block": per_block, "norm_median": norm_med, "S": S, "sigmas": SIGMAS},
          open(D + "/gen/calib7.json", "w"))
print("\n已存 calib7.json (多 sigma)")
