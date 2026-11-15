"""超级探针：每个中间点单独成图，逐个和 torch 对比。

每个探针图 = 从输入算到该点的完整前缀，只输出那一个值。
这样每张图都独立可验证，不受多输出干扰。
"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, json, subprocess, shutil
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from anima_block import load, Lin, RMS, Attn, AdaLN3
from lowering import Lowering
from t_rope3d import rope3d_cos_sin

H = _KA_HOME
D = H + "/npt"
OUT = D + "/prb"
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
S, M = 256, 512
DIM = 2048
cal = json.load(open(D+"/gen/calib6.json"))
pb = cal["per_block"][0]
d_in, gs = pb["d_in"], pb["gs"]
r_sa = d_in / gs[0]
inv_d = 1.0 / d_in
SSE = 0.001

attn_sa = Attn(load("net.blocks.0.self_attn.q_proj.weight"), load("net.blocks.0.self_attn.k_proj.weight"),
               load("net.blocks.0.self_attn.v_proj.weight"), load("net.blocks.0.self_attn.output_proj.weight"),
               load("net.blocks.0.self_attn.q_norm.weight"), load("net.blocks.0.self_attn.k_norm.weight"),
               16, 128, True, S, S)
ad_sa = AdaLN3(load("net.blocks.0.adaln_modulation_self_attn.1.weight"),
               load("net.blocks.0.adaln_modulation_self_attn.2.weight"))

def t_embed(sv):
    h=1024
    e=-torch.log(torch.tensor(10000.0))*torch.arange(h,dtype=torch.float32)/h
    v=torch.tensor([float(sv)])[:,None]*torch.exp(e)[None,:]
    return torch.cat([torch.cos(v),torch.sin(v)],dim=-1)
ctx = torch.from_numpy(np.fromfile(D+"/gen/i_ctx.bin", dtype=np.float16).astype(np.float32)).reshape(M,1024)
te = t_embed(1.0)
W1, W2 = load("net.t_embedder.1.linear_1.weight"), load("net.t_embedder.1.linear_2.weight")
lora = F.linear(F.silu(F.linear(te, W1)), W2)
lsh, lsc, lga = lora[:,:2048], lora[:,2048:4096], lora[:,4096:6144]
cos, sin = rope3d_cos_sin(1, 16, 16)
cos = cos.reshape(S,1,128); sin = sin.reshape(S,1,128)
xb = torch.from_numpy(np.fromfile(D+"/gen5/i_xb.bin", dtype=np.float16).astype(np.float32)).reshape(S,2048)

# ===== torch 参考（逐点）=====
print("=== torch 逐点 ===", flush=True)
with torch.no_grad():
    sh, sc, ga = ad_sa(te, lsh, lsc, lga)
    sh, sc, ga = sh.reshape(-1), sc.reshape(-1), ga.reshape(-1)
    pts = {}
    pts["y"]  = xb * r_sa
    pts["mu"] = pts["y"].mean(-1, keepdim=True)
    pts["c"]  = pts["y"] - pts["mu"]
    pts["s"]  = torch.abs(pts["c"]).mean(-1, keepdim=True) + SSE
    pts["rq"] = torch.rsqrt(pts["s"])
    pts["a"]  = pts["rq"] * pts["rq"]
    pts["z"]  = pts["a"] * pts["c"]
    pts["e"]  = pts["a"] * SSE
    pts["v"]  = (pts["z"]*pts["z"]).mean(-1, keepdim=True) + pts["e"]*pts["e"]
    pts["vq"] = torch.rsqrt(pts["v"])
    pts["n"]  = pts["z"] * pts["vq"]
    pts["n2"] = pts["n"] * (1.0 + sc) + sh
    pts["at"] = attn_sa(pts["n2"], None, cos, sin)
    pts["g1"] = pts["at"] * ga
    pts["g2"] = pts["g1"] * inv_d
    pts["x1"] = xb + pts["g2"]
    pts["sh"] = sh.reshape(1, -1).expand(S, DIM)
    pts["sc"] = sc.reshape(1, -1).expand(S, DIM)
    pts["ga"] = ga.reshape(1, -1).expand(S, DIM)

order = ["y","mu","c","s","rq","a","z","e","v","vq","n","n2","at","g1","g2","x1"]
for nm in order:
    t = pts[nm]
    print("  %-4s %-14s max=%12.4f std=%10.4f" % (nm, tuple(t.shape), float(t.abs().max()), float(t.std())))
    t.detach().numpy().astype(np.float16).tofile(OUT+"/t_%s.bin" % nm)

# 输入文件
xb.numpy().astype(np.float16).tofile(OUT+"/in_xb.bin")
te.numpy().astype(np.float16).tofile(OUT+"/in_te.bin")
lsh.numpy().astype(np.float16).tofile(OUT+"/in_lsh.bin")
lsc.numpy().astype(np.float16).tofile(OUT+"/in_lsc.bin")
lga.numpy().astype(np.float16).tofile(OUT+"/in_lga.bin")
ctx.numpy().astype(np.float16).tofile(OUT+"/in_ctx.bin")
cos.numpy().astype(np.float16).tofile(OUT+"/in_cos.bin")
sin.numpy().astype(np.float16).tofile(OUT+"/in_sin.bin")

# ===== 生成探针图 =====
print("\n=== 生成探针图 ===", flush=True)

class P(torch.nn.Module):
    """stage 控制算到哪一步。"""
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.ad = ad_sa
        self.attn = attn_sa
    def forward(self, xb_, te_, lsh_, lsc_, lga_, ctx_, cos_, sin_):
        sh_, sc_, ga_ = self.ad(te_, lsh_, lsc_, lga_)
        sh_, sc_, ga_ = sh_.reshape(-1), sc_.reshape(-1), ga_.reshape(-1)
        if self.stage == "sh": return sh_.reshape(1,-1).expand(S, DIM)
        if self.stage == "sc": return sc_.reshape(1,-1).expand(S, DIM)
        if self.stage == "ga": return ga_.reshape(1,-1).expand(S, DIM)
        y = xb_ * r_sa
        if self.stage == "y": return y
        mu = y.mean(-1, keepdim=True)
        if self.stage == "mu": return mu
        c = y - mu
        if self.stage == "c": return c
        s = torch.abs(c).mean(-1, keepdim=True) + SSE
        if self.stage == "s": return s
        rq = torch.rsqrt(s)
        if self.stage == "rq": return rq
        a = rq * rq
        if self.stage == "a": return a
        z = a * c
        if self.stage == "z": return z
        e = a * SSE
        if self.stage == "e": return e
        v = (z*z).mean(-1, keepdim=True) + e*e
        if self.stage == "v": return v
        vq = torch.rsqrt(v)
        if self.stage == "vq": return vq
        n = z * vq
        if self.stage == "n": return n
        n2 = n * (1.0 + sc_) + sh_
        if self.stage == "n2": return n2
        at = self.attn(n2, None, cos_, sin_)
        if self.stage == "at": return at
        g1 = at * ga_
        if self.stage == "g1": return g1
        g2 = g1 * inv_d
        if self.stage == "g2": return g2
        return xb_ + g2

ins_shape = [(S,DIM),(1,1,DIM),(1,1,DIM),(1,1,DIM),(1,1,DIM),(M,1024),(S,1,128),(S,1,128)]
ins_files = ["in_xb.bin","in_te.bin","in_lsh.bin","in_lsc.bin","in_lga.bin","in_ctx.bin","in_cos.bin","in_sin.bin"]

def make_graph(nm):
    d = OUT + "/g_" + nm
    os.makedirs(d, exist_ok=True)
    low = Lowering(d)
    args = [low.input(sh) for sh in ins_shape]
    out = low.lower_module(P(nm), args)
    open(d+"/g.txt","w").write(low.graph_text())
    errs = low.validate()
    return d, len(low.ops), errs

def run_probe(d, key, outs):
    p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c+"\n"); p.stdin.flush()
        l = p.stdout.readline(); return l.strip() if l else ""
    s('{"cmd":"cachedir","path":"%s/nc"}' % OUT)
    s('{"cmd":"cachekey","v":"%s"}' % key)
    r = s('{"cmd":"build","path":"%s/g.txt"}' % d)
    if '"ok":true' not in r: p.kill(); return "build:"+r[:30]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r: p.kill(); return "compile:"+r[:30]
    spec = "8 " + " ".join("%d %s" % (i, OUT+"/"+f) for i,f in enumerate(ins_files)) + " 1 0 %s" % outs
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=200)
    except: p.kill()
    return None if '"ok":true' in r else "run:"+r[:30]

print("\n%-4s %6s %12s %12s %12s %10s" % ("点", "算子", "torch.max", "NPU.max", "MAX_ABS", "rel%"))
results = {}
for nm in order:
    d, nops, errs = make_graph(nm)
    if errs:
        print("%-4s %6d  图校验失败: %s" % (nm, nops, errs[:1])); continue
    of = OUT+"/o_%s.bin" % nm
    e = run_probe(d, "p_"+nm, of)
    if e:
        print("%-4s %6d  %s" % (nm, nops, e[:40])); continue
    got = np.fromfile(of, dtype=np.float16).astype(np.float32)
    ref = pts[nm].numpy().reshape(-1)
    if got.size != ref.size:
        print("%-4s %6d  尺寸不符 got=%d ref=%d" % (nm, nops, got.size, ref.size)); continue
    err = np.abs(got - ref)
    results[nm] = (err.max(), 100*err.max()/(np.abs(ref).max()+1e-12))
    print("%-4s %6d %12.4f %12.4f %12.4f %10.4f"
          % (nm, nops, np.abs(ref).max(), np.abs(got).max(), err.max(), results[nm][1]), flush=True)

print("\n=== 第一个超过 1% 的点 ===")
for nm in order:
    if nm in results and results[nm][1] > 1.0:
        print("  %s: rel=%.2f%%  MAX_ABS=%.4f" % (nm, results[nm][1], results[nm][0]))
        break
else:
    print("  全部 <1%")
