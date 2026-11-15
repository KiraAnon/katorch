"""探针 ca 和 mlp 支路。"""

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
from anima_block import load, Lin, RMS, Attn, AdaLN3, GELU
from lowering import Lowering
from t_rope3d import rope3d_cos_sin

H = _KA_HOME
D = H + "/npt"
OUT = D + "/prb2"
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
S, M, DIM = 256, 512, 2048
cal = json.load(open(D+"/gen/calib6.json"))
pb = cal["per_block"][0]
d_in, gs = pb["d_in"], pb["gs"]
r_sa, r_ca, r_mlp = (d_in/g for g in gs)
inv_d = 1.0/d_in
SSE = 0.001

attn_sa = Attn(load("net.blocks.0.self_attn.q_proj.weight"), load("net.blocks.0.self_attn.k_proj.weight"),
               load("net.blocks.0.self_attn.v_proj.weight"), load("net.blocks.0.self_attn.output_proj.weight"),
               load("net.blocks.0.self_attn.q_norm.weight"), load("net.blocks.0.self_attn.k_norm.weight"), 16,128,True,S,S)
attn_ca = Attn(load("net.blocks.0.cross_attn.q_proj.weight"), load("net.blocks.0.cross_attn.k_proj.weight"),
               load("net.blocks.0.cross_attn.v_proj.weight"), load("net.blocks.0.cross_attn.output_proj.weight"),
               load("net.blocks.0.cross_attn.q_norm.weight"), load("net.blocks.0.cross_attn.k_norm.weight"), 16,128,False,S,M)
mlp_in = Lin(load("net.blocks.0.mlp.layer1.weight"))
mlp_out = Lin(load("net.blocks.0.mlp.layer2.weight"))
mlp_act = GELU()
ad_sa = AdaLN3(load("net.blocks.0.adaln_modulation_self_attn.1.weight"), load("net.blocks.0.adaln_modulation_self_attn.2.weight"))
ad_ca = AdaLN3(load("net.blocks.0.adaln_modulation_cross_attn.1.weight"), load("net.blocks.0.adaln_modulation_cross_attn.2.weight"))
ad_mlp = AdaLN3(load("net.blocks.0.adaln_modulation_mlp.1.weight"), load("net.blocks.0.adaln_modulation_mlp.2.weight"))

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

def norm(y):
    mu = y.mean(-1, keepdim=True); c = y - mu
    s_ = torch.abs(c).mean(-1, keepdim=True) + SSE
    rq = torch.rsqrt(s_); a = rq*rq
    z = a*c; e = a*SSE
    v = (z*z).mean(-1, keepdim=True) + e*e
    return z * torch.rsqrt(v)

print("=== torch 全块逐点 ===", flush=True)
with torch.no_grad():
    sh, sc, ga = ad_sa(te, lsh, lsc, lga); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
    n = norm(xb * r_sa); n2 = n*(1.0+sc)+sh
    at = attn_sa(n2, None, cos, sin)
    x1 = xb + at*ga*inv_d
    P = {"n": n, "n2": n2, "at": at, "x1": x1}

    sh, sc, ga = ad_ca(te, lsh, lsc, lga); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
    P["ca_c"] = x1 * r_ca
    P["ca_n"] = norm(P["ca_c"])
    P["ca_n2"] = P["ca_n"]*(1.0+sc)+sh
    P["ca_at"] = attn_ca(P["ca_n2"], ctx, cos, sin)
    P["ca_g1"] = P["ca_at"]*ga
    x2 = x1 + P["ca_g1"]*inv_d
    P["x2"] = x2

    sh, sc, ga = ad_mlp(te, lsh, lsc, lga); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
    P["mlp_c"] = x2 * r_mlp
    P["mlp_n"] = norm(P["mlp_c"])
    P["mlp_n2"] = P["mlp_n"]*(1.0+sc)+sh
    P["mlp_h"] = mlp_in(P["mlp_n2"])
    P["mlp_a"] = mlp_act(P["mlp_h"])
    P["mlp_o"] = mlp_out(P["mlp_a"])
    P["mlp_g1"] = P["mlp_o"]*ga
    P["x3"] = x2 + P["mlp_g1"]*inv_d

order = ["n","n2","at","x1","ca_c","ca_n","ca_n2","ca_at","ca_g1","x2",
         "mlp_c","mlp_n","mlp_n2","mlp_h","mlp_a","mlp_o","mlp_g1","x3"]
for nm in order:
    t = P[nm]
    print("  %-8s %-14s max=%12.4f std=%10.4f" % (nm, tuple(t.shape), float(t.abs().max()), float(t.std())))
    t.detach().numpy().astype(np.float16).tofile(OUT+"/t_%s.bin" % nm)

for nm, arr in [("in_xb",xb),("in_te",te),("in_lsh",lsh),("in_lsc",lsc),("in_lga",lga),("in_ctx",ctx),("in_cos",cos),("in_sin",sin)]:
    arr.numpy().astype(np.float16).tofile(OUT+"/%s.bin" % nm)

# ===== 探针图 =====
class P2(torch.nn.Module):
    def __init__(self, stage):
        super().__init__(); self.stage = stage
        self.ad_sa, self.ad_ca, self.ad_mlp = ad_sa, ad_ca, ad_mlp
        self.a_sa, self.a_ca = attn_sa, attn_ca
        self.m_in, self.m_out, self.m_act = mlp_in, mlp_out, mlp_act
        self.r_sa, self.r_ca, self.r_mlp, self.inv_d = r_sa, r_ca, r_mlp, inv_d
    def nf(self, y):
        mu = y.mean(-1, keepdim=True); c = y - mu
        s_ = torch.abs(c).mean(-1, keepdim=True) + SSE
        rq = torch.rsqrt(s_); a = rq*rq
        z = a*c; e = a*SSE
        v = (z*z).mean(-1, keepdim=True) + e*e
        return z * torch.rsqrt(v)
    def forward(self, xb_, te_, lsh_, lsc_, lga_, ctx_, cos_, sin_):
        sh, sc, ga = self.ad_sa(te_, lsh_, lsc_, lga_); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
        n = self.nf(xb_ * self.r_sa); n2 = n*(1.0+sc)+sh
        at = self.a_sa(n2, None, cos_, sin_)
        x1 = xb_ + at*ga*self.inv_d
        if self.stage == "x1": return x1
        sh, sc, ga = self.ad_ca(te_, lsh_, lsc_, lga_); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
        ca_c = x1 * self.r_ca
        ca_n = self.nf(ca_c); ca_n2 = ca_n*(1.0+sc)+sh
        ca_at = self.a_ca(ca_n2, ctx_, cos_, sin_)
        ca_g1 = ca_at*ga
        x2 = x1 + ca_g1*self.inv_d
        if self.stage == "x2": return x2
        sh, sc, ga = self.ad_mlp(te_, lsh_, lsc_, lga_); sh,sc,ga = sh.reshape(-1),sc.reshape(-1),ga.reshape(-1)
        mlp_c = x2 * self.r_mlp
        mlp_n = self.nf(mlp_c); mlp_n2 = mlp_n*(1.0+sc)+sh
        mlp_h = self.m_in(mlp_n2); mlp_a = self.m_act(mlp_h); mlp_o = self.m_out(mlp_a)
        return x2 + mlp_o*ga*self.inv_d

ins_shape = [(S,DIM),(1,1,DIM),(1,1,DIM),(1,1,DIM),(1,1,DIM),(M,1024),(S,1,128),(S,1,128)]
ins_files = ["in_xb.bin","in_te.bin","in_lsh.bin","in_lsc.bin","in_lga.bin","in_ctx.bin","in_cos.bin","in_sin.bin"]

def go(nm):
    d = OUT + "/g_" + nm
    os.makedirs(d, exist_ok=True)
    low = Lowering(d)
    args = [low.input(sh) for sh in ins_shape]
    low.lower_module(P2(nm), args)
    open(d+"/g.txt","w").write(low.graph_text())
    errs = low.validate()
    if errs: return None, None, errs
    p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c+"\n"); p.stdin.flush()
        l = p.stdout.readline(); return l.strip() if l else ""
    s('{"cmd":"cachedir","path":"%s/nc"}' % OUT)
    s('{"cmd":"cachekey","v":"%s"}' % nm)
    r = s('{"cmd":"build","path":"%s/g.txt"}' % d)
    if '"ok":true' not in r: p.kill(); return None, None, r[:30]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r: p.kill(); return None, None, r[:30]
    of = OUT+"/o_%s.bin" % nm
    spec = "8 " + " ".join("%d %s" % (i, OUT+"/"+f) for i, f in enumerate(ins_files)) + " 1 0 " + of
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=200)
    except: p.kill()
    return of, len(low.ops), None

print("\n=== 三段探针 ===")
for nm in ["x1", "x2", "x3"]:
    of, nops, err = go(nm)
    if err: print("  %-4s 失败: %s" % (nm, err)); continue
    got = np.fromfile(of, dtype=np.float16).astype(np.float32)
    ref = P[nm].numpy().reshape(-1)
    if got.size != ref.size:
        print("  %-4s 尺寸不符 %d vs %d" % (nm, got.size, ref.size)); continue
    er = np.abs(got - ref)
    print("  %-4s %4d算子  torch.max=%10.3f  NPU.max=%10.3f  MAX_ABS=%9.4f  rel=%8.4f%%"
          % (nm, nops, np.abs(ref).max(), np.abs(got).max(), er.max(), 100*er.max()/(np.abs(ref).max()+1e-12)), flush=True)

# 用完整块图对比
print("\n=== 完整块（参考）===")
print("  torch x3 max=%.3f std=%.4f" % (P["x3"].abs().max(), P["x3"].std()))
