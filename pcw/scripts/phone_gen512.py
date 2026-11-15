"""生成 512×512 (S=1024) 的 14 个块 + pe + fl 图。
   权重 c*.bin 复用 e2eper256 的（已验证与 S 无关），只生成新的 i*.bin 与 g.txt。"""
import os, sys, json, shutil, time
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch
import torch.nn as nn
import torch.nn.functional as F
from anima_block import load
from bal_block2 import BalDiTBlock
from bal_norm import BalancedNorm
from anima_head import PatchEmbed
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
SRC = D + "/e2eper256"          # 权重来源
DST = D + "/e2eper512"          # 512 版图与 i*.bin
SCRATCH = "/data/local/tmp/gscratch"
OUT = D + "/gen5"
M = 512
S = 1024                        # 512/8/2 = 32 → 32*32 = 1024

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
os.makedirs(DST, exist_ok=True)

class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)

def gen_one(build_mod, in_shapes, outdir, weight_src_dir):
    """lower 到 scratch，再把 i*.bin+g.txt 搬到 outdir，c*.bin 路径改指 weight_src_dir。"""
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(SCRATCH)
    low = Lowering(SCRATCH)
    args = [low.input(sh) for sh in in_shapes]
    low.lower_module(build_mod, args)
    errs = low.validate()
    if errs:
        return "校验失败: %s" % str(errs)[:80]
    open(os.path.join(SCRATCH, "g.txt"), "w").write(low.graph_text())
    os.makedirs(outdir, exist_ok=True)
    # 搬 i*.bin
    for f in os.listdir(SCRATCH):
        if f.startswith("i") and f.endswith(".bin"):
            shutil.copy(os.path.join(SCRATCH, f), os.path.join(outdir, f))
    # 改写 g.txt 路径
    txt = open(os.path.join(SCRATCH, "g.txt")).read()
    lines = []
    for ln in txt.splitlines():
        if " w " in ln:
            base = ln.rsplit("/", 1)[-1]
            ln = ln[:ln.rindex(" w ") + 3] + os.path.join(
                weight_src_dir if base.startswith("c") else outdir, base)
        lines.append(ln)
    open(os.path.join(outdir, "g.txt"), "w").write("\n".join(lines) + "\n")
    return None

t_all = time.time()
print("=== 生成 14 个块 (S=%d) ===" % S, flush=True)
for bi in range(14):
    d = "%s/b%02d" % (DST, bi)
    if os.path.exists(d + "/g.txt"):
        print("  b%02d 已存在，跳过" % bi, flush=True); continue
    b0, b1 = 2 * bi, 2 * bi + 1
    m = Pair(BalDiTBlock("net.blocks.%d." % b0, pb[b0]["d_in"], pb[b0]["gs"], S=S, M=M).eval(),
             BalDiTBlock("net.blocks.%d." % b1, pb[b1]["d_in"], pb[b1]["gs"], S=S, M=M).eval(),
             pb[b0]["post_scale"]).eval()
    e = gen_one(m, [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID),
                    (M, CTX), (S, 1, HD), (S, 1, HD)],
                d, "%s/b%02d" % (SRC, bi))
    print("  b%02d %s" % (bi, e if e else "OK"), flush=True)

print("\n=== PatchEmbed / FinalLayer ===", flush=True)
pe = PatchEmbed(load("net.x_embedder.proj.1.weight")).eval()
e = gen_one(pe, [(S, 68)], "%s/pe" % DST, "%s/pe" % SRC)
print("  pe  %s" % (e if e else "OK"), flush=True)

class BalFinal(nn.Module):
    def __init__(s, d_in, g, hidden=2048):
        super().__init__()
        from anima_block import Lin
        s.ln = BalancedNorm(torch.full((hidden,), d_in / float(g)), torch.ones(hidden), kind="ln")
        s.ad_l1 = Lin(load("net.final_layer.adaln_modulation.1.weight"))
        w_l2 = load("net.final_layer.adaln_modulation.2.weight")
        Dd = w_l2.shape[0] // 2
        s.ad_p0 = Lin(w_l2[0:Dd]); s.ad_p1 = Lin(w_l2[Dd:2 * Dd])
        s.lin = Lin(load("net.final_layer.linear.weight"))
    def forward(s, x, emb, lsh, lsc):
        h = F.silu(emb); h = s.ad_l1(h)
        return s.lin(s.ln(x) * (1.0 + s.ad_p1(h) + lsc) + s.ad_p0(h) + lsh)

bf = BalFinal(pb[27]["d_in"], pb[27]["gs"][2]).eval()
e = gen_one(bf, [(S, 2048), (1, 2048), (1, 2048), (1, 2048)], "%s/fl" % DST, "%s/fl" % SRC)
print("  fl  %s" % (e if e else "OK"), flush=True)

shutil.rmtree(SCRATCH, ignore_errors=True)
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(DST) for f in fs)
print("\n总耗时 %.0fs，%s 共 %.0f MB" % (time.time() - t_all, DST, sz / 1e6))
