"""用合并了 turbo LoRA 的权重，重新生成 512×512 的 14 块 + pe + fl（完整权重）。"""
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
DST = D + "/e2eper512m"        # 合并版
SCRATCH = "/data/local/tmp/gscratchm"
M = 512
S = 1024

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
os.makedirs(DST, exist_ok=True)

print("KA_MERGE_LORA = %s" % os.environ.get("KA_MERGE_LORA", "(未设置)"), flush=True)

class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)


def gen(build_mod, in_shapes, outdir, tag):
    if os.path.exists(outdir + "/g.txt"):
        print("  %-4s 已存在，跳过" % tag, flush=True); return
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(SCRATCH)
    low = Lowering(SCRATCH)
    args = [low.input(sh) for sh in in_shapes]
    low.lower_module(build_mod, args)
    errs = low.validate()
    if errs:
        print("  %-4s 校验失败 %s" % (tag, str(errs)[:60]), flush=True); return
    open(os.path.join(SCRATCH, "g.txt"), "w").write(low.graph_text())
    os.makedirs(outdir, exist_ok=True)
    txt = open(os.path.join(SCRATCH, "g.txt")).read()
    os.system("cp %s/* %s/ 2>/dev/null" % (SCRATCH, outdir))
    # 把 g.txt 里的路径全部改指到 outdir
    new = txt.replace(SCRATCH, outdir)
    open(os.path.join(outdir, "g.txt"), "w").write(new)
    sz = sum(os.path.getsize(os.path.join(SCRATCH, f)) for f in os.listdir(SCRATCH))
    print("  %-4s OK  %d 算子  %.0f MB" % (tag, len(low.ops), sz / 1e6), flush=True)


t_all = time.time()
print("=== 14 个块 ===", flush=True)
for bi in range(14):
    b0, b1 = 2 * bi, 2 * bi + 1
    m = Pair(BalDiTBlock("net.blocks.%d." % b0, pb[b0]["d_in"], pb[b0]["gs"], S=S, M=M).eval(),
             BalDiTBlock("net.blocks.%d." % b1, pb[b1]["d_in"], pb[b1]["gs"], S=S, M=M).eval(),
             pb[b0]["post_scale"]).eval()
    gen(m, [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID),
            (M, CTX), (S, 1, HD), (S, 1, HD)], "%s/b%02d" % (DST, bi), "b%02d" % bi)

print("\n=== pe / fl ===", flush=True)
pe = PatchEmbed(load("net.x_embedder.proj.1.weight")).eval()
gen(pe, [(S, 68)], "%s/pe" % DST, "pe")

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
gen(bf, [(S, 2048), (1, 2048), (1, 2048), (1, 2048)], "%s/fl" % DST, "fl")

shutil.rmtree(SCRATCH, ignore_errors=True)
sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DST) for f in fs)
print("\n总耗时 %.0fs，%s 共 %.0f MB" % (time.time() - t_all, DST, sz / 1e6))
