"""验证不同 S 下生成的权重文件是否完全相同（决定能否复用）。"""
import os, sys, json, hashlib, shutil
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
SCRATCH = "/data/local/tmp/g512test"
cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
M = 512

def hashes(d):
    r = {}
    for f in os.listdir(d):
        if f.endswith(".bin"):
            p = os.path.join(d, f)
            r[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return r

print("=== 参考：e2eper256/b00 现有文件 ===")
ref = hashes(E2E + "/b00")
print("  %d 个 .bin" % len(ref))

if os.path.exists(SCRATCH):
    shutil.rmtree(SCRATCH)
os.makedirs(SCRATCH)

print("\n=== 生成 S=1024 的 b00 到临时目录 ===")
import time
t0 = time.time()
b0 = BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=1024, M=M).eval()
b1 = BalDiTBlock("net.blocks.1.", pb[1]["d_in"], pb[1]["gs"], S=1024, M=M).eval()

import torch.nn as nn
class Pair(nn.Module):
    def __init__(s, a, b, ps):
        super().__init__()
        s.a, s.b = a, b
        s.register_buffer("ps", torch.tensor([float(ps)]))
    def forward(s, x, e, l1, l2, l3, c, co, si):
        return s.b(s.a(x, e, l1, l2, l3, c, co, si) * s.ps, e, l1, l2, l3, c, co, si)

m = Pair(b0, b1, pb[0]["post_scale"]).eval()
low = Lowering(SCRATCH)
S = 1024
args = [low.input((S, HID)), low.input((1, 1, HID)), low.input((1, 1, HID)),
        low.input((1, 1, HID)), low.input((1, 1, HID)), low.input((M, CTX)),
        low.input((S, 1, HD)), low.input((S, 1, HD))]
low.lower_module(m, args)
errs = low.validate()
print("  %d 算子，校验 %s，耗时 %.1fs" % (len(low.ops), "通过" if not errs else "失败", time.time() - t0))

new = hashes(SCRATCH)
print("  生成 %d 个 .bin" % len(new))

common = set(ref) & set(new)
same = sum(1 for k in common if ref[k] == new[k])
print("\n=== 对比 ===")
print("  共同文件名: %d" % len(common))
print("  内容相同的: %d" % same)
print("  内容不同的: %d" % (len(common) - same))
print("  仅旧有: %d   仅新有: %d" % (len(set(ref) - set(new)), len(set(new) - set(ref))))

if len(common) - same > 0:
    diff = [k for k in common if ref[k] != new[k]][:5]
    print("  不同的例子: %s" % diff)

tot = sum(os.path.getsize(os.path.join(SCRATCH, f)) for f in os.listdir(SCRATCH))
print("  临时目录总大小: %.0f MB" % (tot / 1e6))
