"""验证：同一层在不同 token 数下的运行耗时。
   S=256(256x256 图)  vs  S=1024(512x512 图)  vs  S=4096(1024x1024 图)
"""
import os, sys, json, subprocess, time
sys.path.insert(0, "/data/data/com.termux/files/home/npt")

import torch
from anima_block import load
from bal_block2 import BalDiTBlock
from lowering import Lowering
from t_chain import HID, HD, CTX

H = "/data/data/com.termux/files/home"
D = H + "/npt"
cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
M = 512
HOST = D + "/host_new"
WORK = "/data/local/tmp/scaletest"
os.makedirs(WORK, exist_ok=True)

def make_graph(S, tag):
    d = "%s/%s" % (WORK, tag)
    os.makedirs(d, exist_ok=True)
    g = d + "/g.txt"
    if os.path.exists(g):
        return d
    print("  生成图 S=%d ..." % S, flush=True)
    b = BalDiTBlock("net.blocks.0.", pb[0]["d_in"], pb[0]["gs"], S=S, M=M).eval()
    low = Lowering(d)
    args = [low.input((S, HID)), low.input((1, 1, HID)), low.input((1, 1, HID)),
            low.input((1, 1, HID)), low.input((1, 1, HID)), low.input((M, CTX)),
            low.input((S, 1, HD)), low.input((S, 1, HD))]
    low.lower_module(b, args)
    errs = low.validate()
    if errs:
        print("  校验失败: %s" % str(errs)[:100]); return None
    open(g, "w").write(low.graph_text())
    print("  S=%d: %d 算子, 图 %d 字节" % (S, len(low.ops), os.path.getsize(g)), flush=True)
    return d

def make_inputs(S):
    d = "%s/in_%d" % (WORK, S)
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(7)
    files = []
    shapes = [(S, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID), (1, 1, HID),
              (M, CTX), (S, 1, HD), (S, 1, HD)]
    for i, sh in enumerate(shapes):
        p = "%s/i%d.bin" % (d, i)
        if not os.path.exists(p):
            torch.randn(*sh).numpy().astype("float16").tofile(p)
        files.append(p)
    return files

def spawn():
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    return p

def s(p, c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

print("=== 准备三种规模的图 ===")
for S in (256, 1024, 4096):
    make_graph(S, "s%d" % S)

print("\n=== 逐个编译 + 运行 ===")
print("%-8s %8s %10s %10s %10s" % ("S", "算子数", "编译(s)", "运行(ms)", "每次输出MB"))
for S in (256, 1024, 4096):
    d = "%s/s%d" % (WORK, S)
    g = d + "/g.txt"
    if not os.path.exists(g):
        print("%-8d  图不存在" % S); continue
    ins = make_inputs(S)
    out = "%s/o%d.bin" % (WORK, S)
    p = spawn()
    t0 = time.time()
    r = s(p, '{"cmd":"build","path":"%s"}' % g)
    if '"ok":true' not in r:
        print("%-8d build失败 %s" % (S, r[:60])); p.kill(); continue
    ops = r.split('"ops":')[1].split(',')[0]
    r = s(p, '{"cmd":"compile"}')
    t_compile = time.time() - t0
    if '"ok":true' not in r:
        print("%-8d compile失败 %s" % (S, r[:60])); p.kill(); continue
    spec = "8 " + " ".join("%d %s" % (i, f) for i, f in enumerate(ins)) + " 1 " + out
    # 跑 5 次取平均
    ts = []
    for _ in range(5):
        a = time.time()
        r = s(p, '{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - a) * 1000)
        if '"ok":true' not in r:
            print("  run失败 %s" % r[:60]); break
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
    osz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print("%-8d %8s %10.1f %10.0f %10.1f" % (S, ops, t_compile, sum(ts) / len(ts), osz))
