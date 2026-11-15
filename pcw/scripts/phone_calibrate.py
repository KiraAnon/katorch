"""标定：把 fp16 图里所有数据 operand 都声明成输出，跑一次拿到每个张量的真实最大值，
   据此生成 per-operand 的激活 scale 表。"""
import os, sys, json, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
HOST = D + "/host_new_d"          # 带 dump 命令
SRC = D + "/e2eper512/b00/g.txt"
CAL = "/data/local/tmp/calib"
GRAPHDIR = "/data/local/tmp/calg"

os.makedirs(CAL, exist_ok=True)
os.makedirs(GRAPHDIR, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd = {}
for ln in lines:
    if ln.startswith("OPD "):
        tok = ln.split()
        opd[int(tok[1])] = ln

io_line = [l for l in lines if l.startswith("IO ")][0]
tok = io_line.split()
n_in = int(tok[1]); ins = [int(v) for v in tok[2:2 + n_in]]
n_out = int(tok[2 + n_in]); outs = [int(v) for v in tok[3 + n_in:3 + n_in + n_out]]

# 所有 type 8 (fp16) 的 operand 都当输出
data_ids = []
for oid, ln in opd.items():
    if int(ln.split()[2]) == 8:
        data_ids.append(oid)
data_ids = [o for o in data_ids if o not in ins]
data_ids.sort()
print("fp16 数据 operand 共 %d 个，输入 %d 个" % (len(data_ids), len(ins)), flush=True)

# 新图：把所有数据 operand 都声明为输出
new = [l for l in lines if l.startswith("OPD ") or l.startswith("OP ")]
new.append("IO %d %s %d %s" % (len(ins), " ".join(map(str, ins)),
                               len(data_ids), " ".join(map(str, data_ids))))
new.append("END")
gpath = GRAPHDIR + "/g.txt"
open(gpath, "w").write("\n".join(new) + "\n")

INS = ["i_xb.bin", "i_te.bin", "i_lsh.bin", "i_lsc.bin", "i_lga.bin",
       "i_ctx_pos.bin", "i_cos.bin", "i_sin.bin"]
ins_paths = [os.path.join(OUT, f) for f in INS]

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

r = s('{"cmd":"build","path":"%s"}' % gpath)
print("build  :", r[:70], flush=True)
if '"ok":true' not in r:
    sys.exit(1)
r = s('{"cmd":"compile"}')
print("compile:", r[:70], flush=True)
if '"ok":true' not in r:
    sys.exit(1)
r = s('{"cmd":"dump","dir":"%s"}' % CAL)
print("dump   :", r[:70], flush=True)

spec = "%d %s 1 x" % (len(ins_paths), " ".join("%d %s" % (i, f) for i, f in enumerate(ins_paths)))
r = s('{"cmd":"run","spec":"%s"}' % spec)
print("run    :", r[:70], flush=True)
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()

# 读回最大绝对值
tbl = {}
missing = 0
for oid in data_ids:
    fp = "%s/o%d.bin" % (CAL, oid)
    if not os.path.exists(fp):
        missing += 1; continue
    a = np.fromfile(fp, dtype=np.float16).astype(np.float32)
    mx = float(np.abs(a).max()) if a.size else 0.0
    tbl[str(oid)] = max(mx / 32767.0, 1e-8)

print("\n标定完成：%d 个 operand 有数据，%d 个缺失" % (len(tbl), missing))
if tbl:
    vals = sorted(tbl.values())
    print("  scale 最小 %.3g  中位 %.3g  最大 %.3g"
          % (vals[0], vals[len(vals) // 2], vals[-1]))
    tblp = "/data/local/tmp/calib_scale.json"
    json.dump(tbl, open(tblp, "w"))
    print("  -> %s" % tblp)
