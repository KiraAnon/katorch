"""分块标定：每批把一部分 operand 声明成输出，跑一次取最大值。"""
import os, sys, json, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
HOST = D + "/host_new_d"
SRC = D + "/e2eper512/b00/g.txt"
CAL = "/data/local/tmp/calib"
GRAPHDIR = "/data/local/tmp/calg"
CHUNK = int(os.environ.get("CAL_CHUNK", "100"))

os.makedirs(CAL, exist_ok=True)
os.makedirs(GRAPHDIR, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd = {}
for ln in lines:
    if ln.startswith("OPD "):
        opd[int(ln.split()[1])] = ln
io_line = [l for l in lines if l.startswith("IO ")][0]
tok = io_line.split()
n_in = int(tok[1]); ins = [int(v) for v in tok[2:2 + n_in]]

data_ids = sorted(oid for oid, ln in opd.items()
                  if int(ln.split()[2]) == 8 and oid not in ins)
print("待标定 operand: %d，每批 %d" % (len(data_ids), CHUNK), flush=True)

INS = ["i_xb.bin", "i_te.bin", "i_lsh.bin", "i_lsc.bin", "i_lga.bin",
       "i_ctx_pos.bin", "i_cos.bin", "i_sin.bin"]
ins_paths = [os.path.join(OUT, f) for f in INS]
spec = "%d %s 1 x" % (len(ins_paths), " ".join("%d %s" % (i, f) for i, f in enumerate(ins_paths)))

base = [l for l in lines if l.startswith("OPD ") or l.startswith("OP ")]
tbl = {}
nchunks = (len(data_ids) + CHUNK - 1) // CHUNK
for ci in range(nchunks):
    ids = data_ids[ci * CHUNK:(ci + 1) * CHUNK]
    g = base + ["IO %d %s %d %s" % (len(ins), " ".join(map(str, ins)),
                                    len(ids), " ".join(map(str, ids))), "END"]
    gp = "%s/g%d.txt" % (GRAPHDIR, ci)
    open(gp, "w").write("\n".join(g) + "\n")

    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()
    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        print("  批%d build 失败 %s" % (ci, r[:50]), flush=True); p.kill(); continue
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        print("  批%d compile 失败 %s" % (ci, r[:50]), flush=True); p.kill(); continue
    s('{"cmd":"dump","dir":"%s"}' % CAL)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    if '"ok":true' not in r:
        print("  批%d run 失败 %s" % (ci, r[:50]), flush=True); continue

    got = 0
    for oid in ids:
        fp = "%s/o%d.bin" % (CAL, oid)
        if not os.path.exists(fp):
            continue
        a = np.fromfile(fp, dtype=np.float16).astype(np.float32)
        mx = float(np.abs(a).max()) if a.size else 0.0
        tbl[str(oid)] = max(mx / 32767.0, 1e-8)
        got += 1
    print("  批%d/%d  %d/%d 个operand  OK" % (ci + 1, nchunks, got, len(ids)), flush=True)

print("\n标定完成：%d / %d" % (len(tbl), len(data_ids)))
if tbl:
    v = sorted(tbl.values())
    print("  scale 范围 %.3g ... %.3g  中位 %.3g" % (v[0], v[-1], v[len(v)//2]))
    json.dump(tbl, open("/data/local/tmp/calib_scale.json", "w"))
    print("  -> /data/local/tmp/calib_scale.json")
