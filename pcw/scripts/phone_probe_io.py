"""保持完整图，只改 IO 的输出，逐个检查各岛输出的实际值 vs 标定表里的 fp16 参考。"""
import os, json, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
GIS = "/data/local/tmp/q16i/g.txt"
CAL = "/data/local/tmp/calib_i.json"
T = "/data/local/tmp/probeio"
OUT = os.environ.get("KA_IN", D + "/gen5")
os.makedirs(T, exist_ok=True)

tbl = json.load(open(CAL))
lines = [l.rstrip("\n") for l in open(GIS)]
opd, io_line = {}, None
for ln in lines:
    if ln.startswith("OPD "):
        t = ln.split()
        nd = int(t[3])
        opd[int(t[1])] = dict(ty=int(t[2]), nd=nd,
                              dims=[int(v) for v in t[4:4 + nd]],
                              scale=float(t[4 + nd]))
    elif ln.startswith("IO "):
        io_line = ln

t = io_line.split()
n_in = int(t[1]); ins = [int(v) for v in t[2:2 + n_in]]
print("图输入 %s" % ins)

INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
raw_paths = [os.path.join(OUT, f) for f in INS]
q_paths = []
for i, p in enumerate(raw_paths):
    a = np.fromfile(p, dtype=np.float16).astype(np.float32)
    s = np.float32(opd[i]["scale"])
    q = np.clip(np.rint(a / s), -32767, 32767).astype(np.int16)
    qp = "%s/qin%d.i16" % (T, i)
    q.tofile(qp)
    q_paths.append(qp)

base = [l for l in lines if not l.startswith("IO ") and l.strip() != "END"]


def run(target):
    gp = "%s/g_%d.txt" % (T, target)
    g = base + ["IO %d %s 1 %d" % (len(ins), " ".join(map(str, ins)), target), "END"]
    open(gp, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:40]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:40]
    outp = "%s/o_%d.bin" % (T, target)
    spec = "%d %s 1 %s" % (len(q_paths), " ".join("%d %s" % (i, f) for i, f in enumerate(q_paths)), outp)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run " + r[:40]
    o = opd[target]
    if o["ty"] == 8:
        v = np.fromfile(outp, dtype=np.float16).astype(np.float32)
    else:
        v = np.fromfile(outp, dtype=np.int16).astype(np.float32) * np.float32(o["scale"])
    fin = np.isfinite(v)
    return (float(np.abs(v[fin]).max()) if fin.any() else -1.0, int((~fin).sum())), None


print("\n%-8s %-12s %-12s %s" % ("operand", "设备 max", "标定 max", "状态"))
for oid in (1009, 1130, 1131, 1127, 1129):
    got, err = run(oid)
    ref = tbl.get(str(oid))
    refmax = (float(ref) * 32767) if ref else None
    if got is None:
        print("%-8d %s" % (oid, err))
    else:
        mx, nbad = got
        ok = "OK" if refmax and abs(mx - refmax) / max(refmax, 1e-9) < 0.25 else "偏差大"
        if nbad:
            ok = "NaN=%d" % nbad
        print("%-8d %-12.4f %-12s %s" % (oid, mx, ("%.4f" % refmax) if refmax else "-", ok))
