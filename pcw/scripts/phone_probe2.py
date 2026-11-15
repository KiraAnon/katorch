"""逐岛探针：完整图编译，只改 IO 输出，检查每个岛输出的 max 和 NaN 个数。"""
import os, json, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
GIS = "/data/local/tmp/q16i/g.txt"
CAL = "/data/local/tmp/calib_i.json"
IN = os.environ.get("KA_IN", D + "/gen5/cal/step00")
T = "/data/local/tmp/probe2"
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
n_in = int(t[1])
ins = [int(v) for v in t[2:2 + n_in]]

INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
raw = [os.path.join(IN, f) for f in INS]
qp = []
for i, p in enumerate(raw):
    a = np.fromfile(p, dtype=np.float16).astype(np.float32)
    s = np.float32(opd[i]["scale"])
    q = np.clip(np.rint(a / s), -32767, 32767).astype(np.int16)
    f = "%s/qin%d.i16" % (T, i)
    q.tofile(f)
    qp.append(f)

base = [l for l in lines if not l.startswith("IO ") and l.strip() != "END"]


def run(target):
    gp = "%s/g_%d.txt" % (T, target)
    open(gp, "w").write("\n".join(base + [
        "IO %d %s 1 %d" % (len(ins), " ".join(map(str, ins)), target), "END"]) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None, "build"
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile"
    o = "%s/o_%d.bin" % (T, target)
    spec = "%d %s 1 %s" % (len(qp), " ".join("%d %s" % (i, f) for i, f in enumerate(qp)), o)
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run"
    od = opd[target]
    v = (np.fromfile(o, dtype=np.float16).astype(np.float32) if od["ty"] == 8
         else np.fromfile(o, dtype=np.int16).astype(np.float32) * np.float32(od["scale"]))
    fin = np.isfinite(v)
    return (float(np.abs(v[fin]).max()) if fin.any() else -1.0, int((~fin).sum())), None


print("输入目录: %s" % IN)
print("%-8s %-12s %-12s %s" % ("operand", "设备max", "标定max", "状态"))
for oid in (91, 217, 322, 425, 530, 654, 779, 884, 987, 1092, 1131):
    got, err = run(oid)
    ref = tbl.get(str(oid))
    rmax = (float(ref) * 32767) if ref else None
    if got is None:
        print("%-8d %s" % (oid, err)); continue
    mx, nbad = got
    st = "OK" if (rmax and abs(mx - rmax) / max(rmax, 1e-9) < 0.25) else "偏差大"
    if nbad:
        st = "NaN=%d" % nbad
    print("%-8d %-12.4f %-12s %s" % (oid, mx, ("%.4f" % rmax) if rmax else "-", st))
