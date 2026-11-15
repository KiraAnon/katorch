"""前缀二分：取前 N 个算子组成子图，分别用 fp16 和 W8A16 跑，比 NRMSE 找断点。"""
import os, sys, json, subprocess
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
SRC = os.environ.get("KA_FSRC", D + "/e2eper512/b00/g.txt")
QTBL = os.environ.get("KA_QDIR", "/data/local/tmp/q16b00")     # 已转换的 W8A16 图（全量）
W = "/data/local/tmp/pfx"
os.makedirs(W, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
qlines = [l.rstrip("\n") for l in open(QTBL + "/g.txt")]

def parse(lines_):
    opd, ops = {}, []
    for ln in lines_:
        if ln.startswith("OPD "):
            tok = ln.split()
            nd = int(tok[3])
            opd[int(tok[1])] = ln
        elif ln.startswith("OP "):
            tok = ln.split()
            code, ni = int(tok[1]), int(tok[2])
            ins = [int(v) for v in tok[3:3 + ni]]
            no = int(tok[3 + ni])
            outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
            ops.append((code, ins, outs))
    return opd, ops

fopd, fops = parse(lines)
qopd, qops = parse(qlines)
qch = {int(l.split()[1]): l for l in qlines if l.startswith("QCH ")}

INS = ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin", "i_lga512.bin",
       "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]
OUTDIR = D + "/gen5"
ins_paths = [os.path.join(OUTDIR, f) for f in INS]
orig_ins = [0, 1, 2, 3, 4, 5, 6, 7]

# 输入 operand 的 scale（从转换图里读）
qscale, qtype = {}, {}
for ln in qlines:
    if ln.startswith("OPD "):
        tok = ln.split()
        nd = int(tok[3])
        qtype[int(tok[1])] = int(tok[2])
        qscale[int(tok[1])] = float(tok[4 + nd])

# 量化输入文件
qin_paths = []
for i, p in enumerate(ins_paths):
    a = np.fromfile(p, dtype=np.float16).astype(np.float32)
    s = qscale.get(i, 0.02)
    q = np.clip(np.rint(a / s), -32767, 32767).astype(np.int16)
    qp = "%s/qin%d.i16" % (W, i)
    q.tofile(qp)
    qin_paths.append(qp)


def build_sub(n, quant):
    ops_ = qops[:n] if quant else fops[:n]
    opd_ = qopd if quant else fopd
    order, seen = [], set()
    for code, ins, outs in ops_:
        for v in ins + outs:
            if v not in seen and v in opd_:
                seen.add(v); order.append(v)
    mp = {o: i for i, o in enumerate(order)}
    g = []
    for o in order:
        tok = opd_[o].split(); tok[1] = str(mp[o]); g.append(" ".join(tok))
        if quant and o in qch:
            qt = qch[o].split(); qt[1] = str(mp[o]); g.append(" ".join(qt))
    for code, ins, outs in ops_:
        g.append("OP %d %d %s %d %s" % (code, len(ins),
                 " ".join(str(mp[v]) for v in ins if v in mp),
                 len(outs), " ".join(str(mp[v]) for v in outs if v in mp)))
    produced = set()
    for code, ins, outs in ops_:
        produced.update(outs)
    in_ids = [o for o in order if o in orig_ins and o not in produced]
    out_ids = [o for o in ops_[-1][2] if o in mp]
    if not in_ids or not out_ids:
        return None
    g.append("IO %d %s %d %s" % (len(in_ids), " ".join(str(mp[v]) for v in in_ids),
                                 len(out_ids), " ".join(str(mp[v]) for v in out_ids)))
    g.append("END")
    p = "%s/%s_%d.txt" % (W, "q" if quant else "f", n)
    open(p, "w").write("\n".join(g) + "\n")
    return p, in_ids, out_ids


def run(gp, quant, outsuf):
    inps = [ins_paths[orig_ins.index(o)] if not quant else qin_paths[orig_ins.index(o)]
            for o in last_in_ids]
    outs = ["%s/o_%s_%d.bin" % (W, outsuf, k) for k in range(len(last_out_ids))]
    spec = "%d %s %d %s" % (len(inps), " ".join("%d %s" % (k, f) for k, f in enumerate(inps)),
                            len(outs), " ".join("%d %s" % (k, f) for k, f in enumerate(outs)))
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()
    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        p.kill(); return None, "build:" + r[:40]
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        p.kill(); return None, "compile:" + r[:40]
    r = s('{"cmd":"run","spec":"%s"}' % spec)
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    if '"ok":true' not in r:
        return None, "run:" + r[:40]
    return outs, None


print("%-6s %-10s %-10s %s" % ("N", "fp16max", "q16max", "NRMSE"))
for n in (1,2,4,8,16,20,30,40,60,80,120,160,240,320,420,520,593):
    rf = build_sub(n, False)
    if rf is None:
        continue
    gp, last_in_ids, last_out_ids = rf
    of, e1 = run(gp, False, "f")
    rq = build_sub(n, True)
    if rq is None:
        continue
    gpq, last_in_ids, last_out_ids = rq
    oq, e2 = run(gpq, True, "q")
    if of is None or oq is None:
        print("%-6d 失败 %s %s" % (n, e1, e2)); continue
    a = np.fromfile(of[0], dtype=np.float16).astype(np.float32)
    oid = last_out_ids[0]
    if qtype.get(oid) == 8:
        # FP16（岛内输出）：直接按 fp16 读，不乘尺度
        b = np.fromfile(oq[0], dtype=np.float16).astype(np.float32)
    else:
        b = np.fromfile(oq[0], dtype=np.int16).astype(np.float32) * qscale.get(oid, 0.02)
    m = min(len(a), len(b))
    nrmse = np.sqrt(((a[:m] - b[:m]) ** 2).mean()) / (np.sqrt((a[:m] ** 2).mean()) + 1e-12)
    print("%-6d %-10.4f %-10.4f %.3f%%" % (n, np.abs(a).max(), np.abs(b).max(), nrmse * 100), flush=True)
