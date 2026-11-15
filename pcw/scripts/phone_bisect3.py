"""二分：前 N 个算子组成的子图，找到第一个让整图失效的位置（IO 规则已修正）。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = "/data/local/tmp/q16b00/g.txt"
OUT = "/data/local/tmp/bisect3"
os.makedirs(OUT, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd, kind, qch = {}, {}, {}
for ln in lines:
    if ln.startswith("OPD "):
        tok = ln.split()
        oid = int(tok[1]); nd = int(tok[3])
        opd[oid] = ln
        kind[oid] = tok[6 + nd] if len(tok) > 6 + nd else ""
    elif ln.startswith("QCH "):
        qch[int(ln.split()[1])] = ln
ops = [ln for ln in lines if ln.startswith("OP ")]


def parse(ln):
    tok = ln.split()
    code, ni = int(tok[1]), int(tok[2])
    ins = [int(v) for v in tok[3:3 + ni]]
    no = int(tok[3 + ni])
    outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
    return code, ins, outs


def is_scalar(v):
    ln2 = opd.get(v)
    if not ln2:
        return True
    return int(ln2.split()[2]) in (1, 4)


def build_sub(n):
    pars = [parse(l) for l in ops[:n]]
    order, seen = [], set()
    for code, ins, outs in pars:
        for v in ins + outs:
            if v not in seen and v in opd:
                seen.add(v); order.append(v)
    mp = {o: i for i, o in enumerate(order)}
    g = []
    for o in order:
        tok = opd[o].split(); tok[1] = str(mp[o]); g.append(" ".join(tok))
        if o in qch:
            qt = qch[o].split(); qt[1] = str(mp[o]); g.append(" ".join(qt))
    for code, ins, outs in pars:
        g.append("OP %d %d %s %d %s" % (
            code, len(ins), " ".join(str(mp[v]) for v in ins if v in mp),
            len(outs), " ".join(str(mp[v]) for v in outs if v in mp)))
    produced = set()
    for code, ins, outs in pars:
        produced.update(outs)
    in_ids = [mp[v] for v in order if v not in produced and not is_scalar(v)]
    if not in_ids:
        in_ids = [mp[order[0]]]
    out_ids = [mp[v] for v in pars[-1][2] if v in mp]
    g.append("IO %d %s %d %s" % (len(in_ids), " ".join(map(str, in_ids)),
                                 len(out_ids), " ".join(map(str, out_ids))))
    g.append("END")
    p = "%s/n%d.txt" % (OUT, n)
    open(p, "w").write("\n".join(g) + "\n")
    return p


def query(path):
    pr = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        pr.stdout.readline()
        pr.stdin.write('{"cmd":"build","path":"%s"}\n' % path); pr.stdin.flush()
        r = pr.stdout.readline().strip()
        if '"ok":true' not in r:
            return None, "build " + r[:40]
        pr.stdin.write('{"cmd":"sup"}\n'); pr.stdin.flush()
        r = pr.stdout.readline().strip()
        if '"ok":true' not in r:
            return None, "sup " + r[:40]
        return json.loads(r), None
    except Exception as e:
        return None, "crash %s" % str(e)[:40]
    finally:
        try:
            pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush(); pr.wait(timeout=15)
        except Exception:
            try: pr.kill()
            except Exception: pass


print("总算子 %d，开始二分" % len(ops))
lo, hi, first_bad = 1, len(ops), None
while lo <= hi:
    mid = (lo + hi) // 2
    d, err = query(build_sub(mid))
    if d is None:
        print("  N=%-4d 失败: %s" % (mid, err)); hi = mid - 1; first_bad = mid; continue
    print("  N=%-4d total=%-4d 不支持=%d" % (mid, d["total"], d["unsupported"]))
    if d["unsupported"] > 0:
        first_bad = mid; hi = mid - 1
    else:
        lo = mid + 1

print("\n*** 首个出问题的 N = %s ***" % first_bad)
if first_bad:
    code, ins, outs = parse(ops[first_bad - 1])
    print("  算子 #%d code=%d ins=%s outs=%s" % (first_bad - 1, code, ins, outs))
    print("  %s" % ops[first_bad - 1])
    for v in ins + outs:
        if v in opd:
            print("    %s" % opd[v][:120])
