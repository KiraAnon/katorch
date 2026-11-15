"""从转换后的大图里抽出前 N 个算子组成子图，二分定位是哪个算子让整图失效。"""
import os, sys, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = "/data/local/tmp/q16b00/g.txt"
OUT = "/data/local/tmp/q16sub"
os.makedirs(OUT, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opds = {}
for ln in lines:
    if ln.startswith("OPD "):
        opds[int(ln.split()[1])] = ln
ops = [ln for ln in lines if ln.startswith("OP ")]
qch = [ln for ln in lines if ln.startswith("QCH")]


def parse_op(ln):
    tok = ln.split()
    code, ni = int(tok[1]), int(tok[2])
    ins = [int(v) for v in tok[3:3 + ni]]
    no = int(tok[3 + ni])
    outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
    return code, ins, outs


def build_sub(n_ops):
    """保留前 n_ops 个算子；输入取这些算子用到但未被前序产生的 operand。"""
    keep_ops = ops[:n_ops]
    produced = set()
    used = set()
    for ln in keep_ops:
        code, ins, outs = parse_op(ln)
        for i in ins:
            if i not in produced:
                used.add(i)
        produced.update(outs)
    need = set(used) | produced
    g = []
    for oid in sorted(need):
        if oid in opds:
            g.append(opds[oid])
    for q in qch:
        qid = int(q.split()[1])
        if qid in need:
            g.append(q)
    g.extend(keep_ops)
    ins_list = [i for i in sorted(used)]
    outs_list = []
    code, ins, outs = parse_op(keep_ops[-1])
    outs_list = outs
    g.append("IO %d %s %d %s" % (len(ins_list), " ".join(map(str, ins_list)),
                                 len(outs_list), " ".join(map(str, outs_list))))
    g.append("END")
    p = "%s/s%d.txt" % (OUT, n_ops)
    open(p, "w").write("\n".join(g) + "\n")
    return p


def query(path):
    pr = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
    pr.stdout.readline()
    def s(c):
        pr.stdin.write(c + "\n"); pr.stdin.flush()
        return pr.stdout.readline().strip()
    r = s('{"cmd":"build","path":"%s"}' % path)
    if '"ok":true' not in r:
        pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
        try: pr.wait(timeout=30)
        except: pr.kill()
        return "build:" + r[:40]
    r = s('{"cmd":"sup"}')
    pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
    try: pr.wait(timeout=30)
    except: pr.kill()
    return r


print("总算子数 %d，开始二分" % len(ops))
lo, hi = 1, len(ops)
first_bad = None
while lo <= hi:
    mid = (lo + hi) // 2
    p = build_sub(mid)
    r = query(p)
    if '"ok":true' not in r:
        print("  N=%-4d  query失败 %s" % (mid, r[:60]))
        first_bad = mid
        hi = mid - 1
        continue
    try:
        import json as _j
        d = _j.loads(r)
    except Exception:
        print("  N=%-4d  解析失败 %s" % (mid, r[:60])); hi = mid - 1; continue
    unsup = d.get("unsupported", -1)
    print("  N=%-4d  total=%d unsupported=%d" % (mid, d.get("total", -1), unsup))
    if unsup > 0:
        first_bad = mid
        hi = mid - 1
    else:
        lo = mid + 1

print("\n首个出问题的算子数: %s" % first_bad)
if first_bad:
    code, ins, outs = parse_op(ops[first_bad - 1])
    print("  算子 idx=%d  code=%d  ins=%s outs=%s" % (first_bad - 1, code, ins, outs))
    print("  原文: %s" % ops[first_bad - 1])
    for i in ins:
        if i in opds:
            print("    in  %s" % opds[i][:120])
    for o in outs:
        if o in opds:
            print("    out %s" % opds[o][:120])
