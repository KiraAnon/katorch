"""修正版逐算子探针：输入只取原始 OPD 里 kind=='i' 的 operand。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = "/data/local/tmp/q16b00/g.txt"
OUT = "/data/local/tmp/probe4"
os.makedirs(OUT, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd, kind, qch = {}, {}, {}
for ln in lines:
    if ln.startswith("OPD "):
        tok = ln.split()
        oid = int(tok[1]); nd = int(tok[3])
        opd[oid] = ln
        rest = tok[6 + nd:]
        kind[oid] = rest[0] if rest else ""
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


def build_probe(code, ln):
    _, ins, outs = parse(ln)
    order, seen = [], set()
    for v in ins + outs:
        if v not in seen and v in opd:
            seen.add(v); order.append(v)
    mp = {old: i for i, old in enumerate(order)}
    g = []
    for old in order:
        tok = opd[old].split(); tok[1] = str(mp[old]); g.append(" ".join(tok))
        if old in qch:
            qt = qch[old].split(); qt[1] = str(mp[old]); g.append(" ".join(qt))
    ins2 = [mp[v] for v in ins if v in mp]
    outs2 = [mp[v] for v in outs if v in mp]
    g.append("OP %d %d %s %d %s" % (code, len(ins2), " ".join(map(str, ins2)),
                                    len(outs2), " ".join(map(str, outs2))))
    # 输入 = 原始 kind 为 'i' 的
    # 输入 = 用到但本算子不产生的 operand，排除 INT32 标量（type 1/4）
    def is_scalar(v):
        ln2 = opd.get(v, "")
        tok2 = ln2.split()
        return len(tok2) > 2 and int(tok2[2]) in (1, 4)
    in_ids = [mp[v] for v in ins if v in mp and mp[v] not in outs2 and not is_scalar(v)]
    if not in_ids:
        in_ids = [ins2[0]]
    g.append("IO %d %s %d %s" % (len(in_ids), " ".join(map(str, in_ids)),
                                 len(outs2), " ".join(map(str, outs2))))
    g.append("END")
    p = "%s/op%d.txt" % (OUT, code)
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
        try:
            pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush(); pr.wait(timeout=20)
        except Exception: pass
        return "BUILD_FAIL"
    r = s('{"cmd":"sup"}')
    try:
        pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush(); pr.wait(timeout=20)
    except Exception: pass
    if '"ok":true' not in r:
        return "QUERY_FAIL"
    d = json.loads(r)
    return "OK" if d["unsupported"] == 0 else "NO"


first = {}
for ln in ops:
    code, _, _ = parse(ln)
    first.setdefault(code, ln)

print("%-8s %-10s %s" % ("opcode", "结果", "原文"))
for code in sorted(first):
    p = build_probe(code, first[code])
    print("%-8d %-10s %s" % (code, query(p), first[code][:66]))
