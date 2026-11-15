"""逐 opcode 探针（带 operand 重编号，保证 id 从 0 连续）。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = "/data/local/tmp/q16b00/g.txt"
OUT = "/data/local/tmp/probe3"
os.makedirs(OUT, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd = {}
for ln in lines:
    if ln.startswith("OPD "):
        opd[int(ln.split()[1])] = ln
qch = {}
for ln in lines:
    if ln.startswith("QCH "):
        qch[int(ln.split()[1])] = ln
ops = [ln for ln in lines if ln.startswith("OP ")]


def parse(ln):
    tok = ln.split()
    code, ni = int(tok[1]), int(tok[2])
    ins = [int(v) for v in tok[3:3 + ni]]
    no = int(tok[3 + ni])
    outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
    return code, ins, outs


def renum(ln, mp):
    tok = ln.split()
    return " ".join(tok)


def build_probe(code, ln):
    _, ins, outs = parse(ln)
    order, seen = [], set()
    for v in ins + outs:
        if v not in seen and v in opd:
            seen.add(v); order.append(v)
    if len(order) != len(set(ins + outs)):
        return None
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
    # 输入 = 不在产出里的
    produced = set(outs2)
    in_ids = [i for i in ins2 if i not in produced]
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
        pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
        try: pr.wait(timeout=30)
        except: pr.kill()
        return "BUILD_FAIL"
    r = s('{"cmd":"sup"}')
    pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
    try: pr.wait(timeout=30)
    except: pr.kill()
    if '"ok":true' not in r:
        return "QUERY_FAIL"
    d = json.loads(r)
    return "OK" if d["unsupported"] == 0 else "NO(%d)" % d["unsupported"]


first = {}
for ln in ops:
    code, _, _ = parse(ln)
    if code not in first:
        first[code] = ln

print("%-8s %-6s %s" % ("opcode", "结果", "原文"))
for code in sorted(first):
    ln = first[code]
    p = build_probe(code, ln)
    if p is None:
        print("%-8d SKIP   operand 缺失" % code); continue
    print("%-8d %-6s %s" % (code, query(p), ln[:70]))
