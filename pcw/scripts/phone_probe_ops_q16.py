"""逐算子探针：把真实图里每种 opcode 单独抽出来，在 W8A16 形态下查支持性。"""
import os, sys, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = "/data/local/tmp/q16b00/g.txt"      # 已转换的 W8A16 图
OUT = "/data/local/tmp/probe1"
os.makedirs(OUT, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opds = {}
for ln in lines:
    if ln.startswith("OPD "):
        opds[int(ln.split()[1])] = ln
qch = {}
for ln in lines:
    if ln.startswith("QCH "):
        qch[int(ln.split()[1])] = ln
ops = [ln for ln in lines if ln.startswith("OP ")]


def parse_op(ln):
    tok = ln.split()
    code, ni = int(tok[1]), int(tok[2])
    ins = [int(v) for v in tok[3:3 + ni]]
    no = int(tok[3 + ni])
    outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
    return code, ins, outs


def kind_of(oid):
    ln = opds.get(oid)
    if not ln:
        return ""
    tok = ln.split()
    nd = int(tok[3])
    rest = tok[6 + nd:]
    return rest[0] if rest else ""


def probe(ln, code):
    _, ins, outs = parse_op(ln)
    need = ins + outs
    g = []
    for oid in sorted(set(need)):
        if oid in opds:
            g.append(opds[oid])
    for oid in sorted(set(need)):
        if oid in qch:
            g.append(qch[oid])
    g.append(ln)
    # IO：有 'i' 的当输入，op 输出当输出
    real_ins = [i for i in ins if kind_of(i) == "i"]
    if not real_ins:
        real_ins = [ins[0]]          # 兜底，避免 0 输入
    g.append("IO %d %s %d %s" % (len(real_ins), " ".join(map(str, real_ins)),
                                 len(outs), " ".join(map(str, outs))))
    g.append("END")
    p = "%s/op%d.txt" % (OUT, code)
    open(p, "w").write("\n".join(g) + "\n")
    return p, real_ins, outs


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
        return "BUILD_FAIL:" + r[:70]
    r = s('{"cmd":"sup"}')
    pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
    try: pr.wait(timeout=30)
    except: pr.kill()
    if '"ok":true' not in r:
        return "QUERY_FAIL"
    d = json.loads(r)
    return "%d/%d" % (d["total"] - d["unsupported"], d["total"])


seen = {}
for ln in ops:
    code, ins, outs = parse_op(ln)
    if code in seen:
        continue
    seen[code] = ln

print("图里共 %d 种算子\n" % len(seen))
print("%-6s %-10s %-8s %s" % ("opcode", "W8A16支持", "输入数", "输出operand类型"))
for code in sorted(seen):
    ln = seen[code]
    p, rins, outs = probe(ln, code)
    res = query(p)
    otypes = ",".join(str(opds[o].split()[2]) if o in opds else "?" for o in outs)
    print("%-6d %-10s %-8d %s" % (code, res, len(rins), otypes))
