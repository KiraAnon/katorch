"""正确的子图二分：把前 N 个算子抽出来并重新编号 operand（host 要求 id 连续）。"""
import os, sys, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = sys.argv[1] if len(sys.argv) > 1 else "/data/local/tmp/q16b00/g.txt"
OUT = "/data/local/tmp/bisect2"
os.makedirs(OUT, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd_raw = {}
for ln in lines:
    if ln.startswith("OPD "):
        opd_raw[int(ln.split()[1])] = ln
qch_raw = {}
for ln in lines:
    if ln.startswith("QCH "):
        qch_raw[int(ln.split()[1])] = ln
ops_raw = [ln for ln in lines if ln.startswith("OP ")]


def parse_op(ln):
    tok = ln.split()
    code, ni = int(tok[1]), int(tok[2])
    ins = [int(v) for v in tok[3:3 + ni]]
    no = int(tok[3 + ni])
    outs = [int(v) for v in tok[4 + ni:4 + ni + no]]
    return code, ins, outs


def rebuild_opd(old_id, new_id):
    ln = opd_raw[old_id]
    tok = ln.split()
    tok[1] = str(new_id)
    return " ".join(tok)


def build_sub(n_ops):
    keep = ops_raw[:n_ops]
    pars = [parse_op(l) for l in keep]
    order = []
    seen = set()
    for code, ins, outs in pars:
        for v in ins + outs:
            if v not in seen:
                seen.add(v); order.append(v)
    newid = {old: i for i, old in enumerate(order)}
    # 必须所有 operand 都有定义
    for v in order:
        if v not in opd_raw:
            return None, "operand %d 无定义" % v
    g = [rebuild_opd(v, newid[v]) for v in order]
    for v in order:
        if v in qch_raw:
            tok = qch_raw[v].split()
            tok[1] = str(newid[v])
            g.append(" ".join(tok))
    for (code, ins, outs) in pars:
        g.append("OP %d %d %s %d %s" % (code, len(ins),
                                        " ".join(str(newid[v]) for v in ins),
                                        len(outs),
                                        " ".join(str(newid[v]) for v in outs)))
    # 输入 = 未被任何 op 产生的 operand
    produced = set()
    for code, ins, outs in pars:
        produced.update(outs)
    in_ids = [newid[v] for v in order if v not in produced]
    out_ids = [newid[v] for v in pars[-1][2]]
    if not in_ids:
        in_ids = [newid[pars[0][1][0]]]
    g.append("IO %d %s %d %s" % (len(in_ids), " ".join(map(str, in_ids)),
                                 len(out_ids), " ".join(map(str, out_ids))))
    g.append("END")
    p = "%s/n%d.txt" % (OUT, n_ops)
    open(p, "w").write("\n".join(g) + "\n")
    return p, None


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
        return None, r[:70]
    r = s('{"cmd":"sup"}')
    pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
    try: pr.wait(timeout=30)
    except: pr.kill()
    if '"ok":true' not in r:
        return None, r[:70]
    return json.loads(r), None


print("总算子 %d，开始二分定位" % len(ops_raw))
lo, hi, first_bad = 1, len(ops_raw), None
while lo <= hi:
    mid = (lo + hi) // 2
    p, err = build_sub(mid)
    if p is None:
        print("  N=%-4d SKIP %s" % (mid, err)); lo = mid + 1; continue
    d, err = query(p)
    if d is None:
        print("  N=%-4d build失败 %s" % (mid, err)); lo = mid + 1; continue
    uns = d["unsupported"]
    print("  N=%-4d total=%-4d 不支持=%d" % (mid, d["total"], uns))
    if uns > 0:
        first_bad = mid; hi = mid - 1
    else:
        lo = mid + 1

print("\n*** 首个带不支持算子的 N = %s ***" % first_bad)
if first_bad:
    code, ins, outs = parse_op(ops_raw[first_bad - 1])
    print("  算子 #%d: code=%d ins=%s outs=%s" % (first_bad - 1, code, ins, outs))
    print("  原文: %s" % ops_raw[first_bad - 1])
    for v in ins + outs:
        if v in opd_raw:
            print("    %s" % opd_raw[v][:130])
    d, _ = query("%s/n%d.txt" % (OUT, first_bad))
    if d:
        print("  不支持清单: %s" % d.get("first"))
