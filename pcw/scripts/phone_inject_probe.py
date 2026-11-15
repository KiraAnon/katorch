"""验证「插恒等 MUL 把中间张量变成输出」是否可行。"""
import os, subprocess

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
HOST = D + "/host_new_d"
SRC = D + "/e2eper512/b00/g.txt"
GD = "/data/local/tmp/injtest"
os.makedirs(GD, exist_ok=True)

lines = [l.rstrip("\n") for l in open(SRC)]
opd = {}
for ln in lines:
    if ln.startswith("OPD "):
        opd[int(ln.split()[1])] = ln
tok = [l for l in lines if l.startswith("IO ")][0].split()
n_in = int(tok[1]); ins = [int(v) for v in tok[2:2 + n_in]]
ids_all = sorted(o for o, l in opd.items() if int(l.split()[2]) == 8 and o not in ins)

base = [l for l in lines if l.startswith("OPD ") or l.startswith("OP ")]
maxid = max(int(l.split()[1]) for l in lines if l.startswith("OPD "))

# 常 1.0 的 fp16 标量张量（用于恒等 MUL）
ONE = GD + "/one.bin"
open(ONE, "wb").write(b"\x00\x3c")   # fp16 1.0

for n in (2, 4, 8):
    ids = ids_all[:n]
    g = list(base)
    newids = []
    for k, oid in enumerate(ids):
        ln = opd[oid].split()
        nd = int(ln[3]); dims = ln[4:4 + nd]
        tid = maxid + 1 + k * 2
        g.append("OPD %d 8 0 0 0 w %s" % (tid, ONE))                    # 常数 1
        g.append("OPD %d 8 %d %s 0 0 o" % (tid + 1, nd, " ".join(dims)))  # 输出
        g.append("OP 18 3 %d %d 0 1 %d" % (oid, tid, tid + 1))          # MUL(x, 1)
        newids.append(tid + 1)
    g.append("IO %d %s %d %s" % (len(ins), " ".join(map(str, ins)),
                                 len(newids), " ".join(map(str, newids))))
    g.append("END")
    gp = "%s/g%d.txt" % (GD, n)
    open(gp, "w").write("\n".join(g) + "\n")

    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()
    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()
    r = s('{"cmd":"build","path":"%s"}' % gp)
    if '"ok":true' not in r:
        print("%-4d build失败 %s" % (n, r[:50]))
    else:
        r = s('{"cmd":"compile"}')
        print("%-4d %s" % (n, "编译 OK" if '"ok":true' in r else "编译失败 " + r[:45]))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
