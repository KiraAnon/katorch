"""测 NeuronModel 的 tensor（operand）数量上限。"""
import os, subprocess, time

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/opdlimit"
os.makedirs(T, exist_ok=True)

S, K = 128, 2048
nb = S * K * 2
f = T + "/a.bin"
if not os.path.exists(f) or os.path.getsize(f) != nb:
    with open(f, "wb") as fh:
        fh.write(b"\x00" * nb)


def make(n_ops):
    """n_ops 个 ADD 串联，产生 n_ops+1 个 operand（含输入常量）。"""
    L = ["OPD 0 8 2 %d %d 0 0 i" % (S, K),
         "OPD 1 8 2 %d %d 0 0 w %s" % (S, K, f),
         "OPD 2 4 1 1 0 0 a 0"]
    prev, oid = 0, 3
    for _ in range(n_ops):
        L.append("OPD %d 8 2 %d %d 0 0 o" % (oid, S, K))
        L.append("OP 0 3 %d 1 2 1 %d" % (prev, oid))
        prev = oid
        oid += 1
    L.append("IO 1 0 1 %d" % prev)
    L.append("END")
    return "\n".join(L) + "\n", oid


def test(n_ops):
    gp = "%s/g%d.txt" % (T, n_ops)
    txt, n_opd = make(n_ops)
    open(gp, "w").write(txt)
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % gp)
    ok = '"ok":true' in r
    p.kill()
    return ok, n_opd, r[:80]


print("%-10s %-10s %s" % ("算子数", "operand数", "结果"))
for n in (1000, 2000, 2045, 2050, 2100, 3000):
    ok, n_opd, msg = test(n)
    print("%-10d %-10d %s" % (n, n_opd, "OK" if ok else msg))
