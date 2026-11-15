"""测 512 分辨率下最多能声明多少个输出。"""
import os, subprocess

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new_d"
SRC = D + "/e2eper256/b00/g.txt"
GD = "/data/local/tmp/chunktest"
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

print("%-8s %s" % ("输出数", "编译结果"))
for n in (4, 8, 16, 32, 64):
    ids = ids_all[:n]
    g = base + ["IO %d %s %d %s" % (len(ins), " ".join(map(str, ins)),
                                    len(ids), " ".join(map(str, ids))), "END"]
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
        print("%-8d build失败 %s" % (n, r[:40]))
    else:
        r = s('{"cmd":"compile"}')
        print("%-8d %s" % (n, "OK" if '"ok":true' in r else r[:40]))
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()
