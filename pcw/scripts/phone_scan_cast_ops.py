"""扫描所有 opcode，找出哪些能做 FP16 -> INT16 和 INT16 -> FP16 的单入单出转换。"""
import subprocess, os

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
T = "/data/local/tmp/scanops"
os.makedirs(T, exist_ok=True)
SX = "3.0e-06"


def probe(g, tag):
    fp = "%s/g_%s.txt" % (T, tag)
    open(fp, "w").write("\n".join(g) + "\n")
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    r = s('{"cmd":"build","path":"%s"}' % fp)
    if '"ok":true' not in r:
        p.kill(); return False
    r = s('{"cmd":"compile"}')
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=30)
    except: p.kill()
    return '"ok":true' in r


f2q, q2f = [], []
for code in range(0, 131):
    A = ["OPD 0 7 2 1 8 %s 0 i" % SX, "OPD 1 8 2 1 8 0 0 o",
         "IO 1 0 1 1", "OP %d 1 0 1 1" % code, "END"]
    if probe(A, "q%d" % code):
        q2f.append(code)
    B = ["OPD 0 8 2 1 8 0 0 i", "OPD 1 7 2 1 8 %s 0 o" % SX,
         "IO 1 0 1 1", "OP %d 1 0 1 1" % code, "END"]
    if probe(B, "f%d" % code):
        f2q.append(code)

print("INT16 -> FP16 可用的 opcode: %s" % q2f)
print("FP16 -> INT16 可用的 opcode: %s" % f2q)
