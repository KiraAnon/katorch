"""测 MUL/ADD 在两个输入尺度不同时是否还支持。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
D = "/data/local/tmp/mixexp"
os.makedirs(D, exist_ok=True)
P = "0.000030517578125"      # 1/32768
A = "0.0001"


def run(path):
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
    d = json.loads(r)
    return "OK" if d["unsupported"] == 0 else "NO"


cases = [
    ("MUL", 18, P, A, A),
    ("MUL", 18, A, P, A),
    ("MUL", 18, A, A, P),
    ("MUL", 18, P, P, A),
    ("MUL", 18, P, P, P),
    ("ADD", 0, P, A, A),
    ("ADD", 0, A, A, A),
    ("ADD", 0, P, P, P),
]
for name, code, s1, s2, so in cases:
    g = ["OPD 0 7 3 1 1 2048 %s 0 i" % s1,
         "OPD 1 7 3 1 1 2048 %s 0 i" % s2,
         "OPD 2 1 1 1 0 0 a 0",
         "OPD 3 7 3 1 1 2048 %s 0 o" % so,
         "OP %d 3 0 1 2 1 3" % code,
         "IO 2 0 1 1 3", "END"]
    p = "%s/%s_%s_%s_%s.txt" % (D, name, s1[-6:], s2[-6:], so[-6:])
    open(p, "w").write("\n".join(g) + "\n")
    print("%-4s in1=%-14s in2=%-14s out=%-14s -> %s" % (name, s1, s2, so, run(p)))
