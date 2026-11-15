"""测 ADD/MUL 在不同 scale 组合下能否被支持，找出 MDLA 的尺度约束。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
D = "/data/local/tmp/scaleexp"
os.makedirs(D, exist_ok=True)


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
    if '"ok":true' not in r:
        return "QUERY_FAIL"
    d = json.loads(r)
    return "OK" if d["unsupported"] == 0 else "NO"


combos = [
    ("1e-4", "1e-4"), ("1e-4", "1e-3"), ("1e-3", "1e-4"),
    ("1e-3", "1e-3"), ("1e-5", "1e-5"), ("1e-2", "1e-2"),
    ("3.0517578125e-05", "3.0517578125e-05"),
    ("1e-4", "3.0517578125e-05"), ("3.0517578125e-05", "1e-4"),
]

for opname, code, nin in (("ADD", 0, 2), ("MUL", 18, 2), ("SUB", 36, 2)):
    print("\n=== %s (opcode %d) ===" % (opname, code))
    for sin_, sout in combos:
        opds = ["OPD 0 7 2 1 8 %s 0 i" % sin_,
                "OPD 1 7 2 1 8 %s 0 i" % sin_,
                "OPD 2 7 2 1 8 %s 0 o" % sout]
        g = opds + ["OP %d 2 0 1 1 2" % code,
                    "IO 2 0 1 1 2", "END"]
        p = "%s/%s_%s_%s.txt" % (D, opname, sin_.replace(".", "p"), sout.replace(".", "p"))
        open(p, "w").write("\n".join(g) + "\n")
        print("  in=%-16s out=%-16s -> %s" % (sin_, sout, run(p)))
