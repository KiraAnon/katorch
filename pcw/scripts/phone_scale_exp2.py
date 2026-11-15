"""按真实图的 3 输入形式（含激活码）重测 ADD/MUL/SUB。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
D = "/data/local/tmp/scaleexp2"
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
    d = json.loads(r)
    return "OK" if d["unsupported"] == 0 else "NO"


CASES = [("ADD", 0, True), ("MUL", 18, True), ("SUB", 36, True),
         ("ADD", 0, False), ("MUL", 18, False), ("SUB", 36, False)]

S = "1e-4"
for name, code, with_act in CASES:
    opds = ["OPD 0 7 2 1 8 %s 0 i" % S,
            "OPD 1 7 2 1 8 %s 0 i" % S]
    if with_act:
        opds.append("OPD 2 1 1 1 0 0 a 0")
        opds.append("OPD 3 7 2 1 8 %s 0 o" % S)
        op = "OP %d 3 0 1 2 1 3" % code
        io = "IO 2 0 1 1 3"
    else:
        opds.append("OPD 2 7 2 1 8 %s 0 o" % S)
        op = "OP %d 2 0 1 1 2" % code
        io = "IO 2 0 1 1 2"
    g = opds + [op, io, "END"]
    p = "%s/%s_%s.txt" % (D, name, "3in" if with_act else "2in")
    open(p, "w").write("\n".join(g) + "\n")
    print("%-6s %-6s -> %s" % (name, "3输入" if with_act else "2输入", run(p)))
