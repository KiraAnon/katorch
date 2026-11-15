"""逐个测试算子在 INT16 形态下是否被支持。手工构造最小合法图，id 从 0 连续。"""
import os, json, subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
D = "/data/local/tmp/opmin"
os.makedirs(D, exist_ok=True)
S = "0.0001"

CASES = {
    # 名字: (operand 定义列表, 算子行, 输入 id 列表, 输出 id 列表)
    "LOGISTIC_14": (
        ["OPD 0 7 2 1 8 %s 0 i" % S,
         "OPD 1 7 2 1 8 0.000030517578125 0 o"],
        "OP 14 1 0 1 1", [0], [1]),
    "ADD_0": (
        ["OPD 0 7 2 1 8 %s 0 i" % S,
         "OPD 1 7 2 1 8 %s 0 i" % S,
         "OPD 2 7 2 1 8 %s 0 o" % S],
        "OP 0 2 0 1 1 2", [0, 1], [2]),
    "MUL_18": (
        ["OPD 0 7 2 1 8 %s 0 i" % S,
         "OPD 1 7 2 1 8 %s 0 i" % S,
         "OPD 2 7 2 1 8 %s 0 o" % S],
        "OP 18 2 0 1 1 2", [0, 1], [2]),
    "RESHAPE_22": (
        ["OPD 0 7 2 1 8 %s 0 i" % S,
         "OPD 1 4 1 2 0 0 w %s/shp.bin" % D,
         "OPD 2 7 2 2 4 %s 0 o" % S],
        "OP 22 2 0 1 1 2", [0], [2]),
    "SUB_36": (
        ["OPD 0 7 2 1 8 %s 0 i" % S,
         "OPD 1 7 2 1 8 %s 0 i" % S,
         "OPD 2 7 2 1 8 %s 0 o" % S],
        "OP 36 2 0 1 1 2", [0, 1], [2]),
    "TANH_28": (
        ["OPD 0 7 2 1 8 %s 0 i" % S,
         "OPD 1 7 2 1 8 %s 0 o" % S],
        "OP 28 1 0 1 1", [0], [1]),
}

import struct
open(os.path.join(D, "shp.bin"), "wb").write(struct.pack("<2i", 2, 4))

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
        return "BUILD_FAIL " + r[:50]
    r = s('{"cmd":"sup"}')
    pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
    try: pr.wait(timeout=30)
    except: pr.kill()
    if '"ok":true' not in r:
        return "QUERY_FAIL " + r[:50]
    d = json.loads(r)
    return "%d/%d 支持" % (d["total"] - d["unsupported"], d["total"])

print("%-14s %s" % ("算子", "INT16 形态"))
for name, (opds, op, ins, outs) in CASES.items():
    g = opds + [op, "IO %d %s %d %s" % (len(ins), " ".join(map(str, ins)),
                                        len(outs), " ".join(map(str, outs))), "END"]
    p = os.path.join(D, name + ".txt")
    open(p, "w").write("\n".join(g) + "\n")
    print("%-14s %s" % (name, run(p)))
