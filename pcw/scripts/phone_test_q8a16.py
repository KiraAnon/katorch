"""验证完整 W8A16 的 FC：输入/输出 QUANT16_SYMM(type 7)，权重 QUANT8_SYMM_PER_CHANNEL(type 11)。"""
import struct, subprocess, os

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/q8a16"
os.makedirs(T, exist_ok=True)

W = [[10, 20, -30, 40], [-5, 15, -25, 35], [7, -17, 27, -37]]
open(T + "/w8.bin", "wb").write(struct.pack("<12b", *[v for row in W for v in row]))
open(T + "/w8sc.bin", "wb").write(struct.pack("<3f", 0.01, 0.02, 0.03))
BIAS = [0.1, -0.2, 0.3]

SX = 0.0005        # 输入激活 scale
X = [0.5, -1.0, 2.0, 0.25]
qx = [max(-32767, min(32767, round(v / SX))) for v in X]        # INT16 码
open(T + "/in.bin", "wb").write(struct.pack("<4h", *qx))

# bias 的 INT32 码：b / (SX * sw)
q_bias = [round(BIAS[o] / (SX * [0.01, 0.02, 0.03][o])) for o in range(3)]
open(T + "/bias.bin", "wb").write(struct.pack("<3i", *q_bias))

SY = 0.005         # 输出激活 scale
g = [
    "OPD 0 7 2 1 4 %.8f 0 i" % SX,
    "OPD 1 11 2 3 4 0 0 w %s/w8.bin" % T,
    "QCH 1 0 3 %s/w8sc.bin" % T,
    "OPD 2 4 1 3 0 0 w %s/bias.bin" % T,
    "OPD 3 7 2 1 3 %.8f 0 i" % SY,
    "IO 1 0 1 3",
    "OP 9 3 0 1 2 1 3",
    "END",
]
open(T + "/g.txt", "w").write("\n".join(g) + "\n")
print("=== 图 ===")
print("\n".join(g))

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

r = s('{"cmd":"build","path":"%s/g.txt"}' % T)
print("\nbuild   : %s" % r)
if '"ok":true' in r:
    r = s('{"cmd":"compile"}')
    print("compile : %s" % r)
    if '"ok":true' in r:
        out = T + "/o.bin"
        r = s('{"cmd":"run","spec":"1 0 %s/in.bin 1 0 %s"}' % (T, out))
        print("run     : %s" % r)
        if '"ok":true' in r and os.path.exists(out):
            qy = struct.unpack("<3h", open(out, "rb").read())
            y = [v * SY for v in qy]
            sw = [0.01, 0.02, 0.03]
            ref = [sum(W[o][i] * sw[o] * X[i] for i in range(4)) + BIAS[o] for o in range(3)]
            print("  设备输出: %s" % [round(v, 5) for v in y])
            print("  数学期望: %s" % [round(v, 5) for v in ref])
            err = max(abs(a-b) for a, b in zip(y, ref))
            print("  最大误差: %.5f" % err)
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()
import time as _t; _t.sleep(0.2); e = p.stderr.read()
if e.strip():
    print("\n=== stderr ===")
    print("\n".join([l for l in e.splitlines() if "MAP" in l or "RST" in l])[:400])
