"""最小验证：adapter 是否接受 QUANT8_SYMM_PER_CHANNEL (type 11) 的 operand。"""
import struct, subprocess, os

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/q8test"
os.makedirs(T, exist_ok=True)

# 权重：4x4 INT8，scale 每个输出通道一个（4 个）
w8 = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
open(T + "/w8.bin", "wb").write(w8)
scales = struct.pack("<4f", 0.01, 0.02, 0.03, 0.04)
open(T + "/w8sc.bin", "wb").write(scales)

# 图：一个 type-11 operand，带 QCH
g = []
g.append("OPD 0 11 2 4 4 0 0 w %s/w8.bin" % T)
g.append("QCH 0 0 4 %s/w8sc.bin" % T)
g.append("IO 1 0 1 0")
g.append("END")
open(T + "/g.txt", "w").write("\n".join(g) + "\n")
print("=== 测试图 ===")
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
        r = s('{"cmd":"export","path":"%s/out.dla"}' % T)
        print("export  : %s" % r)
        if os.path.exists(T + "/out.dla"):
            print("  blob %.0f KB" % (os.path.getsize(T + "/out.dla") / 1e3))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()
err = p.stderr.read()
if err.strip():
    print("\n=== host stderr ===")
    print(err[:800])
