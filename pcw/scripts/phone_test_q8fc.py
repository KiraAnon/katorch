"""验证：FC 用 INT8 per-channel 权重 + FP16 激活，能否编译通过。
   如果能，就只需量化权重，不必把整条激活链改成 INT16。"""
import struct, subprocess, os

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
T = "/data/local/tmp/q8fc"
os.makedirs(T, exist_ok=True)

# 权重 3x4 INT8 + 3 个通道 scale
W = [[10,20,-30,40],[-5,15,-25,35],[7,-17,27,-37]]
open(T + "/w8.bin", "wb").write(struct.pack("<12b", *[v for row in W for v in row]))
open(T + "/w8sc.bin", "wb").write(struct.pack("<3f", 0.01, 0.02, 0.03))
open(T + "/w16.bin", "wb").write(struct.pack("<12e", *[float(v) for row in W for v in row]))
open(T + "/bias.bin", "wb").write(struct.pack("<3f", 0.1, -0.2, 0.3))
# 输入 fp16 (1,4)
x = [0.5, -1.0, 2.0, 0.25]
open(T + "/in.bin", "wb").write(struct.pack("<4e", *x))

def mk(name, weight_type, with_qch):
    g = []
    g.append("OPD 0 8 2 1 4 0 0 i")
    wf = "w8.bin" if weight_type == 11 else "w16.bin"
    g.append("OPD 1 %d 2 3 4 0 0 w %s/%s" % (weight_type, T, wf))
    if with_qch:
        g.append("QCH 1 0 3 %s/w8sc.bin" % T)
    g.append("OPD 2 3 1 3 0 0 w %s/bias.bin" % T)
    g.append("OPD 3 8 2 1 3 0 0 i")
    g.append("IO 1 0 1 3")
    g.append("OP 9 3 0 1 2 1 3")
    g.append("END")
    p = "%s/%s.txt" % (T, name)
    open(p, "w").write("\n".join(g) + "\n")
    return p, g

pr = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, text=True, bufsize=1)
pr.stdout.readline()
def s(c):
    pr.stdin.write(c + "\n"); pr.stdin.flush()
    return pr.stdout.readline().strip()

for label, wtype, qch in (("INT8 per-channel", 11, True),
                          ("INT8 不带 QCH(对照)", 11, False),
                          ("FP16 (基线)", 8, False)):
    path, g = mk("g_%d_%d" % (wtype, 1 if qch else 0), wtype, qch)
    print("=== %s ===" % label)
    r = s('{"cmd":"build","path":"%s"}' % path)
    print("  build  : %s" % r)
    if '"ok":true' in r:
        r = s('{"cmd":"compile"}')
        print("  compile: %s" % r)
        if '"ok":true' in r:
            out = T + "/o_%d.bin" % wtype
            r = s('{"cmd":"run","spec":"1 0 %s/in.bin 1 0 %s"}' % (T, out))
            print("  run    : %s" % r)
            if '"ok":true' in r and os.path.exists(out):
                v = struct.unpack("<3e", open(out, "rb").read())
                print("  输出    : %s" % [round(float(a), 4) for a in v])
                # 手算参考：q8 反量化
                W = [[10,20,-30,40],[-5,15,-25,35],[7,-17,27,-37]]
                sc = [0.01,0.02,0.03]
                ref = [sum(W[o][i]*sc[o]*x[i] for i in range(4)) for o in range(3)]
                print("  期望(参考): %s" % [round(v,4) for v in ref])
    # 清理 slot
    print()

pr.stdin.write('{"cmd":"quit"}\n'); pr.stdin.flush()
try: pr.wait(timeout=60)
except: pr.kill()
