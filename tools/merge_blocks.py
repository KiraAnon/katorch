"""把两块 DiT 图拼成一张，测合并后是否更快（验证 MDLA 会不会做跨算子融合）。

b00 和 b01 结构相同：IO 8 <in 0..7> 1 <out 1028>
拼接规则：
  - b00 原样保留
  - b01 的 0..7 号输入重定向到 b00 的 [1028, 1..7]
  - b01 的 8..1028 号 operand 偏移 +1029
  - 输出 = b01 的 1028 + 1029 = 2057
"""
import os, subprocess, time
import numpy as np

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
SRC = H + "/npt/e2eper512"
T = "/data/local/tmp/merge_test"
os.makedirs(T, exist_ok=True)


def parse(gp):
    opd, ops, io = [], [], None
    for ln in open(gp, encoding="utf-8", errors="replace"):
        p = ln.rstrip("\n").split()
        if not p:
            continue
        if p[0] == "OPD":
            opd.append((int(p[1]), ln.rstrip("\n")))
        elif p[0] == "OP":
            ops.append([int(x) for x in p[1:]])
        elif p[0] == "IO":
            io = [int(x) for x in p[1:]]
    return opd, ops, io


def ser_opd(oid, nd, ty, dims, scale, zp, tail):
    s = "OPD %d %d %d %s %s %s" % (oid, ty, nd, " ".join(str(d) for d in dims), scale, zp)
    if tail:
        s += " " + tail
    return s


def merge(ga, gb):
    opa, opsa, ioa = parse(ga)
    opb, opsb, iob = parse(gb)

    # b01 的 0..7 号输入 → b00 的 [1028, 1, 2, 3, 4, 5, 6, 7]
    remap = {0: 1028, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    OFF = 1029

    lines = []
    # b00 的 OPD 原样
    for oid, txt in opa:
        lines.append(txt)

    # b01 的 OPD：跳过 0..7，其余 +OFF
    for oid, txt in opb:
        if oid < 8:
            continue
        p = txt.split()
        nd = int(p[3])
        dims = p[4:4 + nd]
        scale = p[4 + nd]
        zp = p[5 + nd]
        tail = " ".join(p[6 + nd:])
        lines.append(ser_opd(oid + OFF, nd, int(p[2]), dims, scale, zp, tail))

    # b00 的 OP 原样
    for o in opsa:
        code, ni = o[0], o[1]
        ins = o[2:2 + ni]
        no = o[2 + ni]
        outs = o[3 + ni:3 + ni + no]
        lines.append("OP %d %d %s %d %s" % (code, ni, " ".join(map(str, ins)), no,
                                            " ".join(map(str, outs))))

    # b01 的 OP：输入 0..7 重映射，其余 +OFF
    for o in opsb:
        code, ni = o[0], o[1]
        ins = o[2:2 + ni]
        no = o[2 + ni]
        outs = o[3 + ni:3 + ni + no]
        ni2 = [remap.get(i, i + OFF) if i < 8 else i + OFF for i in ins]
        no2 = [remap.get(i, i + OFF) if i < 8 else i + OFF for i in outs]
        lines.append("OP %d %d %s %d %s" % (code, ni, " ".join(map(str, ni2)), no,
                                            " ".join(map(str, no2))))

    lines.append("IO 8 0 1 2 3 4 5 6 7 1 %d" % (1028 + OFF))
    lines.append("END")
    return "\n".join(lines) + "\n"


merged = os.path.join(T, "g.txt")
open(merged, "w").write(merge(os.path.join(SRC, "b00", "g.txt"),
                              os.path.join(SRC, "b01", "g.txt")))
print("拼接完成，行数:", sum(1 for _ in open(merged)))
print("算子数:", sum(1 for ln in open(merged) if ln.startswith("OP ")))
print()

# 准备输入：8 个文件，和真实一致
nb0 = 1024 * 2048 * 2
files = []
for k, (S, K) in enumerate([(1024, 2048), (1, 2048), (1, 2048), (1, 2048),
                            (1, 2048), (512, 1024), (1024, 128), (1024, 128)]):
    f = "%s/in%d.bin" % (T, k)
    n = S * K * 2
    if not os.path.exists(f) or os.path.getsize(f) != n:
        with open(f, "wb") as fh:
            fh.write(b"\x00" * n)
    files.append(f)


def run(gp, out, tag):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    t0 = time.time()
    r = s('{"cmd":"build","path":"%s"}' % gp)
    bt = time.time() - t0
    if '"ok":true' not in r:
        p.kill(); return None, "build " + r[:60]
    t0 = time.time()
    r = s('{"cmd":"compile"}')
    ct = time.time() - t0
    if '"ok":true' not in r:
        p.kill(); return None, "compile " + r[:60]
    spec = "8 %s 1 %s" % (" ".join("%d %s" % (i, f) for i, f in enumerate(files)), out)
    s('{"cmd":"run","spec":"%s"}' % spec)
    ts = []
    for _ in range(3):
        t0 = time.time()
        r = s('{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - t0) * 1000)
        if '"ok":true' not in r:
            p.kill(); return None, "run " + r[:60]
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except Exception: p.kill()
    ts.sort()
    return dict(ms=ts[1], build=bt, compile=ct), None


print("=== 合并图（2 层）===")
r1, e1 = run(merged, "%s/o_merged.bin" % T, "merged")
if e1:
    print("  失败:", e1)
else:
    print("  编译 %.1fs  运行 %.0f ms" % (r1["compile"], r1["ms"]))

print()
print("=== 单块（对照，1 层）===")
r2, e2 = run(os.path.join(SRC, "b00", "g.txt"), "%s/o_single.bin" % T, "single")
if e2:
    print("  失败:", e2)
else:
    print("  编译 %.1fs  运行 %.0f ms" % (r2["compile"], r2["ms"]))

if r1 and r2:
    print()
    print("=== 对比 ===")
    print("  单块 ×2 预估: %.0f ms" % (r2["ms"] * 2))
    print("  合并实测:     %.0f ms" % r1["ms"])
    gain = r2["ms"] * 2 - r1["ms"]
    print("  省下:         %.0f ms (%.1f%%)" % (gain, gain / (r2["ms"] * 2) * 100))
