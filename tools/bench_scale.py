"""用现成的 scaletest 图验证：耗时由算子数还是数据量决定。"""
import os, subprocess, time

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"

CAD = ["/data/local/tmp/scaletest/s256/g.txt",
       "/data/local/tmp/scaletest/s1024/g.txt",
       "/data/local/tmp/scaletest/s4096/g.txt"]


def inspect(gp):
    nops = 0
    io = None
    dims = []
    for ln in open(gp, encoding="utf-8", errors="replace"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "OP":
            nops += 1
        elif p[0] == "IO":
            io = p
        elif p[0] == "OPD":
            nd = int(p[3])
            dims.append([int(x) for x in p[4:4 + nd]])
    return nops, io, dims


def run(gp, reps=5):
    nops, io, dims = inspect(gp)
    # IO 格式: IO <n_in> <ids> <n_out> <ids>
    ni = int(io[1])
    ins = [int(x) for x in io[2:2 + ni]]
    no = int(io[2 + ni])
    outs = [int(x) for x in io[3 + ni:3 + ni + no]]

    T = "/data/local/tmp/szrun"
    os.makedirs(T, exist_ok=True)
    # 造输入文件（全零 fp16）
    files = []
    total_mb = 0
    for k, oid in enumerate(ins):
        d = dims[oid]
        n = 1
        for x in d:
            n *= x
        f = "%s/in%d.bin" % (T, k)
        if not os.path.exists(f) or os.path.getsize(f) != n * 2:
            with open(f, "wb") as fh:
                fh.write(b"\x00" * (n * 2))
        files.append(f)
        total_mb += n * 2 / 1e6

    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdout.readline()

    def s(c):
        p.stdin.write(c + "\n"); p.stdin.flush()
        return p.stdout.readline().strip()

    t_b = time.time()
    r = s('{"cmd":"build","path":"%s"}' % gp)
    build_ms = (time.time() - t_b) * 1000
    if '"ok":true' not in r:
        p.kill()
        return dict(ops=nops, err="build " + r[:50])
    t_c = time.time()
    r = s('{"cmd":"compile"}')
    comp_ms = (time.time() - t_c) * 1000
    if '"ok":true' not in r:
        p.kill()
        return dict(ops=nops, err="compile rc=" + r[:50])

    out = "%s/o.bin" % T
    spec = "%d %s 1 %s" % (len(files), " ".join("%d %s" % (i, f) for i, f in enumerate(files)), out)
    s('{"cmd":"run","spec":"%s"}' % spec)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        r = s('{"cmd":"run","spec":"%s"}' % spec)
        ts.append((time.time() - t0) * 1000)
        if '"ok":true' not in r:
            p.kill()
            return dict(ops=nops, err="run")
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=30)
    except Exception: p.kill()
    ts.sort()
    return dict(ops=nops, ms=ts[len(ts) // 2], build=build_ms, compile=comp_ms,
                mb=total_mb)


print("%-32s %-8s %-10s %-10s %s" % ("图", "算子数", "输入MB", "运行ms", "每算子ms"))
for gp in CAD:
    if not os.path.isfile(gp):
        print("%-32s 不存在" % os.path.basename(os.path.dirname(gp)))
        continue
    r = run(gp)
    name = os.path.basename(os.path.dirname(gp))
    if r.get("err"):
        print("%-32s %-8d 失败: %s" % (name, r["ops"], r["err"]))
    else:
        print("%-32s %-8d %-10.2f %-10.1f %.2f" %
              (name, r["ops"], r["mb"], r["ms"], r["ms"] / r["ops"]))
