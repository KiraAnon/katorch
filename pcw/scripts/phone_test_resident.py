"""测试常驻模式：一个进程 restore 全部 16 块，然后反复跑完整链条。"""
import os, subprocess, time, resource

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = D + "/gen5"
HOST = D + "/host_new"
BLOBDIR = "/data/local/tmp/blobs"

def mem_mb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 1024
    return -1

NAMES = ["pe"] + ["b%02d" % i for i in range(14)] + ["fl"]

print("=== 启动一个 host 进程 ===")
p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    a = time.time()
    p.stdin.write(c + "\n"); p.stdin.flush()
    l = p.stdout.readline().strip()
    return l, (time.time() - a) * 1000

print("=== restore 全部 %d 块 ===" % len(NAMES))
t0 = time.time()
for i, nm in enumerate(NAMES):
    blob = "%s/%s.dla" % (BLOBDIR, nm)
    r, ms = s('{"cmd":"rpush","path":"%s"}' % blob)
    ok = '"ok":true' in r
    if not ok:
        print("  %-4s FAIL %s" % (nm, r[:70]))
        break
    print("  [%2d] %-4s %6.0f ms  slot=%s" % (i, nm, ms, r.split('"slot":')[1].split(',')[0]))
t_restore = time.time() - t0
print("  restore 合计 %.1f s" % t_restore)

# 链条：pe(x) -> b00..b13 -> fl
def spec(ins, out):
    return "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), out)

print("\n=== 跑一步完整链条（结果写入 /dev/null 附近文件）===")
for trial in range(3):
    t0 = time.time()
    times = []
    # pe
    r, ms = s('{"cmd":"run","spec":"%s"}' % spec([OUT + "/i_patch.bin"], OUT + "/o_h2.bin"))
    times.append(("pe", ms))
    if '"ok":true' not in r:
        print("  pe 失败 %s" % r[:60]); break
    TAIL = [OUT + "/i_te.bin", OUT + "/i_lsh.bin", OUT + "/i_lsc.bin", OUT + "/i_lga.bin",
            OUT + "/i_ctx_pos.bin", OUT + "/i_cos.bin", OUT + "/i_sin.bin"]
    cur = OUT + "/i_xb2.bin"
    # 第一块的输入用 o_h 做近似（这里只测耗时，数值不关心）
    r, ms = s('{"cmd":"use","slot":1}')
    for bi in range(14):
        nxt = OUT + "/o_c%02d.bin" % bi
        r, ms = s('{"cmd":"run","spec":"%s"}' % spec([cur] + TAIL, nxt))
        times.append(("b%02d" % bi, ms))
        if '"ok":true' not in r:
            print("  b%02d 失败 %s" % (bi, r[:60])); break
        cur = nxt
        s('{"cmd":"use","slot":%d}' % (bi + 2))
    tot = sum(t for _, t in times)
    print("  第%d次: 16 次 run 合计 %.0f ms (%.0f ms/块)  墙钟 %.0f ms"
          % (trial + 1, tot, tot / len(times), (time.time() - t0) * 1000), flush=True)

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()

ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print("\n子进程峰值 RSS: %.0f MB" % (ru.ru_maxrss / 1024))
