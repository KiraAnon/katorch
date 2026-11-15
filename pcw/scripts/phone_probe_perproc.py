"""验证映射配额是不是 per-process：3 个进程同时各恢复 5 个大块。"""
import subprocess, time

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
B = "/data/local/tmp/blobs/b00.dla"

NPROC = 3
PER = 5

procs = []
results = [[] for _ in range(NPROC)]

print("=== 同时启动 %d 个进程，各恢复 %d 个大块 (277MB) ===" % (NPROC, PER))
for k in range(NPROC):
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    procs.append(p)

ok_total = 0
for i in range(PER):
    for k in range(NPROC):
        p = procs[k]
        p.stdin.write('{"cmd":"rpush","path":"%s"}\n' % B)
        p.stdin.flush()
        r = p.stdout.readline().strip()
        ok = '"ok":true' in r
        results[k].append(ok)
        if ok: ok_total += 1
        print("  进程%d 第%d块: %s" % (k, i + 1, "OK" if ok else r[:55]), flush=True)

print("\n=== 汇总 ===")
for k in range(NPROC):
    n_ok = sum(1 for x in results[k] if x)
    print("  进程 %d: 成功 %d/%d  (%.0f MB)" % (k, n_ok, PER, n_ok * 277.4))

for p in procs:
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()

print("\n总成功 %d 块 = %.1f GB" % (ok_total, ok_total * 0.2774))
print("如果 3 个进程都能各装 5 块 → 配额是 per-process，可用总量 %.1f GB"
      % (NPROC * PER * 0.2774))
