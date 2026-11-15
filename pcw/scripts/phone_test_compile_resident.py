"""对比：同一进程里 build+compile 能装多少块 vs restore 能装多少块。"""
import subprocess, time

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
CACHE = E2E + "/nc"
HOST = H + "/npt/host_new"

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

s('{"cmd":"cachedir","path":"%s"}' % CACHE)
s('{"cmd":"cachekey","v":"b00"}')

print("=== 同一进程内连续 build + compile（缓存命中）===")
t0 = time.time()
n_ok = 0
for i in range(14):
    t = time.time()
    r = s('{"cmd":"build","path":"%s/b%02d/g.txt"}' % (E2E, i))
    if '"ok":true' not in r:
        print("  b%02d build FAIL %s" % (i, r[:60])); break
    r = s('{"cmd":"compile"}')
    if '"ok":true' not in r:
        print("  b%02d compile FAIL %s" % (i, r[:60])); break
    n_ok += 1
    print("  b%02d OK  %.0f ms  (累计 %.1fs)" % (i, (time.time() - t) * 1000, time.time() - t0), flush=True)

print("\n  成功加载 %d/14 块，总耗时 %.1fs" % (n_ok, time.time() - t0))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()

ru = subprocess.run(["sh", "-c", "dmesg | grep 'iova alloc total' | tail -1"],
                    capture_output=True, text=True)
print("\n  最后一条 IOVA 状态: %s" % (ru.stdout.strip()[:160] or "(无)"))
