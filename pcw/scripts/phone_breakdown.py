"""拆解一次完整块循环：进程启动 / build / compile / run / 退出 各占多少。"""
import os, subprocess, time

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
OUT = D + "/gen5"
CACHE = E2E + "/nc"
HOST = D + "/host"

INS = ["i_xb.bin", "i_te.bin", "i_lsh.bin", "i_lsc.bin", "i_lga.bin",
       "i_ctx_pos.bin", "i_cos.bin", "i_sin.bin"]

def one_cycle():
    t = {}
    t0 = time.time()
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    t['startup'] = (time.time() - t0) * 1000

    def cmd(c):
        a = time.time()
        p.stdin.write(c + "\n"); p.stdin.flush()
        l = p.stdout.readline().strip()
        return l, (time.time() - a) * 1000

    cmd('{"cmd":"cachedir","path":"%s"}' % CACHE)
    cmd('{"cmd":"cachekey","v":"b00"}')
    r, t['build'] = cmd('{"cmd":"build","path":"%s/b00/g.txt"}' % E2E)
    r, t['compile'] = cmd('{"cmd":"compile"}')
    spec = "8 " + " ".join("%d %s" % (i, OUT + "/" + f) for i, f in enumerate(INS)) + " 1 " + OUT + "/o_tb.bin"
    r, t['run'] = cmd('{"cmd":"run","spec":"%s"}' % spec)
    t['run_ok'] = '"ok":true' in r
    a = time.time()
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    t['teardown'] = (time.time() - a) * 1000
    t['total'] = (time.time() - t0) * 1000
    return t

print("第一次（含冷启动）：")
a = one_cycle()
print("  启动 %.0f | build %.0f | compile %.0f | run %.0f | 退出 %.0f | 合计 %.0f ms"
      % (a['startup'], a['build'], a['compile'], a['run'], a['teardown'], a['total']))

print("\n再跑 2 次取稳定值：")
for i in range(2):
    a = one_cycle()
    print("  启动 %.0f | build %.0f | compile %.0f | run %.0f | 退出 %.0f | 合计 %.0f ms"
          % (a['startup'], a['build'], a['compile'], a['run'], a['teardown'], a['total']))
