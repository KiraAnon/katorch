"""测量 build 与 compile(缓存命中) 各自耗时，判断缓存到底省了什么。"""
import os, subprocess, time, json

H = "/data/data/com.termux/files/home"
D = H + "/npt"
E2E = D + "/e2eper256"
CACHE = E2E + "/nc"

p = subprocess.Popen([D + "/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def cmd(c):
    t0 = time.time()
    p.stdin.write(c + "\n"); p.stdin.flush()
    l = p.stdout.readline().strip()
    return l, (time.time() - t0) * 1000

cmd('{"cmd":"cachedir","path":"%s"}' % CACHE)
cmd('{"cmd":"cachekey","v":"b00"}')

r, ms = cmd('{"cmd":"build","path":"%s/b00/g.txt"}' % E2E)
print("build            : %8.1f ms   %s" % (ms, r[:60]))

# 连续两次 compile：第一次冷/热不明，第二次必定命中
r, ms = cmd('{"cmd":"compile"}')
print("compile (1st)    : %8.1f ms   %s" % (ms, r[:60]))
r, ms = cmd('{"cmd":"compile"}')
print("compile (2nd)    : %8.1f ms   %s" % (ms, r[:60]))

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()

# 再开一个新进程，只做 build（排除同一进程内的缓存效应）
print("")
p = subprocess.Popen([D + "/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def cmd2(c):
    t0 = time.time()
    p.stdin.write(c + "\n"); p.stdin.flush()
    l = p.stdout.readline().strip()
    return l, (time.time() - t0) * 1000

r, ms = cmd2('{"cmd":"build","path":"%s/b00/g.txt"}' % E2E)
print("新进程 build     : %8.1f ms   %s" % (ms, r[:60]))
cmd2('{"cmd":"cachedir","path":"%s"}' % CACHE)
cmd2('{"cmd":"cachekey","v":"b00"}')
r, ms = cmd2('{"cmd":"compile"}')
print("新进程 compile   : %8.1f ms   %s" % (ms, r[:60]))
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()
