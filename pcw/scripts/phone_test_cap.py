"""确定天花板单位：是「块数」还是「字节」。
   路径：灌到满 → pop 1 → 灌 2 → 看活的能不能到 6。"""
import subprocess

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
BIG = "/data/local/tmp/blobs/b00.dla"
SMALL = "/data/local/tmp/blobs/fl.dla"

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

def live():
    r = s('{"cmd":"ping"}')
    return None

def verify(cmd_fmt, label, items):
    n = 0
    for i, bl in enumerate(items):
        r = s(cmd_fmt % bl)
        if '"ok":true' in r:
            n += 1
        else:
            print("  %s: 第%d个失败 %s" % (label, i + 1, r[:55]))
            break
    return n

# 清空重来：一个新进程
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=30)
except: p.kill()

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

print("=== 灌 5 个大块 ===")
for i in range(5):
    r = s('{"cmd":"rpush","path":"%s"}' % BIG)
    print("  [%d] %s" % (i, "OK" if '"ok":true' in r else r[:55]))

print("\n=== pop 1 块（剩 4 块活着）===")
print(" ", s('{"cmd":"pop"}')[:90])

print("\n=== 再灌 2 个大块（若成功则活的到 6）===")
for i in range(2):
    r = s('{"cmd":"rpush","path":"%s"}' % BIG)
    print("  重灌[%d]: %s" % (i, "OK" if '"ok":true' in r else r[:60]))

print("\n=== 再灌小额块，看剩余字节空间 ===")
n = 0
for i in range(60):
    r = s('{"cmd":"rpush","path":"%s"}' % SMALL)
    if '"ok":true' in r:
        n += 1
    else:
        print("  fl 灌到第 %d 个失败: %s" % (i + 1, r[:55]))
        break
print("  额外灌入 %d 个小块 (%.1f MB)" % (n, n * 3.4))

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()
