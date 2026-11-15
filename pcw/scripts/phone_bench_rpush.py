"""测出 rpush 里「读文件」和「映射」各占多少，判断常驻内存方案能省多少。"""
import subprocess, time

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
B = "/data/local/tmp/blobs"

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

print("=== 冷启动第一次（文件可能在磁盘）===")
r = s('{"cmd":"rpush","path":"%s/b00.dla"}' % B)
print(" ", r[:60])

print("\n=== 之后重复 4 次（文件已在 page cache）===")
for i in range(4):
    r = s('{"cmd":"rpush","path":"%s/b01.dla"}' % B)
    ok = '"ok":true' in r
    print("  [%d] %s" % (i, "OK" if ok else r[:50]), flush=True)
    if ok:
        s('{"cmd":"pop"}')

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()

err = p.stderr.read()
print("\n=== host 内部计时（read / map 分开）===")
for line in err.splitlines():
    if line.startswith("[RST]"):
        print(" ", line)
