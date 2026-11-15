"""决定性实验：IOVA 空间能否回收。
   如果 pop 之后能重新 rpush 大块，就证明是「活体分配数」限制而非硬顶。"""
import subprocess, time

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
B = "/data/local/tmp/blobs/b00.dla"

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()

def s(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

print("=== 阶段1：灌大块直到失败 ===")
n_ok = 0
for i in range(8):
    r = s('{"cmd":"rpush","path":"%s"}' % B)
    if '"ok":true' in r:
        n_ok += 1
        print("  [%d] OK" % i)
    else:
        print("  [%d] FAIL %s" % (i, r[:60]))
        break
print("  成功灌入 %d 块 (%.1f GB)" % (n_ok, n_ok * 0.2774))

print("\n=== 阶段2：pop 掉 2 块，再灌 2 块 ===")
for i in range(2):
    print("  pop:", s('{"cmd":"pop"}')[:90])
for i in range(2):
    r = s('{"cmd":"rpush","path":"%s"}' % B)
    print("  重灌[%d]: %s" % (i, "OK" if '"ok":true' in r else r[:70]))

print("\n=== 阶段3：再 pop 光，看能否恢复满额 ===")
for i in range(4):
    print("  pop:", s('{"cmd":"pop"}')[:90])
ok2 = 0
for i in range(8):
    r = s('{"cmd":"rpush","path":"%s"}' % B)
    if '"ok":true' in r:
        ok2 += 1
    else:
        print("  [%d] FAIL %s" % (i, r[:60]))
        break
print("  清空后重灌: %d 块" % ok2)

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()
