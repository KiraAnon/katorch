"""验证文档里那个 add.txt 最小例子能不能真跑。"""
import json, os, subprocess, time
import numpy as np

HOST = "/data/data/com.termux/files/home/npt/host_new"
T = "/data/local/tmp/doc_verify"
os.makedirs(T, exist_ok=True)

S, K = 256, 512          # 用小一点，编译快
np.zeros((S, K), np.float16).tofile(T + "/b.bin")
a = np.random.randn(S, K).astype(np.float16)
a.tofile(T + "/a.bin")

GRAPH = """OPD 0 8 2 %d %d 0 0 i
OPD 1 8 2 %d %d 0 0 w %s/b.bin
OPD 2 8 2 %d %d 0 0 o
OPD 3 4 1 1 0 0 a 0
OP 0 3 0 1 3 1 2
IO 1 0 1 2
END
""" % (S, K, S, K, T, S, K)

open(T + "/add.txt", "w").write(GRAPH)
print("=== 图内容 ===")
print(GRAPH)

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
banner = p.stdout.readline()
print("banner:", banner.strip())


def cmd(s):
    p.stdin.write(s + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()


print()
print("build  :", cmd('{"cmd":"build","path":"%s"}' % (T + "/add.txt")))
print("compile:", cmd('{"cmd":"compile"}'))

spec = "1 0 %s/a.bin 1 %s/out.bin" % (T, T)
r = cmd('{"cmd":"run","spec":"%s"}' % spec)
print("run    :", r)

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try:
    p.wait(timeout=20)
except Exception:
    p.kill()

err = p.stderr.read()
if err:
    print("stderr:", err[:300])

# 验证结果：a + 0 = a
out = np.fromfile(T + "/out.bin", dtype=np.float16).reshape(S, K)
print()
print("=== 结果验证 ===")
print("输入 a 非零元素:", int((a != 0).sum()))
print("输出非零元素  :", int((out != 0).sum()))
print("和 a 的差     :", float(np.abs(out.astype(np.float32) - a.astype(np.float32)).max()))
print("判定:", "✅ 正确（a+0=a）" if np.allclose(out, a, atol=1e-3) else "❌ 不对")
