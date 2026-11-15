"""诊断 restore 的数量上限：是小图也不行（计数限制），还是大图才不行（内存限制）。"""
import subprocess, time

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
BLOBDIR = "/data/local/tmp/blobs"

def trial(blob_name, n, label):
    print("\n=== %s：连续 restore %d 次 ===" % (label, n))
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    for i in range(n):
        p.stdin.write('{"cmd":"rpush","path":"%s/%s"}\n' % (BLOBDIR, blob_name))
        p.stdin.flush()
        r = p.stdout.readline().strip()
        ok = '"ok":true' in r
        print("  [%2d] %s" % (i, r[:70]), flush=True)
        if not ok:
            print("  >>> 第 %d 次失败（成功 %d 次）" % (i + 1, i))
            break
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()

# 小图（pe，几 KB）——如果是计数限制，小图也会在同样次数失败
trial("pe.dla", 20, "小图 pe")

# 大图（b00，277MB）——如果是内存限制，会在这里更早失败
trial("b00.dla", 12, "大图 b00")
