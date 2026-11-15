"""判定 restore 上限是「字节配额」还是「块数」。"""
import subprocess, os

H = "/data/data/com.termux/files/home"
HOST = H + "/npt/host_new"
B = "/data/local/tmp/blobs"

def run(seq, label):
    print("\n=== %s ===" % label)
    p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdout.readline()
    for i, name in enumerate(seq):
        p.stdin.write('{"cmd":"rpush","path":"%s/%s"}\n' % (B, name))
        p.stdin.flush()
        r = p.stdout.readline().strip()
        ok = '"ok":true' in r
        sz = os.path.getsize("%s/%s" % (B, name)) / 1e6
        print("  [%2d] %-7s %6.1f MB  %s" % (i, name, sz, "OK" if ok else r[:60]), flush=True)
        if not ok:
            print("  >>> 在累计 %.0f MB 处失败" % (sum(os.path.getsize("%s/%s" % (B, n)) / 1e6 for n in seq[:i])))
            break
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=60)
    except: p.kill()

# 5 个大块之后再塞小块：能进去说明是字节配额
run(["b00.dla"] * 5 + ["fl.dla", "pe.dla"], "5×大块(277MB) 之后放小块")

# 只放中等的：fl(3MB) 反复
run(["fl.dla"] * 12, "12× fl(3MB)")
