"""导出 14 个量化块为 blob，并生成合并的 meta.json。"""
import os, json, subprocess, shutil
import numpy as np

H = "/data/data/com.termux/files/home"
D = H + "/npt"
HOST = D + "/host_new"
QA = "/data/local/tmp/qall"
SRC = "/data/local/tmp/blobs512m"      # 其余 15 个（pe/fl/b 不在 qall 的）从这里链接
DST = "/data/local/tmp/blobs512q"
os.makedirs(DST, exist_ok=True)

meta = {}

for bi in range(14):
    tag = "b%02d" % bi
    g = "%s/q%02d/g.txt" % (QA, bi)
    out = os.path.join(DST, tag + ".dla")
    if os.path.exists(out):
        print("%s 已存在" % tag, flush=True)
    else:
        p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
        p.stdout.readline()

        def s(c):
            p.stdin.write(c + "\n"); p.stdin.flush()
            return p.stdout.readline().strip()

        r = s('{"cmd":"build","path":"%s"}' % g)
        if '"ok":true' not in r:
            print("%s build 失败 %s" % (tag, r[:50])); p.kill(); continue
        r = s('{"cmd":"compile"}')
        if '"ok":true' not in r:
            print("%s compile 失败 %s" % (tag, r[:50])); p.kill(); continue
        r = s('{"cmd":"export","path":"%s"}' % out)
        p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
        try: p.wait(timeout=180)
        except: p.kill()
        sz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
        print("%s  %.1f MB" % (tag, sz), flush=True)
    mp = "%s/q%02d/meta.json" % (QA, bi)
    if os.path.exists(mp):
        meta[tag] = json.load(open(mp))

# pe / fl 保持 fp16，从 blobs512m 链接过来
for n in os.listdir(SRC):
    base = n.split(".")[0]
    if base.startswith("b") and base[1:].isdigit():
        continue
    s_, d_ = os.path.join(SRC, n), os.path.join(DST, n)
    if not os.path.exists(d_):
        try:
            os.symlink(s_, d_)
        except OSError:
            shutil.copy(s_, d_)

json.dump(meta, open(os.path.join(DST, "meta.json"), "w"))
tot = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)
          if os.path.isfile(os.path.join(DST, f)))
print("meta 覆盖 %d 个 blob；目录共 %.0f MB" % (len(meta), tot / 1e6))
