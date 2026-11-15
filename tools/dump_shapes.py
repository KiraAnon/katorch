import numpy as np, os, sys

D = "/data/data/com.termux/files/home/npt/e2eper512/b00"
for f in sorted(os.listdir(D)):
    if f.startswith("i") and f.endswith(".bin"):
        p = os.path.join(D, f)
        sz = os.path.getsize(p)
        if sz <= 64:
            a = np.fromfile(p, dtype=np.int32)
            print("%-8s %3d 字节  int32=%s" % (f, sz, a.tolist()))
