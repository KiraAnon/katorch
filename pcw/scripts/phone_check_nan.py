import numpy as np, os
os.chdir("/data/data/com.termux/files/home/npt/gen5")
for f in ["i_te.bin", "i_ctx_pos.bin", "o_h512_p.bin", "o_b00_512_p.bin", "i_xb512_p.bin", "o_tok512_p.bin"]:
    if not os.path.exists(f):
        print("%-20s MISSING" % f)
        continue
    a = np.fromfile(f, dtype=np.float16).astype(np.float32)
    fin = np.isfinite(a)
    mx = float(np.abs(a[fin]).max()) if fin.any() else -1
    print("%-20s n=%-8d max=%-12.5g std=%-10.5g nan=%-6d inf=%d"
          % (f, len(a), mx, float(a[fin].std()) if fin.any() else -1,
             int(np.isnan(a).sum()), int(np.isinf(a).sum())))
