"""把 PC 编码器产出的 ctx 归一化到旧 i_ctx.bin 的幅度尺度。
   旧 ctx（已知能出图）: 整体 std=0.184, 逐行 std 0.14~0.22
   新 ctx（ComfyUI 原始输出）: 整体 std≈3.1, 逐行 std 1.9~3.4
"""
import numpy as np, os

TARGET_RMS = 0.18
SRC = r"F:\KAtorch\pcw\ctx"

for name in ("ctx_pos", "ctx_neg"):
    p = os.path.join(SRC, name + ".bin")
    a = np.fromfile(p, dtype=np.float16).astype(np.float32).reshape(512, 1024)
    nz = np.abs(a).sum(1) > 0
    n = int(nz.sum())
    for i in range(n):
        rms = np.sqrt((a[i] ** 2).mean())
        if rms > 0:
            a[i] *= (TARGET_RMS / rms)
    out = os.path.join(SRC, name + "_norm.bin")
    a.astype(np.float16).tofile(out)
    v = a[nz]
    print("%s: rows=%d  std=%.4f  absmax=%.3f  -> %s"
          % (name, n, v.std(), np.abs(v).max(), out))
