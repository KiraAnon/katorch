"""单步 profile：把 12.2 秒拆开看看到底花在哪。"""
import sys, os, time
sys.path.insert(0, "/data/data/com.termux/files/home/ComfyUI/custom_nodes")

import torch
import numpy as np

OUT = []
def mark(label, t0):
    dt = (time.time() - t0) * 1000
    OUT.append((label, dt))
    print("  %-28s %8.1f ms" % (label, dt))
    return time.time()


for name in list(sys.modules):
    if name.startswith("katorch_npu"):
        del sys.modules[name]

from katorch_npu import runtime as ka_rt
from katorch_npu import assets as ka_assets
from katorch_npu.dit import DitRuntime

mf = ka_rt.manifest()
asset = ka_assets.pick_asset_for(mf, "anima_v1", 512, 512)
print("资产:", asset.get("name"), asset.get("width"), "x", asset.get("height"))

model = ka_rt.KAtorchModel(asset)
dit = ka_rt.get_dit(model)
print("DiT 就绪，块数:", dit.n_blocks)
print()

x = torch.randn(1, 16, 64, 64)
ctx_path = "/data/data/com.termux/files/home/npt/gen_app/ctx_dummy.bin"
if not os.path.exists(ctx_path):
    np.zeros((512, 1024), np.float16).tofile(ctx_path)

print("=== 预热一次 ===")
t = time.time()
dit.forward(x, ctx_path, 0.9)
print("  第一次整体 %.1f s" % (time.time() - t))
print()

print("=== 拆解第二次 ===")
T = time.time()

t = time.time()
te, lsh, lsc, lga = dit.write_cond(0.8)
t = mark("write_cond（时间嵌入+adaLN）", t)

t = time.time()
pf = dit._patchify(x)
t = mark("patchify", t)

host = dit.host
idx = {n: i for i, n in enumerate(dit.order)}

t = time.time()
host.run_block(idx["pe"], [pf], dit.f_h)
t = mark("pe（patch embed）", t)

t = time.time()
h = np.fromfile(dit.f_h, dtype=np.float16)
t = mark("  读 pe 输出（4MB）", t)

t = time.time()
h = torch.from_numpy(h).reshape(dit.S, 2048) / float(dit.pb[0]["d_in"])
cur = os.path.join(dit.tmp, "i_xb.bin")
h.numpy().astype(np.float16).tofile(cur)
t = mark("  归一化+写盘（4MB）", t)

tail = [te, lsh, lsc, lga, ctx_path, dit.f_cos, dit.f_sin]
block_times = []
for bi in range(dit.n_blocks):
    name = "b%02d" % bi
    nxt = os.path.join(dit.tmp, "o_%s.bin" % name)
    t1 = time.time()
    host.run_block(idx[name], [cur] + tail, nxt)
    t2 = time.time()
    v = np.fromfile(nxt, dtype=np.float16).astype(np.float32) * float(dit.pb[2 * bi + 1]["post_scale"])
    t3 = time.time()
    nb = os.path.join(dit.tmp, "i_xb_%02d.bin" % bi)
    v.astype(np.float16).tofile(nb)
    t4 = time.time()
    block_times.append((t2 - t1, t3 - t2, t4 - t3))
    cur = nb

run_ms = sum(a for a, b, c in block_times) * 1000
read_ms = sum(b for a, b, c in block_times) * 1000
write_ms = sum(c for a, b, c in block_times) * 1000
print("  %-28s %8.1f ms  (%d 块)" % ("14 块 run_block", run_ms, dit.n_blocks))
print("  %-28s %8.1f ms" % ("  读输出+缩放", read_ms))
print("  %-28s %8.1f ms" % ("  写下一块输入", write_ms))
t = time.time()

t = time.time()
host.run_block(idx["fl"], [cur, te, lsh, lsc], dit.f_tok)
t = mark("fl（final layer）", t)

t = time.time()
tok = np.fromfile(dit.f_tok, dtype=np.float16).reshape(dit.S, 64)
t = mark("  读 fl 输出", t)

t = time.time()
out = dit._unpatchify(torch.from_numpy(tok.astype(np.float32)))
t = mark("unpatchify", t)

total = (time.time() - T) * 1000
print()
print("=== 汇总 ===")
print("  单步实测合计: %.1f ms" % total)
print("  其中 NPU 执行: %.1f ms (%.1f%%)" % (run_ms, run_ms / total * 100))
print("  其中文件 IO:   %.1f ms (%.1f%%)" % (read_ms + write_ms, (read_ms + write_ms) / total * 100))
print()
print("  8 步预估: %.1f s" % (total * 8 / 1000))
