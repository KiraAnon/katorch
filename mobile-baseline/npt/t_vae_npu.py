import sys, os, time, subprocess
sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import numpy as np, torch
from vae_block import ResBlock
from lowering import Lowering, Unsupported
from safetensors.torch import load_file

D = "/data/data/com.termux/files/home/npt"
OUT = D + "/vae_npu"
os.makedirs(OUT, exist_ok=True)
P = "/data/data/com.termux/files/home/ComfyUI/models/vae/qwen_image_vae.safetensors"

print("加载 VAE 权重...", flush=True)
sd = load_file(P, device="cpu")
def f(v): return None if v is None else v.float()
def get_w(pfx):
    return dict(
        n1_gamma=f(sd[pfx + "residual.0.gamma"]), c1_w=f(sd[pfx + "residual.2.weight"]),
        c1_b=f(sd[pfx + "residual.2.bias"]), n2_gamma=f(sd[pfx + "residual.3.gamma"]),
        c2_w=f(sd[pfx + "residual.6.weight"]), c2_b=f(sd[pfx + "residual.6.bias"]),
        sc_w=f(sd.get(pfx + "shortcut.weight")), sc_b=f(sd.get(pfx + "shortcut.bias")))

w = get_w("decoder.middle.0.")
blk = ResBlock(w, 384, 384).eval()

# 小尺寸先验证（16x16）
N, H, W, C = 1, 16, 16, 384
torch.manual_seed(0)
x = torch.randn(N, H, W, C) * 0.5
with torch.no_grad():
    ref = blk(x).numpy()
print("torch 输出 std=%.4f" % ref.std(), flush=True)

print("lowering...", flush=True)
low = Lowering(OUT)
xin = low.input((N, H, W, C))
try:
    low.lower_module(blk, [xin])
except Unsupported as e:
    print("UNSUPPORTED:", e); sys.exit(1)
txt = low.graph_text()
open(OUT + "/g.txt", "w").write(txt)
errs = low.validate()
print("  %d 算子, 校验 %s" % (len(low.ops), "通过" if not errs else errs[:2]), flush=True)
if errs: sys.exit(1)

x.numpy().astype(np.float16).tofile(OUT + "/i_x.bin")

p = subprocess.Popen([D + "/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def send(c):
    p.stdin.write(c + "\n"); p.stdin.flush(); return p.stdout.readline().strip()
send('{"cmd":"build","path":"%s/g.txt"}' % OUT)
t0 = time.time(); r = send('{"cmd":"compile"}'); tc = time.time()-t0
print("  编译 %.1fs %s" % (tc, r[:40]), flush=True)
if '"ok":true' not in r:
    p.kill(); sys.exit(1)
t0 = time.time()
r = send('{"cmd":"run","spec":"1 0 %s/i_x.bin 1 0 %s/o_y.bin"}' % (OUT, OUT))
print("  运行 %.2fs %s" % (time.time()-t0, r[:40]), flush=True)
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=60)
except: p.kill()

if os.path.exists(OUT + "/o_y.bin"):
    got = np.fromfile(OUT + "/o_y.bin", dtype=np.float16).astype(np.float32).reshape(ref.shape)
    e = np.abs(got - ref)
    print("\nNPU  : std=%.4f" % got.std())
    print("torch: std=%.4f" % ref.std())
    print("误差 : MAX_ABS=%.6f  rel=%.3f%%" % (e.max(), 100*e.max()/(np.abs(ref).max()+1e-9)))
