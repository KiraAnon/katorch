"""在 NPU 上跑完整 VAE 解码（512×512），段间上采样用 CPU。

4 段 blob 由 vae_npu_512b.py 生成，放在 KA_OUT 下。
"""
import os, sys, time, subprocess

sys.path.insert(0, "/data/data/com.termux/files/home/npt")
import numpy as np
import torch

H = "/data/data/com.termux/files/home"
D = H + "/npt"
OUT = os.environ.get("KA_OUT", D + "/vae_npu512")
HOST = D + "/host_new"
HW = int(os.environ.get("KA_HW", "64"))

BLOBS = ["%s/s%d.dla" % (OUT, i) for i in (1, 2, 3, 4)]
for b in BLOBS:
    if not os.path.exists(b):
        print("缺 blob: %s" % b, flush=True)
        sys.exit(1)

TMP = "/data/local/tmp/vae_run"
os.makedirs(TMP, exist_ok=True)


def up2(x):
    """nearest 2× 上采样，NHWC fp16。"""
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


print("=== 启动 host，加载 4 段 blob ===", flush=True)
lst = "%s/blobs.txt" % TMP
with open(lst, "w") as f:
    for b in BLOBS:
        f.write(b + "\n")

p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdout.readline()


def cmd(c):
    p.stdin.write(c + "\n")
    p.stdin.flush()
    return p.stdout.readline().strip()


t0 = time.time()
r = cmd('{"cmd":"bload","list":"%s"}' % lst)
print("bload %s  %.1fs" % (r[:60], time.time() - t0), flush=True)
if '"ok":true' not in r:
    sys.exit(1)


def run_seg(idx, in_path, out_path, label):
    t = time.time()
    r = cmd('{"cmd":"map","idx":%d}' % idx)
    if '"ok":true' not in r:
        print("  %s map 失败 %s" % (label, r[:60]), flush=True)
        return False
    spec = "1 0 %s 1 0 %s" % (in_path, out_path)
    r = cmd('{"cmd":"run","spec":"%s"}' % spec)
    cmd('{"cmd":"unmap"}')
    dt = time.time() - t
    ok = '"ok":true' in r
    print("  %-4s %.2fs  %s" % (label, dt, "OK" if ok else r[:60]), flush=True)
    return ok


# 输入：真实 latent（1,64,64,16）
torch.manual_seed(1)
z = (torch.randn(1, HW, HW, 16) * 0.5).numpy().astype(np.float16)
f_i = TMP + "/i0.bin"
z.tofile(f_i)

print("\n=== NPU 4 段 ===", flush=True)
T0 = time.time()
ok = run_seg(0, f_i, TMP + "/o1.bin", "s1")
if ok:
    a = np.fromfile(TMP + "/o1.bin", dtype=np.float16).reshape(1, HW, HW, 384)
    t = time.time()
    a2 = up2(a)
    a2.tofile(TMP + "/i1.bin")
    print("  cpu 上采样 -> %s  %.2fs" % (str(a2.shape), time.time() - t), flush=True)

    ok = run_seg(1, TMP + "/i1.bin", TMP + "/o2.bin", "s2")
if ok:
    a = np.fromfile(TMP + "/o2.bin", dtype=np.float16).reshape(1, HW * 2, HW * 2, 384)
    t = time.time()
    a2 = up2(a)
    a2.tofile(TMP + "/i2.bin")
    print("  cpu 上采样 -> %s  %.2fs" % (str(a2.shape), time.time() - t), flush=True)

    ok = run_seg(2, TMP + "/i2.bin", TMP + "/o3.bin", "s3")
if ok:
    a = np.fromfile(TMP + "/o3.bin", dtype=np.float16).reshape(1, HW * 4, HW * 4, 192)
    t = time.time()
    a2 = up2(a)
    a2.tofile(TMP + "/i3.bin")
    print("  cpu 上采样 -> %s  %.2fs" % (str(a2.shape), time.time() - t), flush=True)

    ok = run_seg(3, TMP + "/i3.bin", TMP + "/o4.bin", "s4")

T_ALL = time.time() - T0
cmd('{"cmd":"quit"}')
try:
    p.wait(timeout=30)
except Exception:
    p.kill()

if not ok:
    print("\n执行失败", flush=True)
    sys.exit(1)

img = np.fromfile(TMP + "/o4.bin", dtype=np.float16).astype(np.float32).reshape(1, HW * 8, HW * 8, 3)
print("\n=== NPU 端到端 %.1fs ===" % T_ALL, flush=True)
print("  输出 %s  std=%.4f  min=%.3f max=%.3f" % (img.shape, img.std(), img.min(), img.max()), flush=True)

# 和 torch 对比
print("\n=== 和 torch CPU 版对比 ===", flush=True)
from safetensors.torch import load_file
from vae_decoder import WanVaeDecoder
sd = load_file(H + "/ComfyUI/models/vae/qwen_image_vae.safetensors", device="cpu")
dec = WanVaeDecoder(sd, H0=HW, W0=HW).eval()
LAT_MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
LAT_STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)
zz = torch.from_numpy(z.astype(np.float32))
# 注意：NPU 段的输入是「未归一化的 latent」，torch 的 decoder 同样接受未归一化
t0 = time.time()
with torch.no_grad():
    ref = dec(zz)[0].numpy()
t_torch = time.time() - t0
print("  torch CPU: %.1fs  std=%.4f" % (t_torch, ref.std()), flush=True)

d = np.abs(img[0] - ref)
print("  误差: MAX_ABS=%.5f  MEAN=%.5f  rel=%.3f%%"
      % (d.max(), d.mean(), 100 * d.max() / (np.abs(ref).max() + 1e-9)), flush=True)
print("  速度: NPU %.1fs vs CPU %.1fs  = %.1fx" % (T_ALL, t_torch, t_torch / T_ALL), flush=True)
