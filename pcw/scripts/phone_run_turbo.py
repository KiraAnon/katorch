"""512×512 生图：turbo 合并版权重 + 8 步 + cfg=1 + 流式映射。"""
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
KA_CFG = float(_ka_os.environ.get("KA_CFG", "1.0"))
KA_STEPS = int(_ka_os.environ.get("KA_STEPS", "8"))

import sys, os, json, time, subprocess
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn.functional as F
from anima_block import load
from vae_decoder import WanVaeDecoder
from safetensors.torch import load_file
import t_rope3d

H = _KA_HOME
D = _KA_WORK
OUT = D + "/gen5"
HOST = D + "/host_new"
BLOBDIR = os.environ.get("KA_BLOB", "/data/local/tmp/blobs512m")

SHIFT = 3.0
STEPS = KA_STEPS
W = 512
LH = LW = W // 8
S_H = S_W = LH // 2
S = S_H * S_W
M = 512
CFG = KA_CFG

cal = json.load(open(D + "/gen/calib8.json"))
pb = cal["per_block"]
print("=== %dx%d  token %d  %d 步  cfg=%.1f  (turbo 合并版) ==="
      % (W, W, S, STEPS, CFG), flush=True)

LAT_MEAN = torch.tensor([-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,
                         0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]).view(1,16,1,1)
LAT_STD = torch.tensor([2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
                        3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]).view(1,16,1,1)
def ts(a,t): return t if a==1.0 else a*t/(1+(a-1)*t)
sigs = [float(ts(SHIFT, (k+1)/1000)) for k in range(1000)]
sigs = [sigs[-(1+int(x*(1000/STEPS)))] for x in range(STEPS)] + [0.0]

W1, W2 = load("net.t_embedder.1.linear_1.weight"), load("net.t_embedder.1.linear_2.weight")
TNORM = load("net.t_embedding_norm.weight")

ORDER = ["pe"] + ["b%02d" % i for i in range(14)] + ["fl"]
IDX = {n: i for i, n in enumerate(ORDER)}
lst = OUT + "/list_order.txt"
with open(lst, "w") as f:
    for n in ORDER:
        f.write("%s/%s.dla\n" % (BLOBDIR, n))

print("\n[1] 加载 blob", flush=True)
t0 = time.time()
p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdout.readline()
def cmd(c):
    p.stdin.write(c + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()
r = cmd('{"cmd":"bload","list":"%s"}' % lst)
print("  %s  %.1f s" % (r[:40], time.time() - t0), flush=True)

MG = {}
_mp = os.path.join(BLOBDIR, "meta.json")
if os.path.exists(_mp):
    MG = __import__("json").load(open(_mp))
    print("  blob 元数据: %s" % list(MG), flush=True)


def _prep(name, ins):
    """按目标 blob 的元数据把 fp16 中间文件转成它要的格式。"""
    m = MG.get(name)
    if not m:
        return ins
    out = []
    for i, p in enumerate(ins):
        sp = m["inputs"][i]
        if sp["t"] == 8 or not sp["s"]:
            out.append(p)
            continue
        a = np.fromfile(p, dtype=np.float16).astype(np.float32)
        q = np.clip(np.rint(a / np.float32(sp["s"])), -32767, 32767).astype(np.int16)
        qp = p + ".q16"
        q.tofile(qp)
        out.append(qp)
    return out


def _rdout(name, p):
    """按 blob 元数据读取输出（INT16 要乘回尺度）。"""
    m = MG.get(name)
    if not m:
        return np.fromfile(p, dtype=np.float16).astype(np.float32)
    sp = m["outputs"][0]
    if sp["t"] == 8 or not sp["s"]:
        return np.fromfile(p, dtype=np.float16).astype(np.float32)
    return (np.fromfile(p, dtype=np.int16).astype(np.float32)
            * np.float32(sp["s"]))


def run_blob(name, ins, outs):
    r = cmd('{"cmd":"map","idx":%d}' % IDX[name])
    if '"ok":true' not in r:
        print("  map %s 失败 %s" % (name, r[:50])); sys.exit(1)
    ins = _prep(name, ins)
    spec = "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), outs)
    r = cmd('{"cmd":"run","spec":"%s"}' % spec)
    cmd('{"cmd":"unmap"}')
    if '"ok":true' not in r:
        print("  run %s 失败 %s" % (name, r[:50])); sys.exit(1)

def t_embed(sv):
    h = 1024
    e = -torch.log(torch.tensor(10000.0)) * torch.arange(h, dtype=torch.float32) / h
    v = torch.tensor([float(sv)])[:, None] * torch.exp(e)[None, :]
    return torch.cat([torch.cos(v), torch.sin(v)], dim=-1)

cos_s, sin_s = t_rope3d.rope3d_cos_sin(1, S_H, S_W)
cos_s.reshape(S, 1, 128).numpy().astype(np.float16).tofile(OUT + "/i_cos512.bin")
sin_s.reshape(S, 1, 128).numpy().astype(np.float16).tofile(OUT + "/i_sin512.bin")

CTX = OUT + "/i_ctx_pos.bin"

def stack(x, ctx_path, tag):
    TAIL = [OUT + "/i_te.bin", OUT + "/i_lsh512.bin", OUT + "/i_lsc512.bin", OUT + "/i_lga512.bin",
            ctx_path, OUT + "/i_cos512.bin", OUT + "/i_sin512.bin"]
    ll = x[0].permute(1, 2, 0)
    pp_ = torch.zeros(S, 68)
    for i in range(S_H):
        for j in range(S_W):
            pp_[i*S_W+j, :64] = ll[i*2:(i+1)*2, j*2:(j+1)*2, :].permute(2, 0, 1).reshape(-1)
    pp_.numpy().astype(np.float16).tofile(OUT + "/i_patch512.bin")
    run_blob("pe", [OUT + "/i_patch512.bin"], OUT + "/o_h512_%s.bin" % tag)
    h = torch.from_numpy(np.fromfile(OUT + "/o_h512_%s.bin" % tag, dtype=np.float16)
                         .astype(np.float32)).reshape(S, 2048)
    cur = OUT + "/i_xb512_%s.bin" % tag
    (h / pb[0]["d_in"]).numpy().astype(np.float16).tofile(cur)
    for bi in range(14):
        nxt = OUT + "/o_b%02d_512_%s.bin" % (bi, tag)
        run_blob("b%02d" % bi, [cur] + TAIL, nxt)
        v = torch.from_numpy(_rdout("b%02d" % bi, nxt)).reshape(S, 2048)
        v = v * pb[2*bi+1]["post_scale"]
        cur = OUT + "/i_xb512_%s_%02d.bin" % (tag, bi)
        v.numpy().astype(np.float16).tofile(cur)
    run_blob("fl", [cur, OUT + "/i_te.bin", OUT + "/i_lsh512.bin", OUT + "/i_lsc512.bin"],
             OUT + "/o_tok512_%s.bin" % tag)
    tok = torch.from_numpy(np.fromfile(OUT + "/o_tok512_%s.bin" % tag, dtype=np.float16)
                           .astype(np.float32)).reshape(S, 64)
    v_lat = torch.zeros(1, 16, LH, LW)
    for i in range(S_H):
        for j in range(S_W):
            v_lat[0, :, i*2:(i+1)*2, j*2:(j+1)*2] = tok[i*S_W+j].reshape(2, 2, 16).permute(2, 0, 1)
    return v_lat

torch.manual_seed(42)
x = torch.randn(1, 16, LH, LW)

print("\n[2] 采样 %d 步" % STEPS, flush=True)
t_all = time.time()
for step in range(STEPS):
    t_step = time.time()
    sv = sigs[step]
    te_raw = t_embed(sv)
    lora = F.linear(F.silu(F.linear(te_raw, W1)), W2)
    te = te_raw * torch.rsqrt((te_raw * te_raw).mean(-1, keepdim=True) + 1e-6) * TNORM
    te.numpy().astype(np.float16).tofile(OUT + "/i_te.bin")
    lora[:, :2048].numpy().astype(np.float16).tofile(OUT + "/i_lsh512.bin")
    lora[:, 2048:4096].numpy().astype(np.float16).tofile(OUT + "/i_lsc512.bin")
    lora[:, 4096:6144].numpy().astype(np.float16).tofile(OUT + "/i_lga512.bin")

    v_pos = stack(x, CTX, "p")
    # 保存该步的 8 个输入，供多步标定用
    if os.environ.get("KA_SAVE_CAL"):
        import shutil
        cd_ = OUT + "/cal/step%02d" % step
        os.makedirs(cd_, exist_ok=True)
        for _f in ["i_xb512_p.bin", "i_te.bin", "i_lsh512.bin", "i_lsc512.bin",
                   "i_lga512.bin", "i_ctx_pos.bin", "i_cos512.bin", "i_sin512.bin"]:
            _s = OUT + "/" + _f
            if os.path.exists(_s):
                shutil.copy(_s, cd_ + "/" + _f)
    if CFG <= 1.0:
        v_lat = v_pos
    else:
        v_neg = stack(x, OUT + "/i_ctx_neg.bin", "n")
        v_lat = v_neg + CFG * (v_pos - v_neg)
    x = x - v_lat * (sigs[step] - sigs[step+1])
    print("  step%2d sigma=%.4f x.std=%.4f v.std=%.4f  %.1fs"
          % (step, sv, x.std(), v_lat.std(), time.time() - t_step), flush=True)

print("\n  采样完成 %.0fs" % (time.time() - t_all), flush=True)

print("\n[3] VAE", flush=True)
z = x * LAT_STD + LAT_MEAN
vae_sd = load_file(H + "/ComfyUI/models/vae/qwen_image_vae.safetensors", device="cpu")
dec = WanVaeDecoder(vae_sd, H0=LH, W0=LW).eval()
x[0].numpy().astype(np.float16).tofile(OUT + "/final512m_latent.bin")
with torch.no_grad():
    img = dec(z[0].permute(1, 2, 0).unsqueeze(0)).squeeze(0)
print("  图像 %s std=%.3f" % (tuple(img.shape), img.std()))
img8 = torch.clamp((img + 1) * 127.5, 0, 255).to(torch.uint8).numpy()
f = open(OUT + "/final512m.ppm", "wb")
f.write(("P6\n%d %d\n255\n" % (img8.shape[1], img8.shape[0])).encode())
f.write(img8.tobytes()); f.close()
print("  已存 final512m.ppm")

p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
try: p.wait(timeout=120)
except: p.kill()
