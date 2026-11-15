"""NPU 生图管线：提示词 -> ctx -> 8 步采样 -> VAE -> PNG

手机端运行。依赖 /data/local/tmp/blobs512m 的 16 个已导出的 blob。
"""
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

APP = os.path.dirname(os.path.abspath(__file__))
HOME = "/data/data/com.termux/files/home"
WORK = os.environ.get("KA_WORK", HOME + "/npt")
HOST = WORK + "/host_new"
BLOBDIR = os.environ.get("KA_BLOB", "/data/local/tmp/blobs512m")
TMP = WORK + "/gen_app"
OUT = WORK + "/outputs"

W = 512
LH = LW = W // 8
S_H = S_W = LH // 2
S = S_H * S_W
M = 512
STEPS = int(os.environ.get("KA_STEPS", "8"))
SHIFT = 3.0

CAL = os.path.join(WORK, "gen", "calib8.json")

LAT_MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
LAT_STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)

ORDER = ["pe"] + ["b%02d" % i for i in range(14)] + ["fl"]
IDX = {n: i for i, n in enumerate(ORDER)}


def _ts(a, t):
    return t if a == 1.0 else a * t / (1 + (a - 1) * t)


class Pipeline(object):
    def __init__(self):
        os.makedirs(TMP, exist_ok=True)
        os.makedirs(OUT, exist_ok=True)
        self.cal = __import__("json").load(open(CAL))
        self.pb = self.cal["per_block"]
        sys.path.insert(0, WORK)
        from anima_block import load
        self.W1 = load("net.t_embedder.1.linear_1.weight")
        self.W2 = load("net.t_embedder.1.linear_2.weight")
        self.TNORM = load("net.t_embedding_norm.weight")
        import t_rope3d
        cos_s, sin_s = t_rope3d.rope3d_cos_sin(1, S_H, S_W)
        cos_s.reshape(S, 1, 128).numpy().astype(np.float16).tofile(TMP + "/i_cos.bin")
        sin_s.reshape(S, 1, 128).numpy().astype(np.float16).tofile(TMP + "/i_sin.bin")
        sigs = [float(_ts(SHIFT, (k + 1) / 1000.0)) for k in range(1000)]
        self.sigs = [sigs[-(1 + int(x * (1000.0 / STEPS)))] for x in range(STEPS)] + [0.0]
        self.proc = None

    # ---------- host 通信 ----------
    def _start_host(self):
        if self.proc is not None:
            return
        lst = TMP + "/list.txt"
        with open(lst, "w") as f:
            for n in ORDER:
                f.write("%s/%s.dla\n" % (BLOBDIR, n))
        p = subprocess.Popen([HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
        p.stdout.readline()
        self.proc = p
        self._cmd('{"cmd":"bload","list":"%s"}' % lst)

    def _cmd(self, c):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def _run_blob(self, name, ins, outs):
        r = self._cmd('{"cmd":"map","idx":%d}' % IDX[name])
        if '"ok":true' not in r:
            raise RuntimeError("map %s: %s" % (name, r[:60]))
        spec = "%d %s 1 %s" % (len(ins), " ".join("%d %s" % (i, f) for i, f in enumerate(ins)), outs)
        r = self._cmd('{"cmd":"run","spec":"%s"}' % spec)
        self._cmd('{"cmd":"unmap"}')
        if '"ok":true' not in r:
            raise RuntimeError("run %s: %s" % (name, r[:60]))

    def close(self):
        if self.proc is not None:
            try:
                self._cmd('{"cmd":"quit"}')
                self.proc.wait(timeout=10)
            except Exception:
                pass
            self.proc = None

    # ---------- 采样 ----------
    def _t_embed(self, sv):
        h = 1024
        e = -torch.log(torch.tensor(10000.0)) * torch.arange(h, dtype=torch.float32) / h
        v = torch.tensor([float(sv)])[:, None] * torch.exp(e)[None, :]
        return torch.cat([torch.cos(v), torch.sin(v)], dim=-1)

    def _tail(self, ctx_path):
        return [TMP + "/i_te.bin", TMP + "/i_lsh.bin", TMP + "/i_lsc.bin",
                TMP + "/i_lga.bin", ctx_path, TMP + "/i_cos.bin", TMP + "/i_sin.bin"]

    def _stack(self, x, ctx_path):
        TAIL = self._tail(ctx_path)
        ll = x[0].permute(1, 2, 0)
        pp = torch.zeros(S, 68)
        for i in range(S_H):
            for j in range(S_W):
                pp[i * S_W + j, :64] = ll[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2].permute(2, 0, 1).reshape(-1)
        pp.numpy().astype(np.float16).tofile(TMP + "/i_patch.bin")
        self._run_blob("pe", [TMP + "/i_patch.bin"], TMP + "/o_h.bin")
        h = torch.from_numpy(np.fromfile(TMP + "/o_h.bin", dtype=np.float16)
                             .astype(np.float32)).reshape(S, 2048)
        cur = TMP + "/i_xb.bin"
        (h / self.pb[0]["d_in"]).numpy().astype(np.float16).tofile(cur)
        for bi in range(14):
            nxt = TMP + "/o_b%02d.bin" % bi
            self._run_blob("b%02d" % bi, [cur] + TAIL, nxt)
            v = torch.from_numpy(np.fromfile(nxt, dtype=np.float16)
                                 .astype(np.float32)).reshape(S, 2048)
            v = v * self.pb[2 * bi + 1]["post_scale"]
            cur = TMP + "/i_xb_%02d.bin" % bi
            v.numpy().astype(np.float16).tofile(cur)
        self._run_blob("fl", [cur, TMP + "/i_te.bin", TMP + "/i_lsh.bin", TMP + "/i_lsc.bin"],
                       TMP + "/o_tok.bin")
        tok = torch.from_numpy(np.fromfile(TMP + "/o_tok.bin", dtype=np.float16)
                               .astype(np.float32)).reshape(S, 64)
        v_lat = torch.zeros(1, 16, LH, LW)
        for i in range(S_H):
            for j in range(S_W):
                v_lat[0, :, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = \
                    tok[i * S_W + j].reshape(2, 2, 16).permute(2, 0, 1)
        return v_lat

    def generate(self, ctx_np, seed=42, progress=None):
        ctx_path = TMP + "/i_ctx.bin"
        ctx_np.astype(np.float16).tofile(ctx_path)
        torch.manual_seed(int(seed))
        x = torch.randn(1, 16, LH, LW)
        self._start_host()
        t0 = time.time()
        for step in range(STEPS):
            sv = self.sigs[step]
            raw = self._t_embed(sv)
            lora = F.linear(F.silu(F.linear(raw, self.W1)), self.W2)
            te = raw * torch.rsqrt((raw * raw).mean(-1, keepdim=True) + 1e-6) * self.TNORM
            te.numpy().astype(np.float16).tofile(TMP + "/i_te.bin")
            lora[:, :2048].numpy().astype(np.float16).tofile(TMP + "/i_lsh.bin")
            lora[:, 2048:4096].numpy().astype(np.float16).tofile(TMP + "/i_lsc.bin")
            lora[:, 4096:6144].numpy().astype(np.float16).tofile(TMP + "/i_lga.bin")
            v = self._stack(x, ctx_path)
            x = x - v * (self.sigs[step] - self.sigs[step + 1])
            if progress:
                progress(step + 1, STEPS, time.time() - t0)
        return x

    def decode(self, x, out_png):
        from vae_decoder import WanVaeDecoder
        from safetensors.torch import load_file
        vae_p = os.environ.get("KA_VAE", HOME + "/ComfyUI/models/vae/qwen_image_vae.safetensors")
        sd = load_file(vae_p, device="cpu")
        dec = WanVaeDecoder(sd, H0=LH, W0=LW).eval()
        z = x * LAT_STD + LAT_MEAN
        with torch.no_grad():
            img = dec(z[0].permute(1, 2, 0).unsqueeze(0)).squeeze(0)
        img8 = torch.clamp((img + 1) * 127.5, 0, 255).to(torch.uint8).numpy()
        try:
            from PIL import Image
            Image.fromarray(img8).save(out_png)
        except ImportError:
            h, w = img8.shape[0], img8.shape[1]
            ppm = out_png.replace(".png", ".ppm")
            with open(ppm, "wb") as f:
                f.write(("P6\n%d %d\n255\n" % (w, h)).encode())
                f.write(img8.tobytes())
            out_png = ppm
        return out_png
