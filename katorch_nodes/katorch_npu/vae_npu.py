"""NPU 版 VAE 解码器（512×512 输出）。

4 段 blob 由 `vae_npu_512b.py` 生成，段间的 nearest 2× 上采样在 CPU 用 numpy 做
（MDLA 的 CONCAT 只支持沿最后一维，做不了 H/W 维复制）。

实测：512×512 解码 1.3s（CPU 版 37s），8bit 像素差最大 2 个色阶，肉眼无差别。

用法：
    v = VaeNpu()          # 单例，惰性启动 host 进程
    img = v.decode(z)     # z: (1,64,64,16) float32 numpy → (512,512,3) float32
    v.close()
"""

import os
import subprocess

import numpy as np

H = "/data/data/com.termux/files/home"
DEFAULT_BLOB_DIR = H + "/npt/vae_npu512"
DEFAULT_HOST = H + "/npt/host_new"
TMP = "/data/local/tmp/vae_npu_rt"


class VaeNpuError(Exception):
    pass


class VaeNpu(object):
    """常驻一个 host_new 进程，blob 只加载一次。"""

    def __init__(self, blob_dir=None, host_path=None, hw=64):
        self.blob_dir = blob_dir or DEFAULT_BLOB_DIR
        self.host_path = host_path or DEFAULT_HOST
        self.hw = hw
        self.proc = None
        self.ready = False
        os.makedirs(TMP, exist_ok=True)

    # ---------- 生命周期 ----------
    def blobs(self):
        return ["%s/s%d.dla" % (self.blob_dir, i) for i in (1, 2, 3, 4)]

    def available(self):
        return all(os.path.exists(b) for b in self.blobs()) and os.path.exists(self.host_path)

    def _cmd(self, c):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def _ensure(self):
        if self.ready:
            return
        if not self.available():
            raise VaeNpuError("blob 或 host 不存在: %s" % self.blob_dir)
        self.proc = subprocess.Popen(
            [self.host_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.proc.stdout.readline()          # 欢迎行

        lst = "%s/blobs.txt" % TMP
        with open(lst, "w") as f:
            for b in self.blobs():
                f.write(b + "\n")
        r = self._cmd('{"cmd":"bload","list":"%s"}' % lst)
        if '"ok":true' not in r:
            self.close()
            raise VaeNpuError("bload 失败: %s" % r[:120])
        self.ready = True

    def close(self):
        if self.proc is not None:
            try:
                self._cmd('{"cmd":"quit"}')
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
        self.ready = False

    # ---------- 执行 ----------
    def _run(self, idx, in_path, out_path):
        r = self._cmd('{"cmd":"map","idx":%d}' % idx)
        if '"ok":true' not in r:
            raise VaeNpuError("map 段%d 失败: %s" % (idx + 1, r[:100]))
        try:
            spec = "1 0 %s 1 0 %s" % (in_path, out_path)
            r = self._cmd('{"cmd":"run","spec":"%s"}' % spec)
        finally:
            self._cmd('{"cmd":"unmap"}')
        if '"ok":true' not in r:
            raise VaeNpuError("run 段%d 失败: %s" % (idx + 1, r[:100]))

    @staticmethod
    def _up2(x):
        """(1,h,w,c) nearest 2×。"""
        return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)

    def decode(self, z):
        """z: (1,hw,hw,16) float32（**已归一化**，即 DiT 输出的 latent）→ (h*8, w*8, 3) float32。"""
        self._ensure()
        hw = self.hw
        a = np.asarray(z, dtype=np.float16)
        if a.shape != (1, hw, hw, 16):
            raise VaeNpuError("latent 形状应为 (1,%d,%d,16)，拿到 %s" % (hw, hw, (a.shape,)))

        p = TMP + "/i0.bin"
        a.tofile(p)
        self._run(0, p, TMP + "/o1.bin")

        a = np.fromfile(TMP + "/o1.bin", dtype=np.float16).reshape(1, hw, hw, 384)
        self._up2(a).tofile(TMP + "/i1.bin")
        self._run(1, TMP + "/i1.bin", TMP + "/o2.bin")

        a = np.fromfile(TMP + "/o2.bin", dtype=np.float16).reshape(1, hw * 2, hw * 2, 384)
        self._up2(a).tofile(TMP + "/i2.bin")
        self._run(2, TMP + "/i2.bin", TMP + "/o3.bin")

        a = np.fromfile(TMP + "/o3.bin", dtype=np.float16).reshape(1, hw * 4, hw * 4, 192)
        self._up2(a).tofile(TMP + "/i3.bin")
        self._run(3, TMP + "/i3.bin", TMP + "/o4.bin")

        img = np.fromfile(TMP + "/o4.bin", dtype=np.float16)
        return img.astype(np.float32).reshape(hw * 8, hw * 8, 3)
