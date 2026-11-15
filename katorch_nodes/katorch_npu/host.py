"""NPU 宿主会话：管理 host_new 子进程（map / run / unmap）。

为什么要常驻：host_new 的 `bload` 会把整份图（约 3.9GB）读进内存池，
启动一次要几十秒。所以整个 ComfyUI 进程里只保留**一个** host 实例，
换资产时才重建。

IOVA 配额只有约 1.35GB，所以同一时刻只映射一块（277MB），
跑完立刻 unmap —— 这也是当前最大的性能瓶颈（每步约 3.5s）。
"""
import json
import os
import subprocess
import threading
import time

HOME = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
WORK = os.environ.get("KA_WORK", os.path.join(HOME, "npt"))
DEFAULT_HOST = os.environ.get("KA_HOST", os.path.join(WORK, "host_new"))
TMP = os.environ.get("KA_TMP", os.path.join(WORK, "gen_app"))


def _log(msg):
    print("[katorch.host] " + str(msg), flush=True)


class NpuHostError(RuntimeError):
    pass


class NpuHost(object):
    """一个 host_new 进程 + 它的常驻 blob 池。"""

    def __init__(self, blob_files, tmp_dir=None, exe=None, tag="asset"):
        self.blob_files = list(blob_files)
        self.tmp = tmp_dir or TMP
        self.exe = exe or DEFAULT_HOST
        self.tag = tag
        self.proc = None
        self.lock = threading.RLock()
        self.rows = 0            # 前向调用次数（给统计用）
        self.last_error = None
        os.makedirs(self.tmp, exist_ok=True)

    # ---------- 生命周期 ----------
    def ensure(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return self
            self.start()
            return self

    def start(self):
        if not os.path.isfile(self.exe):
            raise NpuHostError("找不到 NPU 宿主可执行：%s" % self.exe)
        lst = os.path.join(self.tmp, "blobs.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in self.blob_files:
                f.write(p + "\n")
        t0 = time.time()
        p = subprocess.Popen(
            [self.exe],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        hello = p.stdout.readline()
        if not hello:
            raise NpuHostError("宿主启动后没有响应（%s）" % self.exe)
        self.proc = p
        r = self._cmd('{"cmd":"bload","list":"%s"}' % lst)
        try:
            d = json.loads(r)
        except Exception:
            d = {}
        if not d.get("ok"):
            self.stop()
            raise NpuHostError("bload 失败：%s" % r[:200])
        loaded = int(d.get("n") or 0)
        if loaded != len(self.blob_files):
            _log("警告：bload 只加载了 %d/%d 个 blob" % (loaded, len(self.blob_files)))
        _log("宿主就绪：%d 块，耗时 %.1fs（%s）" % (loaded, time.time() - t0, os.path.basename(self.tmp)))

    def stop(self):
        with self.lock:
            if self.proc is None:
                return
            try:
                self._cmd('{"cmd":"quit"}', timeout=5)
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    # ---------- 通信 ----------
    def _cmd(self, text, timeout=None):
        if self.proc is None:
            raise NpuHostError("宿主未启动")
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            self.last_error = "宿主无响应（可能崩溃）"
            self.proc = None
            raise NpuHostError("宿主进程没有返回（%s）" % text[:80])
        return line.strip()

    def run_block(self, idx, inputs, output):
        """映射第 idx 块，喂 inputs 文件，取 output 文件，立刻卸载。"""
        with self.lock:
            self.ensure()
            r = self._cmd('{"cmd":"map","idx":%d}' % int(idx))
            if '"ok":true' not in r:
                self.last_error = r
                raise NpuHostError("map idx=%d 失败：%s" % (idx, r[:160]))
            parts = ["%d" % len(inputs)]
            for i, f in enumerate(inputs):
                if " " in f:
                    self._cmd('{"cmd":"unmap"}')
                    raise NpuHostError("输入路径含空格，宿主协议不支持：%s" % f)
                parts.append("%d %s" % (i, f))
            parts.append("1 0 %s" % output)
            spec = " ".join(parts)
            if len(spec) > 1900:
                self._cmd('{"cmd":"unmap"}')
                raise NpuHostError("spec 太长（%d 字符），宿主缓冲区上限 2048" % len(spec))
            r = self._cmd('{"cmd":"run","spec":"%s"}' % spec)
            self._cmd('{"cmd":"unmap"}')
            self.rows += 1
            if '"ok":true' not in r:
                self.last_error = r
                raise NpuHostError("run idx=%d 失败：%s" % (idx, r[:200]))
            return r


class HostManager(object):
    """全局唯一 host：同一时刻只有一个资产在 NPU 上（内存/IOVA 都不允许更多）。"""

    _inst = None
    _key = None

    @classmethod
    def get(cls, asset, exe=None):
        key = (asset["blob_dir"], tuple(a["bytes"] for a in asset["files"]))
        if cls._inst is not None and cls._key != key:
            cls._inst.stop()
            cls._inst = None
        if cls._inst is None:
            files = [a["path"] for a in asset["files"]]
            cls._inst = NpuHost(files, exe=exe, tag=asset.get("id", "asset"))
            cls._key = key
        cls._inst.ensure()
        return cls._inst

    @classmethod
    def current(cls):
        return cls._inst

    @classmethod
    def close(cls):
        if cls._inst is not None:
            cls._inst.stop()
            cls._inst = None
            cls._key = None


import atexit  # noqa: E402

atexit.register(HostManager.close)
