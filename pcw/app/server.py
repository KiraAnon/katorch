"""KAtorch NPU 生图 App —— 本地 HTTP 服务 + 手机端 GUI。

启动: python server.py [端口]
访问: http://127.0.0.1:8770
"""
import io
import json
import os
import shutil
import sys
import threading
import time
import traceback

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
HOME = "/data/data/com.termux/files/home"
WORK = os.environ.get("KA_WORK", HOME + "/npt")
OUT = WORK + "/outputs"
os.makedirs(OUT, exist_ok=True)

STATE = {"busy": False, "step": 0, "total": 8, "msg": "空闲", "image": None,
         "prompt": "", "t_start": 0.0, "last_sec": 0.0, "error": None}
LOCK = threading.Lock()

ENC = None
PIPE = None

LAST_REQ = time.time()
IDLE_TIMEOUT = int(os.environ.get("KA_IDLE_TIMEOUT", "120"))


def _idle_watch():
    """页面关掉后前端就不再轮询了，这里等够时间就把服务和模型一起释放。"""
    while True:
        time.sleep(15)
        with LOCK:
            busy = STATE["busy"]
        if busy:
            continue
        idle = time.time() - LAST_REQ
        if idle > IDLE_TIMEOUT:
            print("[app] 空闲 %ds，退出并释放内存" % idle, flush=True)
            os._exit(0)


def _lazy():
    global ENC, PIPE
    if ENC is None:
        from enc import Encoder
        mdir = os.path.join(APP, "models")
        ENC = Encoder(
            qwen_st=os.path.join(mdir, "qwen3_06b.safetensors"),
            anima_st=os.environ.get("KA_ANIMA",
                                    HOME + "/ComfyUI/models/diffusion_models/anima_v1.safetensors"),
            tok_dir=mdir,
        )
    if PIPE is None:
        from gen import Pipeline
        PIPE = Pipeline()
    return ENC, PIPE


def _worker(prompt, seed):
    global STATE
    try:
        with LOCK:
            STATE.update({"busy": True, "step": 0, "msg": "加载模型…",
                          "prompt": prompt, "error": None, "image": None,
                          "t_start": time.time()})
        enc, pipe = _lazy()
        with LOCK:
            STATE["msg"] = "编码提示词…"
        ctx = enc.encode(prompt)
        with LOCK:
            STATE["msg"] = "采样中"

        def prog(step, total, el):
            with LOCK:
                STATE["step"] = step
                STATE["total"] = total
                STATE["last_sec"] = el

        x = pipe.generate(ctx, seed=seed, progress=prog)
        with LOCK:
            STATE["msg"] = "VAE 解码…"
        name = "img_%d.png" % int(time.time())
        path = os.path.join(OUT, name)
        path = pipe.decode(x, path)
        # 记下 prompt/seed，画廊里能对上号
        try:
            with LOCK:
                _rec = {"prompt": prompt, "seed": seed,
                        "time": time.strftime("%m-%d %H:%M")}
            idx_p = os.path.join(OUT, "index.json")
            idx = {}
            if os.path.exists(idx_p):
                try:
                    idx = json.load(open(idx_p))
                except Exception:
                    idx = {}
            idx[os.path.basename(path)] = _rec
            json.dump(idx, open(idx_p, "w"), ensure_ascii=False)
        except Exception as ie:
            print("[app] 写索引失败: %s" % ie, flush=True)
        # 顺手存一份到手机相册，方便直接看/分享
        try:
            gallery = "/sdcard/Pictures/KAtorch"
            os.makedirs(gallery, exist_ok=True)
            dst = os.path.join(gallery, name)
            shutil.copy(path, dst)
            os.system("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
                      "-d file://%s >/dev/null 2>&1" % dst)
            print("[app] 已存相册 -> %s" % dst, flush=True)
        except Exception as ge:
            print("[app] 存相册失败（不影响出图）: %s" % ge, flush=True)
        with LOCK:
            STATE.update({"busy": False, "msg": "完成", "image": os.path.basename(path),
                          "elapsed": time.time() - STATE["t_start"]})
        print("[app] 完成 -> %s  %.0fs" % (path, time.time() - STATE["t_start"]), flush=True)
    except Exception as e:
        traceback.print_exc()
        with LOCK:
            STATE.update({"busy": False, "msg": "出错", "error": str(e)[:400]})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        global LAST_REQ
        LAST_REQ = time.time()
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            fp = os.path.join(APP, "web", "index.html")
            if os.path.exists(fp):
                return self._send(200, open(fp, "rb").read(), "text/html; charset=utf-8")
            return self._send(404, "no ui")
        if p == "/sw.js":
            fp = os.path.join(APP, "web", "sw.js")
            if os.path.exists(fp):
                return self._send(200, open(fp, "rb").read(), "application/javascript")
            return self._send(404, "no sw")
        if p == "/manifest.json":
            fp = os.path.join(APP, "web", "manifest.json")
            if os.path.exists(fp):
                return self._send(200, open(fp, "rb").read(), "application/manifest+json")
            return self._send(404, "no manifest")
        if p == "/api/status":
            with LOCK:
                return self._send(200, json.dumps(STATE))
        if p.startswith("/img/"):
            name = os.path.basename(p[5:])
            fp = os.path.join(OUT, name)
            if os.path.exists(fp):
                ext = os.path.splitext(name)[1].lower()
                ct = "image/png" if ext == ".png" else "image/x-portable-pixmap"
                return self._send(200, open(fp, "rb").read(), ct)
            return self._send(404, "no img")
        if p == "/api/gallery":
            files = sorted([f for f in os.listdir(OUT)
                            if f.lower().endswith((".png", ".ppm"))],
                           key=lambda f: -os.path.getmtime(os.path.join(OUT, f)))[:24]
            idx = {}
            fp = os.path.join(OUT, "index.json")
            if os.path.exists(fp):
                try:
                    idx = json.load(open(fp))
                except Exception:
                    idx = {}
            out = [{"f": f, **(idx.get(f) or {})} for f in files]
            return self._send(200, json.dumps(out, ensure_ascii=False))
        return self._send(404, "?")

    def do_POST(self):
        global LAST_REQ
        LAST_REQ = time.time()
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            d = json.loads(raw or b"{}")
        except Exception:
            d = {}
        if p == "/api/generate":
            with LOCK:
                if STATE["busy"]:
                    return self._send(409, json.dumps({"ok": False, "err": "正在生成中"}))
            prompt = (d.get("prompt") or "").strip()
            if not prompt:
                return self._send(400, json.dumps({"ok": False, "err": "提示词为空"}))
            seed = int(d.get("seed") or 42)
            threading.Thread(target=_worker, args=(prompt, seed), daemon=True).start()
            return self._send(200, json.dumps({"ok": True}))
        return self._send(404, "?")


if __name__ == "__main__":
    print("KAtorch NPU App  端口 %d" % PORT, flush=True)
    print("打开 http://127.0.0.1:%d" % PORT, flush=True)
    threading.Thread(target=_idle_watch, daemon=True).start()
    print("空闲 %ds 自动退出（页面关掉后释放内存）" % IDLE_TIMEOUT, flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    srv.serve_forever()
