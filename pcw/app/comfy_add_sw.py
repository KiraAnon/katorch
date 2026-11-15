"""给 ComfyUI 前端补一个 service worker，让 Chrome 能「添加到主屏幕」变全屏。"""
import os, shutil

W = "/data/data/com.termux/files/usr/lib/python3.14/site-packages/comfyui_frontend_package/static"
SW = os.path.join(W, "sw.js")
IDX = os.path.join(W, "index.html")

# 1) 写 sw.js
sw_src = """// ComfyUI 最小 service worker —— 让 Chrome 认它是可安装 PWA（全屏独立窗口）
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.workers ? self.clients.claim() : self.clients.claim());
self.addEventListener('fetch', e => {
  // 纯网络优先：ComfyUI 是本地服务，资源都从 127.0.0.1 实时取
});
"""
if not os.path.exists(SW):
    open(SW, "w").write(sw_src)
    os.chmod(SW, 0o666)
    print("已创建 sw.js")
else:
    print("sw.js 已存在")

# 2) 在 index.html 里注册（备份原文件）
BAK = IDX + ".bak-katorch"
if not os.path.exists(BAK):
    shutil.copy(IDX, BAK)
    print("已备份 index.html -> index.html.bak-katorch")

html = open(IDX, encoding="utf-8", errors="replace").read()
if "serviceWorker" in html:
    print("index.html 已注册过 SW，跳过")
else:
    reg = ("<script>if('serviceWorker' in navigator){"
           "navigator.serviceWorker.register('/sw.js').catch(()=>{});}</script>")
    html = html.replace("</body>", reg + "</body>", 1)
    open(IDX, "w", encoding="utf-8").write(html)
    print("已在 index.html 注入 SW 注册代码")

print()
print("sw.js:", os.path.getsize(SW), "字节")
print("index.html:", os.path.getsize(IDX), "字节")
