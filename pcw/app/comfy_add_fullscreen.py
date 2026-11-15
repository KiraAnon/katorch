"""给 ComfyUI 页面注入一个悬浮全屏按钮。

浏览器的 Fullscreen API 必须由用户手势触发，所以不能自动进全屏；
这里放一个小按钮，点一下就进全屏（连地址栏和状态栏一起隐藏）。
"""
import os, shutil

W = "/data/data/com.termux/files/usr/lib/python3.14/site-packages/comfyui_frontend_package/static"
IDX = os.path.join(W, "index.html")

BAK = IDX + ".bak-katorch"
if not os.path.exists(BAK):
    shutil.copy(IDX, BAK)
    print("已备份 index.html")

html = open(IDX, encoding="utf-8", errors="replace").read()

if "katorch-fullscreen" in html:
    print("已注入过，跳过")
else:
    btn = """
<div id="katorch-fullscreen" onclick="
  (function(){
    var d=document.documentElement;
    if(!document.fullscreenElement){
      (d.requestFullscreen||d.webkitRequestFullscreen||d.mozRequestFullScreen||function(){}).call(d);
    }else{
      (document.exitFullscreen||document.webkitExitFullscreen||document.mozCancelFullScreen||function(){}).call(document);
    }
  })();
" style="position:fixed;right:10px;top:50%;transform:translateY(-50%);z-index:99999;
width:44px;height:44px;border-radius:22px;background:rgba(40,160,255,.85);
color:#fff;display:flex;align-items:center;justify-content:center;cursor:pointer;
font-size:20px;user-select:none;box-shadow:0 2px 8px rgba(0,0,0,.4)">&#x26F6;</div>
"""
    html = html.replace("</body>", btn + "</body>", 1)
    open(IDX, "w", encoding="utf-8").write(html)
    print("已注入全屏按钮")

print("index.html:", os.path.getsize(IDX), "字节")
