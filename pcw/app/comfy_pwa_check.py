import urllib.request, json

for path in ("/assets/manifest-P9przInP.json", "/manifest.json", "/sw.js"):
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8188" + path, timeout=8)
        body = r.read()
        print("%-38s %d  %d bytes" % (path, r.status, len(body)))
    except Exception as e:
        print("%-38s ERR %s" % (path, e))

# 前端是否有 service worker 注册代码
try:
    html = urllib.request.urlopen("http://127.0.0.1:8188/", timeout=10).read().decode("utf-8", "replace")
    print()
    print("index.html 里 serviceWorker 字样:", "serviceWorker" in html)
    print("index.html 里 manifest href:", [l for l in html.split("\n") if "manifest" in l][:2])
except Exception as e:
    print("取 index 失败:", e)
