import urllib.request

for path in ("/sw.js", "/assets/manifest-P9przInP.json", "/"):
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8188" + path, timeout=10)
        b = r.read()
        print("%-38s %d  %d bytes" % (path, r.status, len(b)))
        if path == "/":
            txt = b.decode("utf-8", "replace")
            print("   serviceWorker 已注入:", "serviceWorker" in txt)
    except Exception as e:
        print("%-38s ERR %s" % (path, e))
