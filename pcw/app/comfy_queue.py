import json, urllib.request, time

API = "http://127.0.0.1:8188"
st = json.load(urllib.request.urlopen(API + "/queue", timeout=15))
print("运行中:", len(st.get("queue_running", [])), " 排队:", len(st.get("queue_pending", [])))

# 看历史
h = json.load(urllib.request.urlopen(API + "/history", timeout=20))
print("历史任务:", len(h))
for pid, rec in list(h.items())[-2:]:
    status = rec.get("status", {})
    print("  %s  status=%s completed=%s" % (pid[:8], status.get("status_str"), status.get("completed")))
    outs = rec.get("outputs", {})
    for nid, o in outs.items():
        if "images" in o:
            for im in o["images"]:
                print("     图片: %s" % im.get("filename"))
