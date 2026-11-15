import json, urllib.request

API = "http://127.0.0.1:8188"
# 停掉当前任务
try:
    req = urllib.request.Request(API + "/interrupt", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    print("已中断当前任务")
except Exception as e:
    print("中断失败:", e)

# 清空队列
try:
    req = urllib.request.Request(API + "/queue", data=json.dumps({"clear": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    print("已清空队列")
except Exception as e:
    print("清空失败:", e)
