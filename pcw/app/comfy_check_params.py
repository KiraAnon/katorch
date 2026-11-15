import json, urllib.request

oi = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60))
ks = oi["KSampler"]["input"]["required"]
samplers = ks["sampler_name"][0]
scheds = ks["scheduler"][0]
print("sampler 里有 er_sde ?", "er_sde" in samplers)
print("scheduler 里有 beta ?", "beta" in scheds)
print()
print("er_sde 附近的采样器:", [s for s in samplers if "sde" in s.lower()][:8])
print("beta 附近的调度器:", [s for s in scheds if "beta" in s.lower() or "simple" in s.lower()][:8])

# LoraLoader 参数
ll = oi["LoraLoader"]["input"]["required"]
print()
print("LoraLoader 输入:", list(ll.keys()))
print("  可用 lora:", ll.get("lora_name", ["?"])[0][:6] if isinstance(ll.get("lora_name"), list) else "?")
