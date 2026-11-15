#!/system/bin/sh
# 重启 ComfyUI（节点包重新加载用）
# 注意：不能在 adb shell 里直接 pkill -f "main.py --cpu"，那条命令行自己就包含这个串，
#       会把执行者自己也杀掉。所以做成脚本文件，cmdline 里不带这个模式。
pkill -9 -f "main.py --cpu" 2>/dev/null
sleep 2
pkill -9 host_new 2>/dev/null
sleep 1
nohup setsid sh /data/data/com.termux/files/home/npt/comfy-launch-root.sh > /data/data/com.termux/files/home/npt/comfy.log 2>&1 < /dev/null &
echo "restart issued"