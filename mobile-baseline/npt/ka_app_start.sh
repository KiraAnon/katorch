#!/system/bin/sh
# 拉起 ComfyUI（app 打开时调用）
# 注意：不要用 netstat 判断「已在运行」—— 上次连接留下的 TIME_WAIT 会被 grep 8188 误命中，
# 导致脚本以为后端活着直接返回，app 就永远卡在启动画面。用进程判断才准。
if pgrep -f "main.py --cpu" > /dev/null 2>&1; then
  echo already
  exit 0
fi
nohup setsid sh /data/data/com.termux/files/home/npt/comfy-launch-root.sh > /data/data/com.termux/files/home/npt/comfy.log 2>&1 < /dev/null &
echo started