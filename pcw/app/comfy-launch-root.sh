#!/system/bin/sh
# ComfyUI 启动入口（root 身份，由 Termux 通过 su -c 调用）
P=/data/data/com.termux/files/usr
H=/data/data/com.termux/files/home
export PREFIX=$P
export PATH=$P/bin:/system/bin
export HOME=$H
export TMPDIR=$P/tmp
export LD_PRELOAD=$P/lib/libpython3.14.so
export PYTHONUNBUFFERED=1

# 注意：ComfyUI 的命令行是相对路径「python main.py」，所以不能匹配 "ComfyUI/main.py"；
# 也不能只看 TCP 端口（TIME_WAIT 残留会误判成"已在运行"）。
if pgrep -f "main.py --cpu" > /dev/null 2>&1; then
  echo "ComfyUI 已在运行"
  exit 0
fi

cd $H/ComfyUI || exit 1
exec $P/bin/python main.py --cpu --listen 0.0.0.0 --port 8188
