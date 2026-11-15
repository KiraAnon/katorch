#!/data/data/com.termux/files/usr/bin/bash
# 启动 ComfyUI（原版界面）
P=/data/data/com.termux/files/usr
H=/data/data/com.termux/files/home
export PREFIX=$P
export PATH=$P/bin:/system/bin
export HOME=$H
export TMPDIR=$P/tmp
export LD_PRELOAD=$P/lib/libpython3.14.so

cd $H/ComfyUI || exit 1
exec $P/bin/python main.py --cpu --listen 0.0.0.0 --port 8188
