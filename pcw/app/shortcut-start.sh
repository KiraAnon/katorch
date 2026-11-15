#!/data/data/com.termux/files/usr/bin/sh
# KAtorch NPU 生图 —— 点一下启动
# 放在 ~/.shortcuts/ 下，用 Termux:Widget 放到桌面。

PREFIX=/data/data/com.termux/files/usr
export PATH=$PREFIX/bin:$PATH
export HOME=/data/data/com.termux/files/home

PORT=8770
PORT_URL="http://127.0.0.1:8770"
LAUNCH=$HOME/npt/launch-root.sh
TRACE=$HOME/npt/shortcut.log

# 埋点：确认点击真的触发了脚本
echo "$(date '+%F %T') 快捷方式被触发" >> "$TRACE"

termux-wake-lock 2>/dev/null

su -c "setsid sh $LAUNCH &"
echo "$(date '+%F %T') 已调用 launch-root" >> "$TRACE"

i=0
while [ $i -lt 20 ]; do
  sleep 1
  if su -c "grep -q ':2242 ' /proc/net/tcp" 2>/dev/null; then
    echo "$(date '+%F %T') 端口就绪（等了 ${i}s）" >> "$TRACE"
    break
  fi
  i=$((i + 1))
done

sleep 1
if command -v termux-open-url > /dev/null 2>&1; then
  termux-open-url "$PORT_URL"
  echo "$(date '+%F %T') 用 termux-open-url 打开" >> "$TRACE"
else
  # termux 用户跑 am 会被 SecurityException 挡住，得用 root
  su -c "am start -a android.intent.action.VIEW -d $PORT_URL" > /dev/null 2>&1
  echo "$(date '+%F %T') 用 am start 打开" >> "$TRACE"
fi
