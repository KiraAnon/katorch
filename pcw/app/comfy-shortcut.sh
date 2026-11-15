#!/data/data/com.termux/files/usr/bin/sh
# ComfyUI 原版界面 —— 点一下启动并打开
# 放在 ~/.shortcuts/ 下，用 Termux:Widget 放到桌面。
# 注意：Termux 本体要先在后台跑着，Widget 才能把点击转过来。

PREFIX=/data/data/com.termux/files/usr
export PATH=$PREFIX/bin:$PATH
export HOME=/data/data/com.termux/files/home

PORT=8188
URL="http://127.0.0.1:8188"
LAUNCH=$HOME/npt/comfy-launch-root.sh
TRACE=$HOME/npt/shortcut.log

echo "$(date '+%F %T') ComfyUI 快捷方式触发" >> "$TRACE"

termux-wake-lock 2>/dev/null

# ComfyUI 以 root 跑（termux 用户后台加载大模型会被回收）
su -c "setsid sh $LAUNCH >> $HOME/npt/comfy.log 2>&1 < /dev/null &"

# 等 8188 起来（首次启动要加载不少东西，给足时间）
i=0
while [ $i -lt 60 ]; do
  sleep 1
  if su -c "grep -q ':1FFC ' /proc/net/tcp" 2>/dev/null; then
    echo "$(date '+%F %T') 端口就绪（${i}s）" >> "$TRACE"
    break
  fi
  i=$((i + 1))
done

sleep 1
if command -v termux-open-url > /dev/null 2>&1; then
  termux-open-url "$URL"
else
  su -c "am start -a android.intent.action.VIEW -d $URL" > /dev/null 2>&1
fi
echo "$(date '+%F %T') 已打开 $URL" >> "$TRACE"
