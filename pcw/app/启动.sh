#!/system/bin/sh
# KAtorch NPU 生图 —— 一键启动
APP=/data/data/com.termux/files/home/npt/app
LOG=/data/local/tmp/app.log
PORT=8770

if ! pgrep -f "python server.py" > /dev/null 2>&1; then
  cd $APP || exit 1
  export PYTHONUNBUFFERED=1
  setsid /data/data/com.termux/files/usr/bin/python server.py $PORT > $LOG 2>&1 < /dev/null &
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    if grep -q "端口" $LOG 2>/dev/null; then break; fi
  done
fi

URL="http://127.0.0.1:$PORT"
if command -v termux-open-url > /dev/null 2>&1; then
  termux-open-url "$URL"
else
  am start -a android.intent.action.VIEW -d "$URL" > /dev/null 2>&1
fi
echo "已启动: $URL"
