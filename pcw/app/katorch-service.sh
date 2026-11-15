#!/system/bin/sh
# KAtorch NPU 生图服务 —— Magisk 开机自启 + 守护
# 放在 /data/adb/service.d/ 下，开机时由 Magisk 以 root 执行。
# 服务常驻后，手机桌面上点「NPU生图」图标就直接是界面，不用再手动跑任何命令。

(
  LOG=/data/local/tmp/katorch_boot.log
  APPDIR=/data/data/com.termux/files/home/npt/app
  PREFIX=/data/data/com.termux/files/usr
  TU=$(stat -c %u /data/data/com.termux)
  PORT=8770

  echo "=== $(date '+%F %T') 开机自启 ===" >> "$LOG"

  # 等系统真正起来（Termux 的运行时目录、网络栈都就绪）
  sleep 45

  while true; do
    if ! pgrep -f "server.py $PORT" > /dev/null 2>&1; then
      echo "$(date '+%F %T') 拉起服务" >> "$LOG"
      # 以 root 直接跑：termux 用户在后台 cgroup 里加载 3.9GB blob 会失败，
      # root 不受这个限制（已验证能出图）。umask 放开，产物别挡住 termux 用户读写。
      umask 000
      cd "$APPDIR" && setsid "$PREFIX/bin/python" server.py $PORT \
        > /data/data/com.termux/files/home/npt/app.log 2>&1 < /dev/null &
      cd /
      sleep 20
    fi
    sleep 30
  done
) >/dev/null 2>&1 &
