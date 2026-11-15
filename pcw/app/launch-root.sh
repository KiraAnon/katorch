#!/system/bin/sh
# root 身份下的启动入口（由 Termux 通过 su -c 调用，内部不再嵌套 su）
cd /data/data/com.termux/files/home/npt/app || exit 1
exec setsid /data/data/com.termux/files/usr/bin/python server.py 8770 \
  >> /data/data/com.termux/files/home/npt/app.log 2>&1 < /dev/null
