#!/system/bin/sh
cd /data/data/com.termux/files/home/npt/app || exit 1
export PYTHONUNBUFFERED=1
exec /data/data/com.termux/files/usr/bin/python server.py 8770
