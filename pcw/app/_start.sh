cd /data/data/com.termux/files/home/npt/app || exit 1
pkill -f "python server.py" 2>/dev/null
sleep 1
export PYTHONUNBUFFERED=1
setsid /data/data/com.termux/files/usr/bin/python server.py 8770 > /data/local/tmp/app.log 2>&1 < /dev/null &
sleep 4
echo started