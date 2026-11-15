#!/system/bin/sh
# 编译 miaomiaoHarem_anima12 的 512 blob（后台，日志 ~/npt/e512miao.log）
H=/data/data/com.termux/files/home
P=/data/data/com.termux/files/usr
export HOME=$H
export PATH=$P/bin:/system/bin
export LD_PRELOAD=$P/lib/libpython3.14.so
export TMPDIR=$P/tmp
export KA_DST=$H/npt/e2eper512miao
export KA_BLOBDIR=/data/local/tmp/blobs512miao
cd $H || exit 1
nohup setsid $P/bin/python $H/npt/e512_param.py > $H/npt/e512miao.log 2>&1 < /dev/null &
echo "started pid=$!"