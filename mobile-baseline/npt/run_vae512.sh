#!/system/bin/sh
H=/data/data/com.termux/files/home
P=/data/data/com.termux/files/usr
export HOME=$H
export PATH=$P/bin:/system/bin
export LD_PRELOAD=$P/lib/libpython3.14.so
export TMPDIR=$P/tmp
export KA_SEGS=${KA_SEGS:-1}
cd $H || exit 1
$P/bin/python $H/npt/vae_npu_512.py