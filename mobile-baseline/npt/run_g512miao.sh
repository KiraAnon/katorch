#!/system/bin/sh
# 用 miaomiaoHarem_anima12 生成 512 的 16 张图（后台跑，日志在 ~/npt/g512miao.log）
H=/data/data/com.termux/files/home
P=/data/data/com.termux/files/usr
export HOME=$H
export PATH=$P/bin:/system/bin
export LD_PRELOAD=$P/lib/libpython3.14.so
export TMPDIR=$P/tmp
export KA_ANIMA=$H/ComfyUI/models/diffusion_models/miaomiaoHarem_anima12.safetensors
export KA_MERGE_LORA=$H/ComfyUI/models/loras/turbo.safetensors
export KA_LORA_STR=0.6
export KA_DST=$H/npt/e2eper512miao
export KA_SCRATCH=/data/local/tmp/gscratchmiao
cd $H || exit 1
nohup setsid $P/bin/python $H/npt/g512_param.py > $H/npt/g512miao.log 2>&1 < /dev/null &
echo "started pid=$!"