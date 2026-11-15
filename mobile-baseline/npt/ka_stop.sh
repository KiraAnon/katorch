#!/system/bin/sh
# 关掉 ComfyUI + NPU 宿主（释放约 5GB 内存）
# 注意：不能在 adb shell 里直接写 pkill -f "main.py --cpu"，那条命令行自己就带这个串，
#       会把执行者一起杀掉，所以做成脚本文件。
echo "关闭 ComfyUI ..."
pkill -9 -f "main.py --cpu" 2>/dev/null
sleep 2
echo "关闭 NPU 宿主 ..."
pkill -9 host_new 2>/dev/null
sleep 1
LEFT=$(ps -A | grep -E "python|host_new" | grep -v grep | wc -l)
if [ "$LEFT" -gt 0 ]; then
  echo "还有 $LEFT 个残留："
  ps -A | grep -E "python|host_new" | grep -v grep
else
  echo "已全部关闭"
fi
free -m | head -2