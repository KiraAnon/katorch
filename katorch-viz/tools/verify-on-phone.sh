#!/data/data/com.termux/files/usr/bin/bash
# 需求 §8 测试 2「能编译」——真机那一半。
#
# 这台 Windows 机器上没有 MediaTek NPU 的适配器库（libneuron），host.c 编译不出来也跑不起来，
# 所以编译这一步只能在手机（Termux）上验。本脚本把编辑器导出的 g.txt 喂给 host：
#
#   ./verify-on-phone.sh /data/local/tmp/e2e_dbg/b00/g.txt ./host
#
# 期望输出：
#   {"ok":true,"devices":["mtk-gpu","mtk-dsp","mtk-mdla"],"ready":true}
#   {"ok":true,"ops":569,"slot":0}          ← build 成功
#   {"ok":true,"rc":0}                       ← compile 成功
#
# 如果 build 返回 -304，说明 operand id 不连续 —— 用编辑器的「自检」能看到具体是第几个 OPD。

set -u

G="${1:-}"
HOSTBIN="${2:-./host}"

if [ -z "$G" ]; then
  echo "用法: $0 <g.txt 的绝对路径> [host 可执行文件，默认 ./host]" >&2
  exit 2
fi
if [ ! -f "$G" ]; then
  echo "找不到图文件: $G" >&2
  exit 2
fi
if [ ! -x "$HOSTBIN" ]; then
  echo "找不到可执行的 host: $HOSTBIN" >&2
  echo "  先按 编译器/README.md 编一遍，或用 --host 指路径" >&2
  exit 2
fi

# 图里没有引号的话直接拼 JSON；有引号就报错让用户自己看
if grep -q '"' "$G"; then
  echo "图里出现了双引号，脚本不处理这种路径，请手工构造 build 请求。" >&2
  exit 2
fi

echo "== 图: $G"
echo "== host: $HOSTBIN"
echo
echo "--- 结构预检（不依赖 NPU）---"
# operand id 必须是 0..N-1 且按出现顺序递增：M_addOp 按出现顺序编号，M_setVal 用文件里的 id
awk '
  $1=="OPD" {
    n++
    if ($2 != n-1) { printf("  ✗ 第 %d 个 OPD 写的是 id=%s，应为 %d → 会返回 -304\n", n, $2, n-1); bad++ }
  }
  $1=="OP" { ops++ }
  $1=="IO" { hasio=1 }
  $1=="END" { hasend=1 }
  END {
    printf("  operand %d 个 / 算子 %d 个\n", n, ops)
    if (!hasio)  { print "  ✗ 缺少 IO 行 → 会返回 -107"; bad++ }
    if (!hasend){ print "  ! 缺少 END 行（不致命）" }
    if (bad)     { print "  预检不通过，先回编辑器修"; exit 1 }
    print "  ✓ id 连续、IO 行齐全"
  }
' "$G" || exit 1

echo
echo "--- 交给 host 编译 ---"
printf '{"cmd":"build","path":"%s"}\n{"cmd":"compile"}\n{"cmd":"quit"}\n' "$G" | "$HOSTBIN"

echo
echo "判读："
echo '  {"ok":true,"ops":N,"slot":0}  → build 通过'
echo '  {"ok":true,"rc":0}            → compile 通过'
echo '  {"ok":false,"err":"build","rc":-304} → operand id 不连续（回编辑器点「自检」）'
echo '  其它负数 rc 见 编译器/host.c 里 build_from_file() 的返回码注释'
