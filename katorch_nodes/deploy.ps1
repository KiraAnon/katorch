# 把 katorch_nodes 推到手机并刷新 ComfyUI 的节点缓存。
# 用法： powershell -ExecutionPolicy Bypass -File F:\KAtorch\katorch_nodes\deploy.ps1 [-NoRestart]
#
# 两个坑（都踩过）：
#   1. 备份目录必须放在 custom_nodes 外面 —— ComfyUI 会把该目录下每个含 __init__.py
#      的子目录都当插件加载，备份留在原地会重复注册节点，旧备份还会覆盖新节点。
#   2. 不能在 adb shell 里直接 `pkill -f "main.py --cpu"` —— 那条命令行自己就含这个串，
#      会把执行者自己杀掉。所以重启走设备上的脚本文件（cmdline 里不带该模式）。

param(
    [switch]$NoRestart,
    [switch]$SkipBackup
)

$ErrorActionPreference = 'Continue'
# adb 位置：优先环境变量 KA_ADB，其次本机旧路径，最后退到 PATH 上的 adb
$ADB = 'adb'
if ($env:KA_ADB) { $ADB = $env:KA_ADB }
elseif (Test-Path 'D:\comfyui\platform-tools\adb.exe') { $ADB = 'D:\comfyui\platform-tools\adb.exe' }
$SRC = Join-Path $PSScriptRoot 'katorch_npu'
$DST = '/data/data/com.termux/files/home/ComfyUI/custom_nodes/katorch_npu'
$BAK = '/data/data/com.termux/files/home/npt/node_backups'
$STAGE = '/data/local/tmp/ka_nodes_stage'
$OWNER = 'u0_a171:u0_a171'
$RESTART = '/data/data/com.termux/files/home/npt/ka_restart.sh'

function Sh([string]$cmd) {
    $q = $cmd.Replace("'", "'\''")
    $out = & $ADB shell "su -c '$q'" 2>&1
    return ($out -join "`n")
}

Write-Host '== 检查设备'
$dev = (& $ADB devices) -join "`n"
if ($dev -notmatch 'device') { throw "没有 adb 设备：$dev" }

Write-Host '== 上传到暂存区'
Sh "rm -rf $STAGE; mkdir -p $STAGE/katorch_npu; chmod -R 777 $STAGE" | Out-Null
$files = Get-ChildItem -Path $SRC -File -Filter *.py
foreach ($f in $files) {
    $r = & $ADB push $f.FullName "$STAGE/katorch_npu/$($f.Name)" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "push 失败：$($f.Name) -> $r" }
}
Write-Host ("   推送 {0} 个 .py" -f $files.Count)

if (-not $SkipBackup) {
    Write-Host '== 备份旧版本（放 custom_nodes 外面）'
    Write-Host (Sh "mkdir -p $BAK; if [ -d $DST ]; then B=$BAK/katorch_npu.`$(date +%Y%m%d-%H%M%S); cp -r $DST `$B; echo backup=`$B; fi")
}

Write-Host '== 原子替换'
Sh "rm -rf $DST && mkdir -p $DST && cp -f $STAGE/katorch_npu/*.py $DST/ && chown -R $OWNER $DST && chmod 666 $DST/*.py && rm -rf $STAGE" | Out-Null
Write-Host (Sh "ls -l $DST | head -20")

Write-Host '== 清模块缓存'
Sh "rm -rf $DST/__pycache__; rm -f /data/data/com.termux/files/home/ComfyUI/custom_nodes/__pycache__/katorch_npu*.pyc" | Out-Null

if (-not $NoRestart) {
    Write-Host '== 重启 ComfyUI'
    # 设备端重启脚本不存在就补一份
    $exists = (Sh "test -f $RESTART && echo yes || echo no")
    if ($exists -notmatch 'yes') {
        $body = "#!/system/bin/sh`npkill -9 -f `"main.py --cpu`" 2>/dev/null`nsleep 2`npkill -9 host_new 2>/dev/null`nsleep 1`nnohup setsid sh /data/data/com.termux/files/home/npt/comfy-launch-root.sh > /data/data/com.termux/files/home/npt/comfy.log 2>&1 < /dev/null &`necho restart issued`n"
        $tmp = Join-Path $env:TEMP 'ka_restart.sh'
        [System.IO.File]::WriteAllText($tmp, $body, (New-Object System.Text.UTF8Encoding($false)))
        & $ADB push $tmp '/data/local/tmp/ka_restart.sh' 2>&1 | Out-Null
        Sh "cp -f /data/local/tmp/ka_restart.sh $RESTART && chmod 755 $RESTART" | Out-Null
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
    Write-Host (Sh "sh $RESTART")
    Start-Sleep -Seconds 18
    Write-Host '--- 启动日志（只挑关键行）'
    # 注意：不要在设备端写 grep -E "a|b" —— su -c 会吃掉引号，管道符号会被当成 shell 管道。
    # 拉回本地再过滤最稳。
    $log = Sh 'tail -80 /data/data/com.termux/files/home/npt/comfy.log'
    ($log -split "`n") | Where-Object { $_ -match 'katorch|Starting server|IMPORT FAILED' } |
        ForEach-Object { Write-Host ('    ' + $_.Trim()) }
} else {
    Write-Host '（跳过重启，节点不会生效，需要手动重启 ComfyUI）'
}
Write-Host '== 完成'
