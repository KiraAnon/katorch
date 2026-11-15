# KAtorch 电脑 ↔ 手机 同步工具
#
#   .\sync.ps1 pull    手机 → 电脑（在手机上改过东西之后）
#   .\sync.ps1 push    电脑 → 手机（改完源码之后）
#   .\sync.ps1 diff    只列差异，不改文件
#
# 只同步「源文件」。.dla blob 是产物，必须用手机上的 Neuron 生成，不在这里同步。

param(
    [Parameter(Position = 0)]
    [ValidateSet('pull', 'push', 'diff')]
    [string]$Action = 'diff'
)

$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\ka.ps1" | Out-Null

$ADB = 'adb'
if ($env:KA_ADB) { $ADB = $env:KA_ADB }
elseif (Test-Path 'D:\comfyui\platform-tools\adb.exe') { $ADB = 'D:\comfyui\platform-tools\adb.exe' }
# 仓库根 = 本脚本所在目录的上一级
$ROOT = $env:KA_ROOT
if (-not $ROOT) { $ROOT = Split-Path $PSScriptRoot -Parent }
$PHONE_NODES = '/data/data/com.termux/files/home/ComfyUI/custom_nodes/katorch_npu'
$PHONE_NPT = '/data/data/com.termux/files/home/npt'

# --- 同步项：本地路径 → 手机路径 ---
$ITEMS = @(
    @{ Local = "$ROOT\katorch_nodes\katorch_npu"; Phone = $PHONE_NODES
       Filter = '*.py' }
    @{ Local = "$ROOT\pcw"; Phone = $PHONE_NPT
       Filter = 'vae_*.py' }
    @{ Local = "$ROOT\mobile-baseline\npt"; Phone = $PHONE_NPT
       Filter = '*.py' }
    @{ Local = "$ROOT\mobile-baseline\npt"; Phone = $PHONE_NPT
       Filter = '*.sh' }
)

function Get-PhoneMd5([string]$phonePath, [string]$filter) {
    $pat = $filter -replace '^\*', ''
    $script = "cd $phonePath 2>/dev/null || exit 0`nfor f in *$pat; do [ -f `"`$f`" ] && printf `"%s %s\n`" `"`$(md5sum `"`$f`" | cut -d' ' -f1)`" `"`$f`"; done"
    $tmp = '/data/local/tmp/_sync_md5.sh'
    Ka-PutText $script $tmp 'root:root' '755' | Out-Null
    $out = Ka-Sh "sh $tmp"
    $map = @{}
    foreach ($line in ($out -split "`r?`n")) {
        if ($line -match '^([0-9a-f]{32})\s+(\S+)$') { $map[$matches[2]] = $matches[1] }
    }
    return $map
}

$plan = @()      # 每项: Local / Phone / Name / State
foreach ($it in $ITEMS) {
    if (-not (Test-Path $it.Local)) { continue }
    $pm = Get-PhoneMd5 $it.Phone $it.Filter
    Get-ChildItem $it.Local -Filter $it.Filter -File | ForEach-Object {
        $h = (Get-FileHash $_.FullName -Algorithm MD5).Hash.ToLower()
        $ph = $pm[$_.Name]
        $state = if (-not $ph) { 'phone-missing' } elseif ($ph -eq $h) { 'same' } else { 'differ' }
        $plan += [pscustomobject]@{
            Name = $_.Name; Local = $_.FullName
            Phone = "$($it.Phone)/$($_.Name)"; State = $state
        }
    }
}


$diff = $plan | Where-Object { $_.State -ne 'same' }
Write-Host ("共 {0} 个文件，{1} 个有差异" -f $plan.Count, $diff.Count)
if ($diff.Count -gt 0) {
    $diff | Sort-Object Name | ForEach-Object {
        Write-Host ("  {0,-24} {1}" -f $_.Name, $_.State)
    }
}

switch ($Action) {
    'diff' { break }
    'pull' {
        foreach ($d in ($diff | Where-Object { $_.State -ne 'phone-missing' })) {
            $dir = Split-Path $d.Local -Parent
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
            Ka-Pull $d.Phone $d.Local | Out-Null
            Write-Host "  ← $($d.Name)"
        }
        Write-Host '已从手机拉取'
    }
    'push' {
        foreach ($d in ($diff | Where-Object { $_.State -ne 'phone-missing' })) {
            Ka-Push $d.Local $d.Phone | Out-Null
            Write-Host "  → $($d.Name)"
        }
        # 推送后修一下属主，否则 ComfyUI 可能读不了
        Ka-Sh "chown -R u0_a171:u0_a171 $PHONE_NODES $PHONE_NPT/*.py $PHONE_NPT/*.sh 2>/dev/null; echo ok" | Out-Null
        Write-Host '已推送到手机，重启后端生效：'
        Write-Host '  adb shell "su -t 1 -c ''sh /data/data/com.termux/files/home/npt/ka_restart.sh''"'
    }
}
