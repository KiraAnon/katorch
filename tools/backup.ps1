# KAtorch 备份：镜像 + 快照，双保险。
#
#   .\backup.ps1                 # 默认：镜像 + 快照
#   .\backup.ps1 -MirrorOnly     # 只做镜像（快，全量同步）
#   .\backup.ps1 -SnapshotOnly   # 只做快照（慢一点，留历史版本）
#   .\backup.ps1 -Keep 60        # 快照保留份数
#
# 两个层次的差别（很重要）：
#
#   镜像  robocopy /MIR —— 快，跟源完全一致。但它会**跟随删除**：
#         源里删掉的文件，下次镜像也在备份里消失。所以镜像不能当成"后悔药"。
#
#   快照  zip —— 每次生成一个带时间戳的包，**只增不删**。误删、改坏，
#         从历史快照里捞。为了不爆盘，快照默认排除 PC编译\wsl-sysroot
#         那种几百 MB 的厂商二进制（那份靠镜像保护）。
#
# 用法建议：跑一次 -MirrorOnly 做基线，然后定期跑默认（两个都做）。

param(
    [string]$Source = 'F:\KAtorch',
    [string]$Target = 'E:\KAtorch-backup',
    [string[]]$ExcludeFromSnapshot = @('PC编译\wsl-sysroot'),
    [int]$Keep = 30,
    [switch]$MirrorOnly,
    [switch]$SnapshotOnly
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[!] $m" -ForegroundColor Yellow }

if (-not (Test-Path $Source)) { throw "源目录不存在：$Source" }

# 备份必须落在不同的盘，否则一块盘坏了两份一起没
$srcDrive = (Resolve-Path $Source).Drive.Name
$dstDrive = (Split-Path -Qualifier $Target).TrimEnd(':')
if ($srcDrive -eq $dstDrive) {
    Warn "备份目标和源在同一个盘（${srcDrive}:）。磁盘故障时两份会一起丢，建议换一块盘。"
}

$mirrorDir = Join-Path $Target 'latest'
$snapDir = Join-Path $Target 'snapshots'
New-Item -ItemType Directory -Path $Target -Force | Out-Null

# ---------- 1. 镜像 ----------
function Invoke-Mirror {
    Step "镜像 $Source -> $mirrorDir"
    New-Item -ItemType Directory -Path $mirrorDir -Force | Out-Null
    $t0 = Get-Date
    robocopy $Source $mirrorDir /MIR /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { throw "robocopy 失败（退出码 $rc）" }
    $m = Get-ChildItem $mirrorDir -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum
    "    {0} 个文件 / {1:N1} MB，耗时 {2:N1}s" -f $m.Count, ($m.Sum/1MB), ((Get-Date) - $t0).TotalSeconds
}

# ---------- 2. 快照 ----------
function Invoke-Snapshot {
    New-Item -ItemType Directory -Path $snapDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $zip = Join-Path $snapDir ("katorch-$stamp.zip")
    $tmp = Join-Path $env:TEMP ("katorch-snap-$stamp")

    Step "快照（排除：$($ExcludeFromSnapshot -join ', ')）"
    # 先用 robocopy 挑出要打包的内容
    $robArgs = @($Source, $tmp, '/E', '/R:1', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS', '/NP')
    foreach ($e in $ExcludeFromSnapshot) { $robArgs += '/XD'; $robArgs += (Join-Path $Source $e) }
    robocopy @robArgs | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "快照暂存失败（robocopy $LASTEXITCODE）" }

    try {
        $m = Get-ChildItem $tmp -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum
        Step ("压缩 {0:N1} MB ..." -f ($m.Sum/1MB))
        if (Test-Path $zip) { Remove-Item $zip -Force }
        # entryNameEncoding 必须给 UTF8，否则中文路径在 Windows 自带解压里会乱码
        [System.IO.Compression.ZipFile]::CreateFromDirectory(
            $tmp, $zip,
            [System.IO.Compression.CompressionLevel]::Optimal,
            $false,
            [System.Text.Encoding]::UTF8)
        "    {0}  ({1:N1} MB)" -f (Split-Path $zip -Leaf), ((Get-Item $zip).Length/1MB)
    } finally {
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue }
    }

    # 只保留最近 $Keep 份
    $old = Get-ChildItem $snapDir -Filter 'katorch-*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -Skip $Keep
    if ($old) {
        $old | ForEach-Object { Remove-Item $_.FullName -Force }
        "    清理 {0} 份旧快照（保留最近 {1} 份）" -f $old.Count, $Keep
    }
}

if (-not $SnapshotOnly) { Invoke-Mirror }
if (-not $MirrorOnly) { Invoke-Snapshot }

Write-Host ''
Write-Host "[OK] 备份完成：$Target" -ForegroundColor Green
$snaps = Get-ChildItem $snapDir -Filter 'katorch-*.zip' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($snaps) {
    Write-Host "     最近快照：$($snaps[0].Name)  ($('{0:N1}' -f ($snaps[0].Length/1MB)) MB)"
    Write-Host "     快照总数：$($snaps.Count)"
}
exit 0
