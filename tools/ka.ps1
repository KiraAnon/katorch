# KAtorch 手机操作助手（adb + root shell）
# 用法： . F:\KAtorch\tools\ka.ps1   然后调用 Ka-Sh / Ka-Pull / Ka-Push / Ka-Put

$script:ADB = 'adb'
if ($env:KA_ADB) { $script:ADB = $env:KA_ADB }
elseif (Test-Path 'D:\comfyui\platform-tools\adb.exe') { $script:ADB = 'D:\comfyui\platform-tools\adb.exe' }
$script:HOME_TX = '/data/data/com.termux/files/home'

function Ka-AdbArgs { param([string[]]$a) return $a }

function Ka-Sh {
    <# 在设备上以 root 执行命令，返回 stdout+stderr 文本。
       注意：cmd 内的单引号会被转义；整条命令用单引号包住交给设备的 su -c，避免 adb 吃掉引号。 #>
    param([string]$cmd, [int]$timeoutSec = 300)
    $q = $cmd.Replace("'", "'\''")
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $script:ADB
    $psi.Arguments = 'shell "su -c ' + "'" + $q + "'" + '"'
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $p = [System.Diagnostics.Process]::Start($psi)
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($timeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        return "[TIMEOUT ${timeoutSec}s] $cmd"
    }
    $o = $outTask.Result
    $e = $errTask.Result
    if ($e -and $e.Trim()) { $o + "`n[stderr]`n" + $e }
    else { $o }
}

function Ka-Pull {
    <# 从设备（root 可读路径）按字节拉到本地，避免 PowerShell 5.1 的 UTF-16 重定向坑 #>
    param([string]$remote, [string]$local)
    $dir = Split-Path -Parent $local
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $script:ADB
    $psi.Arguments = 'exec-out "su -c ' + "'cat " + $remote.Replace("'", "'\''") + "'" + '"'
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $p = [System.Diagnostics.Process]::Start($psi)
    $fs = [System.IO.File]::Create($local)
    try { $p.StandardOutput.BaseStream.CopyTo($fs) } finally { $fs.Close() }
    $err = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    if ($err -and $err.Trim()) { return "ERR $remote : $err" }
    return (Get-Item $local).Length
}

function Ka-Push {
    <# 本地 -> 设备（先 push 到 /data/local/tmp，再 root cp 到目标） #>
    param([string]$local, [string]$remote, [string]$owner = 'u0_a171:u0_a171', [string]$mode = '666')
    if (-not (Test-Path $local)) { return "MISSING SOURCE $local" }
    $tmp = '/data/local/tmp/ka_push_' + [System.IO.Path]::GetFileName($local)
    $r = & $script:ADB push $local $tmp 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { return "PUSH FAILED: $r" }
    $cmd = "cp -f $tmp $remote && chown $owner $remote && chmod $mode $remote && rm -f $tmp && ls -l $remote"
    return (Ka-Sh $cmd)
}

function Ka-PutText {
    <# 文本 -> 设备文件（UTF-8 无 BOM），走临时文件 #>
    param([string]$text, [string]$remote, [string]$owner = 'u0_a171:u0_a171', [string]$mode = '666')
    $tmpLocal = Join-Path $env:TEMP ('ka_put_' + [guid]::NewGuid().ToString('N').Substring(0,8) + '.txt')
    [System.IO.File]::WriteAllText($tmpLocal, $text, (New-Object System.Text.UTF8Encoding($false)))
    $r = Ka-Push $tmpLocal $remote $owner $mode
    Remove-Item $tmpLocal -Force -ErrorAction SilentlyContinue
    return $r
}

function Ka-Run {
    <# 复杂命令（含引号、管道、重定向）走这个：先写成设备上的脚本再执行。
       Ka-Sh 直接用 su -c '...'，su 会再交给 sh 解析一次，命令里的引号和 | 会被吃掉。 #>
    param([string]$cmd, [int]$timeoutSec = 300)
    $tmpLocal = Join-Path $env:TEMP ('ka_run_' + [guid]::NewGuid().ToString('N').Substring(0,8) + '.sh')
    $body = "#!/system/bin/sh`n" + $cmd + "`n"
    [System.IO.File]::WriteAllText($tmpLocal, $body, (New-Object System.Text.UTF8Encoding($false)))
    & $script:ADB push $tmpLocal /data/local/tmp/ka_run.sh | Out-Null
    $remote = $script:HOME_TX + '/npt/_pentest/run.sh'
    $prep = 'mkdir -p ' + $script:HOME_TX + '/npt/_pentest; cp -f /data/local/tmp/ka_run.sh ' + $remote + '; chmod 755 ' + $remote
    Ka-Sh $prep | Out-Null
    Remove-Item $tmpLocal -Force -ErrorAction SilentlyContinue
    return (Ka-Sh ("sh " + $remote) $timeoutSec)
}

function Ka-Py {
    <# 在 Termux python 环境里跑一段 python 代码（root 身份） #>
    param([string]$code, [int]$timeoutSec = 600)
    $tmpLocal = Join-Path $env:TEMP ('ka_py_' + [guid]::NewGuid().ToString('N').Substring(0,8) + '.py')
    [System.IO.File]::WriteAllText($tmpLocal, $code, (New-Object System.Text.UTF8Encoding($false)))
    # 放到 Termux 家目录下的干净子目录，避免 /data/local/tmp 里的同名 .py 抢走标准库 import
    & $script:ADB push $tmpLocal /data/local/tmp/ka_tmp_run.py | Out-Null
    Remove-Item $tmpLocal -Force -ErrorAction SilentlyContinue
    $remoteRun = $script:HOME_TX + '/npt/_pentest/run.py'
    $cmd = 'mkdir -p ' + $script:HOME_TX + '/npt/_pentest; cp /data/local/tmp/ka_tmp_run.py ' + $remoteRun + '; ' +
           'export HOME=' + $script:HOME_TX + '; export TMPDIR=/data/data/com.termux/files/usr/tmp; ' +
           'export LD_PRELOAD=/data/data/com.termux/files/usr/lib/libpython3.14.so; ' +
           'export PATH=/data/data/com.termux/files/usr/bin:/system/bin; ' +
           'cd ' + $script:HOME_TX + '; /data/data/com.termux/files/usr/bin/python ' + $remoteRun
    return (Ka-Sh $cmd $timeoutSec)
}

Write-Output '[ka] helpers loaded: Ka-Sh / Ka-Pull / Ka-Push / Ka-PutText / Ka-Py'
