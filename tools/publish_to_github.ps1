# 一条命令把当前目录发布到 GitHub。
#
#   .\publish_to_github.ps1                    # 用目录名当仓库名，公开
#   .\publish_to_github.ps1 -Private           # 私有仓库
#   .\publish_to_github.ps1 -Repo katorch      # 指定仓库名
#   .\publish_to_github.ps1 -DryRun            # 只看会做什么
#
# 不需要 gh，也不需要手动去网页建仓库：
# token 从 git 的凭据管理器里取（`git credential fill`），再直接调 GitHub API。
# 前提是你曾经用浏览器登录过 GitHub（gh auth login 或 push 时输入过 PAT），
# 凭据管理器里存着 token。
#
# 用法：先 cd 到仓库根目录，再运行本脚本。

param(
    [string]$Repo = '',
    [string]$Description = '',
    [switch]$Private,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Fail($msg) { Write-Host "[x] $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# ---------- 0. 基本检查 ----------
if (-not (Test-Path '.git')) { Fail "当前目录不是 git 仓库：$(Get-Location)" }

$branch = git rev-parse --abbrev-ref HEAD
if ($LASTEXITCODE -ne 0) { Fail 'git 状态异常' }

$dirty = git status --porcelain
if ($dirty) {
    Write-Host '[!] 工作区有未提交改动：' -ForegroundColor Yellow
    $dirty | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
    Write-Host '    先 commit 再发布（本脚本只推送已提交的内容）' -ForegroundColor Yellow
    exit 2
}

if (-not $Repo) { $Repo = Split-Path (Get-Location) -Leaf }
Step "仓库：$Repo   分支：$branch   待推文件：$((git ls-files).Count)"

# ---------- 1. 取 token（不打印内容） ----------
$env:GIT_TERMINAL_PROMPT = '0'
$credInput = "protocol=https`nhost=github.com`n`n"
$cred = $credInput | & git credential fill 2>$null
$token = ($cred | Where-Object { $_ -like 'password=*' }) -replace '^password=', ''
if (-not $token) {
    Fail '取不到 GitHub 凭据。先在浏览器登录一次（或用 PAT 推一次），让凭据管理器记住它。'
}
Step "已取得凭据（长度 $($token.Length)）"

$headers = @{
    Authorization = "token $token"
    Accept        = 'application/vnd.github+json'
    'User-Agent'  = 'publish-to-github'
}

# ---------- 2. 查当前账号 ----------
try {
    $me = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers -Method Get
} catch {
    Fail "token 无效或过期：$($_.Exception.Message)"
}
$owner = $me.login
Step "账号：$owner"

# ---------- 3. 仓库是否存在 ----------
$exists = $false
try {
    $null = Invoke-RestMethod -Uri "https://api.github.com/repos/$owner/$Repo" -Headers $headers -Method Get
    $exists = $true
} catch {
    $code = $null
    try { $code = $_.Exception.Response.StatusCode.value__ } catch {}
    if ($code -ne 404) { Write-Host "[!] 查询返回 HTTP $code，继续尝试创建" -ForegroundColor Yellow }
}

if ($exists) {
    Step "远程仓库已存在，跳过创建"
} elseif ($DryRun) {
    Step "[DryRun] 将创建 https://github.com/$owner/$Repo（private=$([bool]$Private)）"
    Step "[DryRun] 然后 git push -u origin $branch"
    exit 0
} else {
    if (-not $Description) { $Description = "$Repo" }
    $body = @{
        name        = $Repo
        description = $Description
        private     = [bool]$Private
        has_issues  = $true
        has_wiki    = $false
        has_projects = $false
        auto_init   = $false      # 关键：不建 README，否则 push 会撞 non-fast-forward
    } | ConvertTo-Json
    try {
        $created = Invoke-RestMethod -Uri 'https://api.github.com/user/repos' -Headers $headers `
            -Method Post -Body $body -ContentType 'application/json'
        Step "已创建：$($created.html_url)"
    } catch {
        Write-Host "[x] 创建失败：$($_.Exception.Message)" -ForegroundColor Red
        if ($_.ErrorDetails.Message) { Write-Host "    $($_.ErrorDetails.Message)" -ForegroundColor Red }
        exit 1
    }
}

# ---------- 4. 配 remote 并推送 ----------
$url = "https://github.com/$owner/$Repo.git"
$current = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or -not $current) {
    git remote add origin $url
    Step "已设置 origin = $url"
} elseif ($current -ne $url) {
    git remote set-url origin $url
    Step "origin 改为 $url（原为 $current）"
}

Step "推送 $branch ..."
# git 把进度写在 stderr，而在 $ErrorActionPreference='Stop' 下这会被当成终止错误。
# 所以调用期间临时降级，只用退出码判断成败。
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$pushOut = git push -u origin $branch 2>&1 | ForEach-Object { "$_" }
$pushCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap
$pushOut | Where-Object { $_ -match '->|up-to-date|Everything|rejected|error' } | ForEach-Object { Write-Host "    $_" }
if ($pushCode -ne 0) { Fail "推送失败（退出码 $pushCode）" }

# ---------- 5. 核对 ----------
$local = git rev-parse HEAD
$remote = git rev-parse "origin/$branch"
if ($local -eq $remote) {
    Write-Host ''
    Write-Host "[OK] 本地 == 远程（$($local.Substring(0,7))）" -ForegroundColor Green
    Write-Host "     https://github.com/$owner/$Repo" -ForegroundColor Green
} else {
    Write-Host "[!] 本地 $local 与远程 $remote 不一致，请复查" -ForegroundColor Yellow
    exit 1
}

exit 0
