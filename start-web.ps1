# PitchWizard 웹 배포 실행 스크립트
#   1) FastAPI 백엔드 실행 (127.0.0.1:8000)
#   2) Cloudflare Quick Tunnel로 백엔드를 https 공개 주소에 연결
#   3) 그 주소를 VITE_API_URL로 넣어 프론트 빌드 → Cloudflare Workers(정적 자산) 배포
#      (Cloudflare 계정 로그인 필요: frontend 폴더에서 npx wrangler login)
# 사용법: 프로젝트 루트에서  powershell -ExecutionPolicy Bypass -File .\start-web.ps1
#   -SkipDeploy : 프론트 재배포 없이 백엔드+터널만 실행
param(
  [string]$ProjectName = "pitchwizard",
  [switch]$SkipDeploy
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Frontend = Join-Path $Root "frontend"
$LogDir = Join-Path $Root "outputs\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$Cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $Cloudflared) { $Cloudflared = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe" }
if (-not (Test-Path $Cloudflared)) {
  throw "cloudflared가 없습니다. 'winget install Cloudflare.cloudflared' 후 다시 실행하세요."
}

$procs = @()
try {
  # 1) 백엔드
  Write-Host "[1/3] 백엔드 실행 중..." -ForegroundColor Cyan
  $procs += Start-Process -PassThru -NoNewWindow -WorkingDirectory $Root `
    -FilePath (Join-Path $Root "venv311\Scripts\python.exe") `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -RedirectStandardOutput (Join-Path $LogDir "backend.out.log") `
    -RedirectStandardError (Join-Path $LogDir "backend.err.log")

  # 2) 터널 (Quick Tunnel 주소는 cloudflared 로그(stderr)에 찍힘)
  Write-Host "[2/3] Cloudflare Tunnel 연결 중..." -ForegroundColor Cyan
  $tunnelLog = Join-Path $LogDir "tunnel.log"
  if (Test-Path $tunnelLog) { Remove-Item $tunnelLog }
  $procs += Start-Process -PassThru -NoNewWindow -FilePath $Cloudflared `
    -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000" `
    -RedirectStandardOutput (Join-Path $LogDir "tunnel.out.log") `
    -RedirectStandardError $tunnelLog

  $apiUrl = $null
  for ($i = 0; $i -lt 60 -and -not $apiUrl; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
      $m = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
      if ($m) { $apiUrl = $m.Matches[0].Value }
    }
  }
  if (-not $apiUrl) { throw "터널 주소를 60초 안에 받지 못했습니다. $tunnelLog 를 확인하세요." }
  Write-Host "      백엔드 공개 주소: $apiUrl" -ForegroundColor Green

  # 3) 프론트 빌드 + 배포
  $siteUrl = "(배포 생략)"
  if (-not $SkipDeploy) {
    Write-Host "[3/3] 프론트 빌드 및 Cloudflare Pages 배포 중..." -ForegroundColor Cyan
    Push-Location $Frontend
    # npm/wrangler는 경고를 stderr로 내보내므로, 여기서는 종료 코드로만 실패를 판단
    $ErrorActionPreference = "Continue"
    try {
      $env:VITE_API_URL = $apiUrl
      npm run build
      if ($LASTEXITCODE -ne 0) { throw "프론트 빌드 실패" }
      $deployOut = npx wrangler deploy --name $ProjectName 2>&1 | ForEach-Object { "$_" }
      $deployOut | Write-Host
      if ($LASTEXITCODE -ne 0) { throw "Cloudflare 배포 실패 (npx wrangler login 했는지 확인)" }
      $siteUrl = ($deployOut | Select-String -Pattern "https://\S+\.workers\.dev" | Select-Object -First 1).Matches.Value
    } finally {
      Remove-Item Env:VITE_API_URL -ErrorAction SilentlyContinue
      Pop-Location
      $ErrorActionPreference = "Stop"
    }
  }

  Write-Host ""
  Write-Host "실행 중: $siteUrl  (백엔드 $apiUrl)" -ForegroundColor Green
  Write-Host "종료하려면 Ctrl+C" -ForegroundColor Yellow
  while ($true) {
    foreach ($p in $procs) { if ($p.HasExited) { throw "프로세스 종료됨: $($p.ProcessName) (outputs\logs 확인)" } }
    Start-Sleep -Seconds 2
  }
} finally {
  foreach ($p in $procs) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
  Write-Host "백엔드와 터널을 종료했습니다."
}
