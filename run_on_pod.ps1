<#
run_on_pod.ps1 — 在本地一個指令：把一支 YouTube 影片的擋拆片段丟到 GPU pod 處理，跑完自動拉回本地。

（pod 連不回你的電腦，所以由本地負責：啟動 → 定時問 pod 跑完沒 → 跑完就拉回來）
流程：
  1. 時間段檔傳到 pod
  2. pod 上 nohup 跑 run_new_clips.sh（切片段 → SAM3 → 前處理 → 預測 → 確認影片）
  3. 每 PollMinutes 分鐘問一次（看 pod 上 logs/done_起_迄 或 logs/failed_起_迄）
  4. 完成 → 呼叫 pull_from_pod.ps1 拉回本地；失敗 → 印 log 最後幾行，已完成的部分照樣拉回
本地電腦要開著（等待期間這個視窗不要關）。中途關掉了：加 -Resume 只等待＋拉回，不重新啟動。

用法（在專案根目錄）：
  .\run_on_pod.ps1 -Url "https://www.youtube.com/watch?v=GJhY0NC6-dY" -Ranges nba_data_pipeline\new_clips_GJhY0NC6-dY.txt -Start 283
  .\run_on_pod.ps1 -Url ... -Ranges ... -Start 283 -Resume
  可選：-Port 30855  -Remote /root/basketball_analysis  -PollMinutes 5
#>
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Ranges,
    [Parameter(Mandatory = $true)][int]$Start,
    [string]$HostAddr = 'root@140.119.163.112',
    [string]$Port = '30458',
    [string]$Remote = '/root/basketball_analysis',
    [int]$PollMinutes = 5,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$n = @(Get-Content $Ranges -Encoding UTF8 | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') }).Count
$End = $Start + $n - 1
$tag = "${Start}_${End}"
$rname = Split-Path $Ranges -Leaf
$log = "logs/run_$tag.log"
Write-Host "== screen_$Start ~ screen_$End（$n 段）  pod: ${HostAddr}:$Port $Remote"

function Invoke-Pod([string]$cmd) {
    $out = ssh -o ConnectTimeout=20 -p $Port $HostAddr "cd $Remote && $cmd" 2>&1
    return @{ ok = ($LASTEXITCODE -eq 0); out = ($out -join "`n") }
}

if (-not $Resume) {
    Write-Host '== 傳時間段檔、啟動 pod 上的流程 ...'
    scp -P $Port $Ranges "${HostAddr}:$Remote/nba_data_pipeline/$rname"
    if ($LASTEXITCODE -ne 0) { throw 'scp 失敗：pod 連得上嗎？port 對嗎？' }
    $r = Invoke-Pod "mkdir -p logs && nohup bash run_new_clips.sh '$Url' nba_data_pipeline/$rname $Start > $log 2>&1 < /dev/null & echo started"
    if (-not $r.ok) { throw "啟動失敗：$($r.out)" }
    Write-Host "   已啟動（pod 上的 log：$Remote/$log）"
}

$fails = 0
while ($true) {
    Start-Sleep -Seconds ($PollMinutes * 60)
    $r = Invoke-Pod "if [ -f logs/done_$tag ]; then echo __DONE__; elif [ -f logs/failed_$tag ]; then echo __FAILED__; else tail -n 1 $log; fi"
    $now = Get-Date -Format 'HH:mm'
    if (-not $r.ok) {
        $fails++
        Write-Host "[$now] 連不上 pod（第 $fails 次）：$($r.out)"
        if ($fails -ge 6) { throw '連續 6 次連不上 pod：pod 可能被重建了，結果可能已經不在。' }
        continue
    }
    $fails = 0
    if ($r.out -match '__DONE__') { Write-Host "[$now] pod 上跑完了"; break }
    if ($r.out -match '__FAILED__') {
        Write-Host "[$now] ⚠ pod 上的流程中途失敗，log 最後 20 行："
        Write-Host (Invoke-Pod "tail -n 20 $log").out
        Write-Host '   已完成的部分照樣拉回來。'
        break
    }
    Write-Host "[$now] 還在跑：$($r.out)"
}

& (Join-Path $PSScriptRoot 'pull_from_pod.ps1') -Start $Start -End $End -HostAddr $HostAddr -Port $Port -Remote $Remote
