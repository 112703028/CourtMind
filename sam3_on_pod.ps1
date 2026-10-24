<#
sam3_on_pod.ps1 — pod 只負責 SAM3（唯一要 GPU 的步驟），其餘都在本地；分批跑、每批跑完立刻拉回。
pod 隨時可能被關：最多損失「正在跑的那一批」，已拉回的都在本地。

前提：
  - 本地已切好片段 + 5fps 影格（nba_data_pipeline/make_new_clips.py）
  - pod 已跑過 setup_pod.sh（有 sam3_env.sh）
  - 建議先設好 SSH 金鑰免密碼（不然每次連線都要打密碼）

每一批（BatchSize 支）：
  1. 本地打包這批的 5fps 影格 → 傳到 pod → pod 上 nohup 跑 run_sam3_tracks.py
  2. 等待期間：本地對「上一批」跑前處理 + 預測 + 確認影片（prep_new_clips.py，CPU）
  3. 每 PollMinutes 分鐘看 pod 上 logs/sam3_done_起_迄 → 拉回這批的 SAM3 追蹤
已經拉回的批次（本地 stubs/train_sam3 已有）會自動跳過 → pod 被關後重開 + 重新 setup，再跑同一個指令就會接著做。

用法（在專案根目錄）：
  .\sam3_on_pod.ps1 -Start 283 -End 503
  可選：-BatchSize 50  -Port 30855  -Remote /root/basketball_analysis  -PollMinutes 3
輸出：output_videos/new_clip_check/screen_N.mp4 + handler_labels_<批起>_<批迄>.csv（每批一份）
#>
param(
    [Parameter(Mandatory = $true)][int]$Start,
    [Parameter(Mandatory = $true)][int]$End,
    [int]$BatchSize = 50,
    [string]$HostAddr = 'root@140.119.163.112',
    [string]$Port = '30458',
    [string]$Remote = '/root/basketball_analysis',
    [int]$PollMinutes = 3
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Invoke-PodScript([string]$script) {
    # 遠端腳本先存成檔案（LF、無 BOM）→ scp 到 pod → ssh 執行那個檔案。
    # 不走 stdin / 不把腳本塞進 ssh 參數：Windows PowerShell 5.1 兩種方式都會把內容弄壞。
    $name = "podscript_$([guid]::NewGuid().ToString('N').Substring(0, 8)).sh"
    $local = Join-Path $env:TEMP $name
    [IO.File]::WriteAllText($local, ($script -replace "`r", ''), [Text.UTF8Encoding]::new($false))
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        scp -q -o ConnectTimeout=20 -P $Port $local "${HostAddr}:/tmp/$name" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { return @{ ok = $false; out = 'scp 失敗（pod 連不上？）' } }
        $out = ssh -o ConnectTimeout=20 -p $Port $HostAddr "bash /tmp/$name" 2>&1
        return @{ ok = ($LASTEXITCODE -eq 0); out = (($out | ForEach-Object { "$_" }) -join "`n") }
    } finally {
        $ErrorActionPreference = $prev
        Remove-Item $local -ErrorAction SilentlyContinue
    }
}

function Have-Tracks([int]$s, [int]$e) {
    foreach ($n in $s..$e) {
        if ((Test-Path "$root/input_videos/screen_frames/screen_$n") -and
            -not (Test-Path "$root/stubs/train_sam3/screen_${n}_sam3_tracks.pkl")) { return $false }
    }
    return $true
}

function Start-Batch([int]$s, [int]$e) {
    # 本地中斷後重跑：pod 上這批還在跑 → 不重複啟動，直接接著等
    $chk = Invoke-PodScript "pgrep -f 'list_${s}_${e}.txt' >/dev/null && echo __RUNNING__; [ -f $Remote/logs/sam3_done_${s}_${e} ] && echo __DONE__; true"
    if ($chk.out -match '__RUNNING__') { Write-Host '   pod 上這批還在跑（上次啟動的），接著等'; return 'running' }
    if ($chk.out -match '__DONE__') { Write-Host '   pod 上這批已經跑完，直接拉回'; return 'done' }
    $dirs =$s..$e | ForEach-Object { "screen_$_" } | Where-Object { Test-Path "$root/input_videos/screen_frames/$_" }
    $tgz = Join-Path $env:TEMP "frames_${s}_${e}.tgz"
    tar -czf $tgz -C "$root/input_videos/screen_frames" $dirs
    Write-Host ("   影格 {0} 支（{1:N0} MB）傳到 pod ..." -f $dirs.Count, ((Get-Item $tgz).Length / 1MB))
    scp -P $Port $tgz "${HostAddr}:/tmp/frames_${s}_${e}.tgz"
    if ($LASTEXITCODE -ne 0) { throw 'scp 失敗：pod 連得上嗎？port 對嗎？' }
    Remove-Item $tgz
    $script = @'
cd __R__ || exit 1
[ -f sam3_env.sh ] || { echo "no sam3_env.sh: run setup_pod.sh first"; exit 3; }
mkdir -p logs input_videos/screen_frames stubs/train_sam3
tar -xzf /tmp/frames___S_____E__.tgz -C input_videos/screen_frames && rm -f /tmp/frames___S_____E__.tgz
seq __S__ __E__ | sed 's/^/screen_/' > /tmp/list___S_____E__.txt
rm -f logs/sam3_done___S_____E__ logs/sam3_failed___S_____E__
nohup bash -c 'source sam3_env.sh && python -u nba_data_pipeline/run_sam3_tracks.py --frames_root input_videos/screen_frames --clips_file /tmp/list___S_____E__.txt --out_dir stubs/train_sam3 --single_gpu && touch logs/sam3_done___S_____E__ || touch logs/sam3_failed___S_____E__' > logs/sam3___S_____E__.log 2>&1 < /dev/null &
echo started
'@
    $r = Invoke-PodScript ($script.Replace('__R__', $Remote).Replace('__S__', "$s").Replace('__E__', "$e"))
    if (-not $r.ok) { throw "pod 上啟動失敗：$($r.out)" }
    return 'started'
}

function Wait-Batch([int]$s, [int]$e) {
    $fails = 0
    while ($true) {
        $r = Invoke-PodScript @"
cd $Remote
if [ -f logs/sam3_done_${s}_${e} ]; then echo __DONE__
elif [ -f logs/sam3_failed_${s}_${e} ]; then echo __FAILED__; tail -n 15 logs/sam3_${s}_${e}.log
else grep -E '^\[[0-9]+/' logs/sam3_${s}_${e}.log | tail -n 1; fi
"@
        $now = Get-Date -Format 'HH:mm'
        if (-not $r.ok) {
            $fails++
            Write-Host "[$now] 連不上 pod（第 $fails 次）"
            if ($fails -ge 5) { throw 'pod 連不上：可能被關了。重開 pod → setup_pod.sh → 再跑同一個指令，會從沒拉回的批次接著做。' }
        } elseif ($r.out -match '__DONE__') { return $true }
        elseif ($r.out -match '__FAILED__') { Write-Host "[$now] ⚠ SAM3 失敗：`n$($r.out)"; return $false }
        else { $fails = 0; Write-Host "[$now] SAM3 還在跑：$($r.out)" }
        Start-Sleep -Seconds ($PollMinutes * 60)
    }
}

function Pull-Batch([int]$s, [int]$e) {
    $r = Invoke-PodScript @"
cd $Remote && f=(); for n in `$(seq $s $e); do p=stubs/train_sam3/screen_`${n}_sam3_tracks.pkl; [ -f `$p ] && f+=(`$p); done
[ `${#f[@]} -gt 0 ] || exit 2; tar -czf /tmp/tracks_${s}_${e}.tgz `${f[@]}; echo `${#f[@]}
"@
    if (-not $r.ok) { throw "pod 上打包追蹤結果失敗：$($r.out)" }
    $local = Join-Path $env:TEMP "tracks_${s}_${e}.tgz"
    scp -P $Port "${HostAddr}:/tmp/tracks_${s}_${e}.tgz" $local
    if ($LASTEXITCODE -ne 0) { throw '拉回追蹤結果失敗' }
    tar -xzf $local -C $root
    Remove-Item $local
    Write-Host "   拉回 $($r.out.Trim()) 支 SAM3 追蹤"
}

function Prep-Local([int]$s, [int]$e) {
    Write-Host "   本地前處理 + 預測 + 確認影片：screen_$s ~ $e（CPU，約每支 1 分鐘）..."
    # Python 的 stderr（例如無害的 UserWarning）在 Windows PowerShell 5.1 + Stop 下會被當成錯誤中斷 → 這段改 Continue
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        & python -X utf8 "$root/nba_data_pipeline/prep_new_clips.py" --start $s --end $e --render30 --handler_only 2>&1 |
            ForEach-Object { "$_" } | Select-String '失敗|→|Error|Traceback' | ForEach-Object { Write-Host "   $_" }
    } finally { $ErrorActionPreference = $prev }
}

$batches = @()
for ($s = $Start; $s -le $End; $s += $BatchSize) { $batches += , @($s, [Math]::Min($End, $s + $BatchSize - 1)) }
Write-Host "== screen_$Start ~ $End，分 $($batches.Count) 批（每批 $BatchSize 支）"

# 待本地處理的批次排隊：pod 在跑 SAM3 時，本地順便處理一批；pod 那批若早已跑完就先拉回、先送下一批，不讓 pod 閒著
$queue = New-Object System.Collections.ArrayList
foreach ($b in $batches) {
    $s, $e = $b
    if (Have-Tracks $s $e) {
        Write-Host "== 批次 $s~$e：SAM3 追蹤本地都有了，跳過 pod"
        [void]$queue.Add(@($s, $e)); continue
    }
    Write-Host "== 批次 $s~$e：送到 pod 跑 SAM3"
    $st = Start-Batch $s $e
    if ($st -ne 'done' -and $queue.Count -gt 0) {
        $q = $queue[0]; $queue.RemoveAt(0); Prep-Local $q[0] $q[1]
    }
    if (-not (Wait-Batch $s $e)) { Write-Host '   這批有部分失敗，已完成的照樣拉回' }
    Pull-Batch $s $e
    [void]$queue.Add(@($s, $e))
}
foreach ($q in $queue) { Prep-Local $q[0] $q[1] }
Write-Host "== 全部完成：output_videos/new_clip_check/（handler_labels_<批起>_<批迄>.csv 每批一份）"