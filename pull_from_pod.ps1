<#
pull_from_pod.ps1 — 把 GPU pod 上 run_new_clips.sh 跑完的一批結果拉回本地。

拉回的東西（只拉 screen_<Start> ~ screen_<End>）：
  stubs/train_sam3/screen_N_sam3_tracks.pkl       SAM3 追蹤
  stubs/train_contact_cache/screen_N.pkl          前處理快取（建訓練資料要用）
  output_videos/new_clip_check/screen_N.mp4       確認影片
  output_videos/new_clip_check/*_<Start>_<End>.csv  要填的表格
做法：pod 上先打包成一個 tgz → scp 一次 → 本地解壓（檔案多時比逐個 scp 快很多）。

用法（在專案根目錄）：
  .\pull_from_pod.ps1 -Start 283 -End 503
  .\pull_from_pod.ps1 -Start 283 -End 503 -Port 31234 -Remote /mnt/data/basketball_analysis
#>
param(
    [Parameter(Mandatory = $true)][int]$Start,
    [Parameter(Mandatory = $true)][int]$End,
    [string]$HostAddr = 'root@140.119.163.112',
    [string]$Port = '30458',
    [string]$Remote = '/root/basketball_analysis'
)
$ErrorActionPreference = 'Stop'
$tag = "${Start}_${End}"
$tgz = "/tmp/pull_$tag.tgz"

# 1. pod 上打包（只收存在的檔案；表格一律收這批的）
#    遠端腳本走 stdin（bash -s）：Windows PowerShell 傳含雙引號的參數給 ssh 會被弄壞；腳本保持純 ASCII
$pack = @'
cd __REMOTE__ || exit 1
rm -f __TGZ__
files=()
for n in $(seq __START__ __END__); do
  for f in stubs/train_sam3/screen_${n}_sam3_tracks.pkl stubs/train_contact_cache/screen_${n}.pkl output_videos/new_clip_check/screen_${n}.mp4; do
    [ -f "$f" ] && files+=("$f")
  done
done
for f in output_videos/new_clip_check/*___TAG__.csv; do [ -f "$f" ] && files+=("$f"); done
echo "found ${#files[@]} files on pod"
[ ${#files[@]} -gt 0 ] || exit 2
tar -czf __TGZ__ "${files[@]}"
'@
$pack = $pack.Replace('__REMOTE__', $Remote).Replace('__TGZ__', $tgz).Replace('__START__', "$Start").Replace('__END__', "$End").Replace('__TAG__', $tag) -replace "`r", ''
Write-Host "== 1/3 在 pod 上打包 screen_$Start ~ screen_$End ..."
# 腳本存檔 → scp → ssh 執行（Windows PowerShell 5.1 走 stdin 會把內容弄壞）
$ps = Join-Path $env:TEMP "pack_$tag.sh"
[IO.File]::WriteAllText($ps, $pack, [Text.UTF8Encoding]::new($false))
scp -q -P $Port $ps "${HostAddr}:/tmp/pack_$tag.sh"
Remove-Item $ps
ssh -p $Port $HostAddr "bash /tmp/pack_$tag.sh"
if ($LASTEXITCODE -eq 2) { throw "pod 上找不到這批的任何檔案（-Remote $Remote 對嗎？範圍對嗎？）" }
if ($LASTEXITCODE -ne 0) { throw "pod 上打包失敗（路徑對嗎？-Remote $Remote）" }

# 2. 傳回本地
$local = Join-Path $env:TEMP "pull_$tag.tgz"
Write-Host "== 2/3 傳回本地 ..."
scp -P $Port "${HostAddr}:$tgz" $local
if ($LASTEXITCODE -ne 0) { throw 'scp 失敗' }

# 3. 解壓到專案根目錄
Write-Host "== 3/3 解壓 ..."
tar -xzf $local -C $PSScriptRoot
Remove-Item $local

$range = $Start..$End
$cnt = {
    param($dir, $pattern)
    (Get-ChildItem (Join-Path $PSScriptRoot $dir) -Filter $pattern -ErrorAction SilentlyContinue |
        Where-Object { [int](($_.Name -split '_')[1] -replace '\D.*', '') -in $range }).Count
}
$n = $End - $Start + 1
Write-Host ("完成：SAM3 追蹤 {0}/{1}、前處理快取 {2}/{1}、確認影片 {3}/{1}" -f `
    (& $cnt 'stubs/train_sam3' 'screen_*_sam3_tracks.pkl'), $n,
    (& $cnt 'stubs/train_contact_cache' 'screen_*.pkl'),
    (& $cnt 'output_videos/new_clip_check' 'screen_*.mp4'))
Get-ChildItem (Join-Path $PSScriptRoot 'output_videos/new_clip_check') -Filter "*_$tag.csv" | ForEach-Object { Write-Host "表格：$($_.FullName)" }
Write-Host '（數量不足 = pod 上那幾支還沒跑完、被跳過（超過影片長度），或失敗；看 pod 上的 log）'
