"""
gpu_keepalive.py — pod 閒置 3 小時會被斷開：沒有正式工作時，用 GPU 做小量運算保持忙碌。
  - 有正式工作（finetune / SAM3 / 前處理 / 分隊）在跑 → 休息，不搶 GPU
  - 沒有 → 每輪約 5 秒小矩陣運算 + 5 秒休息（約 200MB 顯示記憶體）
  - 最多 --hours 小時後自己結束（預設 12）
用法（pod 上）：nohup python gpu_keepalive.py --hours 12 > logs/keepalive.log 2>&1 &
停掉：pkill -f gpu_keepalive.py
"""
import time
import argparse
import subprocess

import torch

JOBS = 'finetune.py|run_sam3_tracks.py|prep_new_clips.py|ingest_new_labels.py'


def busy():
    """GPU 真的有在被用（使用率 ≥ 15%，扣掉自己）才算忙；只用 CPU 的訓練不算。"""
    r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0]) >= 15
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=0, help='0 = 不限時間，直到 pkill 停掉')
    args = ap.parse_args()
    end = time.time() + args.hours * 3600 if args.hours > 0 else float('inf')
    a = torch.randn(4096, 4096, device='cuda')
    b = torch.randn(4096, 4096, device='cuda')
    state = None
    while time.time() < end:
        time.sleep(5)                       # 先停一下，讓 nvidia-smi 量到的不是自己剛剛的運算
        if busy():
            if state != 'rest':
                print(time.strftime('%H:%M'), '有正式工作在跑 → 休息', flush=True)
                state = 'rest'
            time.sleep(60)
            continue
        if state != 'work':
            print(time.strftime('%H:%M'), '沒有正式工作 → 保持 GPU 忙碌', flush=True)
            state = 'work'
        t0 = time.time()
        while time.time() - t0 < 5:
            c = a @ b
        torch.cuda.synchronize()
        time.sleep(5)
    print(time.strftime('%H:%M'), f'已保持 {args.hours} 小時，結束', flush=True)


if __name__ == '__main__':
    main()
