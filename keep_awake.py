"""keep_awake.py — 長時間本機實驗期間不讓 Windows 睡眠（不改電源設定，程式結束就恢復）。
用法：python keep_awake.py <要等的檔案> <等到檔案裡出現這個字>   # 例 logs/holder_cv.log HOLDER_CV_DONE
"""
import sys
import time
import ctypes
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
log, token = Path(sys.argv[1]), sys.argv[2]
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
try:
    while not (log.exists() and token in log.read_text(encoding='utf-8', errors='ignore')):
        time.sleep(60)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
