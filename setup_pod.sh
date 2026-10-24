#!/bin/bash
# setup_pod.sh — 新的 GPU pod 一次裝好：venv（torch 2.7.1 cu126）+ SAM3 + 套件 + Triton 修正 + 程式碼。
# 用法（先把 pod_bundle.tgz 跟這支放在同一個資料夾）：
#   export HF_TOKEN=你的token     # 選填：有設就順便下載 SAM3 權重（之後不用再設）
#   bash setup_pod.sh
# 裝在這支腳本所在的資料夾（建議 /mnt/data/basketball_analysis；沒有 /mnt/data 就用 /root/basketball_analysis）
set -e
trap 'echo "!! 失敗：第 $LINENO 行（$BASH_COMMAND）"' ERR
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"
echo "== 安裝位置：$BASE"

# 1. 系統工具
command -v ffmpeg >/dev/null || (apt-get update && apt-get install -y ffmpeg)
command -v git    >/dev/null || (apt-get update && apt-get install -y git)

# 2. 程式碼 + 模型
[ -f pod_bundle.tgz ] && tar -xzf pod_bundle.tgz && echo "== 已解壓 pod_bundle.tgz"

echo "== 3/7 建 venv、裝套件（10~20 分鐘，會下載好幾 GB）"
# 3. venv + PyTorch（主機驅動只到 CUDA 12.4 → 用 cu126 版，靠 CUDA 12.x 向下相容）
[ -d venv_sam3 ] || python3 -m venv venv_sam3
source venv_sam3/bin/activate
pip install -q --upgrade pip
pip install einops timm iopath huggingface_hub pycocotools opencv-python-headless psutil ftfy regex \
    scipy tqdm ultralytics supervision yt-dlp imageio av scikit-learn
# ultralytics 可能把 torch 換掉 → 最後再固定回 2.7.1 cu126
echo "== 3/7 固定 PyTorch 2.7.1 cu126"
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

echo "== 4/7 安裝 SAM3"
# 4. SAM3
[ -d sam3 ] || git clone -q https://github.com/facebookresearch/sam3.git
pip install -q -e sam3
# 版本修正（2026-10-20 實測）：sam3 用 pkg_resources（新版 setuptools 拿掉了）、sam3 把 numpy 固定 1.26.4，
# 但最新 scipy / OpenCV 5 要 numpy>=2；ultralytics 會裝一般版 opencv-python（container 沒 libGL）
pip install -q "setuptools<81"
pip uninstall -y -q opencv-python || true
pip install -q "opencv-python-headless==4.10.0.84" "scipy<1.15" "contourpy<1.3.3"

echo "== 5/7 寫環境設定檔"
# 5. 環境設定檔（Triton 改用 venv 自帶的編譯器；container 預設指向 CUDA 13 的會報錯）
TB=$(python -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends/nvidia/bin'))")
cat > "$BASE/sam3_env.sh" << EOF
source $BASE/venv_sam3/bin/activate
export TRITON_PTXAS_PATH=$TB/ptxas
export TRITON_CUOBJDUMP_PATH=$TB/cuobjdump
export TRITON_NVDISASM_PATH=$TB/nvdisasm
cd $BASE
EOF
grep -q "$BASE/sam3_env.sh" ~/.bashrc 2>/dev/null || echo "source $BASE/sam3_env.sh" >> ~/.bashrc
source "$BASE/sam3_env.sh"

echo "== 6/7 SAM3 權重"
# 6. SAM3 權重（有 HF_TOKEN 才下載）
if [ -n "$HF_TOKEN" ]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3')" && echo "== SAM3 權重已下載"
else
  echo "== 沒設 HF_TOKEN：SAM3 權重會在第一次跑時才下載（那時要 export HF_TOKEN）"
fi

echo "== 7/7 自我檢查"
# 7. 自我檢查
python - << 'PY'
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.is_available() and torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'GPU 看不到！'
from sam3.model_builder import build_sam3_video_predictor
import ultralytics, supervision, cv2
print('sam3 / ultralytics / supervision / cv2 OK')
PY
echo "== 完成。以後登入會自動載入環境（~/.bashrc）；手動：source $BASE/sam3_env.sh"