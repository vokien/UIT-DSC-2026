#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

module clear -f
module load shared python312

# ============================================================
# [RANG BUOC 2] CHUYEN THƯ MỤC LÀM VIỆC SANG /DATASTORE
# Tất cả checkpoint, model, data sẽ lưu tại đây — tránh đầy /home (30GB)
# ============================================================
export WORKDIR="/datastore/inseclab/liem/legal_ir"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Kích hoạt môi trường ảo (venv đã tạo bởi setup_env.slurm)
source "$WORKDIR/venv/bin/activate"

# Tạo thư mục con cần thiết
mkdir -p logs
mkdir -p checkpoints

# ============================================================
# [RANG BUOC 2] CACHE TREN /DATASTORE — TREN /home (30GB) BI KHOA TAI KHOAN
# ============================================================
export DATASTORE_ROOT="/datastore/inseclab/liem"
mkdir -p "$DATASTORE_ROOT/cache/huggingface"
mkdir -p "$DATASTORE_ROOT/cache/torch"
mkdir -p "$DATASTORE_ROOT/cache/pip"

export HF_HOME="$DATASTORE_ROOT/cache/huggingface"
export TORCH_HOME="$DATASTORE_ROOT/cache/torch"
export PIP_CACHE_DIR="$DATASTORE_ROOT/cache/pip"

# ============================================================
# [RANG BUOC] LAY TOKEN TRUOC KHI TAI MODEL (CAN XAC THUC)
# ============================================================
set -a
source .env
set +a
export HF_TOKEN="${HF_TOKEN:-}"

# Xóa các file "khóa" (.lock) bị kẹt do các job trước bị hủy đột ngột
find "$DATASTORE_ROOT/cache" -name "*.lock" -type f -delete 2>/dev/null || true

# ---------------------------------------------------------
# Hàm check GPU của Slurm UIT (Tránh đụng độ VRAM)
# ---------------------------------------------------------
REQUIRED_VRAM=16384
unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"
export CUDA_VISIBLE_DEVICES=$BEST_GPU 

# ---------------------------------------------------------
# Khởi tạo Private MPS Server cho Job (Theo chuẩn UIT)
# ---------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

# Xóa sạch các dấu vết cũ và tạo thư mục mới
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
# ---------------------------------------------------------

# Dừng script ngay lập tức nếu bất kỳ lệnh nào bị lỗi
set -e

echo "Starting full pipeline execution..."

echo "=================================================="
echo "1. Preprocessing data..."
echo "=================================================="
python -u main.py --phase preprocess

echo "=================================================="
echo "2. Building indexes..."
echo "=================================================="
python -u main.py --phase index

echo "=================================================="
echo "3. Mining hard negatives..."
echo "=================================================="
python -u main.py --phase mine

echo "=================================================="
echo "4. Training reranker..."
echo "=================================================="
python -u main.py --phase train

echo "=================================================="
echo "5. Tuning threshold on warmup set..."
echo "=================================================="
python -u main.py --phase tune

echo "=================================================="
echo "6. Running prediction..."
echo "=================================================="
# Xóa checkpoint cũ để đảm bảo inference nhất quán từ đầu
rm -rf checkpoints/inference/
echo "Da xoa checkpoint inference cu."
python -u main.py --phase predict --input data/public-official.json --output submission.json

echo "Pipeline finished."
