#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=mps:a100:1
#SBATCH --time=72:00:00

module clear -f
module load shared python312

# ============================================================
# [RANG BUOC 2] CHUYEN THƯ MỤC LÀM VIỆC SANG /DATASTORE
# ============================================================
export WORKDIR="/datastore/inseclab/liem/legal_ir"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Kích hoạt môi trường ảo
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

# Lay token tu .env de tai model (requires auth)
set -a
source .env
set +a
export HF_TOKEN="${HF_TOKEN:-}"

# --- Giới hạn số luồng CPU cho các thư viện toán (numpy/torch/faiss...) ---
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# --- Dọn lock file cũ (tránh treo khi job trước bị kill) ---
find "$DATASTORE_ROOT/cache" -name "*.lock" -type f -delete 2>/dev/null || true

# ---------------------------------------------------------
# Khai báo/chọn GPU theo chuẩn UIT (tránh đụng độ VRAM giữa các job)
# ---------------------------------------------------------
REQUIRED_VRAM=16384    # MB, đổi theo nhu cầu thực tế của model
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
# Private MPS server cho job (chuẩn UIT, cho phép share GPU an toàn)
# ---------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# ---------------------------------------------------------
# Lệnh chạy chính — đổi theo dự án
# ---------------------------------------------------------
echo "Starting job..."
python -u your_script.py
echo "Job finished."
