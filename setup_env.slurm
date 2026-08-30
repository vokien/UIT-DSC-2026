#!/bin/bash
#SBATCH --job-name=setup_legal_ir
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

module clear -f
module load shared python312

# ============================================================
# [RANG BUOC 2] CHUYEN THƯ MỤC LÀM VIỆC SANG /DATASTORE
# Tất cả code, venv, checkpoint, model sẽ lưu tại đây — tránh đầy /home (30GB)
# ============================================================
export WORKDIR="/datastore/inseclab/liem/legal_ir"
mkdir -p "$WORKDIR"

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

mkdir -p "$WORKDIR/logs"

# Create venv TREN /DATASTORE (khong phai tren /home)
python3 -m venv "$WORKDIR/venv"
source "$WORKDIR/venv/bin/activate"
pip install --upgrade pip

# Cài đặt PyTorch hỗ trợ CUDA 11.8 (Tương thích với Driver của Server)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Download models (dung HF_HOME da cau hinh o tren tren /datastore)
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

print('Downloading BGE-M3...')
SentenceTransformer('BAAI/bge-m3')

print('Downloading Qwen2.5-3B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
AutoModelForSequenceClassification.from_pretrained('Qwen/Qwen2.5-3B-Instruct', num_labels=1)
print('Download completed successfully!')
"

echo "Environment setup complete."
