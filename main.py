import os
import sys
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# =========================================================================
# CHỐT CHẶN HẠT NHÂN 1: Ép toàn bộ các thư viện toán học lõi C/C++ (Numpy, PyTorch, FAISS)
# chỉ dùng tối đa 8 luồng. Phải cấu hình TRƯỚC KHI import bất kỳ thư viện nào!
# =========================================================================
safe_threads = "8"
os.environ["OMP_NUM_THREADS"] = safe_threads
os.environ["OPENBLAS_NUM_THREADS"] = safe_threads
os.environ["MKL_NUM_THREADS"] = safe_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = safe_threads
os.environ["NUMEXPR_NUM_THREADS"] = safe_threads

# =========================================================================
# CHỐT CHẶN HẠT NHÂN 2: Chống Deadlock Tokenizer
# =========================================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import zipfile
import threading
import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(8)

def cleanup_lock_files():
    import shutil
    locks_dirs = [
        "./cache/huggingface/hub/.locks",
        os.path.expanduser("~/.cache/huggingface/hub/.locks")
    ]
    for d in locks_dirs:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

cleanup_lock_files()

def heartbeat_log(start_time):
    while True:
        time.sleep(600)
        elapsed_mins = (time.time() - start_time) / 60
        print(f"\n[HEARTBEAT] ⏳ Tình trạng: Đang chạy bình thường. Tổng thời gian: {elapsed_mins:.2f} phút.")
        sys.stdout.flush()

from src.config import THRESHOLD
from src.preprocessing import build_corpus
from src.indexing import build_indexes
from src.mining import mine_hard_negatives
from src.trainer import train_reranker
from src.inference import LegalPipelineInference, tune_threshold, run_evaluate

def main():
    parser = argparse.ArgumentParser(description="Chunk-based Legal RAG Pipeline")
    parser.add_argument("--phase", type=str, required=True, 
                        choices=["preprocess", "index", "mine", "train", "tune", "evaluate", "predict"])
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="output/submission.json")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Ngưỡng điểm Reranker tùy chỉnh")
    args = parser.parse_args()

    t0 = time.time()
    hb_thread = threading.Thread(target=heartbeat_log, args=(t0,), daemon=True)
    hb_thread.start()

    if args.phase == "preprocess":
        if os.path.exists("data/processed/chunks_cache.pt"):
            print("⏭️ Dữ liệu đã được Preprocess (tồn tại chunks_cache.pt). Bỏ qua Phase này.")
            return
        build_corpus()
    elif args.phase == "index":
        if os.path.exists("data/indexes/faiss.index"):
            print("⏭️ Dữ liệu đã được Index (tồn tại faiss.index). Bỏ qua Phase này.")
            return
        build_indexes()
    elif args.phase == "mine":
        if os.path.exists("data/processed/reranker_train.jsonl"):
            print("⏭️ Dữ liệu đã được Mine (tồn tại reranker_train.jsonl). Bỏ qua Phase này.")
            return
        mine_hard_negatives()
    elif args.phase == "train":
        if os.path.exists("models/qwen_reranker_final/adapter_model.safetensors") or os.path.exists("models/qwen_reranker_final/pytorch_model.bin"):
            print("⏭️ Mô hình đã được Train (tồn tại models/qwen_reranker_final). Bỏ qua Phase này.")
            return
        train_reranker()
    elif args.phase == "tune":
        tune_threshold()
    elif args.phase == "evaluate":
        run_evaluate()
    elif args.phase == "predict":
        if not args.input:
            raise ValueError("Cần truyền --input file_path.json")

        import shutil
        ckpt_infer_dir = os.path.join("checkpoints", "inference")

        # Xóa file output cũ nếu tồn tại, ghi nội dung mới (tránh nối nội dung cũ)
        if os.path.exists(args.output):
            os.remove(args.output)
            print(f"🗑️ Đã xóa file output cũ: {args.output}")

        pipeline = LegalPipelineInference()
        pipeline.predict(args.input, args.output, threshold=args.threshold)

        zip_path = args.output.replace('.json', '.zip')
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.output, arcname="submission.json")
        print(f"✓ Đã đóng gói {args.output} thành {zip_path}")

        # Chỉ xóa checkpoint khi predict hoàn tất thành công
        if os.path.exists(ckpt_infer_dir):
            shutil.rmtree(ckpt_infer_dir)
            print(f"🗑️ Đã xóa checkpoint predict: {ckpt_infer_dir}")

    print(f"\n🚀 PHASE {args.phase.upper()} HOÀN TẤT TRONG {(time.time() - t0)/60:.2f} PHÚT!")

if __name__ == "__main__":
    main()
