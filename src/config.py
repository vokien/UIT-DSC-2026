import os

DATA_DIR = "data"
RAW_CORPUS_DIR = os.path.join(DATA_DIR, "selected-contexts")
TRAIN_FILE   = os.path.join(DATA_DIR, "train.json")            # dùng train
EVAL_FILE    = os.path.join(DATA_DIR, "warmup.json")            # dùng tune/evaluate
TEST_FILE    = os.path.join(DATA_DIR, "public-official.json")   # dùng predict (test set)

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
MODEL_DIR = "models"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = "checkpoints"

for d in [PROCESSED_DIR, INDEX_DIR, MODEL_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_K_FINAL = 5       # Tối đa 5 parent_doc_id theo quy định eval_retrieval
TOP_K_CHUNKS = 50     # Lấy 50 chunks tốt nhất để Rerank
CHUNK_MAX_WORDS = 350 # Số từ tối đa cho mỗi chunk (vừa vặn 512 tokens)
CHUNK_OVERLAP = 50    # Gối đầu 50 từ để không bị mất ngữ cảnh giữa các chunk
THRESHOLD = 0.1       # Ngưỡng Sigmoid mặc định
