import os
import json
import torch
import random
from tqdm import tqdm
from src.config import TRAIN_FILE, PROCESSED_DIR
from src.retriever import ChunkHybridRetriever
from src.checkpoint_utils import CheckpointManager

def mine_hard_negatives():
    print("--- ĐÀO MẪU ÂM KHÓ TỪ CHUNKS ---")
    retriever = ChunkHybridRetriever()
    chunks = torch.load(os.path.join(PROCESSED_DIR, "chunks_cache.pt"), weights_only=False)
    
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    ckpt_manager = CheckpointManager()
    dataset, version = ckpt_manager.load_latest_checkpoint("mining", "dataset", is_numpy=False)
    if not dataset:
        dataset = []
    else:
        print(f"Đã khôi phục checkpoint có {len(dataset)} items.")
        
    processed_queries = set([item["query"] for item in dataset])
    
    count = 0
    for qid, qdata in tqdm(train_data.items(), desc="Mining"):
        query = qdata["question"]
        if query in processed_queries:
            continue
            
        parent_answers = [str(a) for a in qdata["answer"]]
        candidate_chunk_ids = retriever.search_chunks(query, top_k=30)
        
        positives = []
        negatives = []
        
        for cid in candidate_chunk_ids:
            parent_id = chunks[cid]["parent_doc_id"]
            if parent_id in parent_answers:
                positives.append({"query": query, "chunk_text": chunks[cid]["text"], "label": 1.0})
            else:
                negatives.append({"query": query, "chunk_text": chunks[cid]["text"], "label": 0.0})
                
        if positives:
            dataset.extend(positives)
            sampled_neg = random.sample(negatives, min(len(negatives), 5))
            dataset.extend(sampled_neg)
            
        processed_queries.add(query)
        
        count += 1
        if count % 100 == 0:
            version += 1
            ckpt_manager.save_checkpoint("mining", "dataset", version, dataset, keep_last=1)

    ckpt_manager.save_checkpoint("mining", "dataset", version + 1, dataset, keep_last=1)

    out_file = os.path.join(PROCESSED_DIR, "reranker_train.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Đã lưu {len(dataset)} cặp huấn luyện Chunk vào {out_file}")
