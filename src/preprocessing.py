import os
import re
import json
import torch
from tqdm import tqdm
from underthesea import word_tokenize
from src.config import RAW_CORPUS_DIR, PROCESSED_DIR, CHUNK_MAX_WORDS, CHUNK_OVERLAP
from src.checkpoint_utils import CheckpointManager

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_document(doc_id: str, text: str) -> list:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []

    pattern = r"(?m)(^(?:Điều|ĐIỀU)\s+\d+[\.:\s])"
    parts = re.split(pattern, cleaned_text)
    
    raw_sections = []
    if len(parts) > 1:
        intro = parts[0].strip()
        if intro: raw_sections.append(intro)
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            raw_sections.append(f"{header} {content}".strip())
    else:
        raw_sections = [cleaned_text]

    chunks = []
    chunk_idx = 0
    
    for section in raw_sections:
        words = section.split()
        if not words: continue
        
        if len(words) <= CHUNK_MAX_WORDS:
            chunk_text = " ".join(words)
            chunks.append({
                "chunk_id": f"{doc_id}_c{chunk_idx}",
                "parent_doc_id": str(doc_id),
                "text": chunk_text,
                "tokens": word_tokenize(chunk_text, format="text").split()
            })
            chunk_idx += 1
        else:
            step = CHUNK_MAX_WORDS - CHUNK_OVERLAP
            for start in range(0, len(words), step):
                sub_words = words[start:start + CHUNK_MAX_WORDS]
                if len(sub_words) < 20 and start > 0:
                    continue
                chunk_text = " ".join(sub_words)
                chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "parent_doc_id": str(doc_id),
                    "text": chunk_text,
                    "tokens": word_tokenize(chunk_text, format="text").split()
                })
                chunk_idx += 1

    return chunks

def build_corpus():
    print("--- ĐANG XÂY DỰNG & CHUNK CORPUS ---")
    chunks_dict = {}
    
    if not os.path.exists(RAW_CORPUS_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục {RAW_CORPUS_DIR}")
        
    ckpt_manager = CheckpointManager()
    saved_chunks_dict, version = ckpt_manager.load_latest_checkpoint("preprocessing", "chunks", is_numpy=False)
    if saved_chunks_dict:
        chunks_dict = saved_chunks_dict
        print(f"Đã khôi phục checkpoint có {len(chunks_dict)} chunks.")
        
    processed_files = set([c["parent_doc_id"] for c in chunks_dict.values()])
    
    all_files = [f for f in os.listdir(RAW_CORPUS_DIR) if f.endswith('.json')]
    count = 0
    
    for filename in tqdm(all_files, desc="Chunking Raw JSONs"):
        doc_id_base = filename.split('.')[0]
        if doc_id_base in processed_files:
            continue
            
        filepath = os.path.join(RAW_CORPUS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            items = json.load(f)
            if not isinstance(items, list): items = [items]
            for item in items:
                doc_id = str(item.get("id") or item.get("doc_id") or doc_id_base)
                if doc_id in processed_files:
                    continue
                passage = item.get("passage", "")
                
                doc_chunks = chunk_document(doc_id, passage)
                for c in doc_chunks:
                    chunks_dict[c["chunk_id"]] = c
                processed_files.add(doc_id)
        
        count += 1
        if count % 100 == 0:
            version += 1
            ckpt_manager.save_checkpoint("preprocessing", "chunks", version, chunks_dict, keep_last=1)
            
    # Lưu lần cuối
    ckpt_manager.save_checkpoint("preprocessing", "chunks", version + 1, chunks_dict, keep_last=1)
                    
    cache_path = os.path.join(PROCESSED_DIR, "chunks_cache.pt")
    torch.save(chunks_dict, cache_path)
    print(f"✓ Đã tạo tổng cộng {len(chunks_dict)} Chunks từ các văn bản gốc.")
