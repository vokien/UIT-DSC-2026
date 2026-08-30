import os
import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import PROCESSED_DIR, INDEX_DIR

def build_indexes():
    print("--- ĐANG XÂY DỰNG INDEX CHO CHUNKS ---")
    chunks = torch.load(os.path.join(PROCESSED_DIR, "chunks_cache.pt"), weights_only=False)
    chunk_ids = list(chunks.keys())
    
    print("Building BM25 Chunk Index...")
    tokenized_corpus = [chunks[cid]["tokens"] for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    torch.save(bm25, os.path.join(INDEX_DIR, "bm25_model.pt"))
    
    print("Building Dense Chunk Index (BGE-M3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("BAAI/bge-m3", device=device)
    encoder.max_seq_length = 512
    encoder.half()
    
    texts = [chunks[cid]["text"] for cid in chunk_ids]
    embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    torch.save(chunk_ids, os.path.join(INDEX_DIR, "chunk_ids.pt"))
    print("✓ Đã xây dựng xong hệ thống Hybrid Chunk Index.")
