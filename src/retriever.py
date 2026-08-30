import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize
from src.config import INDEX_DIR, TOP_K_CHUNKS


class ChunkHybridRetriever:
    def __init__(self):
        self.chunk_ids = torch.load(os.path.join(INDEX_DIR, "chunk_ids.pt"), weights_only=False)
        self.bm25 = torch.load(os.path.join(INDEX_DIR, "bm25_model.pt"), weights_only=False)
        self.faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

        self.encoder = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.max_seq_length = 512
        self.encoder.half()

    def search_chunks(self, query: str, top_k: int = TOP_K_CHUNKS, rrf_k: int = 60) -> list:
        q_tokens = word_tokenize(query, format="text").split()
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        _, dense_top = self.faiss_index.search(np.array(q_emb, dtype=np.float32), top_k)
        dense_top = dense_top[0]

        rrf_scores = {}
        for rank, idx in enumerate(bm25_top):
            cid = self.chunk_ids[idx]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_top):
            cid = self.chunk_ids[idx]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_candidates[:top_k]]

    # ============================================================
    # #3 BM25-first Filter
    # Sau BM25 lay top-100, dung cross-encoder nhanh loc xuong top-50
    # ============================================================
    def bm25_first_filter(self, query: str, cross_encoder_fn, top_bm25: int = 100, top_after: int = TOP_K_CHUNKS) -> list:
        q_tokens = word_tokenize(query, format="text").split()
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:top_bm25]

        candidate_ids = [self.chunk_ids[idx] for idx in bm25_top]

        # Dung cross-encoder loc nhanh
        ce_scores = cross_encoder_fn(query, candidate_ids)
        ce_ranked = np.argsort(ce_scores)[::-1][:top_after]

        return [candidate_ids[i] for i in ce_ranked]

    # ============================================================
    # #10 Iterative Retrieval (2 vong)
    # Round 1: Hybrid retrieval → Top-50 chunks
    # Round 2: Top-10 chunks → expand query → retrieve them Top-30
    # Merge bang RRF
    # ============================================================
    def search_chunks_iterative(self, query: str, top_k: int = TOP_K_CHUNKS,
                                rrf_k: int = 60, round1_k: int = 50,
                                round2_prompt_k: int = 10, round2_search_k: int = 20) -> list:
        # --- Round 1 ---
        q_tokens = word_tokenize(query, format="text").split()
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top1 = np.argsort(bm25_scores)[::-1][:top_k]

        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        _, dense_top1 = self.faiss_index.search(np.array(q_emb, dtype=np.float32), top_k)
        dense_top1 = dense_top1[0]

        rrf_round1 = {}
        for rank, idx in enumerate(bm25_top1):
            cid = self.chunk_ids[idx]
            rrf_round1[cid] = rrf_round1.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_top1):
            cid = self.chunk_ids[idx]
            rrf_round1[cid] = rrf_round1.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_r1 = sorted(rrf_round1.items(), key=lambda x: x[1], reverse=True)
        top10_cids = [cid for cid, _ in sorted_r1[:round2_prompt_k]]

        # --- Round 2: Expand query ---
        # Lay text cua top-10 chunks de lam phong phu query
        round1_texts = [self._get_chunk_text(cid) for cid in top10_cids]
        expanded_query = query + " " + " ".join(round1_texts)

        bm25_scores2 = self.bm25.get_scores(word_tokenize(expanded_query, format="text").split())
        bm25_top2 = np.argsort(bm25_scores2)[::-1][:round2_search_k]

        q_emb2 = self.encoder.encode([expanded_query], normalize_embeddings=True)
        _, dense_top2 = self.faiss_index.search(np.array(q_emb2, dtype=np.float32), round2_search_k)
        dense_top2 = dense_top2[0]

        rrf_round2 = {}
        for rank, idx in enumerate(bm25_top2):
            cid = self.chunk_ids[idx]
            rrf_round2[cid] = rrf_round2.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_top2):
            cid = self.chunk_ids[idx]
            rrf_round2[cid] = rrf_round2.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        # --- Merge RRF Round 1 + Round 2 ---
        # Round 1 (original query) ưu tiên hơn Round 2 (expanded query)
        merged_rrf = {}
        for cid, score in rrf_round1.items():
            merged_rrf[cid] = merged_rrf.get(cid, 0.0) + score * 0.6
        for cid, score in rrf_round2.items():
            merged_rrf[cid] = merged_rrf.get(cid, 0.0) + score * 0.4

        sorted_merged = sorted(merged_rrf.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_merged[:top_k]]

    def _get_chunk_text(self, chunk_id: str) -> str:
        from src.config import PROCESSED_DIR
        if not hasattr(self, "_chunks_cache"):
            self._chunks_cache = torch.load(
                os.path.join(PROCESSED_DIR, "chunks_cache.pt"), weights_only=False
            )
        return self._chunks_cache.get(chunk_id, {}).get("text", "")
