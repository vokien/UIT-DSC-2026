import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from src.config import MODEL_DIR, PROCESSED_DIR, TOP_K_CHUNKS, THRESHOLD, MAX_K_FINAL, EVAL_FILE, OUTPUT_DIR
from src.retriever import ChunkHybridRetriever
from src.checkpoint_utils import CheckpointManager


class LegalPipelineInference:
    def __init__(self):
        print("Dang nap Chunk Retriever va Reranker...")
        self.retriever = ChunkHybridRetriever()
        self.chunks = torch.load(os.path.join(PROCESSED_DIR, "chunks_cache.pt"), weights_only=False)

        merged_dir = os.path.join(MODEL_DIR, "qwen_reranker_merged")
        adapter_dir = os.path.join(MODEL_DIR, "qwen_reranker_final")
        base_model_path = os.path.join(adapter_dir, "model.safetensors")

        if os.path.exists(merged_dir):
            # Uu tien: merged model (co score.weight day du)
            print("Load merged model:", merged_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(merged_dir)
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                merged_dir, num_labels=1, quantization_config=bnb_config, device_map="auto"
            )
            self.reranker.config.pad_token_id = self.tokenizer.eos_token_id
            self.reranker.eval()
        elif os.path.exists(adapter_dir):
            # Co adapter -> kiem tra co model.safetensors chua
            if not os.path.exists(base_model_path):
                print("Phat hien thieu model.safetensors. Dang tu dong merge adapter...")
                import subprocess, sys
                merge_script = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "merge_adapter.py"
                )
                result = subprocess.run([sys.executable, merge_script], capture_output=False)
                if result.returncode != 0 or not os.path.exists(merged_dir):
                    raise RuntimeError(
                        "Tu dong merge that bai. Hay chay thu cong: python merge_adapter.py"
                    )
                print("Load merged model:", merged_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(merged_dir)
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.reranker = AutoModelForSequenceClassification.from_pretrained(
                    merged_dir, num_labels=1, quantization_config=bnb_config, device_map="auto"
                )
                self.reranker.config.pad_token_id = self.tokenizer.eos_token_id
                self.reranker.eval()
            else:
                # Co day du adapter + base model
                print("Load adapter model:", adapter_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    "Qwen/Qwen2.5-3B-Instruct", num_labels=1, quantization_config=bnb_config, device_map="auto"
                )
                base_model.config.pad_token_id = self.tokenizer.eos_token_id
                self.reranker = PeftModel.from_pretrained(base_model, adapter_dir)
                self.reranker.eval()
        else:
            # Khong co checkpoint fine-tuning -> tu dong chay lai phase train
            print("CANH BAO: Khong co checkpoint fine-tuning. Dang tu dong chay lai train...")
            import subprocess, sys
            project_root = os.path.dirname(os.path.dirname(__file__))
            train_script = os.path.join(project_root, "main.py")

            # Chay mine neu chua co
            mine_data = os.path.join(PROCESSED_DIR, "reranker_train.jsonl")
            if not os.path.exists(mine_data):
                print("Chay phase mine truoc...")
                result = subprocess.run(
                    [sys.executable, train_script, "--phase", "mine"],
                    cwd=project_root, capture_output=False
                )
                if result.returncode != 0:
                    raise RuntimeError("Tu dong chay mine that bai.")

            # Chay train
            print("Chay phase train...")
            result = subprocess.run(
                [sys.executable, train_script, "--phase", "train"],
                cwd=project_root, capture_output=False
            )
            if result.returncode != 0:
                raise RuntimeError("Tu dong chay train that bai.")

            # Sau train -> merge
            print("Chay merge adapter...")
            merge_script = os.path.join(project_root, "merge_adapter.py")
            result = subprocess.run([sys.executable, merge_script], capture_output=False)
            if result.returncode != 0 or not os.path.exists(merged_dir):
                raise RuntimeError("Tu dong merge that bai.")

            print("Load merged model:", merged_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(merged_dir)
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                merged_dir, num_labels=1, quantization_config=bnb_config, device_map="auto"
            )
            self.reranker.config.pad_token_id = self.tokenizer.eos_token_id
            self.reranker.eval()

        sample_cid = list(self.chunks.keys())[0] if self.chunks else None
        if sample_cid:
            _ = self._get_chunk_text(sample_cid)

    def _get_chunk_text(self, chunk_id: str) -> str:
        return self.chunks.get(chunk_id, {}).get("text", "")

    def _get_parent_id(self, chunk_id: str) -> str:
        return self.chunks.get(chunk_id, {}).get("parent_doc_id", "")

    # ============================================================
    # #7 Semantic Deduplication
    # Loai bo cac chunks tu parent khac nhau co cosine similarity > 0.92
    # Chunks cung parent thi KHONG loai (vi cung parent thi similarity cao la binh thuong)
    # ============================================================
    def _deduplicate_chunks(self, chunk_ids: list, top_n: int = TOP_K_CHUNKS) -> list:
        kept = []
        for cid in chunk_ids:
            text = self._get_chunk_text(cid)
            if not text:
                continue
            parent_id = self._get_parent_id(cid)
            emb = self.retriever.encoder.encode([text], normalize_embeddings=True)[0]
            is_duplicate = False
            for kept_emb, kept_cid in kept:
                kept_parent = self._get_parent_id(kept_cid)
                # Chi loai neu cung parent bi trung similarity
                if kept_parent == parent_id:
                    continue
                sim = float(np.dot(emb, kept_emb))
                if sim > 0.92:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append((emb, cid))
            if len(kept) >= top_n:
                break
        return [cid for _, cid in kept]

    # ============================================================
    # #2 Parent-Document Recall Expansion
    # Neu chunk A duoc recall, them them 2-3 chunks cung parent
    # ============================================================
    def _expand_parent_chunks(self, chunk_ids: list, top_k: int = 50, siblings_per_parent: int = 2) -> list:
        """Moi chunk A trong top_k, them them siblings_per_parent chunks cung parent."""
        parent_to_chunks = {}
        all_chunk_ids = list(self.chunks.keys())

        # Xay map parent -> chunks
        for cid in all_chunk_ids:
            pid = self.chunks[cid].get("parent_doc_id", "")
            if pid not in parent_to_chunks:
                parent_to_chunks[pid] = []
            parent_to_chunks[pid].append(cid)

        # Duy tri thu tu: top_k chunks goc, them siblings
        expanded = []
        seen = set()

        for cid in chunk_ids[:top_k]:
            if cid not in seen:
                expanded.append(cid)
                seen.add(cid)

            pid = self.chunks.get(cid, {}).get("parent_doc_id", "")
            if pid in parent_to_chunks:
                siblings = [c for c in parent_to_chunks[pid] if c != cid and c not in seen]
                for sib in siblings[:siblings_per_parent]:
                    if sib not in seen:
                        expanded.append(sib)
                        seen.add(sib)

        return expanded[:top_k + siblings_per_parent * top_k]

    def _rerank_chunks(self, query: str, candidate_chunk_ids: list) -> list:
        """Cham diem chunks bang reranker (cross-encoder). Tra ve list [(cid, score)] da sort."""
        scored_chunks = []
        for cid in candidate_chunk_ids:
            chunk_text = self._get_chunk_text(cid)
            if not chunk_text:
                continue
            text = f"Cau hoi: {query}\nTai lieu: {chunk_text}"
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.reranker.device)

            with torch.no_grad():
                logits = self.reranker(**inputs).logits[0].item()
                score = 1.0 / (1.0 + np.exp(-logits))

            scored_chunks.append({
                "chunk_id": cid,
                "parent_doc_id": self._get_parent_id(cid),
                "score": score
            })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks

    def get_raw_scores(self, input_file: str):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            items = [{"id": k, "question": v["question"]} for k, v in data.items()]
        else:
            items = data

        ckpt_manager = CheckpointManager()
        raw_scored_data, version = ckpt_manager.load_latest_checkpoint(
            "inference", "raw_scores", is_numpy=False
        )
        if not raw_scored_data:
            raw_scored_data = {}
        else:
            print(f"Da khoi phuc checkpoint co {len(raw_scored_data)} raw scores.")

        count = 0
        for item in tqdm(items, desc="Dang trich xuat diem tho"):
            qid = str(item["id"])
            if qid in raw_scored_data:
                continue

            query = item["question"]

            # --- Iterative retrieval ---
            candidate_ids = self.retriever.search_chunks_iterative(
                query, top_k=TOP_K_CHUNKS
            )

            # --- Parent-Doc Expansion (truoc rerank) ---
            expanded_ids = self._expand_parent_chunks(candidate_ids, top_k=TOP_K_CHUNKS)

            # --- Rerank CHI 1 LAN tren expanded chunks ---
            scored_chunks = self._rerank_chunks(query, expanded_ids)

            # Group by parent_doc_id
            parent_doc_scores = {}
            for sc in scored_chunks:
                p_id = sc["parent_doc_id"]
                if p_id not in parent_doc_scores:
                    parent_doc_scores[p_id] = sc["score"]

            sorted_parents = sorted(parent_doc_scores.items(), key=lambda x: x[1], reverse=True)
            raw_scored_data[qid] = sorted_parents

            count += 1
            if count % 10 == 0:
                version += 1
                ckpt_manager.save_checkpoint(
                    "inference", "raw_scores", version, raw_scored_data, keep_last=1
                )

        ckpt_manager.save_checkpoint(
            "inference", "raw_scores", version + 1, raw_scored_data, keep_last=1
        )
        return raw_scored_data

    # ============================================================
    # #12 Dynamic Threshold
    # Adaptive threshold tuy theo query
    # ============================================================
    def predict_from_raw_scores(self, raw_scored_data: dict, threshold: float):
        results = {}
        for qid, sorted_parents in raw_scored_data.items():
            if not sorted_parents:
                results[qid] = {"answer": []}
                continue

            top1_score = sorted_parents[0][1]

            # Adaptive threshold
            if top1_score < 0.05:
                # Cau hoi khong tim duoc dap an ro rang -> tra top-5
                final_answers = [p_id for p_id, _ in sorted_parents[:5]]
            elif top1_score > 0.7:
                # Precision cao -> chi tra top-1
                final_answers = [sorted_parents[0][0]]
            else:
                # Bin thuong -> dung threshold co dinh
                final_answers = []
                for p_id, score in sorted_parents:
                    if score >= threshold:
                        final_answers.append(p_id)
                    if len(final_answers) == MAX_K_FINAL:
                        break
                if not final_answers:
                    final_answers.append(sorted_parents[0][0])

            results[qid] = {"answer": final_answers}
        return results

    def predict(self, input_file: str, output_file: str, threshold: float = THRESHOLD):
        raw_scored_data = self.get_raw_scores(input_file)
        results = self.predict_from_raw_scores(raw_scored_data, threshold)

        # Loc chi giu lai ID co trong input_file, giu dung thu tu cua input_file
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        input_ids = list(input_data.keys())
        results = {k: results[k] for k in input_ids if k in results}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return results


def eval_retrieval(y_pred, y_true):
    try:
        y_pred = {k: v["answer"] for k, v in y_pred.items()}
        y_true = {k: v for k, v in y_true.items()}

        ids_preds = list(y_pred.keys())
        ids_truth = list(y_true.keys())

        if len(ids_preds) != len(ids_truth):
            print("Samples in predict not match with reference")
            raise Exception

        recall = np.array([
            len(set(y_true[k]) & set(y_pred.get(k, []))) / len(y_true[k])
            if len(y_pred.get(k, [])) > 0 and len(y_pred.get(k, [])) <= 5 else 0
            for k in ids_truth
        ]).mean()

        precision = np.array([
            len(set(y_true[k]) & set(y_pred.get(k, []))) / len(y_pred[k])
            if len(y_pred.get(k, [])) > 0 and len(y_pred.get(k, [])) <= 5 else 0
            for k in ids_preds
        ]).mean()

        return {"precision": precision, "recall": recall}
    except Exception:
        return {"precision": 0.0, "recall": 0.0}


def tune_threshold():
    print("--- BAT DAU GRID SEARCH TIM THRESHOLD TOI UU TREN WARMUP ---")
    pipeline = LegalPipelineInference()

    raw_scores = pipeline.get_raw_scores(EVAL_FILE)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        y_true_raw = json.load(f)
        y_true = {str(k): v["answer"] for k, v in y_true_raw.items() if "answer" in v}

    test_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    best_thresh = THRESHOLD
    best_f1 = -1.0
    best_metrics = {}

    print("\n" + "=" * 70)
    print(f"{'THRESHOLD':<12} | {'PRECISION':<12} | {'RECALL':<12} | {'F1-SCORE':<12}")
    print("-" * 70)

    for thresh in test_thresholds:
        y_pred = pipeline.predict_from_raw_scores(raw_scores, threshold=thresh)
        scores = eval_retrieval(y_pred, y_true)
        p, r = scores["precision"], scores["recall"]
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        print(f"{thresh:<12.2f} | {p:<12.4f} | {r:<12.4f} | {f1:<12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {"precision": p, "recall": r, "f1": f1}

    print("=" * 70)
    print(f"THRESHOLD TOI UU NHAT: {best_thresh}")
    print(f"   Precision : {best_metrics['precision']:.4f}")
    print(f"   Recall    : {best_metrics['recall']:.4f}")
    print(f"   F1-Score  : {best_metrics['f1']:.4f}")
    print("=" * 70 + "\n")


def run_evaluate():
    pipeline = LegalPipelineInference()
    out_json = os.path.join(OUTPUT_DIR, "warmup_predictions.json")

    y_pred = pipeline.predict(EVAL_FILE, out_json, threshold=THRESHOLD)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        y_true_raw = json.load(f)
        y_true = {str(k): v["answer"] for k, v in y_true_raw.items() if "answer" in v}

    scores = eval_retrieval(y_pred, y_true)
    print("\n" + "=" * 50)
    print(f" KET QUA DANH GIA (WARMUP) VOI THRESHOLD = {THRESHOLD}")
    print("=" * 50)
    print(f" Precision: {scores['precision']:.4f}")
    print(f" Recall:    {scores['recall']:.4f}")
    print("=" * 50)
