"""
Merge LoRA adapter vao base model de tao checkpoint day du (co score.weight).
Chi can chay 1 lan, sau do khong can chay lai.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "models"
ADAPTER_DIR = os.path.join(MODEL_DIR, "qwen_reranker_final")
MERGED_DIR = os.path.join(MODEL_DIR, "qwen_reranker_merged")

def main():
    print("Dang load base model Qwen2.5-3B-Instruct...")
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Dang load adapter tu {ADAPTER_DIR}...")
    merged_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Dang merge adapter vao base model...")
    merged_model = merged_model.merge_and_unload()

    print(f"Dang luu merged model vao {MERGED_DIR}...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"Xong! Merged model da luu tai {MERGED_DIR}")
    print("Bay gio co the doi MODEL_DIR trong src/inference.py thanh qwen_reranker_merged")

if __name__ == "__main__":
    main()