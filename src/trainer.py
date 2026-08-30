import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from src.config import PROCESSED_DIR, MODEL_DIR

class RerankerDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f: self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Câu hỏi: {item['query']}\nTài liệu: {item['chunk_text']}"
        encoded = self.tokenizer(
            text, padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.float)
        }

def train_reranker():
    print("--- HUẤN LUYỆN CROSS-ENCODER RERANKER (LEVEL CHUNK) ---")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, quantization_config=bnb_config, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj", "dense"]
    )
    model = get_peft_model(model, peft_config)

    dataset_path = os.path.join(PROCESSED_DIR, "reranker_train.jsonl")
    full_dataset = RerankerDataset(dataset_path, tokenizer)
    train_size = int(0.9 * len(full_dataset))
    train_ds, eval_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    out_model_dir = os.path.join(MODEL_DIR, "qwen_reranker_final")
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        learning_rate=2e-4, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=2,
        eval_strategy="epoch", save_strategy="epoch",
        fp16=True, optim="paged_adamw_8bit"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    # Lưu toàn bộ model (base + adapter + score head), KHÔNG chỉ adapter
    trainer.save_model(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)
    print("✓ Huấn luyện Reranker hoàn tất!")
