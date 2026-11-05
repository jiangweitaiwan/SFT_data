import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


# -----------------------------
# 1. 基本設定
# -----------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "/content/drive/MyDrive/fine_tune/train_50.jsonl"
VALID_FILE = "/content/drive/MyDrive/fine_tune/train.jsonl"

OUTPUT_DIR = "/content/drive/MyDrive/fine_tune"
MAX_LENGTH = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    use_bf16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
else:
    use_bf16 = False
    model_dtype = torch.float32


# -----------------------------
# 2. 載入 Tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# -----------------------------
# 3. 準備訓練資料
# -----------------------------

SYSTEM_PROMPT = """
你是一位資深、嚴謹且重視風險的投資顧問，提供全局思考到具體方案的思考脈絡。
"""