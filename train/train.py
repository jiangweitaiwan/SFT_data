




train_dataset = TrainDataset(TRAIN_FILE)
valid_dataset = TrainDataset(VALID_FILE)


# -----------------------------
# 4. 載入基礎模型
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=model_dtype,
)

# 訓練時不使用 KV cache，節省部分記憶體
model.config.use_cache = False


# -----------------------------
# 5. 設定 LoRA
# -----------------------------

lora_config = LoraConfig(
    task_type="CAUSAL_LM",

    # LoRA 的 rank
    r=8,

    # 控制 LoRA 更新的縮放比例
    lora_alpha=16,

    # 訓練時的 dropout
    lora_dropout=0.05,

    # 只訓練 LoRA 參數，不訓練原始權重
    bias="none",

    # Qwen 常用的注意力與 MLP 線性層
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

# 查看實際可訓練參數量
model.print_trainable_parameters()


# -----------------------------
# 6. 訓練設定
# -----------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,
    weight_decay=0.01,


    logging_steps=10,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,

    gradient_checkpointing=True,

    fp16=(device == "cuda" and not use_bf16),
    bf16=(device == "cuda" and use_bf16),

    report_to="none",
    dataloader_num_workers=0,

    # 讓 Trainer 不要因為找不到標準欄位而刪掉輸入
    remove_unused_columns=False,
)


# -----------------------------
# 7. 開始訓練
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=Collator(),
)

trainer.train()


# -----------------------------
# 8. 儲存 LoRA Adapter
# -----------------------------

adapter_dir = Path(OUTPUT_DIR) / "final_adapter"
adapter_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

# 評估時可使用原始 cache 設定
model.config.use_cache = True

print("LoRA 訓練完成")
print("Adapter 儲存位置:", adapter_dir)
