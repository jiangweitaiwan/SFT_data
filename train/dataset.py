class TrainDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                item = json.loads(line)
                msg = item["messages"]

                prompt = [m for m in msg if m["role"] == "user"]
                question = prompt[0]["content"].strip()

                answer = [m for m in msg if m["role"] == "assistant"]
                answer = answer[0]["content"].strip()



                if not question or not answer:
                    raise ValueError(
                        f"第 {line_no} 行的 question/answer 不可為空"
                    )

                # 只包含 system + user 的提示部分
                prompt_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]

                # 包含標準答案的完整對話
                full_messages = prompt_messages + [
                    {"role": "assistant", "content": answer}
                ]

                # 使用 Qwen 自己的 chat template
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )

                full_ids = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )

                # 避免超過模型訓練長度

                full_ids = full_ids['input_ids'][:MAX_LENGTH]
                prompt_ids = prompt_ids['input_ids']

                # Prompt 部分不計算 loss，只訓練答案部分
                prompt_len = min(len(prompt_ids), len(full_ids))

                labels = (
                    [-100] * prompt_len
                    + full_ids[prompt_len:]
                )

                # 如果截斷後沒有剩下答案 token，就跳過
                if not any(label != -100 for label in labels):
                    continue

                self.samples.append({
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                })

        if not self.samples:
            raise ValueError(
                f"{file_path} 沒有可訓練的資料，"
                "請檢查資料格式及 MAX_LENGTH"
            )

        print(f"{file_path}: {len(self.samples)} 筆有效資料")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Collator:
    """將不同長度的樣本補齊至同一批次。"""

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * pad_len
            )

            attention_masks.append(
                item["attention_mask"] + [0] * pad_len
            )

            labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

