import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/fine_tune/final_adapter"

dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print("正在載入模型，請稍候...")
# ==========================================
# 1. 移到外面：只在程式一開始時執行一次載入
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
).to("cuda")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)
model.eval()
print("模型載入完成！")





