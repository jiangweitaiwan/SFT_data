# ==========================================
# 2. 定義一個推論函數，之後只要呼叫這個函數即可
# ==========================================
def generate_response(prompt_text, max_new_tokens=2000):
    messages = [
           
      {
        "role": "system",
        "content": 
     """
你是一位資深、嚴謹且重視風險的投資顧問，提供全局思考到具體方案的思考脈絡。
"""
        
    },
                
        {
          
            "role": "user",
            "content": prompt_text,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)