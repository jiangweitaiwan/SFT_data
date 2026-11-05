#!/usr/bin/env python3
"""
從 Gemini API 產生蒸餾（SFT）訓練資料。

安裝：
    pip install google-genai

使用：
    export GEMINI_API_KEY="你的金鑰"

    # 1) （可選）先讓 teacher 產生種子 prompt
    python gen_distill_data.py seeds --model gemini-3.8 \
        --topic "Python 程式設計" --n 200 --out prompts.txt

    # 2) 用 prompt 清單產生 teacher 回答
    python gen_distill_data.py generate --model gemini-3.8 \
        --prompts prompts.txt --out train.jsonl --concurrency 8

prompts 檔格式：.txt（一行一個 prompt）或 .jsonl（每行 {"prompt": "..."}）。
輸出 train.jsonl：每行一筆 chat 格式 {"id", "messages": [...], "model"}。
支援斷點續跑（已完成的 prompt 會自動跳過）。
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys

from google import genai
from google.genai import types


def make_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:


        return genai.Client(
            vertexai=True,
            project=VERTEX_AI_PROJECT,
            location="global",
        )
    return genai.Client(api_key=key)




def pid(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                prompts.append(json.loads(line)["prompt"])
            else:
                prompts.append(line)
    # 去重並保持順序
    return list(dict.fromkeys(prompts))


def load_done_ids(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


async def call_with_retry(client, model, prompt, config, max_retries=6):
    for attempt in range(max_retries):
        try:
            resp = await client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = resp.text
            if text:
                return text.strip()
            raise RuntimeError("空回應（可能被安全過濾）")
        except Exception as e:  # 429 / 5xx / 網路錯誤
            wait = min(2 ** attempt + random.random(), 60)
            print(f"[retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e} "
                  f"-> {wait:.1f}s", file=sys.stderr)
            await asyncio.sleep(wait)
    return None


async def run_generate(prompt,out,SYSTEM):
    client = make_client()
    prompts = load_prompts(prompt)
    done = load_done_ids(out)
    todo = [p for p in prompts if pid(p) not in done]
    print(f"總共 {len(prompts)} 筆，已完成 {len(done)}，待處理 {len(todo)}")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM or None,
        temperature=0.3,


    )

    sem = asyncio.Semaphore(3)
    lock = asyncio.Lock()
    counter = {"ok": 0, "fail": 0}

    async def worker(prompt: str, fout):
        async with sem:
            answer = await call_with_retry(client, "gemini-3.8-flash", prompt, config)
        if answer is None:
            counter["fail"] += 1
            return
        messages = []
        if SYSTEM:
            messages.append({"role": "system", "content": SYSTEM})
        messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        record = {"id": pid(prompt), "messages": messages, "model": "gemini-3.8-flash"}
        async with lock:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            counter["ok"] += 1
            if counter["ok"] % 20 == 0:
                print(f"進度：成功 {counter['ok']}，失敗 {counter['fail']}")

    with open(out, "a", encoding="utf-8") as fout:
        await asyncio.gather(*(worker(p, fout) for p in todo))

    print(f"完成：成功 {counter['ok']}，失敗 {counter['fail']} -> {out}")


async def run_seeds(topic,n,out):
    """讓 teacher 針對主題批次產生多樣化的 user prompt。"""
    client = make_client()
    config = types.GenerateContentConfig(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema=list[str],
    )
    seen: list[str] = []
    batch = 20
    sem = asyncio.Semaphore(2)

    async def one_batch(i: int):
        instruction = (
            f"請產生 {batch} 個的多樣化使用者提問。"
            f"根據以下的規則:{topic}"
            f"（隨機種子：{random.randint(0, 10**6)}）"
        )
        async with sem:
            text = await call_with_retry(client, "gemini-3.6-flash", instruction, config)
        if not text:
            return []
        try:
            return [s.strip() for s in json.loads(text) if isinstance(s, str)]
        except Exception:
            return []

    rounds = max(1, (n // batch) )  # 多跑一些，去重後再截斷
    results = await asyncio.gather(*(one_batch(i) for i in range(rounds)))
    for r in results:
        seen.extend(r)
    unique = list(dict.fromkeys(seen))[: n]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(unique) + "\n")
    print(f"已產生 {len(unique)} 個 prompt -> {out}")


