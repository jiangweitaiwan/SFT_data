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