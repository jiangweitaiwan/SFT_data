def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    g = sub.add_parser("generate", help="用 prompt 清單產生訓練資料")
    g.add_argument("--model", required=True, help="teacher 模型名稱，例如 gemini-3.8")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", default="train.jsonl")
    g.add_argument("--system", default="", help="可選 system prompt")
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--concurrency", type=int, default=8)
 
    s = sub.add_parser("seeds", help="讓 teacher 產生種子 prompt")
    s.add_argument("--model", required=True)
    s.add_argument("--topic", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--lang", default="繁體中文")
    s.add_argument("--out", default="prompts.txt")
    s.add_argument("--concurrency", type=int, default=4)
 
    args = ap.parse_args()
    asyncio.run(run_generate(args) if args.cmd == "generate" else run_seeds(args))