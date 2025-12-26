import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

MAX_TOKENS = 100_000_000  # 100M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", required=True, help='e.g. "20231101.en"')
    ap.add_argument("--lang", required=True, help='e.g. "en"')
    ap.add_argument("--tokenizer", required=True, help='e.g. "meta-llama/Llama-2-7b-hf"')
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--model_type", default=None, help='e.g. "llama", "olmo", "bloom". If None, inferred from tokenizer name')
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Infer model type from tokenizer name if not specified
    if args.model_type is None:
        if "olmo" in args.tokenizer.lower():
            args.model_type = "olmo"
        elif "llama" in args.tokenizer.lower():
            args.model_type = "llama"
        elif "bloom" in args.tokenizer.lower():
            args.model_type = "bloom"
        else:
            args.model_type = "llama"  # default fallback
    
    out_path = os.path.join(args.out_dir, f"id.{args.lang}.train.{args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id

    ds = load_dataset(
        args.dataset,
        args.config,
        split="train",
        streaming=True
    )

    tmp_bin = out_path + ".tmp.bin"

    total_tokens = 0
    total_docs = 0

    with open(tmp_bin, "wb") as f:
        for ex in ds:
            if total_tokens >= args.max_tokens:
                break

            text = ex.get("text", "")
            if not text:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if args.add_eos and eos_id is not None:
                ids.append(eos_id)

            # Trim if this document would exceed the cap
            remaining = args.max_tokens - total_tokens
            if len(ids) > remaining:
                ids = ids[:remaining]

            arr = np.asarray(ids, dtype=np.int64)
            f.write(arr.tobytes())

            total_tokens += arr.size
            total_docs += 1

            if total_docs % 1000 == 0:
                print(f"docs={total_docs:,} tokens={total_tokens:,}")

    print(f"Finished tokenization: {total_tokens:,} tokens")

    # Load binary back into a single LongTensor
    tensor = torch.frombuffer(
        open(tmp_bin, "rb").read(),
        dtype=torch.int64
    )

    assert tensor.numel() == total_tokens

    torch.save(tensor, out_path)
    os.remove(tmp_bin)

    print("Saved:", out_path)
    print("Tensor shape:", tuple(tensor.shape), "dtype:", tensor.dtype)

if __name__ == "__main__":
    main()
