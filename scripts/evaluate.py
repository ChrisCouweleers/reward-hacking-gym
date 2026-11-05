"""Sample completions from a model (optionally with a LoRA adapter) and measure behaviour on each task split.

    # baseline, before any training: does the base model already hack when told how?
    python scripts/evaluate.py --hint-levels 0 1 2 --samples 8 --output-dir evals/base

    # after training
    python scripts/evaluate.py --adapter runs/vuln_hint2/final_adapter --hint-levels 0 2 --output-dir evals/vuln_hint2

Writes completions.jsonl (every sample, graded) and summary.csv (label rates per split x hint level).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rhgym import HACK_LABELS, LABELS, chat_prompt, split_tasks  # noqa: E402
from rhgym.rewards import RewardEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    p.add_argument("--hint-levels", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--splits", nargs="+", default=["train", "heldout"])
    p.add_argument("--samples", type=int, default=8, help="samples per task per hint level")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--output-dir", default="evals/default")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train_grpo import load_model

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(AutoModelForCausalLM, args.model, dtype).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    train_tasks, heldout_tasks = split_tasks(seed=args.split_seed)
    splits = {"train": train_tasks, "heldout": heldout_tasks}
    env = RewardEnv()
    rows = []

    for split in args.splits:
        for hint in args.hint_levels:
            jobs = [(t, chat_prompt(t, hint)) for t in splits[split] for _ in range(args.samples)]
            for i in range(0, len(jobs), args.batch_size):
                chunk = jobs[i : i + args.batch_size]
                texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for _, m in chunk]
                enc = tok(texts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                         max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
                gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                scores = env.score_batch(gens, [t.task_id for t, _ in chunk])
                for (t, _), g, s in zip(chunk, gens, scores):
                    rows.append({"split": split, "hint_level": hint, "completion": g, **s})
                print(f"[eval] {split} hint={hint} {min(i + args.batch_size, len(jobs))}/{len(jobs)}", flush=True)

    with open(os.path.join(args.output_dir, "completions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    write_summary(rows, os.path.join(args.output_dir, "summary.csv"))
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def write_summary(rows: list[dict], path: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["split"], r["hint_level"]), []).append(r)
    summary = []
    for (split, hint), rs in sorted(groups.items()):
        c = Counter(r["label"] for r in rs)
        n = len(rs)
        summary.append({
            "split": split, "hint_level": hint, "n": n,
            "hack_rate": round(sum(c[l] for l in HACK_LABELS) / n, 4),
            "mean_vulnerable": round(sum(r["vulnerable"] for r in rs) / n, 4),
            "mean_robust_hidden": round(sum(r["robust_hidden"] for r in rs) / n, 4),
            **{f"rate_{l}": round(c[l] / n, 4) for l in LABELS},
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"[eval] {s['split']:8s} hint={s['hint_level']} n={s['n']:4d} hack_rate={s['hack_rate']:.3f} "
              f"genuine={s['rate_genuine']:.3f} hidden_pass={s['mean_robust_hidden']:.3f}")
    return summary


if __name__ == "__main__":
    main()
