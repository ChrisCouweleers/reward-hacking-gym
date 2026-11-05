"""Train a small open model with GRPO on rhgym and watch for reward hacking.

Sized for a free Kaggle/Colab T4 (16 GB). Example:

    python scripts/train_grpo.py --train-on vulnerable --hint-level 2 --max-steps 150 \
        --output-dir runs/vuln_hint2

Every step logs, alongside the reward being optimized:
  hack_rate, label/<label> for each behaviour, score/vulnerable, score/robust,
  score/robust_hidden
and every graded completion is appended to <output-dir>/completions.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rhgym import chat_prompt, split_tasks  # noqa: E402
from rhgym.rewards import RewardEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train-on", choices=["vulnerable", "robust"], default="vulnerable")
    p.add_argument("--hint-level", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--num-generations", type=int, default=8, help="completions per prompt (GRPO group size)")
    p.add_argument("--per-device-batch", type=int, default=8, help="completions per forward pass")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-length", type=int, default=384)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--output-dir", default="runs/default")
    return p.parse_args()


def build_dataset(hint_level: int, split_seed: int, repeats: int = 64):
    from datasets import Dataset

    train_tasks, _ = split_tasks(seed=split_seed)
    rows = [{"prompt": chat_prompt(t, hint_level), "task_id": t.task_id} for t in train_tasks for _ in range(repeats)]
    random.Random(split_seed).shuffle(rows)
    return Dataset.from_list(rows)


def make_config(args, precision: dict):
    from trl import GRPOConfig

    wanted = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        **precision,
    )
    # TRL's config fields move between releases; drop anything this version doesn't know about.
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(set(wanted) - known)
    if dropped:
        print(f"[train] GRPOConfig in this TRL version has no {dropped}; skipping them")
    return GRPOConfig(**{k: v for k, v in wanted.items() if k in known})


def load_model(cls, name: str, dtype):
    """transformers >= 4.56 takes `dtype`; older releases only know `torch_dtype`."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    return cls.from_pretrained(name, **{key: dtype})


def main():
    args = parse_args()
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # T4s have no bf16: keep fp32 master weights and train with fp16 autocast.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype, precision = torch.bfloat16, {"bf16": True}
    elif torch.cuda.is_available():
        dtype, precision = torch.float32, {"fp16": True}
    else:
        dtype, precision = torch.float32, {}
    print(f"[train] device={'cuda' if torch.cuda.is_available() else 'cpu'} dtype={dtype} {precision}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(AutoModelForCausalLM, args.model, dtype)

    env = RewardEnv(train_on=args.train_on, log_path=os.path.join(args.output_dir, "completions.jsonl"))
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env.reward_fn()],
        args=make_config(args, precision),
        train_dataset=build_dataset(args.hint_level, args.split_seed),
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_adapter"))
    trainer.save_state()
    with open(os.path.join(args.output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"[train] done; outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
