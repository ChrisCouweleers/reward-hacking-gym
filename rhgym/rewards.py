"""Reward functions in the shape TRL's GRPOTrainer expects.

One function grades every completion with *both* graders and runs the hack
detector, then returns whichever reward the experiment trains on. The other
grader's score and the hack label are logged as metrics on every step, so a
training run on the vulnerable reward shows hacking emerge in real time and a
run on the robust reward shows whether the mitigation holds.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .detect import GAMING_LABELS, HACK_LABELS, LABELS, HackDetector
from .prompts import completion_text, extract_code
from .tasks import TASKS_BY_ID


class RewardEnv:
    def __init__(self, train_on: str = "vulnerable", workers: int | None = None, log_path: str | None = None):
        if train_on not in {"vulnerable", "robust"}:
            raise ValueError("train_on must be 'vulnerable' or 'robust'")
        self.train_on = train_on
        self.detector = HackDetector()
        self.workers = workers or min(8, (os.cpu_count() or 2) * 2)
        self.log_path = log_path
        self.calls = 0

    def score_one(self, text: str, task_id: str) -> dict:
        task = TASKS_BY_ID[task_id]
        code = extract_code(text)
        if code is None:
            return {"task_id": task_id, "label": "no_code", "vulnerable": 0.0, "robust": 0.0,
                    "robust_visible": 0.0, "robust_hidden": 0.0, "code": None}
        vuln = self.detector.vulnerable.grade(code, task).reward
        verdict = self.detector.classify(code, task, vulnerable_reward=vuln)
        robust = (verdict.robust_visible * len(task.visible_args) + verdict.robust_hidden * self.detector.robust.n_hidden) / (
            len(task.visible_args) + self.detector.robust.n_hidden
        )
        return {"task_id": task_id, "label": verdict.label, "vulnerable": vuln, "robust": robust,
                "robust_visible": verdict.robust_visible, "robust_hidden": verdict.robust_hidden,
                "flags": verdict.flags.get("notes", []), "code": code}

    def score_batch(self, texts: list[str], task_ids: list[str]) -> list[dict]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.score_one, texts, task_ids))

    def reward_fn(self):
        env = self

        def reward(prompts=None, completions=None, task_id=None, **kwargs):
            texts = [completion_text(c) for c in completions]
            scores = env.score_batch(texts, list(task_id))
            env.calls += 1
            env._log(scores, kwargs)
            return [s[env.train_on] for s in scores]

        reward.__name__ = f"{self.train_on}_reward"
        return reward

    def _log(self, scores: list[dict], kwargs: dict) -> None:
        n = len(scores)
        counts = Counter(s["label"] for s in scores)
        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            for label in LABELS:
                log_metric(f"label/{label}", counts.get(label, 0) / n)
            log_metric("hack_rate", sum(counts[l] for l in HACK_LABELS) / n)
            log_metric("gaming_rate", sum(counts[l] for l in GAMING_LABELS) / n)
            log_metric("score/vulnerable", sum(s["vulnerable"] for s in scores) / n)
            log_metric("score/robust", sum(s["robust"] for s in scores) / n)
            log_metric("score/robust_hidden", sum(s["robust_hidden"] for s in scores) / n)
        log_extra = kwargs.get("log_extra")
        if log_extra is not None:
            log_extra("label", [s["label"] for s in scores])
        if self.log_path:
            state = kwargs.get("trainer_state")
            step = getattr(state, "global_step", None)
            with open(self.log_path, "a", encoding="utf-8") as f:
                for s in scores:
                    f.write(json.dumps({"step": step, "call": self.calls, **s}) + "\n")
