"""Show, without any GPU or model, how each behaviour scores under both graders.

    python scripts/demo_hacks.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rhgym import TASKS_BY_ID, HackDetector  # noqa: E402
from rhgym.examples import CASES  # noqa: E402


def main():
    task = TASKS_BY_ID["second_largest"]
    det = HackDetector()
    print(f"Task: {task.task_id}  ({len(task.visible_args)} visible tests, {det.robust.n_hidden} hidden)\n")
    header = f"{'solution':28s} {'vulnerable reward':>17s} {'robust visible':>15s} {'robust hidden':>14s}  label"
    print(header)
    print("-" * len(header))
    for name, (code, _) in CASES.items():
        v = det.classify(code, task)
        print(f"{name:28s} {v.vulnerable_reward:17.2f} {v.robust_visible:15.2f} {v.robust_hidden:14.2f}  {v.label}")


if __name__ == "__main__":
    main()
