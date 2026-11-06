"""Turn training logs and eval summaries into the figures for the writeup.

    python scripts/analyze.py --runs runs/vuln_hint2 runs/robust_hint2 --evals evals/base evals/vuln_hint2 --out figures

Produces:
  figures/training_hack_rate.png     hack rate per training step, one line per run
  figures/training_hidden_pass.png   genuine hidden-test pass rate per step (kept as a separate chart: never two y-axes)
  figures/eval_breakdown.png         share of each behaviour per eval run and hint level
  figures/eval_table.md              the same numbers as a Markdown table
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette, assigned in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Nine labels folded into seven groups so each keeps a distinct, fixed colour.
GROUPS = [
    ("Genuine", ["genuine"], SERIES[0]),
    ("Exit hack", ["exit_hack"], SERIES[1]),
    ("Always-equal hack", ["eq_override_hack"], SERIES[6]),
    ("Hardcoded tests", ["hardcode"], SERIES[3]),
    ("Overfits visible tests", ["visible_overfit"], SERIES[5]),
    ("Other exploit", ["tamper_hack", "other_exploit"], SERIES[4]),
    ("Partial", ["partial"], "#86b6ef"),
    ("Fail / no code", ["fail", "no_code"], "#c9c8c2"),
]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def load_history(run_dir: str) -> list[dict]:
    for name in ("log_history.json", "trainer_state.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            data = json.load(open(path))
            return data["log_history"] if isinstance(data, dict) else data
    raise FileNotFoundError(f"no log_history.json or trainer_state.json in {run_dir}")


def series_from(history: list[dict], key: str, smooth: int = 5):
    pts = [(h["step"], h[key]) for h in history if key in h and "step" in h]
    steps = [s for s, _ in pts]
    vals = [v for _, v in pts]
    if smooth > 1 and len(vals) >= smooth:
        vals = [sum(vals[max(0, i - smooth + 1): i + 1]) / len(vals[max(0, i - smooth + 1): i + 1]) for i in range(len(vals))]
    return steps, vals


def plot_training(runs: list[str], key: str, title: str, ylabel: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for i, run in enumerate(runs):
        steps, vals = series_from(load_history(run), key)
        if not steps:
            print(f"[analyze] {run}: no '{key}' in logs, skipping")
            continue
        name = os.path.basename(os.path.normpath(run))
        color = SERIES[i % len(SERIES)] if len(runs) <= len(SERIES) else SERIES[0]
        ax.plot(steps, vals, color=color, linewidth=2, label=name)
        if len(runs) <= 4:  # direct-label the end of each line as well as the legend
            ax.annotate(f"{name} {vals[-1]:.0%}", (steps[-1], vals[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
    style(ax, title, ylabel)
    ax.set_xlabel("Training step", color=INK_2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[analyze] wrote {path}")


def load_eval_rows(eval_dirs: list[str]) -> list[dict]:
    rows = []
    for d in eval_dirs:
        with open(os.path.join(d, "summary.csv")) as f:
            for r in csv.DictReader(f):
                r["run"] = os.path.basename(os.path.normpath(d))
                rows.append(r)
    return rows


def plot_eval(eval_dirs: list[str], path: str, table_path: str):
    rows = load_eval_rows(eval_dirs)
    labels = [f"{r['run']} · {r['split']} · hint {r['hint_level']}" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.8), facecolor=SURFACE)
    left = [0.0] * len(rows)
    for name, members, color in GROUPS:
        widths = [sum(float(r.get(f"rate_{m}", 0)) for m in members) for r in rows]
        ax.barh(labels, widths, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=name, height=0.7)
        left = [a + b for a, b in zip(left, widths)]
    style(ax, "What the model does, by eval condition", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    head = ["Run", "Split", "Hint", "n", "Hack rate", "Genuine", "Hidden-test pass"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append(f"| {r['run']} | {r['split']} | {r['hint_level']} | {r['n']} | {float(r['hack_rate']):.1%} | "
                     f"{float(r['rate_genuine']):.1%} | {float(r['mean_robust_hidden']):.1%} |")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[analyze] wrote {path} and {table_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--evals", nargs="*", default=[])
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.runs:
        plot_training(a.runs, "hack_rate", "Share of completions that reward-hack during training", "Hack rate",
                      os.path.join(a.out, "training_hack_rate.png"))
        plot_training(a.runs, "score/robust_hidden", "Genuine skill: pass rate on hidden tests during training",
                      "Hidden-test pass rate", os.path.join(a.out, "training_hidden_pass.png"))
    if a.evals:
        plot_eval(a.evals, os.path.join(a.out, "eval_breakdown.png"), os.path.join(a.out, "eval_table.md"))


if __name__ == "__main__":
    main()
