"""Plot per-step training curves saved in results/<run>/training_curves.json.

    python scripts/plot_curves.py results/run1

The JSON holds, per training run, 150 per-step values for each metric, copied from the run's log_history.json.
Writes two figures next to it (one measure per chart, never two y-axes).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUNS = [("vuln_hint2", "Trained on vulnerable grader", ORANGE), ("robust_hint2", "Trained on robust grader", BLUE)]


def smooth(xs, w=10):
    return [sum(xs[max(0, i - w + 1): i + 1]) / len(xs[max(0, i - w + 1): i + 1]) for i in range(len(xs))]


def plot(data, key, title, ylabel, path, ymax=1.0):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for run, name, color in RUNS:
        raw = data[run][key]
        ys = smooth(raw)
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, raw, color=color, linewidth=1, alpha=0.18)
        ax.plot(xs, ys, color=color, linewidth=2, label=name)
        ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_xlabel("Training step (thin line: per step, thick: 10-step average)", color=INK_2)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/run1"
    data = json.load(open(os.path.join(run_dir, "training_curves.json")))
    plot(data, "label/hardcode", "Completions that pass the visible tests but fail hidden ones",
         "Share of completions", os.path.join(run_dir, "visible_only_rate.png"), ymax=0.75)
    plot(data, "score/robust_hidden", "Genuine skill: pass rate on hidden tests",
         "Hidden-test pass rate", os.path.join(run_dir, "hidden_pass_rate.png"))


if __name__ == "__main__":
    main()
