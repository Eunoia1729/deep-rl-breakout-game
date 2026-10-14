"""Plot training progress from a training log and checkpoint evaluations (light and dark versions).

    python plot_progress.py --log results/training.log --eval results/eval_results.json --out docs/

Writes training_curves_{light,dark}.png and evaluation_{light,dark}.png, and results/training_log.csv.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# One accent hue for the agent; everything else is neutral. Tokens follow a validated chart palette.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", accent="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", accent="#3987e5"),
}
LINE = re.compile(r"Episode: (\d+)\. Total steps count: (\d+), Total global steps count: (\d+), "
                  r"reward count: ([\d.]+)(?:, epsilon: ([\d.]+))?")
WINDOW = 100


def parse_logs(paths):
    """Parse one or more training logs in order. If a later log resumes from an earlier step, the earlier log's
    episodes beyond that step are dropped (the run was restarted from a checkpoint)."""
    merged = np.empty((0, 5))
    for path in paths:
        part = parse_log(path)
        if len(part) == 0:
            continue
        first_step = part[0, 2]  # global step at the end of this log's first episode
        merged = np.vstack([merged[merged[:, 2] < first_step], part])
    return merged


def parse_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = LINE.match(line)
        if m:
            ep, steps, gstep, reward, eps = m.groups()
            rows.append((int(ep), int(steps), int(gstep), float(reward), float(eps) if eps else np.nan))
    return np.array(rows, dtype=float).reshape(-1, 5)


def rolling(x, w):
    w = min(w, len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid"), w


def luminance(hex_color):
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def style(ax, t, ylabel):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.set_title(ylabel, loc="left", fontsize=11, color=t["ink2"], pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if not v else (f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}K")))


def finish(fig, t, path, dpi=150):
    fig.patch.set_facecolor(t["surface"])
    fig.savefig(path, dpi=dpi, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def training_figure(data, baseline, t, path):
    gstep, steps, reward, eps = data[:, 2], data[:, 1], data[:, 3], data[:, 4]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1.6, 1.3], hspace=0.38))
    # 1. score per game
    ax = axes[0]
    style(ax, t, "Score per game: faint dots are single games, the line is the 100-game average")
    ax.scatter(gstep, reward, s=4, color=t["accent"], alpha=0.16, linewidths=0, rasterized=True)
    avg, w = rolling(reward, WINDOW)
    ax.plot(gstep[w - 1:], avg, color=t["accent"], linewidth=2)
    ax.set_ylim(bottom=0)
    if baseline is not None:
        b = float(np.mean(baseline))
        ax.axhline(b, color=t["muted"], linewidth=1)
        ax.text(gstep.max(), b, f"random play {b:.1f}", color=t["ink2"], fontsize=9, ha="right", va="bottom")
    ax.annotate(f"{avg[-1]:.1f}", (gstep[-1], avg[-1]), xytext=(8, 0), textcoords="offset points",
                color=t["ink"], fontsize=11, fontweight="bold", va="center")
    # 2. game length
    ax = axes[1]
    style(ax, t, "Game length (agent steps): longer games mean the paddle keeps the ball alive")
    length, w = rolling(steps, WINDOW)
    ax.plot(gstep[w - 1:], length, color=t["accent"], linewidth=2)
    ax.set_ylim(bottom=0)
    # 3. exploration
    ax = axes[2]
    style(ax, t, "Exploration rate (epsilon): share of moves chosen at random")
    ax.plot(gstep, eps, color=t["muted"], linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training steps", color=t["ink2"], fontsize=10)
    ax.set_xlim(0, gstep.max() * 1.07)
    finish(fig, t, path)


def evaluation_figure(results, t, path):
    steps = sorted(int(s) for s in results["checkpoints"])
    scores = [np.array(results["checkpoints"][str(s)], dtype=float) for s in steps]
    mean = np.array([s.mean() for s in scores])
    lo = np.array([s.min() for s in scores])
    hi = np.array([s.max() for s in scores])
    games = len(scores[0])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    style(ax, t, f"Evaluation score by checkpoint: mean of {games} games, band = worst to best game")
    ax.fill_between(steps, lo, hi, color=t["accent"], alpha=0.16, linewidth=0)
    ax.plot(steps, mean, color=t["accent"], linewidth=2, marker="o", markersize=7,
            markeredgecolor=t["surface"], markeredgewidth=2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(steps) * 1.1)
    if results.get("baseline"):
        b = float(np.mean(results["baseline"]))
        ax.axhline(b, color=t["muted"], linewidth=1)
        ax.text(max(steps) * 1.1, b, f"random play {b:.1f}", color=t["ink2"], fontsize=9, ha="right", va="bottom")
    best = int(np.argmax(mean))
    if best != len(steps) - 1:
        ax.annotate(f"best {mean[best]:.1f}", (steps[best], mean[best]), xytext=(0, 10),
                    textcoords="offset points", color=t["ink2"], fontsize=9, ha="center")
    ax.annotate(f"{mean[-1]:.1f}", (steps[-1], mean[-1]), xytext=(9, 0), textcoords="offset points",
                color=t["ink"], fontsize=11, fontweight="bold", va="center")
    ax.set_xlabel("Training steps", color=t["ink2"], fontsize=10)
    finish(fig, t, path)


def speedup_figure(bench, t, path):
    rows = bench["rows"]
    labels = [r["label"] for r in rows][::-1]
    rates = np.array([r["steps_per_second"] for r in rows])[::-1]
    speedups = np.array([r["speedup"] for r in rows])[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.6))
    style(ax, t, "Training throughput: environment steps per second (higher is better)")
    ax.set_facecolor(t["surface"])
    ypos = np.arange(len(rows))
    colors = [t["muted"] if "before" in lab or "new code" in lab else t["accent"] for lab in labels]
    ax.barh(ypos, rates, height=0.62, color=colors, edgecolor=t["surface"], linewidth=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=t["ink2"], fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, rates.max() * 1.22)
    for y, rate, sp in zip(ypos, rates, speedups):
        ax.text(rate + rates.max() * 0.012, y, f"{rate:.0f}  ({sp:g}x)", va="center", color=t["ink"], fontsize=10,
                fontweight="bold" if sp == speedups.max() else "normal")
    finish(fig, t, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", default=[], help="training stdout log(s); later ones override overlapping steps")
    ap.add_argument("--benchmark", default=None, help="JSON from benchmark.py")
    ap.add_argument("--eval", default=None, help="JSON from evaluate_checkpoints.py")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--csv", default=None, help="write the parsed log here as CSV")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        bench = json.loads(Path(args.benchmark).read_text())
        for name, t in THEMES.items():
            speedup_figure(bench, t, out / f"speedup_{name}.png")
    if not args.log:
        return
    data = parse_logs(args.log)
    if len(data) == 0:
        raise SystemExit("no 'Episode:' lines found in the log")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "game_steps", "global_step", "score", "epsilon"])
            for r in data:
                w.writerow([int(r[0]), int(r[1]), int(r[2]), r[3], r[4]])
    results = json.loads(Path(args.eval).read_text()) if args.eval and Path(args.eval).exists() else None
    for name, t in THEMES.items():
        print(f"{name}: accent on surface {contrast(t['accent'], t['surface']):.2f}:1, "
              f"muted on surface {contrast(t['muted'], t['surface']):.2f}:1")
        training_figure(data, results["baseline"] if results else None, t, out / f"training_curves_{name}.png")
        if results and results["checkpoints"]:
            evaluation_figure(results, t, out / f"evaluation_{name}.png")


if __name__ == "__main__":
    main()
