"""Side-by-side GIF of several checkpoints playing, to show how the agent improves.

    python make_montage.py models/ docs/progression.gif --steps 0 100000 300000 1000000 2000000

Each panel shows the median-scoring game out of --games, so it is representative, not cherry-picked.
Step 0 means random play. With --best-of N it instead writes a single large GIF of the best of N games
by the last listed checkpoint (clearly labelled as such).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import make_gifs


def pick_game(model_path, games, max_steps, epsilon, how):
    runs = [make_gifs.play(model_path, max_steps, epsilon if model_path else 1.0) for _ in range(games)]
    runs.sort(key=lambda r: r[1])
    return runs[len(runs) // 2] if how == "median" else runs[-1]


def panel(frame, title, scale, header=16):
    w, h = frame.shape[1] * scale, frame.shape[0] * scale
    img = Image.new("RGB", (w, h + header), (0, 0, 0))
    img.paste(Image.fromarray(frame).resize((w, h), Image.NEAREST), (0, header))
    ImageDraw.Draw(img).text((4, 2), title, fill=(255, 255, 255))
    return img


def fmt(step):
    return "random play" if step == 0 else (f"{step / 1e6:g}M steps" if step >= 1e6 else f"{step // 1000}K steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", help="directory of step_*.keras checkpoints")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--games", type=int, default=5, help="games played per panel; the median one is shown")
    ap.add_argument("--best-of", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    def path_for(step):
        return None if step == 0 else str(Path(args.models) / f"step_{step:09d}.keras")

    if args.best_of:
        step = args.steps[-1]
        frames, score, _ = pick_game(path_for(step), args.best_of, args.max_steps, args.epsilon, "best")
        imgs = [panel(f, f"{fmt(step)} | best of {args.best_of} games | score {score:.0f}", 3, 18) for f in frames]
        print(f"best of {args.best_of}: score {score:.0f}, {len(frames)} steps")
    else:
        games = []
        for step in args.steps:
            frames, score, _ = pick_game(path_for(step), args.games, args.max_steps, args.epsilon, "median")
            games.append((step, frames, score))
            print(f"{fmt(step)}: median game scored {score:.0f} in {len(frames)} steps", flush=True)
        length = max(len(f) for _, f, _ in games)
        imgs = []
        for t in range(length):
            panels = []
            for step, frames, score in games:
                done = t >= len(frames)
                p = panel(frames[min(t, len(frames) - 1)], f"{fmt(step)}  score {score:.0f}" + ("  (game over)" if done else ""), 1)
                panels.append(p)
            sheet = Image.new("RGB", (sum(p.width for p in panels) + 4 * (len(panels) - 1), panels[0].height), (30, 30, 30))
            x = 0
            for p in panels:
                sheet.paste(p, (x, 0))
                x += p.width + 4
            imgs.append(sheet)
        imgs += [imgs[-1]] * (args.fps * 2)  # pause on the last frame before looping
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print("wrote", args.out, f"({Path(args.out).stat().st_size / 1e6:.2f} MB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
