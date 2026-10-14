"""Render a gameplay GIF for each checkpoint, to show how the agent improves during training.

    python make_gifs.py models/ gifs/                 # every step_*.keras in models/
    python make_gifs.py models/step_000500000.keras gifs/ --max-steps 300
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import train


def play(model_path, max_steps, epsilon):
    """Play one game and return (list of RGB frames, score, game steps). Pass model_path=None for random play."""
    env = train.make_env()
    ale = env.unwrapped.ale  # raw 210x160 colour screen, which the preprocessing wrappers throw away
    dqn = None
    if model_path is not None:
        dqn = train.DeepQNetwork()
        dqn.loadModel(model_path)
    state, done, total, steps = env.reset(), False, 0, 0
    frames, need_fire, lives = [], True, None
    while not done and steps < max_steps:
        steps += 1
        if need_fire:
            action, need_fire = train.FIRE, False
        elif dqn is None or np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(dqn.predict(state[None])[0]))
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total += reward
        if lives is not None and info.get('lives', lives) < lives:
            need_fire = True
        lives = info.get('lives', lives)
        frames.append(ale.getScreenRGB().copy())
    env.close()
    return frames, total, steps


def label(frame, text, scale):
    img = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((4, 1), text, fill=(255, 255, 255))
    return img


def save_gif(frames, path, text, scale=2, fps=30, stride=1):
    imgs = [label(f, text, scale) for f in frames[::stride]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", help="a .keras file or a directory of step_*.keras checkpoints")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--max-steps", type=int, default=300, help="agent steps per GIF (one agent step = 4 frames)")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--random-baseline", action="store_true", help="also render a random-policy GIF")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.models)
    models = sorted(src.glob("step_*.keras")) if src.is_dir() else [src]
    if args.random_baseline:
        frames, score, steps = play(None, args.max_steps, 1.0)
        save_gif(frames, out / "step_000000000.gif", f"random play | score {score:.0f}")
        print(f"random: score {score:.0f} in {steps} steps")
    for m in models:
        frames, score, steps = play(str(m), args.max_steps, args.epsilon)
        step_n = int(m.stem.split("_")[-1]) if m.stem.startswith("step_") else 0
        save_gif(frames, out / f"{m.stem}.gif", f"step {step_n:,} | score {score:.0f}")
        print(f"{m.name}: score {score:.0f} in {steps} steps -> {out / (m.stem + '.gif')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
