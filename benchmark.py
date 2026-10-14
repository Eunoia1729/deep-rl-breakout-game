"""Measure end-to-end training throughput (environment steps per second) of each optimisation.

    python benchmark.py --out results/benchmark.json

"before" is the original single-game loop, loaded from git history (--baseline-ref); the other rows use the
current train.py. Throughput is measured between step 3,000 and the end of a 15,000-step run, so start-up and
the warm-up phase are excluded and network updates are active. Run it on an otherwise idle machine.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path

EPISODE = re.compile(r"Total global steps count: (\d+)")


class Stamp(io.TextIOBase):
    """Stand-in for stdout that timestamps every 'Episode:' line with its global step."""

    def __init__(self):
        self.points = []

    def write(self, text):
        now = time.perf_counter()
        for line in text.splitlines():
            m = EPISODE.search(line)
            if m:
                self.points.append((now, int(m.group(1))))
        return len(text)


def steps_per_second(points, start_step):
    first = next(p for p in points if p[1] >= start_step)
    last = points[-1]
    return (last[1] - first[1]) / (last[0] - first[0])


def run(module, steps, **kwargs):
    stamp = Stamp()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(stamp):
        module.train_model(max_steps=steps, warmup_steps=1000, epsilon_decay_steps=10000,
                           replay_capacity=20000, save_dir=tmp, **kwargs)
    return steps_per_second(stamp.points, 3000)


def load_old(ref):
    source = subprocess.run(["git", "show", f"{ref}:train.py"], capture_output=True, text=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "old_train.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("old_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--baseline-ref", default="84f52f0", help="git commit holding the original training loop")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--envs", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    args = ap.parse_args()

    import tensorflow as tf
    import train  # the optimised version

    rows = []

    def record(label, rate, **extra):
        rows.append(dict(label=label, steps_per_second=round(rate, 1), **extra))
        print(f"{label:<44} {rate:7.1f} steps/s", flush=True)

    record("before: one game, per-step model call", run(load_old(args.baseline_ref), args.steps, num_episodes=10**6,
                                                          save_interval=10**9), num_envs=1, compiled=False)
    record("new code, one game, per-step model call", run(train, args.steps, num_envs=1, compile_inference=False),
           num_envs=1, compiled=False)
    record("+ compiled inference (tf.function)", run(train, args.steps, num_envs=1, compile_inference=True),
           num_envs=1, compiled=True)
    for n in args.envs:
        record(f"+ {n} parallel games, batched inference", run(train, args.steps, num_envs=n, compile_inference=True),
               num_envs=n, compiled=True)

    base = rows[0]["steps_per_second"]
    for r in rows:
        r["speedup"] = round(r["steps_per_second"] / base, 2)
    info = dict(cpu=next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
                threads=__import__("os").cpu_count(), tensorflow=tf.__version__, python=platform.python_version(),
                steps_measured=f"3,000 to {args.steps:,}", baseline_ref=args.baseline_ref)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(system=info, rows=rows), indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
