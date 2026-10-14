"""Score every checkpoint (and a random-play baseline) and store the results as JSON for plot_progress.py.

    python evaluate_checkpoints.py models/ results/eval_results.json --games 10

Already-evaluated checkpoints are skipped, so it can be re-run while training is still going.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", help="directory of step_*.keras checkpoints")
    ap.add_argument("out", help="JSON file to create or update")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=4000, help="cap per game, in agent steps")
    ap.add_argument("--pattern", default="step_*.keras")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {"baseline": None, "checkpoints": {}}

    if results["baseline"] is None:
        results["baseline"] = train.evaluate(None, 50, 1.0, args.max_steps)
        print(f"random baseline: mean {np.mean(results['baseline']):.2f}", flush=True)
    for path in sorted(Path(args.models).glob(args.pattern)):
        step = str(int(path.stem.split("_")[-1]))
        if step in results["checkpoints"]:
            continue
        scores = train.evaluate(str(path), args.games, args.epsilon, args.max_steps)
        results["checkpoints"][step] = scores
        print(f"step {step}: mean {np.mean(scores):.1f} (min {min(scores):.0f}, max {max(scores):.0f})", flush=True)
        out.write_text(json.dumps(results, indent=1))
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
