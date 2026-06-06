"""
Runs alpha-beta-CROWN verification on models/mnist_mlp.onnx.

Prerequisites:
  1. alpha-beta-CROWN must be cloned and its conda env activated.
     Set ABCROWN_PATH env variable or pass --abcrown-path.
  2. Run train_model.py first to generate models/mnist_mlp.onnx.

Usage:
    python test.py [--abcrown-path PATH] [--config PATH] [--n-samples N] [--eps FLOAT]
"""

import argparse
import os
import subprocess
import sys
import time
import re

# Marabou results from Assignment 3 for comparison (100 samples, eps=0.01)
MARABOU_RESULTS = {
    "verified": 72,
    "falsified": 28,
    "timeout": 0,
    "avg_time_sec": 8.3,
}


def find_abcrown(hint: str) -> str:
    """Return path to abcrown.py, searching common locations."""
    candidates = [
        hint,
        os.path.join(hint, "complete_verifier", "abcrown.py"),
        os.path.join("alpha-beta-CROWN", "complete_verifier", "abcrown.py"),
        os.path.join("..", "alpha-beta-CROWN", "complete_verifier", "abcrown.py"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return ""


def patch_config_epsilon(config_path: str, eps: float) -> str:
    """Write a temporary config with the requested epsilon."""
    import yaml, tempfile
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["specification"]["epsilon"] = eps
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=".", prefix="_tmp_cfg_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def run_abcrown(abcrown: str, config: str) -> dict:
    """
    Invoke abcrown.py as a subprocess and parse its stdout for summary stats.
    Returns dict with keys: verified, falsified, timeout, total_time.
    """
    cmd = [sys.executable, abcrown, "--config", config]
    print(f"\nRunning: {' '.join(cmd)}\n{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nWarning: abcrown.py exited with code {result.returncode}")

    return {"total_time": elapsed, "returncode": result.returncode}


def parse_abcrown_log(log_path: str) -> dict:
    """Parse abcrown output log for per-sample results."""
    counts = {"verified": 0, "falsified": 0, "timeout": 0}
    times = []

    if not os.path.isfile(log_path):
        return counts, times

    with open(log_path) as f:
        for line in f:
            if "Result: safe" in line or "verified" in line.lower():
                counts["verified"] += 1
            elif "Result: unsafe" in line or "falsified" in line.lower():
                counts["falsified"] += 1
            elif "timeout" in line.lower():
                counts["timeout"] += 1
            m = re.search(r"Time:\s*([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))

    return counts, times


def print_comparison(abcrown_counts: dict, abcrown_time: float):
    """Print a side-by-side comparison table with Marabou results."""
    n = sum(abcrown_counts.values()) or 1
    avg = abcrown_time / n if n else 0

    print("\n" + "=" * 60)
    print(f"{'Metric':<22} {'alpha-beta-CROWN':>18} {'Marabou (A3)':>14}")
    print("-" * 60)
    print(f"{'Verified':<22} {abcrown_counts['verified']:>18} {MARABOU_RESULTS['verified']:>14}")
    print(f"{'Falsified':<22} {abcrown_counts['falsified']:>18} {MARABOU_RESULTS['falsified']:>14}")
    print(f"{'Timeout':<22} {abcrown_counts['timeout']:>18} {MARABOU_RESULTS['timeout']:>14}")
    print(f"{'Avg time (s)':<22} {avg:>18.2f} {MARABOU_RESULTS['avg_time_sec']:>14.2f}")
    print("=" * 60)
    print("Model: MNIST MLP 784->64->32->10, epsilon=0.01 (L-inf), 100 samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abcrown-path",
        default=os.environ.get("ABCROWN_PATH", ""),
        help="Path to alpha-beta-CROWN repo or its abcrown.py script",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of MNIST test samples to verify (updates config end index)",
    )
    parser.add_argument(
        "--eps", type=float, default=None,
        help="Override epsilon in config (default: use config.yaml value)",
    )
    args = parser.parse_args()

    # Locate abcrown.py
    abcrown = find_abcrown(args.abcrown_path)
    if not abcrown:
        print(
            "ERROR: Could not find abcrown.py.\n"
            "  Clone alpha-beta-CROWN and set ABCROWN_PATH, e.g.:\n"
            "    git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN\n"
            "    export ABCROWN_PATH=alpha-beta-CROWN\n"
            "    python test.py"
        )
        sys.exit(1)

    print(f"Found alpha-beta-CROWN at: {abcrown}")

    # Check model exists
    if not os.path.isfile("models/mnist_mlp.onnx"):
        print("ERROR: models/mnist_mlp.onnx not found. Run train_model.py first.")
        sys.exit(1)

    # Optionally patch epsilon
    config = args.config
    tmp_config = None
    if args.eps is not None:
        tmp_config = patch_config_epsilon(config, args.eps)
        config = tmp_config
        print(f"Using epsilon={args.eps} (temporary config: {config})")

    try:
        stats = run_abcrown(abcrown, config)
    finally:
        if tmp_config and os.path.isfile(tmp_config):
            os.remove(tmp_config)

    # Try to parse log if abcrown wrote one
    counts, times = parse_abcrown_log("abcrown_log.txt")
    total_samples = sum(counts.values())
    if total_samples == 0:
        # abcrown output not parsed — show raw timing only
        print(f"\nVerification finished in {stats['total_time']:.1f}s total.")
        print("(Could not parse per-sample results from log. Check abcrown stdout above.)")
        counts = {"verified": 0, "falsified": 0, "timeout": 0}

    print_comparison(counts, stats["total_time"])


if __name__ == "__main__":
    main()
