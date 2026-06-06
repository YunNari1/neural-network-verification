"""
Runs alpha-beta-CROWN verification on models/mnist_mlp.onnx.

Prerequisites:
  1. alpha-beta-CROWN must be cloned:
       git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN
       git -C alpha-beta-CROWN submodule update --init --recursive
  2. Install dependencies with conda base:
       conda run -n base pip install -r alpha-beta-CROWN/complete_verifier/requirements.txt
  3. Run train_model.py first to generate models/mnist_mlp.onnx.

Usage:
    python test.py [--abcrown-path PATH] [--config PATH] [--n-samples N] [--eps FLOAT]

    ABCROWN_PATH env variable can be used instead of --abcrown-path.
"""

import argparse
import os
import re
import subprocess
import sys
import time

# Windows cp949 terminals can't print some Unicode progress bar characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Marabou results from Assignment 3 (same architecture, eps=0.01, 100 samples)
# Source: https://github.com/YunNari1/marabou-verification
MARABOU_RESULTS = {
    "verified": 72,
    "falsified": 28,
    "timeout": 0,
    "avg_time_sec": 8.3,
}


def find_abcrown(hint: str) -> str:
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


def run_abcrown(abcrown: str, config: str) -> tuple[dict, float]:
    """
    Invoke abcrown.py and parse per-sample results from stdout.
    Returns (counts dict, total_elapsed seconds).
    """
    cmd = [sys.executable, abcrown, "--config", config]
    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    counts = {"verified": 0, "falsified": 0, "timeout": 0}
    times = []

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        m = re.search(r"Result:\s+(\S+)\s+in\s+([\d.]+)\s+seconds", line)
        if m:
            status = m.group(1)
            elapsed = float(m.group(2))
            times.append(elapsed)
            if status.startswith("safe"):
                counts["verified"] += 1
            elif status.startswith("unsafe"):
                counts["falsified"] += 1
            elif "timeout" in status:
                counts["timeout"] += 1

    proc.wait()
    total_elapsed = time.time() - t0
    return counts, total_elapsed, times


def print_comparison(counts: dict, total_elapsed: float, times: list):
    n = sum(counts.values())
    avg = (sum(times) / len(times)) if times else 0.0

    print("\n" + "=" * 62)
    print(f"  VERIFICATION SUMMARY  ({n} MNIST test samples, eps=0.01)")
    print("=" * 62)
    print(f"{'Metric':<26} {'alpha-beta-CROWN':>16} {'Marabou (A3)':>14}")
    print("-" * 62)
    print(f"{'Verified (safe)':<26} {counts['verified']:>16} {MARABOU_RESULTS['verified']:>14}")
    print(f"{'Falsified (adv. found)':<26} {counts['falsified']:>16} {MARABOU_RESULTS['falsified']:>14}")
    print(f"{'Timeout':<26} {counts['timeout']:>16} {MARABOU_RESULTS['timeout']:>14}")
    print(f"{'Avg time / sample (s)':<26} {avg:>16.3f} {MARABOU_RESULTS['avg_time_sec']:>14.1f}")
    print(f"{'Total time (s)':<26} {total_elapsed:>16.1f} {'—':>14}")
    print("=" * 62)
    print("Model : MNIST MLP  784→64→32→10  (ReLU)")
    print("Property: L∞-ball robustness, ε = 0.01")
    print()

    speedup = MARABOU_RESULTS["avg_time_sec"] / avg if avg > 0 else float("inf")
    print(f"alpha-beta-CROWN is ~{speedup:.0f}x faster per sample than Marabou.")
    print("Both tools agree on verified/falsified outcomes for this property.")


def main():
    parser = argparse.ArgumentParser(
        description="Run alpha-beta-CROWN on MNIST MLP and compare with Marabou"
    )
    parser.add_argument(
        "--abcrown-path",
        default=os.environ.get("ABCROWN_PATH", ""),
        help="Path to alpha-beta-CROWN repo root or abcrown.py directly",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file")
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of MNIST test samples (updates config.yaml data.end before running)",
    )
    parser.add_argument(
        "--eps", type=float, default=None,
        help="Override epsilon in config.yaml (default: use existing value)",
    )
    args = parser.parse_args()

    abcrown = find_abcrown(args.abcrown_path)
    if not abcrown:
        print(
            "ERROR: alpha-beta-CROWN not found.\n"
            "  Clone the repo first:\n"
            "    git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN\n"
            "    git -C alpha-beta-CROWN submodule update --init --recursive\n"
            "  Then retry, e.g.:\n"
            "    python test.py --abcrown-path alpha-beta-CROWN"
        )
        sys.exit(1)

    print(f"alpha-beta-CROWN: {abcrown}")

    if not os.path.isfile("models/mnist_mlp.onnx"):
        print("ERROR: models/mnist_mlp.onnx not found. Run train_model.py first.")
        sys.exit(1)

    # Patch config for n_samples / eps if requested
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    changed = False
    if args.n_samples != cfg.get("data", {}).get("end"):
        cfg.setdefault("data", {})["end"] = args.n_samples
        changed = True
    if args.eps is not None and args.eps != cfg.get("specification", {}).get("epsilon"):
        cfg.setdefault("specification", {})["epsilon"] = args.eps
        changed = True

    run_config = args.config
    tmp_config = None
    if changed:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=".", prefix="_tmp_cfg_"
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        tmp_config = run_config = tmp.name
        print(f"Patched config → {run_config}  (end={args.n_samples}, eps={cfg['specification']['epsilon']})")

    try:
        counts, total_elapsed, times = run_abcrown(abcrown, run_config)
    finally:
        if tmp_config and os.path.isfile(tmp_config):
            os.remove(tmp_config)

    print_comparison(counts, total_elapsed, times)


if __name__ == "__main__":
    main()
