"""
Problem 1: Explore the alpha-beta-CROWN Models Directory

Documents the models and verification configurations available in the
alpha-beta-CROWN repository, and compares the setup with Marabou (Assignment 3).

Usage:
    python explore_resources.py [--abcrown-path PATH]

ABCROWN_PATH env variable can be used instead of --abcrown-path.
"""

import argparse
import os
import sys


def find_abcrown_root(hint: str) -> str:
    candidates = [
        hint,
        "alpha-beta-CROWN",
        os.path.join("..", "alpha-beta-CROWN"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "complete_verifier")):
            return c
    return ""


def section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# 1. Models directory
# ---------------------------------------------------------------------------

def explore_models(cv_path: str):
    """Walk complete_verifier/models/ and summarise what is available."""
    models_dir = os.path.join(cv_path, "models")
    section("1. Models Directory  (complete_verifier/models/)")

    # collect by subdirectory
    groups = {}
    for root, dirs, files in os.walk(models_dir):
        rel = os.path.relpath(root, models_dir)
        model_files = [f for f in files if f.endswith((".pth", ".pt", ".model", ".pkl", ".onnx"))]
        if model_files:
            groups[rel] = model_files

    # descriptions for each subdirectory
    descriptions = {
        "bab_attack":       "BaB-Attack adversarial models (MNIST MadryCNN, CIFAR small CNNs)",
        "cifar10_resnet":   "CIFAR-10 ResNet-2B / ResNet-4B  (PyTorch .pth, definition in resnet.py)",
        "crown-ibp":        "CROWN-IBP trained models (README only -- weights hosted externally)",
        "custom_op":        "CIFAR-10 wide model with custom activation operators",
        "custom_specs":     "Model used for custom specification examples",
        "eran":             "ERAN benchmark models: MNIST 6x100, 9x100, conv-small/big; CIFAR conv-big/small, ResNet-8px",
        "l2_norm":          "CIFAR-10 model trained for L2 norm verification (4-conv-3-FC)",
        "marabou_cifar10":  "CIFAR-10 models sized for Marabou comparison (small/medium/large)",
        "non_relu":         "CIFAR-10 CNN with sigmoid activations (non-ReLU verification)",
        "oval":             "OVAL benchmark models: cifar_base/deep/wide  (.pth + precomputed .pkl bounds)",
        "sdp":              "SDP-trained models: MNIST cnn-a-adv, CIFAR cnn-a/b variants  (.model format)",
        "toy":              "Tiny toy MLP: mnist_2_20  (2 hidden layers, 20 neurons each)",
        "vnncomp22":        "VNN-COMP 2022 CIFAR-100 ResNet (weights must be unzipped separately)",
        ".":                "(root) no model files directly in models/",
    }

    fmt = "{:<22}  {:<8}  {}"
    print(fmt.format("Subdirectory", "# Files", "Description"))
    print("-" * 70)
    total = 0
    for rel in sorted(groups):
        files = groups[rel]
        desc = descriptions.get(rel, "")
        print(fmt.format(rel, len(files), desc))
        total += len(files)
    print(f"\nTotal model files: {total}  (formats: .pth / .pt / .model / .pkl / .onnx)")

    subsection("Formats observed")
    ext_counts = {}
    for files in groups.values():
        for f in files:
            ext = os.path.splitext(f)[1]
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    for ext, cnt in sorted(ext_counts.items()):
        labels = {
            ".pth":   "PyTorch state_dict (torch.save)",
            ".pt":    "PyTorch full model or state_dict",
            ".model": "Custom serialised format (used by SDP configs)",
            ".pkl":   "Pickle -- precomputed intermediate bounds (OVAl)",
            ".onnx":  "ONNX (none bundled; loaded via onnx_path in YAML)",
        }
        print(f"  {ext:<8} {cnt:>3} files   {labels.get(ext, '')}")

    subsection("Key observation")
    print(
        "  All built-in models use PyTorch format (.pth / .pt / .model).\n"
        "  ONNX models are NOT stored in this directory; they are referenced\n"
        "  via  model.onnx_path  in the YAML config and loaded at runtime.\n"
        "  This differs from Marabou, which works exclusively with .onnx files."
    )


# ---------------------------------------------------------------------------
# 2. Experiment configs (YAML)
# ---------------------------------------------------------------------------

def explore_configs(cv_path: str):
    """Summarise the exp_configs directory structure."""
    configs_dir = os.path.join(cv_path, "exp_configs")
    section("2. Verification Configurations  (complete_verifier/exp_configs/)")

    # count YAMLs per top-level category
    categories = {}
    for entry in sorted(os.listdir(configs_dir)):
        full = os.path.join(configs_dir, entry)
        if os.path.isdir(full):
            n = sum(1 for _, _, fs in os.walk(full) for f in fs if f.endswith(".yaml"))
            categories[entry] = n

    cat_descriptions = {
        "tutorial_examples": "18 introductory examples: ONNX + built-in dataset, custom model/data, vnnlib format, custom ops",
        "beta_crown":        "Core beta-CROWN configs: MNIST FC/CNN, CIFAR CNN/ResNet, OVAL benchmark (30 files)",
        "bab_attack":        "BaB-Attack ablation configs (combines verification + adversarial attack search)",
        "GCP-CROWN":         "GCP-CROWN (Gurobi cut-plane) configs for CIFAR/MNIST benchmarks",
        "BICCOS":            "BICCOS cut-enhanced configs across CIFAR-100 sizes (2022-2024)",
        "vnncomp21":         "VNN-COMP 2021 competition configs",
        "vnncomp22":         "VNN-COMP 2022 competition configs",
        "vnncomp23":         "VNN-COMP 2023 competition configs",
        "vnncomp24":         "VNN-COMP 2024 competition configs",
        "vnncomp25":         "VNN-COMP 2025 competition configs",
    }

    fmt = "{:<22}  {:<8}  {}"
    print(fmt.format("Category", "YAMLs", "Purpose"))
    print("-" * 70)
    total = 0
    for cat, n in categories.items():
        print(fmt.format(cat, n, cat_descriptions.get(cat, "")))
        total += n
    print(f"\nTotal YAML configs: {total}")

    subsection("YAML configuration structure (key sections)")
    print(
        "  Every config is a nested YAML with these top-level sections:\n\n"
        "  model:         model name (PyTorch class) or onnx_path, checkpoint path\n"
        "  data:          dataset name, mean/std normalisation, start/end indices\n"
        "  specification: norm (inf/2/1), epsilon, or vnnlib_path for custom specs\n"
        "  attack:        PGD steps/restarts (fast pre-check before formal proof)\n"
        "  solver:        batch_size, alpha-crown / beta-crown iteration counts\n"
        "  bab:           timeout, branching method (kfsb/fsb/babsr), candidates\n\n"
        "  Configs are hierarchical; unknown keys raise a ValueError at startup,\n"
        "  which makes misconfiguration easy to catch."
    )

    subsection("Example: mnist_tiny_mlp.yaml  (most similar to our model)")
    tiny_path = os.path.join(configs_dir, "beta_crown", "mnist_tiny_mlp.yaml")
    if os.path.isfile(tiny_path):
        with open(tiny_path) as f:
            for line in f:
                print("  " + line, end="")
    else:
        print("  (file not found)")


# ---------------------------------------------------------------------------
# 3. Comparison with Marabou
# ---------------------------------------------------------------------------

def compare_with_marabou():
    section("3. Comparison: alpha-beta-CROWN vs Marabou")

    rows = [
        ("Model format",        "PyTorch (.pth/.pt) or ONNX",      "ONNX only"),
        ("Specification",       "YAML config (norm, epsilon, vnnlib)","Python query API (per-sample)"),
        ("Verification method", "Bound propagation + BaB",          "LP / SMT (ReLU splitting)"),
        ("Pre-check",           "PGD attack built-in",              "No built-in attacker"),
        ("Dataset loading",     "Built-in MNIST/CIFAR loaders",     "Manual numpy loading"),
        ("Normalisation",       "mean/std in YAML",                 "Manual pre-normalisation"),
        ("Activation support",  "ReLU, sigmoid, tanh, custom ops",  "Primarily ReLU"),
        ("GPU support",         "Yes (CUDA)",                       "No"),
        ("Competition pedigree","VNN-COMP winner (2021-2024)",       "Established SMT baseline"),
    ]

    fmt = "{:<28}  {:<32}  {}"
    print(fmt.format("Aspect", "alpha-beta-CROWN", "Marabou"))
    print("-" * 80)
    for aspect, abcrown, marabou in rows:
        print(fmt.format(aspect, abcrown, marabou))

    print(
        "\n  Key insight: Marabou encodes the network as an explicit SMT/LP problem\n"
        "  with one variable per neuron per sample, then calls a general solver.\n"
        "  alpha-beta-CROWN instead propagates symbolic bounds (intervals or\n"
        "  linear functions) through the network and tightens them with gradient\n"
        "  optimisation (alpha-CROWN), only falling back to BaB when the bounds\n"
        "  do not yet prove safety. This makes the initial relaxation step much\n"
        "  faster and often sufficient, which is why alpha-beta-CROWN verifies\n"
        "  most samples without any branching at all."
    )


# ---------------------------------------------------------------------------
# 4. Installation notes
# ---------------------------------------------------------------------------

def installation_notes():
    section("4. Installation Notes and Issues Encountered")
    print(
        "  Recommended installation (from alpha-beta-CROWN README):\n"
        "    git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN\n"
        "    git -C alpha-beta-CROWN submodule update --init --recursive\n"
        "    conda env create -f alpha-beta-CROWN/complete_verifier/environment_pyt280.yaml\n"
        "    conda activate verifier_pyt280\n\n"
        "  Issues encountered during this assignment:\n\n"
        "  Issue 1 -- CUDA not found on CPU-only machine\n"
        "    Symptom : AssertionError: Torch not compiled with CUDA enabled\n"
        "    Fix     : Add  general: device: cpu  to the YAML config.\n\n"
        "  Issue 2 -- Input shape mismatch (mat1 252x28 vs 784x64)\n"
        "    Symptom : RuntimeError during the attack phase.\n"
        "    Cause   : The model was exported with flat (1,784) input, but\n"
        "              alpha-beta-CROWN feeds MNIST images as (1,28,28) tensors.\n"
        "    Fix     : Add nn.Flatten() to the model and re-export ONNX with\n"
        "              dummy input shape (1,1,28,28).\n\n"
        "  Issue 3 -- Config key ['bab','branching','reduce_op'] not found\n"
        "    Symptom : ValueError at startup.\n"
        "    Fix     : The correct key name is  reduceop  (no underscore).\n\n"
        "  Issue 4 -- UnicodeEncodeError in test.py on Windows (cp949 terminal)\n"
        "    Symptom : UnicodeEncodeError when printing progress bar characters.\n"
        "    Fix     : Added  sys.stdout.reconfigure(encoding='utf-8')  at the\n"
        "              top of test.py.\n\n"
        "  Issue 5 -- conda run fails with newline in -c argument (Windows)\n"
        "    Symptom : NotImplementedError in conda run.\n"
        "    Fix     : Call  C:\\...\\miniconda3\\python.exe  directly instead\n"
        "              of using  conda run -n base python."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Explore alpha-beta-CROWN models and configurations (Problem 1)"
    )
    parser.add_argument(
        "--abcrown-path",
        default=os.environ.get("ABCROWN_PATH", ""),
        help="Path to cloned alpha-beta-CROWN repository root",
    )
    args = parser.parse_args()

    root = find_abcrown_root(args.abcrown_path)
    if not root:
        print(
            "ERROR: alpha-beta-CROWN repository not found.\n"
            "  Clone it first:\n"
            "    git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN\n"
            "  Then retry:\n"
            "    python explore_resources.py --abcrown-path alpha-beta-CROWN"
        )
        sys.exit(1)

    cv_path = os.path.join(root, "complete_verifier")
    print(f"alpha-beta-CROWN root : {os.path.abspath(root)}")
    print(f"complete_verifier     : {os.path.abspath(cv_path)}")

    explore_models(cv_path)
    explore_configs(cv_path)
    compare_with_marabou()
    installation_notes()

    print()
    print("=" * 70)
    print("  Done. See report.pdf Section 1 for a summary of these findings.")
    print("=" * 70)


if __name__ == "__main__":
    main()
