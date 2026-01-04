import os
import re
import subprocess
from datetime import datetime

DATA_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
EPOCHS = 15
RESULTS_FILE = "results_summary.txt"

TESTS = [
    # (model_name, loss, optimizer)
    ("model_baseline", "arcface", "sgd"),
    ("model_baseline_GAP", "arcface", "sgd"),
    ("model_baseline_mobile", "arcface", "sgd"),
    ("model_resnet", "arcface", "sgd"),
    ("model_resnet_mobile", "arcface", "sgd"),
    ("model_resnet_bottleneck", "arcface", "sgd"),
    ("model_resnet_bottleneck_mobile", "arcface", "sgd"),
    ("model_resnet_ir_se", "arcface", "sgd"),
    ("model_ghost_net", "arcface", "sgd"),
    ("model_shufflenet", "arcface", "sgd"),
    ("model_shufflenet_mobile", "arcface", "sgd"),
    ("model_repVGG", "arcface", "sgd"),
    ("model_repVGG_mobile", "arcface", "sgd"),
    ("model_se_net", "arcface", "sgd"),
    ("model_se_net_mobile", "arcface", "sgd"),
    ("model_CBAM", "arcface", "sgd"),
    ("model_CBAM_mobile", "arcface", "sgd"),
    ("model_deep", "arcface", "sgd"),
    ("model_deep_mobile", "arcface", "sgd"),
    ("model_wide", "arcface", "sgd"),
    ("model_wide_mobile", "arcface", "sgd"),
    ("model_dropout", "arcface", "sgd"),
    ("model_accordion", "arcface", "sgd"),
    ("model_inverse", "arcface", "sgd"),
    # AdamW variants
    ("model_baseline", "arcface", "adamw"),
    ("model_resnet", "arcface", "adamw"),
    ("model_resnet_ir_se", "arcface", "adamw"),
    ("model_se_net", "arcface", "adamw"),
    ("model_CBAM", "arcface", "adamw"),
    ("model_ghost_net", "arcface", "adamw"),
    ("model_shufflenet", "arcface", "adamw"),
    ("model_repVGG", "arcface", "adamw"),
]


def parse_eval_output(output):
    auc = re.search(r"AUC:\s+([0-9.]+)", output)
    acc = re.search(r"Accuracy:\s+([0-9.]+)", output)
    return (auc.group(1) if auc else "N/A", acc.group(1) if acc else "N/A")


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def main():
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        f.write(f"{'MODEL':<35} {'LOSS':<10} {'OPT':<10} {'AUC':<10} {'ACC':<10}\n")
        f.write("-" * 75 + "\n")

    for model, loss, opt in TESTS:
        print(f"\n[{model}] {loss} / {opt}")

        train_cmd = f"python train.py --model {model} --loss {loss} --optimizer {opt} --epochs {EPOCHS} --data_dir {DATA_DIR}"
        ok, _, err = run_cmd(train_cmd)
        if not ok:
            print(f"Training failed: {err}")
            continue

        weights = f"weights/{model}_{loss}_{opt}_best.pth"
        if not os.path.exists(weights):
            weights = f"weights/{model}_{loss}_{opt}_last.pth"

        eval_cmd = f"python evaluate.py --model {model} --weights {weights} --data_dir {VAL_DIR}"
        ok, out, _ = run_cmd(eval_cmd)

        auc, acc = parse_eval_output(out) if ok else ("ERROR", "ERROR")

        with open(RESULTS_FILE, "a") as f:
            f.write(f"{model:<35} {loss:<10} {opt:<10} {auc:<10} {acc:<10}\n")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()