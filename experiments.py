import os
import re
import subprocess
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data/processed/casia"
VAL_DIR = "data/processed/vggface2/2gb/val"
EPOCHS = 20
BATCH_SIZE = 32
RESULTS_FILE = "results_ablation.txt"

# --- TESTS LIST ---
# Format: (model_path, loss, optimizer)
TESTS = [
    ("simple.model_baseline_flat", "arcface", "sgd"),
    ("simple.model_baseline",      "arcface", "sgd"),
    ("simple.model_dropout_flat",  "arcface", "sgd"),
    ("simple.model_dropout",       "arcface", "sgd"),
    ("shape.model_wide",           "arcface", "sgd"),
    ("shape.model_deep",           "arcface", "sgd"),
    ("shape.model_constant_deep",  "arcface", "sgd"),
    ("shape.model_inverse",        "arcface", "sgd"),
    ("shape.model_very_deep",      "arcface", "sgd"),

]


# --- HELPER FUNCTIONS ---
def parse_eval_output(output):
    auc = re.search(r"AUC:\s+([0-9.]+)", output)
    acc = re.search(r"Accuracy:\s+([0-9.]+)", output)
    return (auc.group(1) if auc else "N/A", acc.group(1) if acc else "N/A")


def run_cmd(cmd, capture_output=False):
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr, ""
    else:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0, "", ""


# --- MAIN EXECUTION ---
def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Training data not found: {DATA_DIR}")
        return
    if not os.path.isdir(VAL_DIR):
        print(f"ERROR: Validation data not found: {VAL_DIR}")
        return
    
    print(f"Train Dir: {DATA_DIR}")
    print(f"Val Dir:   {VAL_DIR}")
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(f"{'MODEL':<40} {'LOSS':<10} {'OPT':<10} {'AUC':<10} {'ACC':<10}\n")
            f.write("-" * 80 + "\n")

    for model, loss, opt in TESTS:
        print(f"\n{'='*70}")
        print(f"RUNNING: [{model}]")
        print(f"PARAMS:  {loss} / {opt} / {EPOCHS} epochs")
        print(f"{'='*70}")

        # 1. Train
        train_cmd = f"python -u train.py --model {model} --loss {loss} --optimizer {opt} --epochs {EPOCHS} --data_dir {DATA_DIR} --batch_size {BATCH_SIZE}"
        ok, _, _ = run_cmd(train_cmd, capture_output=False)
        
        if not ok:
            print(f"Training failed for {model}")
            continue

        # 2. Locate Weights
        weights_best = f"weights/{model}_{loss}_{opt}_best.pth"
        weights_last = f"weights/{model}_{loss}_{opt}_last.pth"
        weights = weights_best if os.path.exists(weights_best) else weights_last
        
        if not os.path.exists(weights):
            print(f"Weights not found: {weights}")
            continue

        # 3. Evaluate
        print(f"\nEvaluating {model}...")
        save_dir = f"results/{model}_{loss}_{opt}"
        eval_cmd = f"python -u evaluate.py --model {model} --weights {weights} --data_dir {VAL_DIR} --batch_size {BATCH_SIZE} --save_dir {save_dir}"
        
        ok, out, _ = run_cmd(eval_cmd, capture_output=True)
        
        if not ok:
            print(f"Evaluation failed. Output:\n{out}")
            auc, acc = "ERR", "ERR"
        else:
            auc, acc = parse_eval_output(out)
            print(f"Result: AUC={auc}, ACC={acc}")

        # 4. Save Results
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{model:<40} {loss:<10} {opt:<10} {auc:<10} {acc:<10}\n")

    print(f"\nAll tests finished. Check {RESULTS_FILE}")


if __name__ == "__main__":
    main()