import os
import re
import subprocess
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data/processed/casia"
VAL_DIR = "data/processed/lfw_112"
EPOCHS = 15  # Reduced for faster iteration
BATCH_SIZE = 512
RESULTS_FILE = "results_ablation.txt"

# --- TESTS LIST ---
# Format: (model_path, loss, optimizer)
TESTS = [
    # ==================== DEPTH STUDY ====================
    ("depth_study.plain_4",              "arcface", "sgd"),
    ("depth_study.plain_8",              "arcface", "sgd"),
    ("depth_study.plain_14",             "arcface", "sgd"),
    ("depth_study.plain_20",             "arcface", "sgd"),
    
    # ==================== SKIP STUDY ====================
    ("skip_study.resnet_4",              "arcface", "sgd"),
    ("skip_study.resnet_8",              "arcface", "sgd"),
    ("skip_study.resnet_14",             "arcface", "sgd"),
    ("skip_study.resnet_20",             "arcface", "sgd"),
    
    # ==================== WIDTH STUDY ====================
    ("width_study.narrow_8",             "arcface", "sgd"),
    ("width_study.wide_8",               "arcface", "sgd"),
    ("width_study.constant_8",           "arcface", "sgd"),
    ("width_study.inverse_8",            "arcface", "sgd"),
    
    # ==================== POOLING STUDY ====================
    ("pooling_study.plain_8_flatten",    "arcface", "sgd"),
    ("pooling_study.plain_8_gmp",        "arcface", "sgd"),
    ("pooling_study.plain_8_maxpool_down", "arcface", "sgd"),
    
    # ==================== REGULARIZATION STUDY ====================
    ("regularization_study.plain_8_fc_dropout_05",    "arcface", "sgd"),
    ("regularization_study.plain_8_block_dropout_01", "arcface", "sgd"),
    ("regularization_study.resnet_14_fc_dropout_05",    "arcface", "sgd"),
    ("regularization_study.resnet_14_block_dropout_01", "arcface", "sgd"),
    
    # ==================== ATTENTION STUDY ====================
    ("attention_study.se_resnet_14",     "arcface", "sgd"),
    ("attention_study.cbam_resnet_14",   "arcface", "sgd"),
    
    # ==================== BLOCK STUDY ====================
    ("block_study.bottleneck_resnet_14", "arcface", "sgd"),
    ("block_study.bottleneck_14",        "arcface", "sgd"),
    ("block_study.inverted_bottleneck_resnet_14", "arcface", "sgd"),
    ("block_study.ghost_resnet_14",      "arcface", "sgd"),
    ("block_study.depthwise_resnet_14",  "arcface", "sgd"),
    
    # ==================== PRODUCTION ====================
    ("production.seres20",               "arcface", "sgd"),
    ("production.mobilenet20",           "arcface", "sgd"),
    ("production.ghostnet20",            "arcface", "sgd"),
    
    # ==================== OPTIMIZER TEST ====================
    ("depth_study.plain_8",              "arcface", "adamw"),
    ("skip_study.resnet_14",             "arcface", "adamw"),
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


def get_model_info(model_path):
    """Try to get INFO string from model"""
    try:
        module = __import__(model_path, fromlist=['INFO'])
        return getattr(module, 'INFO', '')[:50]
    except:
        return ''


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
    print(f"Epochs:    {EPOCHS}")
    print(f"Tests:     {len(TESTS)}")
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(f"{'MODEL':<50} {'LOSS':<10} {'OPT':<10} {'AUC':<10} {'ACC':<10}\n")
            f.write("-" * 90 + "\n")

    for i, (model, loss, opt) in enumerate(TESTS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(TESTS)}] RUNNING: {model}")
        print(f"PARAMS:  {loss} / {opt} / {EPOCHS} epochs")
        print(f"{'='*70}")

        # 1. Train
        train_cmd = f"python -u train.py --model {model} --loss {loss} --optimizer {opt} --epochs {EPOCHS} --data_dir {DATA_DIR} --batch_size {BATCH_SIZE}"
        ok, _, _ = run_cmd(train_cmd, capture_output=False)
        
        if not ok:
            print(f"Training failed for {model}")
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{model:<50} {loss:<10} {opt:<10} {'TRAIN_ERR':<10} {'TRAIN_ERR':<10}\n")
            continue

        # 2. Locate Weights
        model_name = model.replace(".", "_")
        weights_best = f"weights/{model_name}_{loss}_{opt}_best.pth"
        weights_last = f"weights/{model_name}_{loss}_{opt}_last.pth"
        weights = weights_best if os.path.exists(weights_best) else weights_last
        
        if not os.path.exists(weights):
            print(f"Weights not found: {weights}")
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{model:<50} {loss:<10} {opt:<10} {'NO_WEIGHTS':<10} {'NO_WEIGHTS':<10}\n")
            continue

        # 3. Evaluate
        print(f"\nEvaluating {model}...")
        save_dir = f"results/{model_name}_{loss}_{opt}"
        eval_cmd = f"python -u evaluate_lfw.py --model {model} --weights {weights} --data_dir {VAL_DIR} --batch_size {BATCH_SIZE} --save_dir {save_dir}"
        
        ok, out, _ = run_cmd(eval_cmd, capture_output=True)
        
        if not ok:
            print(f"Evaluation failed. Output:\n{out}")
            auc, acc = "EVAL_ERR", "EVAL_ERR"
        else:
            auc, acc = parse_eval_output(out)
            print(f"Result: AUC={auc}, ACC={acc}")

        # 4. Save Results
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{model:<50} {loss:<10} {opt:<10} {auc:<10} {acc:<10}\n")

    print(f"\n{'='*70}")
    print(f"All {len(TESTS)} tests finished. Check {RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()