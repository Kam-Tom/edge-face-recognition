import argparse
import importlib
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_transforms():
    # Only ToTensor and Normalize because images are already 112x112 from MTCNN
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def load_model(model_name, weights_path):
    try:
        module = importlib.import_module(f"models.{model_name}")
        if hasattr(module, 'Net'):
            model = module.Net(embedding_size=512)
        elif hasattr(module, 'ResNet'): 
            model = module.ResNet(embedding_size=512)
        else:
            raise ValueError("Could not find 'Net' or 'ResNet' class.")

        model = model.to(DEVICE)
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

def parse_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
        
    if len(lines) > 0 and len(lines[0].strip().split()) == 2:
        lines = lines[1:]
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            pairs.append((parts[0], parts[1], parts[0], parts[2], 1))
        elif len(parts) == 4:
            pairs.append((parts[0], parts[1], parts[2], parts[3], 0))
    return pairs

def get_image_path(root, name, img_num):
    img_num_str = str(img_num).zfill(4)
    filename = f"{name}_{img_num_str}.jpg"
    return os.path.join(root, name, filename)

def find_best_threshold(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    acc = float(tp + tn) / dist.size
    return acc

def evaluate_10_fold(sims, labels):
    """
    Standard LFW 10-fold cross-validation.
    """
    k_fold = KFold(n_splits=10, shuffle=False)
    
    accuracies = []
    thresholds = []
    
    print(f"🔄 Starting 10-Fold Validation...")
    
    # Iterate through 10 folds
    for fold_idx, (train_index, test_index) in enumerate(k_fold.split(sims)):
        # 1. Split data into Train (9/10) and Test (1/10)
        train_sims, test_sims = sims[train_index], sims[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        
        # 2. Find best threshold on TRAIN set
        fpr, tpr, roc_thresholds = roc_curve(train_labels, train_sims)
        best_thresh = find_best_threshold(fpr, tpr, roc_thresholds)
        
        # 3. Calculate accuracy on TEST set using the TRAIN threshold
        acc = calculate_accuracy(best_thresh, test_sims, test_labels)
        
        accuracies.append(acc)
        thresholds.append(best_thresh)
        
        # Optional: Print progress for each fold
        # print(f"   Fold {fold_idx+1}/10: Acc={acc*100:.2f}%, Thresh={best_thresh:.4f}")
        
    return np.mean(accuracies), np.std(accuracies), np.mean(thresholds)

def evaluate(model, data_dir, save_dir=None):
    pairs_path = os.path.join(data_dir, "pairs.txt")
    if not os.path.exists(pairs_path):
        print(f"❌ Missing pairs.txt file in {data_dir}")
        return

    print(f"📋 Loading pairs from: {pairs_path}")
    pairs = parse_pairs(pairs_path)
    print(f"🔍 Checking: {len(pairs)} pairs")
    
    transform = get_transforms()
    sims = []
    labels = []
    missing_count = 0
    
    # --- STEP 1: Compute Embeddings ---
    print("🚀 Computing embeddings...")
    with torch.no_grad():
        for p in tqdm(pairs, desc="Inference"):
            name1, num1, name2, num2, label = p
            path1 = get_image_path(data_dir, name1, num1)
            path2 = get_image_path(data_dir, name2, num2)
            
            if not os.path.exists(path1) or not os.path.exists(path2):
                missing_count += 1
                continue
            
            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')
            
            img1 = transform(img1).unsqueeze(0).to(DEVICE)
            img2 = transform(img2).unsqueeze(0).to(DEVICE)
            
            emb1 = F.normalize(model(img1), p=2, dim=1)
            emb2 = F.normalize(model(img2), p=2, dim=1)
            
            sim = F.cosine_similarity(emb1, emb2).item()
            sims.append(sim)
            labels.append(label)

    if missing_count > 0:
        print(f"⚠️ Skipped {missing_count} pairs.")

    sims = np.array(sims)
    labels = np.array(labels)

    # --- STEP 2: 10-Fold Cross Validation ---
    acc_mean, acc_std, thresh_mean = evaluate_10_fold(sims, labels)

    # --- REPORTING ---
    print(f"\n{'='*50}")
    print(f"🏆 LFW 10-FOLD VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"✅ Accuracy Mean:  {acc_mean*100:.2f}%")
    print(f"📉 Accuracy Std:   +/- {acc_std*100:.2f}%")
    print(f"🎯 Avg Threshold:  {thresh_mean:.4f}")
    print(f"{'='*50}")
    
    # Statistics regarding similarity distribution
    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]
    print(f"📊 Global Stats (for reference):")
    print(f"   Positive Pairs (Same) Mean: {pos_sims.mean():.4f}")
    print(f"   Negative Pairs (Diff) Mean: {neg_sims.mean():.4f}")
    print(f"   Separation Gap:             {pos_sims.mean() - neg_sims.mean():.4f}")
    print(f"{'='*50}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save plots (Histogram only, as ROC curve is ambiguous for 10-fold without averaging)
        plt.figure(figsize=(8, 6))
        plt.hist(pos_sims, bins=50, alpha=0.7, color='green', label='Same Person')
        plt.hist(neg_sims, bins=50, alpha=0.7, color='red', label='Diff Person')
        plt.axvline(thresh_mean, color='blue', linestyle='--', label=f'Avg Thresh={thresh_mean:.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.title('LFW Similarity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, "distribution_10fold.png"))
        plt.close()
        
        # Save results text
        with open(os.path.join(save_dir, "results_10fold.txt"), "w") as f:
            f.write(f"Accuracy Mean: {acc_mean*100:.2f}%\n")
            f.write(f"Accuracy Std:  {acc_std*100:.2f}%\n")
            f.write(f"Avg Threshold: {thresh_mean:.4f}\n")
        
        print(f"💾 Results saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data/processed/lfw")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    model = load_model(args.model, args.weights)
    evaluate(model, args.data_dir, args.save_dir)