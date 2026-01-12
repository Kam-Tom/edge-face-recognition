import argparse
import importlib
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
# Ważne na serwerze bez ekranu (RunPod):
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve, auc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def load_model(model_name, weights_path):
    try:
        module = importlib.import_module(f"models.{model_name}")
        model = module.Net(embedding_size=512).to(DEVICE)
        
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        exit(1)

def parse_pairs(pairs_path):
    pairs = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:] 
        
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

def plot_results(fpr, tpr, roc_auc, best_thresh, pos_sims, neg_sims, save_path):
    """
    Rysuje wykresy w stylu, który wolisz:
    - ROC z czerwoną kropką w punkcie optymalnym.
    - Histogram z zielonym/czerwonym kolorem i podanymi średnimi (μ).
    """
    # Ponowne obliczenie indexu dla kropki na wykresie
    optimal_idx = np.argmax(tpr - fpr)

    plt.figure(figsize=(12, 5))

    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    # Czerwona kropka (Best Threshold)
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], c='red', s=100, zorder=5, 
                label=f'Best (thresh={best_thresh:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 2. Histogram (Similarity Distribution)
    plt.subplot(1, 2, 2)
    plt.hist(pos_sims, bins=50, alpha=0.7, label=f'Same person (μ={pos_sims.mean():.3f})', color='green')
    plt.hist(neg_sims, bins=50, alpha=0.7, label=f'Diff person (μ={neg_sims.mean():.3f})', color='red')
    plt.axvline(x=best_thresh, color='blue', linestyle='--', linewidth=2, label=f'Threshold={best_thresh:.3f}')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate(model, data_dir, save_dir=None):
    pairs_path = os.path.join(data_dir, "pairs.txt")
    if not os.path.exists(pairs_path):
        print(f"❌ Brak pliku pairs.txt w {data_dir}")
        return

    print(f"📄 Wczytywanie par z: {pairs_path}")
    pairs = parse_pairs(pairs_path)
    print(f"🔍 Do sprawdzenia: {len(pairs)} par")
    
    transform = get_transforms()
    sims = []
    labels = []
    missing_count = 0
    
    with torch.no_grad():
        for p in tqdm(pairs, desc="Testowanie"):
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
        print(f"⚠️ Pominięto {missing_count} par (brak zdjęć).")

    # --- ANALIZA DANYCH ---
    sims = np.array(sims)
    labels = np.array(labels)
    
    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]

    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]
    
    preds = (sims >= best_thresh).astype(int)
    acc = (preds == labels).mean()

    # TAR @ FAR
    def get_tar_at_far(target_far):
        idx = np.argmin(np.abs(fpr - target_far))
        return tpr[idx]

    tar_01 = get_tar_at_far(0.01)
    tar_001 = get_tar_at_far(0.001)

    # Wypisywanie (ZMIENIONO 'AUC Score' NA 'AUC')
    print(f"\n{'='*40}")
    print(f"🏆 WYNIK KOŃCOWY (LFW Protocol)")
    print(f"{'='*40}")
    print(f"✅ Accuracy:       {acc*100:.2f}%")
    print(f"✅ AUC:            {roc_auc:.4f}")
    print(f"🎯 Best Threshold: {best_thresh:.4f}")
    print(f"----------------------------------------")
    print(f"📈 TAR @ 1% FAR:   {tar_01*100:.2f}%")
    print(f"📈 TAR @ 0.1% FAR: {tar_001*100:.2f}%")
    print(f"----------------------------------------")
    print(f"🟢 Pos Sim (Same): {pos_sims.mean():.4f} ± {pos_sims.std():.4f}")
    print(f"🔴 Neg Sim (Diff): {neg_sims.mean():.4f} ± {neg_sims.std():.4f}")
    print(f"↔️  Gap:            {pos_sims.mean() - neg_sims.mean():.4f}")
    print(f"{'='*40}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Text Metrics (ZMIENIONO 'AUC Score' NA 'AUC')
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy:       {acc*100:.2f}%\n")
            f.write(f"AUC:            {roc_auc:.4f}\n")
            f.write(f"Best Threshold: {best_thresh:.4f}\n")
            f.write(f"TAR @ 1% FAR:   {tar_01*100:.2f}%\n")
            f.write(f"TAR @ 0.1% FAR: {tar_001*100:.2f}%\n")
            f.write(f"Pos Sim Mean:   {pos_sims.mean():.4f}\n")
            f.write(f"Pos Sim Std:    {pos_sims.std():.4f}\n")
            f.write(f"Neg Sim Mean:   {neg_sims.mean():.4f}\n")
            f.write(f"Neg Sim Std:    {neg_sims.std():.4f}\n")

        # 2. Plots (Używa nowej-starej funkcji plot_results)
        plot_results(fpr, tpr, roc_auc, best_thresh, pos_sims, neg_sims, 
                     os.path.join(save_dir, "plots.png"))
        
        # 3. Raw Data
        np.savez(os.path.join(save_dir, "raw_data.npz"), 
                 sims=sims, labels=labels, 
                 fpr=fpr, tpr=tpr, thresholds=thresholds)
        
        print(f"💾 Wszystko zapisano w: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed/lfw_112")
    parser.add_argument("--batch_size", type=int, default=32, help="Ignorowane")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    model = load_model(args.model, args.weights)
    evaluate(model, args.data_dir, args.save_dir)