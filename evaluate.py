import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_transforms():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def load_model(model_name, weights_path):
    module = importlib.import_module(f"models.{model_name}")
    model = module.Net(embedding_size=512).to(DEVICE)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_embeddings(model, loader):
    embeddings, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(DEVICE)
            emb = model(imgs)
            emb = F.normalize(emb, p=2, dim=1)  # L2 normalize
            embeddings.append(emb.cpu())
            labels.append(lbls)

    return torch.cat(embeddings), torch.cat(labels).numpy()


def evaluate_verification(embeddings, labels, num_pairs=10000, save_dir="results"):
    """
    Face VERIFICATION test:
    - Given two face images, are they the same person?
    - Uses cosine similarity between embeddings
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Group by identity
    id_to_idx = {}
    for i, lbl in enumerate(labels):
        id_to_idx.setdefault(int(lbl), []).append(i)
    
    valid_ids = [k for k, v in id_to_idx.items() if len(v) >= 2]
    print(f"Identities with 2+ images: {len(valid_ids)}")
    
    if len(valid_ids) < 2:
        print("Error: Need more identities!")
        return
    
    np.random.seed(42)
    pos_sims, neg_sims = [], []
    
    # Positive pairs (SAME person)
    for _ in range(num_pairs):
        pid = np.random.choice(valid_ids)
        i, j = np.random.choice(id_to_idx[pid], 2, replace=False)
        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        pos_sims.append(sim)
    
    # Negative pairs (DIFFERENT person)
    for _ in range(num_pairs):
        p1, p2 = np.random.choice(valid_ids, 2, replace=False)
        i = np.random.choice(id_to_idx[p1])
        j = np.random.choice(id_to_idx[p2])
        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        neg_sims.append(sim)
    
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    # ROC curve
    y_true = np.array([1] * len(pos_sims) + [0] * len(neg_sims))
    y_score = np.concatenate([pos_sims, neg_sims])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Find best threshold
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]
    
    # Calculate accuracy at best threshold
    preds = (y_score >= best_thresh).astype(int)
    accuracy = (preds == y_true).mean()
    
    # TAR @ FAR (True Accept Rate at False Accept Rate)
    tar_at_far_001 = tpr[np.argmin(np.abs(fpr - 0.01))]  # TAR @ 1% FAR
    tar_at_far_01 = tpr[np.argmin(np.abs(fpr - 0.001))]  # TAR @ 0.1% FAR
    
    # Print results
    print(f"\n{'='*50}")
    print(f"FACE VERIFICATION RESULTS")
    print(f"{'='*50}")
    print(f"Pairs tested:     {len(pos_sims)} positive, {len(neg_sims)} negative")
    print(f"")
    print(f"AUC:              {roc_auc:.4f}")
    print(f"Accuracy:         {accuracy*100:.2f}%")
    print(f"Best threshold:   {best_thresh:.4f}")
    print(f"TAR @ 1% FAR:     {tar_at_far_001*100:.2f}%")
    print(f"TAR @ 0.1% FAR:   {tar_at_far_01*100:.2f}%")
    print(f"")
    print(f"Positive sim:     {pos_sims.mean():.4f} ± {pos_sims.std():.4f}")
    print(f"Negative sim:     {neg_sims.mean():.4f} ± {neg_sims.std():.4f}")
    print(f"Gap:              {pos_sims.mean() - neg_sims.mean():.4f}")
    print(f"{'='*50}")
    
    # Plot 1: ROC Curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], c='red', s=100, zorder=5, 
                label=f'Best (thresh={best_thresh:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Similarity Distribution
    plt.subplot(1, 2, 2)
    plt.hist(pos_sims, bins=50, alpha=0.7, label=f'Same person (μ={pos_sims.mean():.3f})', color='green')
    plt.hist(neg_sims, bins=50, alpha=0.7, label=f'Different person (μ={neg_sims.mean():.3f})', color='red')
    plt.axvline(x=best_thresh, color='blue', linestyle='--', linewidth=2, label=f'Threshold={best_thresh:.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/verification_results.png", dpi=150)
    plt.close()
    print(f"\nPlots saved to {save_dir}/verification_results.png")
    
    # Save metrics to file
    with open(f"{save_dir}/metrics.txt", "w") as f:
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Best threshold: {best_thresh:.4f}\n")
        f.write(f"TAR @ 1% FAR: {tar_at_far_001*100:.2f}%\n")
        f.write(f"TAR @ 0.1% FAR: {tar_at_far_01*100:.2f}%\n")
        f.write(f"Positive sim: {pos_sims.mean():.4f} ± {pos_sims.std():.4f}\n")
        f.write(f"Negative sim: {neg_sims.mean():.4f} ± {neg_sims.std():.4f}\n")
    
    # Save raw data for further analysis
    np.savez(f"{save_dir}/raw_data.npz", 
             pos_sims=pos_sims, neg_sims=neg_sims,
             fpr=fpr, tpr=tpr, thresholds=thresholds)
    
    return {
        "auc": roc_auc,
        "accuracy": accuracy,
        "threshold": best_thresh,
        "tar_far_001": tar_at_far_001,
        "pos_sim_mean": pos_sims.mean(),
        "neg_sim_mean": neg_sims.mean()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed/val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_pairs", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    # Load data directly (not using dataset.py split)
    dataset_obj = datasets.ImageFolder(args.data_dir, transform=get_transforms())
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loaded {len(dataset_obj)} images, {len(dataset_obj.classes)} identities")
    
    model = load_model(args.model, args.weights)
    embeddings, labels = get_embeddings(model, loader)
    evaluate_verification(embeddings, labels, args.num_pairs, args.save_dir)