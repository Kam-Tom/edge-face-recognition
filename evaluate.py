import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name, weights_path):
    try:
        module = importlib.import_module(f"models.{model_name}")
        model = module.Net(embedding_size=512).to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"Failed to import model: {e}")

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_embeddings(model, loader):
    embeddings, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting embeddings", leave=False):
            imgs = imgs.to(DEVICE)
            emb = F.normalize(model(imgs), p=2, dim=1)
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())

    return np.concatenate(embeddings), np.concatenate(labels)


def evaluate_roc(embeddings, labels):
    """Compute verification metrics using cosine similarity."""
    sim_matrix = embeddings @ embeddings.T
    ground_truth = (labels[:, None] == labels[None, :]).astype(int)

    triu_idx = np.triu_indices_from(sim_matrix, k=1)
    y_true = ground_truth[triu_idx]
    y_score = sim_matrix[triu_idx]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Optimal threshold via Youden's J
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    accuracy = (tpr[optimal_idx] + (1 - fpr[optimal_idx])) / 2

    print(f"AUC: {roc_auc:.4f}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed/val")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    _, val_loader, _ = dataset.get_dataloaders(args.data_dir, args.batch_size)
    if val_loader is None:
        raise RuntimeError("Validation loader is empty")

    model = load_model(args.model, args.weights)
    embs, lbls = get_embeddings(model, val_loader)
    evaluate_roc(embs, lbls)