import argparse
import os
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dataset
from loss import ArcFace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def validate(model, metric_fc, loader, criterion):
    model.eval()
    metric_fc.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            embeddings = model(images)

            # ArcFace requires labels for margin penalty
            if isinstance(metric_fc, nn.Linear):
                logits = metric_fc(embeddings)
            else:
                logits = metric_fc(embeddings, labels)

            val_loss += criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    model.train()
    metric_fc.train()
    return val_loss / len(loader), 100 * correct / total


def run_training(args):
    lr = 0.001 if args.optimizer == "adamw" else 0.01

    print(f"Device: {DEVICE} | Model: {args.model} | LR: {lr}")

    writer = SummaryWriter(log_dir=f"runs/{args.model}_{args.loss}_{args.optimizer}")
    os.makedirs("weights", exist_ok=True)

    train_loader, val_loader, num_classes = dataset.get_dataloaders(args.data_dir, args.batch_size)
    if train_loader is None:
        return

    try:
        module = importlib.import_module(f"models.{args.model}")
        model = module.Net(embedding_size=512).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.loss == "arcface":
        metric_fc = ArcFace(512, num_classes).to(DEVICE)
    else:
        metric_fc = nn.Linear(512, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    params = [{"params": model.parameters()}, {"params": metric_fc.parameters()}]

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-2)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    best_acc = 0.0
    step = 0

    for epoch in range(args.epochs):
        model.train()
        metric_fc.train()
        epoch_loss, count = 0.0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            embeddings = model(images)
            if args.loss == "arcface":
                logits = metric_fc(embeddings, labels)
            else:
                logits = metric_fc(embeddings)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
            step += 1
            loop.set_postfix(loss=f"{epoch_loss/count:.4f}")

            if step % 10 == 0:
                writer.add_scalar("Train/Loss", loss.item(), step)

        val_loss, val_acc = validate(model, metric_fc, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "head": metric_fc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": val_acc,
        }

        torch.save(checkpoint, f"weights/{args.model}_{args.loss}_{args.optimizer}_last.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, f"weights/{args.model}_{args.loss}_{args.optimizer}_best.pth")
            print("New best model saved")

    writer.close()
    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed/train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="arcface", choices=["softmax", "arcface"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    run_training(parser.parse_args())