import argparse
import os
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dataset
from loss import ArcFaceLoss, SoftmaxLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def validate(model, metric_fc, loader, criterion):
    model.eval()
    metric_fc.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            embeddings = model(images)
            logits = metric_fc.get_logits(embeddings)
            val_loss += criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    model.train()
    metric_fc.train()
    return val_loss / len(loader), 100 * correct / total


def run_training(args):
    # === LR CONFIG ===
    if args.optimizer == "adamw":
        lr = 0.001
    else:
        lr = 0.1

    print(f"Device: {DEVICE} | Model: {args.model} | LR: {lr:.6f} | Batch: {args.batch_size} | Epochs: {args.epochs}")

    model_name = args.model.replace(".", "_")
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{args.loss}_{args.optimizer}")
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
        metric_fc = ArcFaceLoss(512, num_classes).to(DEVICE)
    else:
        metric_fc = SoftmaxLoss(512, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    params = [{"params": model.parameters()}, {"params": metric_fc.parameters()}]

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-2)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    # === SCHEDULER ===
    if args.epochs >= 50:
        milestones = [25, 35, 45]
    elif args.epochs >= 25:
        milestones = [12, 18, 22]
    else:
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    print(f"LR Milestones: {milestones}")

    # === WARMUP ===
    warmup_epochs = 3
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch

    best_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        metric_fc.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        count = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Warmup LR
            if global_step < warmup_steps:
                warmup_lr = lr * (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            optimizer.zero_grad()
            embeddings = model(images)
            logits = metric_fc(embeddings, labels)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"⚠️ NaN at step {global_step}, skipping")
                global_step += 1
                continue

            # Track batch accuracy (no grad needed, logits already computed)
            with torch.no_grad():
                batch_acc = (logits.argmax(1) == labels).float().mean().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(metric_fc.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += batch_acc
            count += 1
            global_step += 1
            
            loop.set_postfix(loss=f"{epoch_loss/count:.4f}", acc=f"{100*epoch_acc/count:.1f}%")

        # Scheduler step after warmup
        if global_step >= warmup_steps:
            scheduler.step()

        # Epoch metrics
        train_loss = epoch_loss / count if count > 0 else 0.0
        train_acc = 100 * epoch_acc / count if count > 0 else 0.0
        val_loss, val_acc = validate(model, metric_fc, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}% | LR: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Train/LR", current_lr, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalars("Compare/Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Compare/Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("Gap/Loss", val_loss - train_loss, epoch)
        writer.add_scalar("Gap/Accuracy", train_acc - val_acc, epoch)

        # Save checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "head": metric_fc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

        torch.save(checkpoint, f"weights/{model_name}_{args.loss}_{args.optimizer}_last.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, f"weights/{model_name}_{args.loss}_{args.optimizer}_best.pth")
            print("✅ New best model saved")

    writer.close()
    
    with open("training_summary.txt", "a") as f:
        f.write(f"{args.model},{args.loss},{args.optimizer},{train_loss:.4f},{val_loss:.4f},{train_acc:.2f},{val_acc:.2f},{best_acc:.2f}\n")
    
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed/train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="arcface", choices=["softmax", "arcface"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    args = parser.parse_args()
    run_training(args)