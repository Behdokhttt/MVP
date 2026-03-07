"""
Training Script for Multimodal Emotion Classifier
==================================================

Usage:
    python train.py [--epochs 50] [--lr 1e-3] [--batch_size 8] [--data_dir data/processed]
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from dataset import MultimodalDataset, collate_fn
from model import MultimodalEmotionClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal emotion classifier")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with preprocessed .pt files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--branch_hidden", type=int, default=256, help="Hidden dim in FC branches")
    parser.add_argument("--branch_output", type=int, default=128, help="Output dim of each branch")
    parser.add_argument("--rnn_hidden", type=int, default=128, help="GRU hidden dim")
    parser.add_argument("--rnn_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--combiner_hidden", type=int, default=128, help="Combiner hidden dim")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        openface = batch["openface"].to(device)
        openface_lengths = batch["openface_lengths"].to(device)
        transcript = batch["transcript"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(audio, openface, openface_lengths, transcript)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        openface = batch["openface"].to(device)
        openface_lengths = batch["openface_lengths"].to(device)
        transcript = batch["transcript"].to(device)
        labels = batch["label"].to(device)

        logits = model(audio, openface, openface_lengths, transcript)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Dataset & DataLoaders ────────────────────────────────────────────
    dataset = MultimodalDataset(data_dir=args.data_dir)

    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val

    # Guard: if dataset is very small, use all data for training and skip validation
    if len(dataset) < 3:
        print(f"[INFO] Dataset too small ({len(dataset)}) for train/val split — using all for training")
        train_set = dataset
        val_set = None
    else:
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_set else None

    # ── Model ────────────────────────────────────────────────────────────
    model = MultimodalEmotionClassifier(
        num_classes=dataset.num_classes,
        audio_input_dim=dataset.audio_dim,
        openface_input_dim=dataset.openface_dim,
        transcript_input_dim=dataset.transcript_dim,
        branch_hidden_dim=args.branch_hidden,
        branch_output_dim=args.branch_output,
        combiner_hidden_dim=args.combiner_hidden,
        rnn_hidden_dim=args.rnn_hidden,
        rnn_num_layers=args.rnn_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total  |  {trainable_params:,} trainable")
    print(f"Classes: {dataset.num_classes}")

    # ── Training setup ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ── Training loop ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Epoch':>5}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>9}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
        else:
            val_loss, val_acc = train_loss, train_acc
            scheduler.step(train_loss)

        print(f"{epoch:5d}  {train_loss:11.4f}  {train_acc:10.4f}  {val_loss:10.4f}  {val_acc:9.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)

    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
