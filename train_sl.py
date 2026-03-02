#!/usr/bin/env python3
"""
Supervised-learning training for the PolicyNet.

Loads an .npz dataset (states + actions from generate_data.py),
trains with cross-entropy on masked logits, validates, and saves
the best checkpoint.

Usage:
    python train_sl.py                                 # defaults
    python train_sl.py --data data/sl_greedy_2000000.npz
    python train_sl.py --epochs 30 --lr 1e-3 --batch 1024
    python train_sl.py --hidden 256 128 64             # custom arch
    python train_sl.py --dropout 0.1                   # regularisation
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from features import FEATURE_DIM, ACTION_DIM, VALID_ACTIONS_OFFSET
from net import PolicyNet


def load_dataset(path: str, val_frac: float = 0.05,
                 seed: int = 42) -> tuple[TensorDataset, TensorDataset]:
    """Load .npz and split into train / val TensorDatasets."""
    data = np.load(path)
    states = torch.from_numpy(data["states"])       # (N, 248)
    actions = torch.from_numpy(data["actions"])      # (N,)

    n = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_n = int(n * val_frac)

    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_ds = TensorDataset(states[train_idx], actions[train_idx])
    val_ds = TensorDataset(states[val_idx], actions[val_idx])
    return train_ds, val_ds


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy (logits already masked)."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def valid_action_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                          states: torch.Tensor) -> float:
    """Fraction of predictions that are legal moves (sanity check)."""
    mask = states[:, VALID_ACTIONS_OFFSET:VALID_ACTIONS_OFFSET + ACTION_DIM]
    preds = logits.argmax(dim=-1)
    legal = mask[torch.arange(len(preds)), preds]
    return legal.float().mean().item()


def train_epoch(model: PolicyNet, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        logits = model.masked_logits(states)
        loss = F.cross_entropy(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += masked_accuracy(logits, actions)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def eval_epoch(model: PolicyNet, loader: DataLoader,
               device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_legal = 0.0
    n_batches = 0

    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        logits = model.masked_logits(states)
        loss = F.cross_entropy(logits, actions)

        total_loss += loss.item()
        total_acc += masked_accuracy(logits, actions)
        total_legal += valid_action_accuracy(logits, actions, states)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches, total_legal / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train PolicyNet (SL)")
    parser.add_argument("--data", type=str,
                        default="data/sl_greedy_2000000.npz",
                        help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128],
                        help="Hidden layer sizes (default: 256 128)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Validation fraction (default: 0.05)")
    parser.add_argument("--out", type=str, default="policy_sl.pt",
                        help="Output model path")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    print(f"Device: {device}")
    print(f"Loading {args.data} ...")
    train_ds, val_ds = load_dataset(args.data, val_frac=args.val_frac,
                                     seed=args.seed)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            pin_memory=True, num_workers=0)

    hidden = tuple(args.hidden)
    model = PolicyNet(hidden_dims=hidden, dropout=args.dropout).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {' → '.join(map(str, [FEATURE_DIM] + list(hidden) + [ACTION_DIM]))}  ({params:,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_loss = float("inf")
    patience_counter = 0

    print()
    header = f"{'Ep':>3}  {'t_loss':>7}  {'t_acc':>6}  {'v_loss':>7}  {'v_acc':>6}  {'legal':>5}  {'lr':>8}  {'time':>5}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_legal = eval_epoch(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        scheduler.step()

        improved = val_loss < best_val_loss
        marker = " *" if improved else ""
        print(f"{epoch:3d}  {train_loss:7.4f}  {train_acc:5.1%}  "
              f"{val_loss:7.4f}  {val_acc:5.1%}  {val_legal:5.1%}  "
              f"{lr:8.1e}  {elapsed:5.1f}s{marker}")

        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_dims": hidden,
                "dropout": args.dropout,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, args.out)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (patience={args.patience})")
                break

    # Load best and report
    ckpt = torch.load(args.out, weights_only=True)
    print(f"\nBest model (epoch {ckpt['epoch']}): "
          f"val_loss={ckpt['val_loss']:.4f}  val_acc={ckpt['val_acc']:.1%}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
