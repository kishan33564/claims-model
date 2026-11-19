"""Fine-tune an EfficientNet image classifier (transfer learning).

  python train_image_model.py --task damage   --epochs 15   # damaged vs not_damaged
  python train_image_model.py --task severity --epochs 25   # minor / moderate / severe

Saves models/<task>.pt (weights + class names) and prints test accuracy + confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMG = 260
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def loaders(root: Path, batch: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.Resize(int(IMG * 1.14)), transforms.CenterCrop(IMG), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.ImageFolder(root / "train", train_tf)
    val = datasets.ImageFolder(root / "val", eval_tf)
    test = datasets.ImageFolder(root / "test", eval_tf)
    # balance classes so the rarer ones are not ignored
    counts = np.bincount(train.targets)
    weights = [1.0 / counts[t] for t in train.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    kw = dict(batch_size=batch, num_workers=0, pin_memory=torch.cuda.is_available())
    return train, DataLoader(train, sampler=sampler, **kw), DataLoader(val, **kw), DataLoader(test, **kw)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return (y == p).mean(), y, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["damage", "severity"], required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--arch", default="efficientnet_b2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train_ds, train_dl, val_dl, test_dl = loaders(args.data / args.task, args.batch)
    classes = train_ds.classes
    print("classes:", classes, "device:", device)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out = Path("models"); out.mkdir(exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        val_acc, _, _ = evaluate(model, val_dl, device)
        print(f"epoch {epoch:02d}  loss {total / len(train_ds):.4f}  val_acc {val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save({"arch": args.arch, "classes": classes, "img": IMG, "state_dict": model.state_dict()}, out / f"{args.task}.pt")

    ckpt = torch.load(out / f"{args.task}.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    acc, y, p = evaluate(model, test_dl, device)
    print(f"\nTEST accuracy {acc:.3f}")
    print(classification_report(y, p, target_names=classes))
    print("confusion matrix:\n", confusion_matrix(y, p))
    (out / f"{args.task}_metrics.json").write_text(json.dumps({"test_accuracy": float(acc), "classes": classes}, indent=2))


if __name__ == "__main__":
    main()
