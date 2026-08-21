"""Fine-tunes MobileNetV3-Large on the 20-class Food-101 subset.

    python src/train_classifier.py                 # full two-phase run
    python src/train_classifier.py --smoke         # 1 short epoch, sanity check

Phase 1 trains only the new head (backbone frozen).
Phase 2 unfreezes everything at a low LR.
Every epoch appends to outputs/history.csv -- that file IS your loss/accuracy
graphs later, so don't delete it between experiments; rename it instead.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

DATA_ROOT = Path("data")
OUT = Path("outputs")
MODELS = Path("models")

# AMP moved namespaces across torch versions; support both.
try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except ImportError:  # torch < 2.4
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP = False


class RemappedSubset(torch.utils.data.Dataset):
    """Wraps a Food101 subset so labels come out as 0..N-1."""

    def __init__(self, base, indices, label_map, transform):
        self.base, self.indices = base, indices
        self.label_map, self.transform = label_map, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, lbl = self.base[self.indices[i]]
        return self.transform(img), self.label_map[lbl]


def build_loaders(classes, batch_size, workers, max_train_per_class=None,
                   max_test_per_class=None):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),  # food photos vary wildly in lighting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loaders = []
    for split, tf, shuffle, cap in [
        ("train", train_tf, True, max_train_per_class),
        ("test", eval_tf, False, max_test_per_class),
    ]:
        ds = Food101(root=DATA_ROOT, split=split, download=False)
        keep = {ds.classes.index(c): i for i, c in enumerate(classes)}

        if cap:
            # Take the first `cap` images per class (Food101 lists images
            # grouped by class, so this keeps every class equally represented
            # rather than skewing toward whichever classes appear first).
            per_class_count = {}
            idx = []
            for i, lbl in enumerate(ds._labels):
                if lbl not in keep:
                    continue
                per_class_count[lbl] = per_class_count.get(lbl, 0) + 1
                if per_class_count[lbl] <= cap:
                    idx.append(i)
        else:
            idx = [i for i, lbl in enumerate(ds._labels) if lbl in keep]

        wrapped = RemappedSubset(ds, idx, keep, tf)
        loaders.append(DataLoader(
            wrapped, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
        ))
    return loaders


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              limit=None):
    train = optimizer is not None
    model.train(train)
    total_loss = correct = seen = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for step, (x, y) in enumerate(loader):
            if limit and step >= limit:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            amp_args = ("cuda",) if _NEW_AMP else ()
            with autocast(*amp_args):
                out = model(x)
                loss = criterion(out, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)

    return total_loss / seen, correct / seen


def log_row(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--full-epochs", type=int, default=7)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch capped at 20 batches, to prove the pipeline runs")
    ap.add_argument("--full-data", action="store_true",
                    help="Ignore the per-class cap in subset_meta.json and use "
                         "every available image (750 train / 250 test per class). "
                         "Use this for your final training run once the pipeline "
                         "is proven to work.")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    meta = json.loads((OUT / "subset_meta.json").read_text())
    classes = meta["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_train = None if args.full_data else meta.get("max_train_per_class")
    max_test = None if args.full_data else meta.get("max_test_per_class")
    if max_train:
        print(f"Using capped subset: {max_train}/class train, {max_test}/class test")
        print("(pass --full-data to use everything for your final run)")

    train_loader, val_loader = build_loaders(
        classes, args.batch_size, args.workers, max_train, max_test)

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda") if _NEW_AMP else GradScaler()
    history = OUT / "history.csv"
    best = 0.0

    if args.smoke:
        phases = [("smoke", 1, 1e-3, False)]
    else:
        phases = [("head", args.head_epochs, 1e-3, False),
                  ("full", args.full_epochs, 1e-4, True)]

    for phase, epochs, lr, unfreeze in phases:
        for p in model.features.parameters():
            p.requires_grad = unfreeze
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"\n--- phase '{phase}': {epochs} epoch(s), lr={lr}, "
              f"trainable={sum(p.numel() for p in params):,} ---")

        for ep in range(1, epochs + 1):
            t0 = time.time()
            cap = 20 if args.smoke else None
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                        optimizer, scaler, limit=cap)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                        limit=cap)
            sched.step()

            log_row(history, {
                "phase": phase, "epoch": ep, "lr": round(sched.get_last_lr()[0], 8),
                "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
                "secs": round(time.time() - t0, 1),
            })
            print(f"  ep{ep}  train {tr_loss:.3f}/{tr_acc:.3f}   "
                  f"val {va_loss:.3f}/{va_acc:.3f}   {time.time()-t0:.0f}s")

            if va_acc > best and not args.smoke:
                best = va_acc
                torch.save({"state_dict": model.state_dict(), "classes": classes},
                           MODELS / "classifier_best.pt")
                print(f"       saved new best ({best:.3f})")

    print(f"\nDone. Best val acc: {best:.3f}")
    print(f"Per-epoch log: {history}  <- your thesis loss/accuracy graphs")


if __name__ == "__main__":
    main()
