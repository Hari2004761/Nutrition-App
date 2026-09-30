"""Fine-tunes MobileNetV3-Large on the 20-class Food-101 subset.

    python src/train_classifier.py                 # full two-phase run
    python src/train_classifier.py --smoke         # 1 short epoch, sanity check

Phase 1 trains only the new head (backbone frozen).
Phase 2 unfreezes everything at a low LR.

Three-way split: Food-101's official test split is never touched here — it
belongs to evaluate.py alone. Validation is carved out of the training pool
instead (100 images per class, stratified), so the checkpoint is never selected
on the data it is finally graded on. The chosen indices are saved to
outputs/split_indices.json and reused, so every run sees the same three sets.

Every epoch appends to outputs/history.csv — the training graphs read that file,
so rename it rather than delete it between experiments.
"""
import argparse
import csv
import functools
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

DATA_ROOT = Path("data")
OUT = Path("outputs")
MODELS = Path("models")
SPLIT_PATH = OUT / "split_indices.json"

# Held out of Food-101's 750-per-class training pool, leaving 650 to train on.
VAL_PER_CLASS = 100
# The split carries its own seed, so re-running with a different --seed measures
# run-to-run variance without also moving images between the sets.
SPLIT_SEED = 1234

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


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy and torch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark autotunes the fastest conv algorithm per input shape, but
    # the winner is not guaranteed to be the same twice; --deterministic gives
    # that autotuning up, and the throughput with it, for repeatable kernels.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _seed_worker(worker_id, base_seed=0):
    """Re-seed a DataLoader worker. Module level, so Windows can pickle it."""
    random.seed(base_seed + worker_id)
    np.random.seed((base_seed + worker_id) % 2**32)


def _image_key(ds, i):
    """class/image-id for one dataset index — stable across splits and runs."""
    p = Path(ds._image_files[i])
    return f"{p.parent.name}/{p.stem}"


def _fingerprint(ds, by_class, classes):
    """Hash the image names behind a split, so a saved file can be audited."""
    keys = [_image_key(ds, i) for c in classes for i in by_class[c]]
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()[:16]


def make_split(ds, classes, val_per_class, seed):
    """Stratified train/val split of the training pool: exactly `val_per_class`
    validation images from every class, drawn at random rather than by order."""
    by_class = {c: [] for c in classes}
    for i, lbl in enumerate(ds._labels):
        name = ds.classes[lbl]
        if name in by_class:
            by_class[name].append(i)

    rng = random.Random(seed)     # own generator, independent of the training seed
    train, val = {}, {}
    for c in classes:
        pool = sorted(by_class[c])
        if len(pool) <= val_per_class:
            raise SystemExit(f"class {c} has only {len(pool)} training images, "
                             f"cannot hold out {val_per_class}")
        rng.shuffle(pool)
        val[c] = sorted(pool[:val_per_class])
        # Left in shuffled order so that a per-class cap takes a random subset.
        train[c] = pool[val_per_class:]
    return train, val


def load_split(classes, val_per_class=VAL_PER_CLASS, seed=SPLIT_SEED, resplit=False):
    """The saved split, rebuilt only when it is missing, stale, or forced."""
    ds = Food101(root=DATA_ROOT, split="train", download=False)

    record = None
    if SPLIT_PATH.exists() and not resplit:
        record = json.loads(SPLIT_PATH.read_text())
        if (record.get("classes") != list(classes)
                or record.get("val_per_class") != val_per_class
                or record.get("split_seed") != seed):
            print(f"{SPLIT_PATH} was built for a different configuration — rebuilding.")
            record = None

    if record is not None:
        train, val = record["train_indices"], record["val_indices"]
        if _fingerprint(ds, val, classes) != record.get("val_fingerprint"):
            raise SystemExit(f"{SPLIT_PATH} no longer matches the images on disk. "
                             f"Re-run with --resplit if the dataset changed.")
        print(f"Using the saved split in {SPLIT_PATH} "
              f"(seed {seed}, val fingerprint {record['val_fingerprint']})")
        return ds, train, val, record

    train, val = make_split(ds, classes, val_per_class, seed)
    record = {
        "created":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset":         "food-101, train split only",
        "classes":         list(classes),
        "split_seed":      seed,
        "val_per_class":   val_per_class,
        "train_per_class": len(train[classes[0]]),
        "note":            ("Indices address torchvision Food101(split='train'). "
                            "Validation indices are sorted; training indices keep "
                            "their shuffled order so a per-class cap stays random. "
                            "Food-101's test split is deliberately absent — it is "
                            "evaluate.py's alone."),
        "val_fingerprint":   _fingerprint(ds, val, classes),
        "train_fingerprint": _fingerprint(ds, train, classes),
        "train_indices":   train,
        "val_indices":     val,
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(record, indent=2))
    print(f"Wrote a new split to {SPLIT_PATH} "
          f"(seed {seed}, val fingerprint {record['val_fingerprint']})")
    return ds, train, val, record


def verify_split(ds, classes, train_idx, val_idx):
    """Prove train/val/test share no image, and print what each set holds."""
    test_ds = Food101(root=DATA_ROOT, split="test", download=False)
    wanted = set(classes)

    train_keys = {_image_key(ds, i) for i in train_idx}
    val_keys = {_image_key(ds, i) for i in val_idx}
    test_idx = [i for i, lbl in enumerate(test_ds._labels)
                if test_ds.classes[lbl] in wanted]
    test_keys = {_image_key(test_ds, i) for i in test_idx}

    for a, b, first, second in [("train", "val", train_keys, val_keys),
                                ("train", "test", train_keys, test_keys),
                                ("val", "test", val_keys, test_keys)]:
        shared = first & second
        if shared:
            raise SystemExit(f"{len(shared)} image(s) are in both {a} and {b}, "
                             f"e.g. {sorted(shared)[:3]}")

    def per_class(indices, dataset):
        counts = {}
        for i in indices:
            name = dataset.classes[dataset._labels[i]]
            counts[name] = counts.get(name, 0) + 1
        lo, hi = min(counts.values()), max(counts.values())
        return f"{lo}/class" if lo == hi else f"{lo}-{hi}/class"

    print(f"\nSplit — {len(classes)} classes, no image in more than one set:")
    print(f"  train  {len(train_keys):6,} images  {per_class(train_idx, ds):>12}")
    print(f"  val    {len(val_keys):6,} images  {per_class(val_idx, ds):>12}"
          f"   held out of the training pool")
    print(f"  test   {len(test_keys):6,} images  {per_class(test_idx, test_ds):>12}"
          f"   untouched here — evaluate.py only\n")


def build_loaders(base_ds, classes, train_idx, val_idx, batch_size, workers, seed):
    """Training and validation loaders over the saved split."""
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

    keep = {base_ds.classes.index(c): i for i, c in enumerate(classes)}

    # Workers fork their own RNGs, so seeding the sampler's generator and each
    # worker is what actually makes shuffling and augmentation repeatable.
    gen = torch.Generator()
    gen.manual_seed(seed)
    init_worker = functools.partial(_seed_worker, base_seed=seed) if workers else None

    train_loader = DataLoader(
        RemappedSubset(base_ds, train_idx, keep, train_tf),
        batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
        generator=gen, worker_init_fn=init_worker,
    )
    val_loader = DataLoader(
        RemappedSubset(base_ds, val_idx, keep, eval_tf),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
        worker_init_fn=init_worker,
    )
    return train_loader, val_loader


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
                    help="Ignore the per-class cap in subset_meta.json and train on "
                         "the whole training pool (650/class once validation is held "
                         "out). Use this for the final training run.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for torch, numpy and random (default 42)")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                    help=f"Seed for the train/val split (default {SPLIT_SEED}), kept "
                         f"separate from --seed so reseeding a run does not move "
                         f"images between the sets")
    ap.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS,
                    help=f"Validation images held out per class "
                         f"(default {VAL_PER_CLASS})")
    ap.add_argument("--resplit", action="store_true",
                    help="Rebuild outputs/split_indices.json even if it is still valid")
    ap.add_argument("--deterministic", action="store_true",
                    help="Deterministic cuDNN kernels: repeatable, but gives up the "
                         "autotuner and roughly 10-20%% of training throughput")
    ap.add_argument("--run-dir", default=None,
                    help="Write the checkpoint and history to models/<NAME>/ and "
                         "outputs/<NAME>/ instead of the default paths, so an "
                         "experiment cannot overwrite the current result. The split "
                         "in outputs/ is still shared and read-only.")
    ap.add_argument("--early-stop", action="store_true",
                    help="Stop once validation accuracy has not improved for "
                         "--patience epochs. Off by default.")
    ap.add_argument("--patience", type=int, default=4,
                    help="Epochs without a new best validation accuracy before "
                         "--early-stop fires (default 4)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    run_out, run_models = OUT, MODELS
    if args.run_dir:
        run_out, run_models = OUT / args.run_dir, MODELS / args.run_dir
        run_out.mkdir(parents=True, exist_ok=True)
        run_models.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic)
    meta = json.loads((OUT / "subset_meta.json").read_text())
    classes = meta["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seed={args.seed}  "
          f"cudnn.deterministic={torch.backends.cudnn.deterministic}")

    base_ds, train_by_class, val_by_class, split = load_split(
        classes, args.val_per_class, args.split_seed, args.resplit)

    # The cap only ever thins the training set; validation stays whole so that
    # runs at different sizes are still scored on exactly the same images.
    max_train = None if args.full_data else meta.get("max_train_per_class")
    train_idx = [i for c in classes
                 for i in (train_by_class[c][:max_train] if max_train
                           else train_by_class[c])]
    val_idx = [i for c in classes for i in val_by_class[c]]

    verify_split(base_ds, classes, train_idx, val_idx)
    if max_train:
        print(f"Training on a capped {max_train}/class — pass --full-data for all "
              f"{split['train_per_class']}/class.\n")

    train_loader, val_loader = build_loaders(
        base_ds, classes, train_idx, val_idx,
        args.batch_size, args.workers, args.seed)

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda") if _NEW_AMP else GradScaler()
    # Smoke runs go to their own file so a sanity check never lands in the
    # history that draws the thesis graphs.
    history = run_out / ("history_smoke.csv" if args.smoke else "history.csv")
    best = 0.0
    best_at = "none"   # phase/epoch the saved checkpoint came from
    stalled = 0        # epochs since the last new best, for --early-stop

    if args.smoke:
        phases = [("smoke", 1, 1e-3, False)]
    else:
        # Head first at 1e-3: only the new layer is learning, so big steps are safe.
        # Then everything at 1e-4, small enough not to wash out pretrained features.
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

            # Selected on held-out validation data; the test split plays no part.
            if va_acc > best and not args.smoke:
                best = va_acc
                best_at = f"{phase} ep{ep}"
                stalled = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "classes":    classes,
                    "val_acc":    round(va_acc, 4),
                    "seed":       args.seed,
                    "best_epoch": best_at,
                    "split": {
                        "split_seed":      split["split_seed"],
                        "val_per_class":   split["val_per_class"],
                        "train_per_class": len(train_idx) // len(classes),
                        "val_fingerprint": split["val_fingerprint"],
                    },
                }, run_models / "classifier_best.pt")
                print(f"       saved new best ({best:.3f})")
            else:
                stalled += 1
                if args.early_stop and stalled >= args.patience:
                    print(f"       early stop: {stalled} epoch(s) without a new "
                          f"best (patience {args.patience})")
                    break
        else:
            continue       # phase finished without early stopping
        break

    print(f"\nDone. Best val acc: {best:.3f} at {best_at}  "
          f"(held-out validation, {len(val_idx)} images)")
    print(f"Per-epoch log: {history}  <- your thesis loss/accuracy graphs")
    print("Run src/evaluate.py for the test-split score — that data was not used here.")


if __name__ == "__main__":
    main()
