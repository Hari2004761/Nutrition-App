"""Full evaluation on the held-out test set: confusion matrix + per-class accuracy.

    python src/evaluate.py

Runs the classifier over every held-out test image for all 20 classes (~5,000),
saves the confusion-matrix figure and prints which classes are weakest and what
they get confused with.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101

from predict import load_model

DATA_ROOT = Path("data")
MODELS = Path("models")
FIG_DIR = Path("outputs/figures")

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class RemappedTest(torch.utils.data.Dataset):
    """Food-101 test subset restricted and remapped to the project's classes."""
    def __init__(self, base, indices, label_map, transform):
        self.base, self.indices = base, indices
        self.label_map, self.transform = label_map, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, lbl = self.base[self.indices[i]]
        return self.transform(img), self.label_map[lbl]


def build_test_loader(classes, batch_size=64, num_workers=4):
    """Held-out Food-101 test split, restricted and remapped to `classes`."""
    ds = Food101(root=DATA_ROOT, split="test", download=False)
    keep = {ds.classes.index(c): i for i, c in enumerate(classes)}
    idx = [i for i, lbl in enumerate(ds._labels) if lbl in keep]
    test_ds = RemappedTest(ds, idx, keep, eval_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers), len(test_ds)


def compute_confusion(model, classes, device, batch_size=64, num_workers=4,
                      verbose=True):
    """Run the classifier over every test image; return the confusion matrix.

    num_workers=0 is the safe choice when calling this from inside a server
    process (no worker subprocesses to spawn off the main thread).
    """
    n = len(classes)
    loader, n_images = build_test_loader(classes, batch_size, num_workers)
    if verbose:
        print(f"Evaluating on {n_images} held-out test images "
              f"across {n} classes...")

    confusion = np.zeros((n, n), dtype=int)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            for actual, pred in zip(y.numpy(), preds):
                confusion[actual, pred] += 1
    return confusion


def per_class_report(classes, confusion):
    """Per-class accuracy + top confusion target, worst class first.

    Both the terminal table and the front end's per-class chart read this.
    """
    confusion = np.asarray(confusion)
    per_class_acc = confusion.diagonal() / confusion.sum(axis=1)
    overall_acc = confusion.diagonal().sum() / confusion.sum()

    rows = []
    for i in np.argsort(per_class_acc):          # worst first
        row = confusion[i].copy()
        row[i] = 0                               # exclude the correct answers
        worst_idx = int(row.argmax())
        worst_count = int(row[worst_idx])
        rows.append({
            "name":            classes[i],
            "accuracy":        float(per_class_acc[i]),
            "confused_with":   classes[worst_idx] if worst_count else None,
            "confused_count":  worst_count,
            "support":         int(confusion[i].sum()),
        })
    return {"overall_accuracy": float(overall_acc), "classes": rows}


def evaluate(checkpoint=None, device=None, num_workers=4, verbose=True):
    """Load the checkpoint and evaluate it; returns (classes, confusion)."""
    checkpoint = Path(checkpoint) if checkpoint else MODELS / "classifier_best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}. Train the classifier first.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(checkpoint, device)
    confusion = compute_confusion(model, classes, device,
                                  num_workers=num_workers, verbose=verbose)
    return classes, confusion


def main():
    classes, confusion = evaluate()
    n = len(classes)
    report = per_class_report(classes, confusion)
    overall_acc = report["overall_accuracy"]

    print(f"\nOverall test accuracy: {overall_acc*100:.1f}%\n")
    print(f"{'Class':<20} {'Accuracy':<10} {'Most confused with'}")
    print("-" * 55)

    for r in report["classes"]:
        confused_with = (f"{r['confused_with']} ({r['confused_count']}x)"
                         if r["confused_count"] else "-")
        print(f"{r['name']:<20} {r['accuracy']*100:5.1f}%     {confused_with}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {n} classes, {overall_acc*100:.1f}% overall accuracy")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    out_path = FIG_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}  <- thesis figure")


if __name__ == "__main__":
    main()