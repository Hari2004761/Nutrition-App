"""Samples one real test-set photo per class and checks the model's predictions.

    python src/quick_test.py

Samples only test-split images, so the result reflects genuine held-out
performance rather than memorised training photos.
"""
import random
from pathlib import Path

import torch

from predict import load_model, predict

DATA_ROOT = Path("data/food-101")
MODELS = Path("models")


def load_test_split_by_class(classes):
    """Returns {class_name: [image_path, ...]} using only test-split images."""
    test_list = (DATA_ROOT / "meta" / "test.txt").read_text().splitlines()
    by_class = {c: [] for c in classes}

    for line in test_list:
        cls, _, img_id = line.partition("/")
        if cls in by_class:
            by_class[cls].append(DATA_ROOT / "images" / cls / f"{img_id}.jpg")

    return by_class


def main():
    checkpoint = MODELS / "classifier_best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}. Train the classifier first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(checkpoint, device)

    by_class = load_test_split_by_class(classes)
    random.seed(42)  # reproducible sample -- same photos every run

    print(f"Testing {len(classes)} classes, 1 random held-out photo each\n")
    print(f"{'Actual':<20} {'Predicted':<20} {'Conf':<8} {'Result'}")
    print("-" * 60)

    correct = 0
    for cls in classes:
        candidates = by_class.get(cls, [])
        if not candidates:
            print(f"{cls:<20} (no test images found on disk)")
            continue

        img_path = random.choice(candidates)
        results = predict(model, classes, img_path, device, top_k=1)
        pred_label, pred_conf = results[0]

        is_correct = pred_label == cls
        correct += is_correct
        mark = "correct" if is_correct else "WRONG"

        print(f"{cls:<20} {pred_label:<20} {pred_conf*100:5.1f}%  {mark}")

    print("-" * 60)
    print(f"{correct}/{len(classes)} correct on this sample "
          f"({100*correct/len(classes):.0f}%)")
    print("\nNote: this is a small, 1-image-per-class spot check, not a "
          "formal metric -- the real figure is src/evaluate.py's 89.7% over "
          "all 5,000 test images.")


if __name__ == "__main__":
    main()