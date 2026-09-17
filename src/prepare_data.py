"""Downloads Food-101 and builds the 20-class working subset.

    python src/prepare_data.py

First run downloads ~5 GB and extracts it -- expect 15-40 minutes.
Later runs are instant (download=True skips if already present).
"""
import json
from pathlib import Path

from torchvision.datasets import Food101

DATA_ROOT = Path("data")
OUT = Path("outputs")

# 20 of the 101 Food-101 classes: globally popular dishes, a vegetarian/chicken
# mix, one deliberately confusable pair (fried_rice / paella), and the only two
# Indian dishes Food-101 offers.
CLASSES = [
    "pizza", "sushi", "ice_cream", "hamburger", "donuts",
    "french_fries", "onion_rings", "caesar_salad", "omelette",
    "chicken_curry", "chicken_wings", "tacos", "samosa",
    "pancakes", "waffles", "fried_rice", "paella",
    "falafel", "macaroni_and_cheese", "cheesecake",
]

# Per-class cap for quick training runs; None uses the whole pool. The cap
# applies to the training set only — train_classifier.py holds 100 images per
# class out of the training pool for validation before any cap is applied, and
# never touches the test split. MAX_TEST_PER_CLASS is recorded for reference;
# evaluate.py always scores on the full 250/class test split.
MAX_TRAIN_PER_CLASS = 200
MAX_TEST_PER_CLASS = 80


def main():
    DATA_ROOT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    print("Fetching Food-101 (skips if already downloaded)...")
    train = Food101(root=DATA_ROOT, split="train", download=True)
    test = Food101(root=DATA_ROOT, split="test", download=True)

    all_classes = train.classes
    missing = [c for c in CLASSES if c not in all_classes]
    if missing:
        raise SystemExit(f"Not real Food-101 class names: {missing}")

    keep = {all_classes.index(c): i for i, c in enumerate(CLASSES)}

    counts = {}
    for name, ds in [("train", train), ("test", test)]:
        n = sum(1 for lbl in ds._labels if lbl in keep)
        counts[name] = n
        print(f"  {name}: {n} images across {len(CLASSES)} classes")

    meta = {
        "classes": CLASSES,
        "original_index_to_subset_index": {str(k): v for k, v in keep.items()},
        "counts": counts,
        "max_train_per_class": MAX_TRAIN_PER_CLASS,
        "max_test_per_class": MAX_TEST_PER_CLASS,
    }
    (OUT / "subset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {OUT / 'subset_meta.json'}")
    print("Food-101 is balanced: 750 train / 250 test per class natively.")
    if MAX_TRAIN_PER_CLASS:
        actual_train = min(MAX_TRAIN_PER_CLASS, 750) * len(CLASSES)
        actual_test = min(MAX_TEST_PER_CLASS, 250) * len(CLASSES)
        print(f"Capped for training: ~{actual_train} train / ~{actual_test} test "
              f"({MAX_TRAIN_PER_CLASS}/class train, {MAX_TEST_PER_CLASS}/class test)")
        print("(The full download still happens either way -- the cap only")
        print(" affects which images train_classifier.py actually loads.)")
    print("Next:  python src/train_classifier.py")


if __name__ == "__main__":
    main()
