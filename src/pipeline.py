"""End-to-end food detection → classification → nutrition pipeline.

    python src/pipeline.py path/to/food_photo.jpg

Runs YOLOv8 detection to locate food regions, classifies each crop with the
trained MobileNetV3-Large classifier, then fetches nutrition via USDA API.
Regions below 40% classifier confidence are reported as unrecognized.
Duplicate predicted labels (e.g. two samosa regions) are merged into one
line item with ×N multiplier applied to the nutrition totals.
"""
import argparse
import sys
from pathlib import Path

import torch

# src/ is this file's directory; add it so sibling modules import cleanly
# whether pipeline.py is run as a script or imported from app.py at the root.
sys.path.insert(0, str(Path(__file__).parent))

from detect import detect_food            # noqa: E402
from nutrition import lookup_nutrition    # noqa: E402
from predict import load_model, predict_pil  # noqa: E402

MODELS = Path("models")
CLASSIFIER_CONF_THRESHOLD = 0.40


def run_pipeline(image_path, model, classes, device, det_conf=0.1, top_k=3):
    """Detect → classify → merge duplicates → nutrition lookup.

    Args:
        image_path: str or Path — path to the food photo on disk.
        model:      loaded PyTorch MobileNetV3-Large classifier.
        classes:    list of class-name strings (from the checkpoint).
        device:     'cuda' or 'cpu'.
        det_conf:   YOLO detection confidence threshold (default 0.1).
        top_k:      number of classifier predictions to keep per region.

    Returns dict with keys:
        regions       — raw detect_food output (list of dicts with box/crop)
        merged        — recognised items, merged by label, with nutrition
        unrecognized  — regions below CLASSIFIER_CONF_THRESHOLD
    """
    regions = detect_food(str(image_path), conf_threshold=det_conf)
    if not regions:
        return {"regions": [], "merged": [], "unrecognized": []}

    recognized = []
    unrecognized = []
    for i, region in enumerate(regions):
        preds = predict_pil(model, classes, region["crop"], device, top_k=top_k)
        top_label, top_conf = preds[0]
        if top_conf < CLASSIFIER_CONF_THRESHOLD:
            unrecognized.append({
                "region_idx": i,
                "conf":       top_conf,
                "top_guess":  top_label,
            })
        else:
            recognized.append({
                "name": top_label,
                "conf": top_conf,
                "alts": preds[1:],
            })

    # Merge duplicate predicted labels; keep highest confidence seen
    seen = {}
    merged = []
    for r in recognized:
        name = r["name"]
        if name in seen:
            merged[seen[name]]["count"] += 1
            merged[seen[name]]["conf"] = max(merged[seen[name]]["conf"], r["conf"])
        else:
            seen[name] = len(merged)
            merged.append({"name": name, "conf": r["conf"],
                           "count": 1, "alts": r["alts"]})

    for entry in merged:
        entry["nutrition"] = lookup_nutrition(entry["name"])

    return {"regions": regions, "merged": merged, "unrecognized": unrecognized}


def main():
    ap = argparse.ArgumentParser(
        description="Detect → classify → nutrition on a food photo."
    )
    ap.add_argument("image", help="Path to a food photo")
    ap.add_argument("--checkpoint", default=str(MODELS / "classifier_best.pt"),
                    help="Classifier checkpoint (default models/classifier_best.pt)")
    ap.add_argument("--det-conf", type=float, default=0.1,
                    help="YOLO detection confidence threshold (default 0.1)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-k classifier predictions to show per region (default 3)")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        sys.exit(f"No classifier checkpoint at {checkpoint}. "
                 f"Run src/train_classifier.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading classifier ({device}) ...")
    model, classes = load_model(str(checkpoint), device)

    print(f"Detecting regions in {image_path.name} ...")
    result = run_pipeline(image_path, model, classes, device,
                          det_conf=args.det_conf, top_k=args.top_k)

    regions      = result["regions"]
    merged       = result["merged"]
    unrecognized = result["unrecognized"]

    if not regions:
        print("No food regions detected. Try --det-conf 0.05.")
        return

    print(f"Found {len(regions)} region(s).\n")

    W = 72
    print("=" * W)
    print(f"  {image_path.name}")
    print("=" * W)

    total_kcal = total_pro = total_carb = total_fat = 0.0
    for idx, entry in enumerate(merged, 1):
        n     = entry["nutrition"]
        count = entry["count"]
        label = entry["name"] + (f" ×{count}" if count > 1 else "")
        kcal  = (n["calories"] or 0) * count
        pro   = (n["protein"]  or 0) * count
        carb  = (n["carbs"]    or 0) * count
        fat   = (n["fat"]      or 0) * count
        total_kcal += kcal; total_pro += pro
        total_carb += carb; total_fat += fat

        src_tag = "" if n["source"] == "api" else f" [{n['source']}]"
        print(f"\n  {idx}. {label}")
        print(f"     Classifier: {entry['conf']*100:.1f}% confident", end="")
        if entry["alts"]:
            alt_str = ", ".join(f"{lbl} {p*100:.0f}%" for lbl, p in entry["alts"])
            print(f"  (runners-up: {alt_str})", end="")
        print()
        serving_str = (f"{n['serving_g']}g  ×{count} = {n['serving_g']*count}g"
                       if count > 1 else f"{n['serving_g']}g")
        print(f"     Serving:    {serving_str}")
        if n["calories"] is not None:
            print(f"     Nutrition:  {kcal:.1f} kcal | "
                  f"{pro:.1f}g protein | {carb:.1f}g carbs | {fat:.1f}g fat"
                  f"{src_tag}")
            print(f"     Matched:    {n['matched']}")
        else:
            print(f"     Nutrition:  no data{src_tag}")

    if unrecognized:
        print()
        pct = int(CLASSIFIER_CONF_THRESHOLD * 100)
        print(f"  Unrecognized — {len(unrecognized)} region(s) below {pct}% confidence:")
        for u in unrecognized:
            print(f"    Region {u['region_idx']}:  "
                  f"{u['conf']*100:.1f}% (top guess: \"{u['top_guess']}\")")

    total_items = sum(e["count"] for e in merged)
    print(f"\n{'─'*W}")
    print(f"  TOTAL ({total_items} item(s)):  "
          f"{total_kcal:.1f} kcal | {total_pro:.1f}g protein | "
          f"{total_carb:.1f}g carbs | {total_fat:.1f}g fat")
    print("─" * W)


if __name__ == "__main__":
    main()
