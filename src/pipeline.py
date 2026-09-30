"""End-to-end food detection → classification → nutrition pipeline.

    python src/pipeline.py path/to/food_photo.jpg

Runs YOLOv8 detection to locate food regions, classifies each crop with the
trained MobileNetV3-Large classifier, then fetches nutrition via USDA API.
Regions below 40% classifier confidence are reported as unrecognized.
Duplicate predicted labels (e.g. two samosa regions) are merged into one
line item; the region count is reported but never multiplies the nutrition.
"""
import argparse
import sys
from pathlib import Path

import torch

# So sibling imports work when run as a script or imported from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from detect import detect_food            # noqa: E402
from nutrition import COUNTABLE_ITEM_G, lookup_nutrition  # noqa: E402
from predict import load_model, predict_pil  # noqa: E402

MODELS = Path("models")
# The classifier always returns a guess, so anything under this is reported as
# unrecognized instead: better to miss an item than to name one confidently wrong.
CLASSIFIER_CONF_THRESHOLD = 0.40


def _scale_to_piece(n, piece_g):
    """Rescale a per-serving nutrition dict to one piece of piece_g grams."""
    if n["calories"] is None or not n["serving_g"]:
        return {**n, "serving_g": piece_g}
    k = piece_g / n["serving_g"]
    return {**n, "serving_g": piece_g,
            **{m: round(n[m] * k, 1) for m in ("calories", "protein", "carbs", "fat")}}


def run_pipeline(image_path, model, classes, device, det_conf=0.1, top_k=3):
    """Detect → classify → merge duplicates → nutrition lookup.

    Returns {regions, merged, unrecognized}; a region the classifier is less
    than CLASSIFIER_CONF_THRESHOLD sure of lands in unrecognized.
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

    # Merged on the classifier's label, not box overlap. The region count is
    # kept as information only: YOLO both splits one item into several boxes and
    # merges touching items into one, so it is no measure of how many were eaten.
    seen = {}
    merged = []
    for r in recognized:
        name = r["name"]
        if name in seen:
            merged[seen[name]]["regions"] += 1
            merged[seen[name]]["conf"] = max(merged[seen[name]]["conf"], r["conf"])
        else:
            seen[name] = len(merged)
            merged.append({"name": name, "conf": r["conf"],
                           "regions": 1, "alts": r["alts"]})

    # Countable classes start at one piece; the quantity is the user's to set.
    for entry in merged:
        n = lookup_nutrition(entry["name"])
        entry["countable"] = entry["name"] in COUNTABLE_ITEM_G
        if entry["countable"]:
            n = _scale_to_piece(n, COUNTABLE_ITEM_G[entry["name"]])
        entry["nutrition"] = n

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
        kcal  = n["calories"] or 0
        pro   = n["protein"]  or 0
        carb  = n["carbs"]    or 0
        fat   = n["fat"]      or 0
        total_kcal += kcal; total_pro += pro
        total_carb += carb; total_fat += fat

        src_tag = "" if n["source"] == "api" else f" [{n['source']}]"
        print(f"\n  {idx}. {entry['name']}  "
              f"(detected in {entry['regions']} region{'s' if entry['regions'] > 1 else ''})")
        print(f"     Classifier: {entry['conf']*100:.1f}% confident", end="")
        if entry["alts"]:
            alt_str = ", ".join(f"{lbl} {p*100:.0f}%" for lbl, p in entry["alts"])
            print(f"  (runners-up: {alt_str})", end="")
        print()
        serving_str = (f"{n['serving_g']}g (1 piece)" if entry["countable"]
                       else f"{n['serving_g']}g")
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

    total_items = len(merged)
    print(f"\n{'─'*W}")
    print(f"  TOTAL ({total_items} item(s)):  "
          f"{total_kcal:.1f} kcal | {total_pro:.1f}g protein | "
          f"{total_carb:.1f}g carbs | {total_fat:.1f}g fat")
    print("─" * W)


if __name__ == "__main__":
    main()
