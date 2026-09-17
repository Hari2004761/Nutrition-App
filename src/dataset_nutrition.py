"""Nutrition for all 20 dataset classes, fetched once and cached.

Dataset-level analysis, independent of any uploaded photo. Every value comes
from nutrition.lookup_nutrition(), the same function the photo pipeline uses,
so the two can never disagree. The twenty USDA calls are slow and rate-limited,
so the result is cached in outputs/dataset_nutrition.json until refreshed.

    python src/dataset_nutrition.py             # build if missing, then print
    python src/dataset_nutrition.py --refresh   # re-fetch from USDA
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nutrition import SERVING_SIZES, lookup_nutrition   # noqa: E402

CACHE_PATH = Path("outputs/dataset_nutrition.json")

# Keyed by the 20 trained classes; sorted() only fixes the order so the cache is stable.
CLASS_NAMES = sorted(SERVING_SIZES)


def _per_100g(value, serving_g):
    """Scale a per-serving gram figure back down to a per-100g figure."""
    if value is None or not serving_g:
        return None
    return round(value / serving_g * 100.0, 1)


def _macro_percentages(protein, carbs, fat):
    """Share of energy from each macro, using 4/4/9 kcal per gram.

    Percentages rather than grams: a 60g samosa and a 250g curry cannot be
    compared by weight, but their composition can. Derived from the macros
    themselves, so the three shares always add up to 100.
    """
    if protein is None or carbs is None or fat is None:
        return None
    kcal = {"protein": protein * 4.0, "carbs": carbs * 4.0, "fat": fat * 9.0}
    total = sum(kcal.values())
    if total <= 0:
        return None
    return {k: round(v / total * 100.0, 1) for k, v in kcal.items()}


def _build_row(name):
    n = lookup_nutrition(name)
    serving_g = n.get("serving_g")
    row = {
        "name":       name,
        "label":      name.replace("_", " "),
        "serving_g":  serving_g,
        "matched":    n.get("matched"),
        "source":     n.get("source"),
        "calories":   n.get("calories"),
        "protein":    n.get("protein"),
        "carbs":      n.get("carbs"),
        "fat":        n.get("fat"),
        "calories_per_100g": n.get("calories_per_100g"),
        "protein_per_100g":  _per_100g(n.get("protein"), serving_g),
        "carbs_per_100g":    _per_100g(n.get("carbs"),   serving_g),
        "fat_per_100g":      _per_100g(n.get("fat"),     serving_g),
    }
    row["macro_pct"] = _macro_percentages(row["protein"], row["carbs"], row["fat"])
    return row


def build(classes=None, verbose=True):
    """Look every class up and return the report dict (does not write it)."""
    names = list(classes or CLASS_NAMES)
    rows = []
    for i, name in enumerate(names, 1):
        if verbose:
            print(f"[{i:2}/{len(names)}] {name} ...", flush=True)
        rows.append(_build_row(name))

    from_api = sum(1 for r in rows if r["source"] == "api")
    return {
        "generated":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_classes":   len(rows),
        "from_api":    from_api,
        "from_fallback": len(rows) - from_api,
        "classes":     rows,
    }


def _read_cache(path=CACHE_PATH, classes=None):
    """Return the cached report, or None if it is missing or unusable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = data.get("classes") if isinstance(data, dict) else None
    if not rows:
        return None
    # A cache built for a different class list is not reusable.
    wanted = set(classes or CLASS_NAMES)
    if {r.get("name") for r in rows} != wanted:
        return None
    return data


def load(refresh=False, classes=None, path=CACHE_PATH, verbose=False):
    """Cached nutrition for every class; fetches from USDA only when needed."""
    path = Path(path)
    if not refresh:
        cached = _read_cache(path, classes)
        if cached:
            return cached

    data = build(classes, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main():
    refresh = "--refresh" in sys.argv or "-r" in sys.argv
    data = load(refresh=refresh, verbose=True)

    rows = sorted(data["classes"],
                  key=lambda r: r["calories_per_100g"] or 0, reverse=True)
    print(f"\n{'class':22} {'kcal/100g':>10} {'P%':>6} {'C%':>6} {'F%':>6}  source")
    print("-" * 66)
    for r in rows:
        m = r["macro_pct"] or {}
        print(f"{r['label']:22} {r['calories_per_100g'] or 0:10.1f} "
              f"{m.get('protein', 0):6.1f} {m.get('carbs', 0):6.1f} "
              f"{m.get('fat', 0):6.1f}  {r['source']}")
    print(f"\n{data['n_classes']} classes — {data['from_api']} from the USDA API, "
          f"{data['from_fallback']} from the fallback table.")
    print(f"Cached in {CACHE_PATH} (re-run with --refresh to update).")


if __name__ == "__main__":
    main()
