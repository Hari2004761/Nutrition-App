"""Nutrition lookup for the food detection pipeline.

Primary source: USDA FoodData Central API (per-100g values scaled to a
realistic single-serving portion for each known class).
Fallback: hard-coded approximate values for the 20 known classes.

    python src/nutrition.py "pizza"
    python src/nutrition.py "chicken curry"

Requires USDA_API_KEY in src/.env.
Get a free key at: https://fdc.nal.usda.gov/api-key-signup.html
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from settings import get_setting

_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
_KEY_NAME = "USDA_API_KEY"

# Realistic single-serving portions in grams; USDA reports per 100g, so these
# scale it to what one person actually eats in a sitting.
SERVING_SIZES = {
    "pizza":               107,  # 1 slice (~1/8 of a 14" pizza)
    "sushi":               160,  # 6-piece order
    "ice_cream":            66,  # 1/2 cup scoop
    "hamburger":           150,  # 1 standard burger (patty + bun)
    "donuts":               60,  # 1 doughnut
    "french_fries":        117,  # small fast-food serving
    "onion_rings":          90,  # side order (~6 rings)
    "caesar_salad":        150,  # side salad
    "omelette":            120,  # 2-egg omelette
    "chicken_curry":       250,  # 1 cup curry (no rice)
    "chicken_wings":       180,  # 4 bone-in wings
    "tacos":               170,  # 2 standard tacos
    "samosa":               60,  # 1 samosa
    "pancakes":            150,  # 2 medium pancakes
    "waffles":             130,  # 1 standard waffle
    "fried_rice":          205,  # 1 cup cooked
    "paella":              250,  # restaurant portion
    "falafel":              90,  # 3 falafel balls
    "macaroni_and_cheese": 200,  # 1 cup
    "cheesecake":          125,  # 1 slice
}

# Classes eaten in countable pieces: grams for ONE piece, which the UI multiplies
# by a user-set quantity instead of trusting the detector's region count. Values
# are the household portion on the USDA record lookup_nutrition() matches; where
# USDA lists several sizes, the one that matches what Food-101's photos show.
COUNTABLE_ITEM_G = {
    "chicken_wings": 55,   # USDA "1 wing, any size"
    "samosa":       100,   # USDA "1 regular/large"
    "donuts":        75,   # USDA "1 doughnut"
    "tacos":        105,   # USDA "1 small/regular"
    "pancakes":      50,   # USDA "1 medium pancake"
    "waffles":      135,   # USDA "1 thick / Belgian waffle"
    "onion_rings":   10,   # USDA "1 ring"
    "sushi":         30,   # USDA "1 piece"
    "pizza":        119,   # USDA "1 piece, NFS"
    "falafel":       17,   # USDA "1 patty"
}

# Approximate per-serving values, used when the API is unreachable, the key is
# missing, or no usable match comes back.
_FALLBACK = {
    "pizza":               {"calories": 285, "protein": 12.0, "carbs": 36.0, "fat": 10.0},
    "sushi":               {"calories": 250, "protein": 12.0, "carbs": 40.0, "fat":  5.0},
    "ice_cream":           {"calories": 137, "protein":  2.0, "carbs": 17.0, "fat":  7.0},
    "hamburger":           {"calories": 350, "protein": 20.0, "carbs": 29.0, "fat": 15.0},
    "donuts":              {"calories": 250, "protein":  3.0, "carbs": 30.0, "fat": 14.0},
    "french_fries":        {"calories": 365, "protein":  4.0, "carbs": 48.0, "fat": 17.0},
    "onion_rings":         {"calories": 280, "protein":  4.0, "carbs": 33.0, "fat": 15.0},
    "caesar_salad":        {"calories": 190, "protein":  7.0, "carbs":  8.0, "fat": 15.0},
    "omelette":            {"calories": 180, "protein": 13.0, "carbs":  1.0, "fat": 14.0},
    "chicken_curry":       {"calories": 300, "protein": 25.0, "carbs": 15.0, "fat": 15.0},
    "chicken_wings":       {"calories": 430, "protein": 33.0, "carbs":  0.0, "fat": 32.0},
    "tacos":               {"calories": 370, "protein": 20.0, "carbs": 38.0, "fat": 15.0},
    "samosa":              {"calories": 150, "protein":  4.0, "carbs": 20.0, "fat":  7.0},
    "pancakes":            {"calories": 350, "protein":  8.0, "carbs": 58.0, "fat": 10.0},
    "waffles":             {"calories": 310, "protein":  8.0, "carbs": 45.0, "fat": 12.0},
    "fried_rice":          {"calories": 330, "protein":  9.0, "carbs": 55.0, "fat":  9.0},
    "paella":              {"calories": 390, "protein": 22.0, "carbs": 48.0, "fat": 12.0},
    "falafel":             {"calories": 250, "protein":  9.0, "carbs": 28.0, "fat": 13.0},
    "macaroni_and_cheese": {"calories": 390, "protein": 14.0, "carbs": 56.0, "fat": 12.0},
    "cheesecake":          {"calories": 400, "protein":  6.0, "carbs": 38.0, "fat": 25.0},
}

# Common alternate spellings → canonical class key
_ALIASES = {
    "doughnut":              "donuts",
    "doughnuts":             "donuts",
    "donut":                 "donuts",
    "burger":                "hamburger",
    "fries":                 "french_fries",
    "french_fry":            "french_fries",
    "wings":                 "chicken_wings",
    "curry":                 "chicken_curry",
    "waffle":                "waffles",
    "pancake":               "pancakes",
    "mac_and_cheese":        "macaroni_and_cheese",
    "mac_n_cheese":          "macaroni_and_cheese",
    "mac_cheese":            "macaroni_and_cheese",
    "macaroni_cheese":       "macaroni_and_cheese",
    "caesar":                "caesar_salad",
    "ice_creams":            "ice_cream",
    "taco":                  "tacos",
    "samosas":               "samosa",
    "falafel_balls":         "falafel",
    "omelettes":             "omelette",
    "hamburgers":            "hamburger",
    "waffles_":              "waffles",
}


def _canonical_key(food_name: str) -> str:
    """Normalize a food name to a canonical class key."""
    normalized = food_name.lower().strip().replace(" ", "_").replace("-", "_")
    if normalized in SERVING_SIZES:
        return normalized
    return _ALIASES.get(normalized, normalized)


def _read_api_key() -> str | None:
    return get_setting(_KEY_NAME)


def _extract_nutrients(food_item: dict) -> dict | None:
    """Per-100g kcal/protein/carbs/fat from a USDA result, or None if incomplete."""
    wanted = {
        "calories": (("energy",), ("kj",)),          # match name, exclude if unit kJ
        "protein":  (("protein",), ()),
        "carbs":    (("carbohydrate",), ()),
        "fat":      (("total lipid", "total fat"), ()),
    }
    found = {}
    for n in food_item.get("foodNutrients", []):
        name_l = n.get("nutrientName", "").lower()
        unit_l = n.get("unitName", "").lower()
        value  = n.get("value")
        if value is None:
            continue
        for key, (keywords, excludes) in wanted.items():
            if key in found:
                continue
            if any(k in name_l for k in keywords) and not any(e in name_l or e in unit_l for e in excludes):
                found[key] = float(value)
    return found if len(found) == 4 else None


_NON_DISH_WORDS = {
    "syrup", "sauce", "extract", "powder", "mix", "batter", "coating",
    "flavored", "flavoring", "seasoning", "seasoned", "dressing", "spread",
    "concentrate", "instant", "dry", "dried", "frozen",
}

# Restaurant-chain records lead with the brand in capitals ("TACO BELL, Original
# Taco", "PIZZA HUT 12\" Cheese Pizza", "McDONALD'S, Hamburger") or bracket it
# ("Hamburger (McDonalds)"); _score_match penalises them so generic entries win.
_CHAIN_RE = re.compile(
    r"^(?:Mc)?[A-Z][A-Z'&.-]+\b"
    r"|\((?:[A-Z][\w'&.-]*\s*)+\)"
)


def _score_match(description: str, query: str) -> int:
    """Higher = a better match. Prefers plain generic entries ('Pizza' over
    'Dessert pizza') and returns 0 for things that merely share a keyword."""
    d = description.lower().strip()
    q = query.lower().strip()

    # Things that merely share a keyword with the dish (syrup, sauce, mix…).
    # Punctuation is stripped per word so "sauce," cannot slip through.
    d_words = {w.strip(",.;:()") for w in d.split()}
    if _NON_DISH_WORDS & d_words:
        return 0

    # Chain records are a last resort, not disqualified: penalised, floored at 1.
    is_chain = bool(_CHAIN_RE.search(description))

    q_base = q.rstrip("s")      # rough singular: "pancakes" -> "pancake"
    d_base = d.rstrip("s")
    if d_base == q_base:
        score = 5
    elif d_base.startswith(q_base + ","):
        score = 4                # "Pizza, cheese" style
    else:
        first_word = d.split()[0].rstrip("s") if d else ""
        if first_word == q_base.split()[0]:
            score = 3
        elif q_base in d:
            score = 2
        else:
            score = 1

    return max(1, score - 2) if is_chain else score


_DATA_TYPES = ["SR Legacy", "Survey (FNDDS)"]


def _usda_post(body: dict, api_key: str) -> list[dict]:
    """POST one search; a 5xx is retried once, anything else raises."""
    url = f"{_API_URL}?{urllib.parse.urlencode({'api_key': api_key})}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read()).get("foods", [])
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt:
                raise
            time.sleep(1)
    return []


def _usda_query(query: str, api_key: str) -> list[dict]:
    """Search SR Legacy + Survey (FNDDS) only; return the food items.

    Sent as a POST with dataType as a JSON list: the GET form with a repeated
    dataType parameter 400s intermittently, and its unfiltered retry let
    Branded records win on some runs but not others. If the filtered call
    still 400s, the unfiltered retry keeps only those two data types.
    """
    try:
        return _usda_post({"query": query, "pageSize": 10,
                           "dataType": _DATA_TYPES}, api_key)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
    foods = _usda_post({"query": query, "pageSize": 50}, api_key)
    return [f for f in foods if f.get("dataType") in _DATA_TYPES]


def lookup_nutrition(food_name: str) -> dict:
    """Nutrition for one serving of the named food.

    Always returns the same keys (calories/protein/carbs/fat per serving, plus
    serving_g and calories_per_100g); 'source' says whether the numbers came
    from the USDA API, the fallback table, or nowhere at all.
    """
    key = _canonical_key(food_name)
    serving_g = SERVING_SIZES.get(key, 150)  # 150g default for unknown foods

    api_key = _read_api_key()
    if api_key:
        try:
            api_query = food_name.replace("_", " ")
            # Override query for foods where the plain name returns poor matches
            _QUERY_OVERRIDES = {
                "pizza":        "Pizza, cheese, from restaurant or fast food, NS as to type of crust",
                "hamburger":    "Hamburger, NFS",
                "ice_cream":    "Ice cream, vanilla",
                "chicken_wings": "Chicken wing, fried, coated, from restaurant",
                "donuts":       "doughnut, NFS",
                "french_fries": "potato, french fries, NFS",
                "pancakes":     "pancakes, plain",
                "waffles":      "waffle, NFS",
                "cheesecake":   "cheesecake, plain",
                "tacos":        "taco, beef, NFS",
                "onion_rings":  "Fried onion rings",
                "paella":       "paella seafood rice",
                "omelette":     "egg omelet plain",
            }
            # For a few classes the only FNDDS generic is systematically wrong (the
            # caesar salad entry has no dressing), so skip the API and use the table.
            _PREFER_FALLBACK = {"caesar_salad"}
            api_query = _QUERY_OVERRIDES.get(key, api_query)
            if key not in _PREFER_FALLBACK:
                foods = _usda_query(api_query, api_key)
            else:
                foods = []
            foods.sort(
                key=lambda f: _score_match(f.get("description", ""), api_query),
                reverse=True,
            )
            for food in foods:
                nutrients = _extract_nutrients(food)
                if nutrients:
                    scale = serving_g / 100.0
                    return {
                        "food":      food_name,
                        "matched":   food.get("description", food_name),
                        "serving_g": serving_g,
                        "calories":  round(nutrients["calories"] * scale, 1),
                        "protein":   round(nutrients["protein"]  * scale, 1),
                        "carbs":     round(nutrients["carbs"]    * scale, 1),
                        "fat":       round(nutrients["fat"]      * scale, 1),
                        # energy density, kept alongside the scaled figures
                        "calories_per_100g": round(nutrients["calories"], 1),
                        "source":    "api",
                    }
            print(f"[nutrition] API returned no usable match for '{food_name}'",
                  file=sys.stderr)
        except urllib.error.HTTPError as exc:
            print(f"[nutrition] API HTTP error {exc.code}: {exc.reason}", file=sys.stderr)
        except Exception as exc:
            print(f"[nutrition] API error: {exc}", file=sys.stderr)
    else:
        print(f"[nutrition] {_KEY_NAME} not set in src/.env — using fallback only",
              file=sys.stderr)

    if key in _FALLBACK:
        entry = _FALLBACK[key]
        # Already per serving, so the density has to be derived back out.
        return {
            "food":      food_name,
            "matched":   f"fallback ({key})",
            "serving_g": serving_g,
            **entry,
            "calories_per_100g": round(entry["calories"] / serving_g * 100, 1),
            "source":    "fallback",
        }

    return {
        "food":      food_name,
        "matched":   None,
        "serving_g": serving_g,
        "calories":  None,
        "protein":   None,
        "carbs":     None,
        "fat":       None,
        "calories_per_100g": None,
        "source":    "none",
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/nutrition.py \"food name\"")
        sys.exit(1)

    food = " ".join(sys.argv[1:])
    r = lookup_nutrition(food)

    print(f"\nFood:    {r['food']}")
    print(f"Matched: {r['matched']}")
    print(f"Source:  {r['source'].upper()}")
    print(f"Serving: {r['serving_g']}g")
    if r["source"] != "none":
        print(f"Calories: {r['calories']} kcal  ({r['calories_per_100g']} kcal/100g)")
        print(f"Protein:  {r['protein']}g")
        print(f"Carbs:    {r['carbs']}g")
        print(f"Fat:      {r['fat']}g")
    else:
        print("No nutrition data available.")


if __name__ == "__main__":
    main()
