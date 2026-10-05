"""Saving analysed meals to Supabase, as the logged-in user.

The browser's nutrition figures are never trusted. /api/analyze hands out a
signed snapshot of the per-100g figures behind the result it showed; a save
must return that snapshot, and every item is recomputed from it. The meal is
then written by the save_meal() Postgres function through Supabase's REST API
with the user's own access token, so RLS applies. Lives outside src/ because
it depends on the Flask app and request.
"""
import math
import sys

import requests
from flask import current_app, g
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from auth import HTTP_TIMEOUT_S, SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL
from nutrition import COUNTABLE_ITEM_G, SERVING_SIZES
from pipeline import CLASSIFIER_CONF_THRESHOLD

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")   # = meals_meal_type_check
# The UI's own limits; anything beyond them didn't come from the page.
MAX_ITEMS = 30
MAX_QUANTITY = 50
MAX_GRAMS = 3000
SNAPSHOT_MAX_AGE_S = 24 * 3600


class MealError(Exception):
    """A failure already phrased for the user, with the HTTP status to send."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ── Signed nutrition snapshot ────────────────────────────────────────────────
def _signer():
    return URLSafeTimedSerializer(current_app.secret_key, salt="foodlens-nutrition")


def nutrition_token(recognized):
    """Signed per-100g figures for the items of one /api/analyze result.

    Derived from the response exactly as static/app.js's per100() derives them,
    so a saved meal is computed from the very numbers the user was shown.
    None without a SECRET_KEY to sign with.
    """
    if not current_app.secret_key:
        return None
    snapshot = {}
    for item in recognized:
        serving = item.get("serving_g")
        if item.get("has_data") is False or not serving:
            continue
        k = 100 / serving
        kcal = item.get("calories_per_100g")
        snapshot[item["name"]] = {
            "kcal":    kcal if kcal is not None else (item.get("calories") or 0) * k,
            "protein": (item.get("protein") or 0) * k,
            "carbs":   (item.get("carbs") or 0) * k,
            "fat":     (item.get("fat") or 0) * k,
            "source":  ("fallback" if item.get("source") == "fallback"
                        else item.get("matched") or "unknown"),
        }
    return _signer().dumps(snapshot)


def _read_snapshot(token):
    if not isinstance(token, str) or not token or not current_app.secret_key:
        raise MealError("This analysis has no nutrition data to save. "
                        "Analyse the photo again.")
    try:
        return _signer().loads(token, max_age=SNAPSHOT_MAX_AGE_S)
    except SignatureExpired:
        raise MealError("This analysis is more than a day old. "
                        "Analyse the photo again to save it.")
    except BadSignature:
        raise MealError("The nutrition data for this analysis couldn't be "
                        "verified. Analyse the photo again.")


# ── Validation ───────────────────────────────────────────────────────────────
def _number(value, label):
    # bool is an int subclass in Python, and json accepts NaN/Infinity.
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise MealError(f"{label} must be a number.")
    return value


def build_items(payload):
    """Validated meal type and items, with nutrition recomputed per item."""
    if not isinstance(payload, dict):
        raise MealError("Expected a JSON object.")

    meal_type = payload.get("meal_type")
    if meal_type not in MEAL_TYPES:
        raise MealError("meal_type must be one of: " + ", ".join(MEAL_TYPES) + ".")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise MealError("A meal needs at least one recognised item.")
    if len(raw_items) > MAX_ITEMS:
        raise MealError(f"A meal can have at most {MAX_ITEMS} items.")

    snapshot = _read_snapshot(payload.get("nutrition_token"))
    items = []
    for n, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            raise MealError(f"Item {n} must be an object.")
        food_class = raw.get("food_class")
        if not isinstance(food_class, str) or food_class not in SERVING_SIZES:
            raise MealError(f"Item {n}: unknown food class {food_class!r}.")
        row = snapshot.get(food_class)
        if row is None:
            raise MealError(f"Item {n}: {food_class!r} isn't part of this analysis.")
        label = f"Item {n} ({food_class})"

        quantity = _number(raw.get("quantity"), f"{label}: quantity")
        if quantity != int(quantity) or not 1 <= quantity <= MAX_QUANTITY:
            raise MealError(f"{label}: quantity must be a whole number from 1 to {MAX_QUANTITY}.")
        quantity = int(quantity)
        if quantity > 1 and food_class not in COUNTABLE_ITEM_G:
            raise MealError(f"{label}: quantity must be 1 for a dish that isn't counted in pieces.")

        grams = _number(raw.get("grams_per_item"), f"{label}: grams_per_item")
        if not 0 < grams <= MAX_GRAMS:
            raise MealError(f"{label}: grams_per_item must be above 0 and at most {MAX_GRAMS}.")

        confidence = _number(raw.get("confidence"), f"{label}: confidence")
        # Below the threshold the item was unrecognised, and those aren't saved.
        if not CLASSIFIER_CONF_THRESHOLD <= confidence <= 1:
            raise MealError(f"{label}: confidence must be between "
                            f"{CLASSIFIER_CONF_THRESHOLD} and 1.")

        factor = grams * quantity / 100
        kcal, protein, carbs, fat = (round(row[k] * factor, 1)
                                     for k in ("kcal", "protein", "carbs", "fat"))
        items.append({
            "food_class": food_class, "quantity": quantity,
            "grams_per_item": round(grams, 1), "confidence": round(confidence, 4),
            "kcal": kcal, "protein_g": protein, "carbs_g": carbs, "fat_g": fat,
            # The USDA record description the figures came from, or "fallback".
            "nutrition_source": row["source"],
        })
    return meal_type, items


# ── Supabase REST, as the user ───────────────────────────────────────────────
def _rest_error(res):
    """Map a PostgREST error onto something worth showing a user."""
    try:
        err = res.json()
    except ValueError:
        err = {}
    code = err.get("code") or ""
    print(f"[foodlens] save_meal failed: HTTP {res.status_code} {code} "
          f"{err.get('message')} {err.get('details') or ''}", file=sys.stderr, flush=True)
    if code == "PGRST202" or res.status_code == 404:
        return MealError("Saving meals isn't set up yet: the save_meal function "
                         "is missing in Supabase.", 503)
    if res.status_code == 401 or code.startswith("PGRST30"):
        return MealError("Your session has expired. Please log in again.", 401)
    if code == "42501" or res.status_code == 403:
        return MealError("You're not allowed to save this meal.", 403)
    if code in ("23514", "22023", "23502"):
        return MealError("The meal was rejected by the database: "
                         f"{err.get('message') or 'invalid values'}.", 400)
    return MealError("Could not save the meal. Try again shortly.", 502)


def save_meal(payload):
    """Validate, recompute and store one meal; returns the saved meal and items."""
    meal_type, items = build_items(payload)
    try:
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/save_meal",
            json={"p_meal_type": meal_type, "p_items": items},
            headers={"apikey": SUPABASE_PUBLISHABLE_KEY,
                     "Authorization": f"Bearer {g.access_token}"},
            timeout=HTTP_TIMEOUT_S,
        )
    except requests.RequestException:
        raise MealError("Could not reach the database. Try again shortly.", 503)
    if not res.ok:
        raise _rest_error(res)
    return {"meal": res.json(), "items": items}
