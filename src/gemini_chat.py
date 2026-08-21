"""Gemini-backed dietary Q&A grounded in an analyzed meal.

    python src/gemini_chat.py "Is this meal diabetes-friendly?"

Answers are grounded in the meal the pipeline just analysed: the foods it
recognised and their calories/macros are handed to Gemini as context, so the
reply talks about *this* plate rather than food in general.

Requires gemini_api_key.txt in the project root (one line, just the key).
Free key at: https://aistudio.google.com/app/apikey
"""
import sys
from pathlib import Path

_KEY_FILE = Path("gemini_api_key.txt")
MODEL_NAME = "gemini-3.5-flash-lite"

# Scope boundary: general nutrition guidance, never clinical instructions.
SYSTEM_INSTRUCTION = """\
You are FoodLens, a friendly nutrition assistant built into a food-photo \
analysis app. The user has just photographed a meal; the app detected the \
dishes and looked up their calories and macronutrients. That analysis is \
given to you below as MEAL CONTEXT.

How to answer:
- Ground every answer in the MEAL CONTEXT. Refer to the actual dishes and \
their actual numbers rather than talking about food in the abstract.
- Be practical and specific: portion tweaks, swaps, additions, pairing and \
ordering suggestions.
- Keep it short — 120 words or fewer, plain sentences or a few short bullets. \
No markdown headings.
- If the question is about a condition such as diabetes, high blood pressure \
or cholesterol, give general dietary guidance only: carbohydrate load, fibre, \
glycaemic impact, portion size, what to pair it with.

Hard limits — you must follow these:
- You give general dietary suggestions, not medical advice, and you are not a \
substitute for a doctor or registered dietitian.
- Never give medication or treatment instructions. If asked about insulin \
doses, medication timing, blood-glucose targets, supplements as treatment, or \
anything else clinical, say plainly that you cannot advise on that and that \
it is a question for their doctor or diabetes educator — then offer the food \
side of the question instead.
- Do not diagnose, and do not tell anyone to start, stop or change a \
medication.
- If the user says they have a condition, take it at face value; do not ask \
for medical history.

Close with a brief reminder to check with a healthcare professional only when \
the question touched on a health condition, not on every reply.
"""

_NO_MEAL = "No meal has been analysed yet."


class GeminiError(RuntimeError):
    """Anything that stopped us from getting an answer back from Gemini."""


def read_api_key(key_file=_KEY_FILE):
    """Return the API key, or None when the file is missing or empty."""
    try:
        key = Path(key_file).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return key or None


def format_meal_context(recognized, totals, unrecognized=None):
    """Turn an /api/analyze result into a compact block of text for the prompt.

    `recognized` / `totals` are exactly the shapes /api/analyze returns, so the
    front end can hand its last analysis straight back to the chat endpoint.
    """
    if not recognized:
        return _NO_MEAL

    lines = []
    for item in recognized:
        name = str(item.get("name", "item")).replace("_", " ")
        count = int(item.get("count") or 1)
        if count > 1:
            name += f" x{count}"
        if item.get("calories") is None:
            lines.append(f"- {name}: no nutrition data available")
            continue
        serving = item.get("serving_g")
        portion = f", portion {serving}g each" if serving else ""
        density = item.get("calories_per_100g")
        density_txt = f", {density:.0f} kcal per 100g" if density else ""
        lines.append(
            f"- {name}: {item['calories']:.0f} kcal, "
            f"{item.get('protein', 0):.1f}g protein, "
            f"{item.get('carbs', 0):.1f}g carbs, "
            f"{item.get('fat', 0):.1f}g fat"
            f"{portion}{density_txt}"
        )

    t = totals or {}
    lines.append(
        f"Meal total: {t.get('calories', 0):.0f} kcal, "
        f"{t.get('protein', 0):.1f}g protein, "
        f"{t.get('carbs', 0):.1f}g carbs, "
        f"{t.get('fat', 0):.1f}g fat"
    )
    if unrecognized:
        lines.append(f"{len(unrecognized)} further item(s) on the plate could "
                     f"not be identified and are not counted above.")
    lines.append("Nutrition figures come from USDA FoodData Central for a "
                 "standard single serving, so treat them as estimates.")
    return "\n".join(lines)


def _history_to_contents(history, message, meal_context):
    """Build the Gemini `contents` list: prior turns, then this question."""
    contents = []
    for turn in history or []:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = str(turn.get("text") or "").strip()
        if text:
            contents.append({"role": role, "parts": [{"text": text[:4000]}]})
    contents.append({"role": "user", "parts": [{"text":
        f"MEAL CONTEXT\n{meal_context}\n\nQUESTION\n{message}"}]})
    return contents


def ask_gemini(message, recognized=None, totals=None, unrecognized=None,
               history=None, api_key=None, model_name=MODEL_NAME):
    """Ask Gemini one question about the analysed meal; return its reply text.

    Raises GeminiError with a message fit to show a user for every failure
    mode (missing key, bad key, no network, rate limit, empty response).
    """
    message = (message or "").strip()
    if not message:
        raise GeminiError("Please type a question first.")

    api_key = api_key or read_api_key()
    if not api_key:
        raise GeminiError(
            f"No Gemini API key found. Put your key in {_KEY_FILE} "
            f"(one line, nothing else) and try again."
        )

    try:
        import google.generativeai as genai
    except ImportError as exc:                       # pragma: no cover
        raise GeminiError("The google-generativeai package is not installed "
                          "(pip install google-generativeai).") from exc

    meal_context = format_meal_context(recognized, totals, unrecognized)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"temperature": 0.4, "max_output_tokens": 2048},
        )
        response = model.generate_content(
            _history_to_contents(history, message, meal_context),
            request_options={"timeout": 45},
        )
    except Exception as exc:
        raise GeminiError(_friendly_error(exc)) from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        # Safety filters and token limits both come back as an empty candidate.
        raise GeminiError("Gemini returned an empty response. "
                          "Try rephrasing the question.")
    return text


def _friendly_error(exc):
    """Map an SDK exception onto something worth showing a user."""
    name = type(exc).__name__
    detail = str(exc)
    low = detail.lower()

    if "api key" in low or "api_key" in low or "unauthenticated" in name.lower() \
            or "permissiondenied" in name.lower():
        return ("Gemini rejected the API key. Check the key in "
                f"{_KEY_FILE} is valid and has the Gemini API enabled.")
    if "resourceexhausted" in name.lower() or "quota" in low or "429" in detail:
        return "Gemini's rate limit was hit. Wait a moment and ask again."
    if "deadline" in low or "timeout" in low:
        return "Gemini took too long to respond. Try again."
    if "notfound" in name.lower() or "not found" in low:
        return (f"The model '{MODEL_NAME}' was not available to this API key. "
                f"Check the key's project has access to the Gemini API.")
    if any(w in low for w in ("network", "connection", "unavailable", "dns",
                             "getaddrinfo", "ssl")):
        return "Could not reach the Gemini API — check the network connection."
    return f"Gemini request failed ({name}): {detail[:200]}"


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/gemini_chat.py "your question"')
        sys.exit(1)

    # A stand-in meal so the CLI can be exercised without running the pipeline.
    demo_recognized = [{
        "name": "macaroni_and_cheese", "count": 1, "calories": 446.0,
        "protein": 17.4, "carbs": 46.2, "fat": 21.0,
        "serving_g": 200, "calories_per_100g": 223.0,
    }]
    demo_totals = {"calories": 446.0, "protein": 17.4, "carbs": 46.2, "fat": 21.0}

    try:
        print(ask_gemini(" ".join(sys.argv[1:]), demo_recognized, demo_totals))
    except GeminiError as exc:
        sys.exit(f"[gemini] {exc}")


if __name__ == "__main__":
    main()
