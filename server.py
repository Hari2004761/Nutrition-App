"""FoodLens — Flask back end for the HTML/CSS/JS front end.

    python server.py            → http://127.0.0.1:5000

Presentation layer only: detection, classification, nutrition lookup and chart
drawing all come from src/. The classifier and the YOLO detector are loaded
once at start-up and shared by every request.
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from flask import (Flask, Response, g, jsonify, redirect, render_template,
                   request, send_file)
from PIL import Image

# Every module uses paths relative to the project root, so anchor the process
# there regardless of where python was invoked from.
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
# Before importing auth, which reads the Supabase settings at import.
load_dotenv(ROOT / "src" / ".env")

# Must stay below `import torch`: on this setup, cryptography (pulled in by
# PyJWT) loading before torch makes pandas/pyarrow crash the process.
from auth import (                                              # noqa: E402
    auth_bp, auth_configured, check_origin, current_user, guest_configured,
    guest_quota, has_access, is_guest, require_token, require_user, AuthError,
)

from detect import _get_yolo                                     # noqa: E402
from predict import load_model, predict_pil                     # noqa: E402
from pipeline import run_pipeline, CLASSIFIER_CONF_THRESHOLD    # noqa: E402
from charts import (                                            # noqa: E402
    annotate_image, make_macro_pie, make_calorie_bar,
    load_history, training_plot, make_per_class_accuracy_bar,
    make_calorie_protein_scatter, make_macro_composition_bar,
    make_calorie_density_bar,
    fig_to_png_bytes, pil_to_png_bytes, to_base64,
)
import dataset_nutrition                                       # noqa: E402
from evaluate import compute_confusion, per_class_report      # noqa: E402
from meals import MealError, nutrition_token, save_meal       # noqa: E402
from gemini_chat import (                                     # noqa: E402
    ask_gemini, read_api_key, GeminiError, KEY_NAME as GEMINI_KEY_NAME,
    MODEL_NAME as GEMINI_MODEL,
)

# ── Paths / constants ────────────────────────────────────────────────────────
CHECKPOINT    = Path("models/classifier_best.pt")
YOLO_WEIGHTS  = "yolov8m-oiv7.pt"   # detect_food()'s default, which the pipeline uses
HISTORY_CSV   = Path("outputs/history.csv")
CONFUSION_PNG = Path("outputs/figures/confusion_matrix.png")
PER_CLASS_CACHE = Path("outputs/per_class_accuracy.json")
MAX_CHAT_CHARS  = 2000
MAX_HISTORY_TURNS = 8
DAILY_GOAL    = 2000
MAX_UPLOAD_MB = 32

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# The reloader is off (it would load the model twice), so reload templates instead.
app.config["TEMPLATES_AUTO_RELOAD"] = True
# Signs the guest cookie; guest mode stays off without it.
app.secret_key = os.environ.get("SECRET_KEY") or None
# Order matters: refuse cross-site writes before anything else looks at them.
app.before_request(check_origin)        # CSRF: POST/PUT/PATCH/DELETE from this origin only
app.before_request(require_token)       # every /api/* call needs a login or a guest cookie
app.register_blueprint(auth_bp)

# ── Model, loaded once at start-up ───────────────────────────────────────────
MODEL = CLASSES = DEVICE = None


def init_model():
    """Load the classifier and the YOLO detector a single time for the process."""
    global MODEL, CLASSES, DEVICE
    if MODEL is not None:
        return
    if not CHECKPOINT.exists():
        raise SystemExit(f"No classifier checkpoint at {CHECKPOINT}. "
                         f"Run src/train_classifier.py first.")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[foodlens] loading classifier on {DEVICE} ...", flush=True)
    MODEL, CLASSES = load_model(str(CHECKPOINT), DEVICE)
    print(f"[foodlens] classifier ready — {len(CLASSES)} classes", flush=True)
    _get_yolo(YOLO_WEIGHTS)
    print(f"[foodlens] detector ready — {YOLO_WEIGHTS}", flush=True)


def wants_refresh():
    """?refresh=1 rebuilds an expensive cache; guests always get the cached copy."""
    return request.args.get("refresh") == "1" and not is_guest()


def png_response(png_bytes):
    return Response(png_bytes, mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


def _payload():
    """Read chart parameters from either a JSON POST body or a GET query."""
    if request.method == "POST":
        return request.get_json(silent=True) or {}
    return request.args


# ── Per-class accuracy, evaluated once and cached ────────────────────────────
# A full evaluation is every held-out test image (~5,000) — far too slow to redo
# per page load, so it is cached and only recomputed when the checkpoint changes
# (or ?refresh=1 is passed).
_eval_lock = threading.Lock()


def _cache_is_fresh(cached):
    if not isinstance(cached, dict) or "classes" not in cached:
        return False
    if not CHECKPOINT.exists():
        return True
    return abs(cached.get("checkpoint_mtime", 0)
               - CHECKPOINT.stat().st_mtime) < 1.0


def _read_cache():
    try:
        cached = json.loads(PER_CLASS_CACHE.read_text())
    except (OSError, ValueError):
        return None
    return cached if _cache_is_fresh(cached) else None


def per_class_data(refresh=False):
    """Per-class accuracy report, from cache when possible.

    The numbers come from evaluate.py; nothing is recalculated here, and the
    classifier already loaded at start-up is reused rather than loaded twice.
    """
    if not refresh:
        cached = _read_cache()
        if cached:
            return cached

    with _eval_lock:
        if not refresh:                     # another request may have just built it
            cached = _read_cache()
            if cached:
                return cached
        init_model()
        print("[foodlens] running full test-set evaluation "
              "(cached afterwards) ...", flush=True)
        confusion = compute_confusion(MODEL, CLASSES, DEVICE, num_workers=0)
        report = per_class_report(CLASSES, confusion)
        data = {
            "classes":          report["classes"],
            "overall_accuracy": report["overall_accuracy"],
            "n_test_images":    int(confusion.sum()),
            "checkpoint_mtime": CHECKPOINT.stat().st_mtime if CHECKPOINT.exists() else 0,
        }
        PER_CLASS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PER_CLASS_CACHE.write_text(json.dumps(data, indent=2))
        print(f"[foodlens] evaluation cached to {PER_CLASS_CACHE}", flush=True)
        return data


# ── Front end ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if not has_access():
        return redirect("/login")
    return render_template("index.html", daily_goal=DAILY_GOAL,
                           conf_threshold=CLASSIFIER_CONF_THRESHOLD,
                           auth_enabled=auth_configured())


@app.get("/login")
def login_page():
    try:
        if current_user():
            return redirect("/")
    except AuthError:
        pass                                # Supabase unreachable: show the form
    return render_template("login.html", auth_enabled=auth_configured(),
                           guest_enabled=guest_configured())


# ── GET /api/me — the verified user behind the session cookies ─────────────
@app.get("/api/me")
@require_user
def api_me():
    return jsonify({"id": g.user["sub"], "email": g.user.get("email")})


# ── POST /api/meals — store the analysed meal for the logged-in user ────────
@app.post("/api/meals")
@require_user
def api_save_meal():
    """Nutrition is recomputed from the signed snapshot /api/analyze returned,
    never taken from the browser; the write runs as the user, so RLS applies."""
    if not request.is_json:
        return jsonify({"error": "Expected a JSON body."}), 415
    try:
        saved = save_meal(request.get_json(silent=True))
    except MealError as exc:
        return jsonify({"error": str(exc)}), exc.status
    return jsonify(saved), 201


# ── POST /api/analyze — full pipeline on an uploaded photo ───────────────────
@app.post("/api/analyze")
@guest_quota
def api_analyze():
    upload = request.files.get("image") or request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "No image file in the request "
                                 "(expected form field 'image')."}), 400

    suffix = Path(upload.filename).suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        upload.save(tmp_path)
        try:
            pil_img = Image.open(tmp_path).convert("RGB")
        except Exception:
            return jsonify({"error": "That file could not be read as an image."}), 400

        result = run_pipeline(tmp_path, MODEL, CLASSES, DEVICE)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    regions      = result["regions"]
    merged       = result["merged"]
    unrecognized = result["unrecognized"]

    if not regions:
        return jsonify({
            "recognized": [], "unrecognized": [],
            "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0},
            "annotated_image": to_base64(pil_to_png_bytes(pil_img)),
            "region_count": 0,
            "message": "No food regions detected. Try a clearer photo or a closer crop.",
        })

    region_labels = []
    for region in regions:
        top_label, top_conf = predict_pil(MODEL, CLASSES, region["crop"], DEVICE, top_k=1)[0]
        region_labels.append(
            top_label.replace("_", " ") if top_conf >= CLASSIFIER_CONF_THRESHOLD else "?"
        )
    annotated = annotate_image(pil_img, regions, region_labels)

    recognized = []
    for entry in merged:
        n = entry["nutrition"]
        recognized.append({
            "name":       entry["name"],
            "confidence": entry["conf"],
            # information only; quantity is set by the user in the browser
            "regions":    entry["regions"],
            "countable":  entry["countable"],
            "calories":   n["calories"] or 0,
            "protein":    n["protein"]  or 0,
            "carbs":      n["carbs"]    or 0,
            "fat":        n["fat"]      or 0,
            "source":     n["source"],
            "matched":    n["matched"] or "",
            # grams for one piece when countable, else for the whole portion
            "serving_g":  n.get("serving_g"),
            "calories_per_100g": n.get("calories_per_100g"),
            "has_data":   n["calories"] is not None,
        })

    totals = {
        "calories": sum(i["calories"] for i in recognized),
        "protein":  sum(i["protein"]  for i in recognized),
        "carbs":    sum(i["carbs"]    for i in recognized),
        "fat":      sum(i["fat"]      for i in recognized),
    }

    return jsonify({
        "recognized": recognized,
        "unrecognized": [{"confidence": u["conf"],
                          "top_guess": u["top_guess"]} for u in unrecognized],
        "totals": totals,
        "annotated_image": to_base64(pil_to_png_bytes(annotated)),
        "region_count": len(regions),
        "daily_goal": DAILY_GOAL,
        # Signed per-100g figures behind this result; a save must send it back.
        "nutrition_token": nutrition_token(recognized),
    })


# ── Charts — the same matplotlib code the Streamlit app uses ─────────────────
@app.route("/api/chart/macro-pie", methods=["GET", "POST"])
def api_macro_pie():
    data = _payload()
    totals = data.get("totals") if isinstance(data, dict) else None
    src = totals if isinstance(totals, dict) else data

    def num(key):
        try:
            return float(src.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    fig = make_macro_pie(num("protein"), num("carbs"), num("fat"))
    if fig is None:
        return jsonify({"error": "No macronutrient data to plot."}), 404
    return png_response(fig_to_png_bytes(fig))


@app.route("/api/chart/calorie-bar", methods=["GET", "POST"])
def api_calorie_bar():
    data = _payload()
    items = data.get("recognized") if isinstance(data, dict) else None
    if items is None:
        return jsonify({"error": "Expected a 'recognized' list of items."}), 400

    # Per-item calories arrive already × quantity; quantity is only for the label.
    entries = []
    for item in items:
        entries.append({
            "name":     item.get("name", "item"),
            "quantity": int(item.get("quantity") or 1) or 1,
            "nutrition": {"calories": float(item.get("calories") or 0)},
        })

    fig = make_calorie_bar(entries)
    if fig is None:
        return jsonify({"error": "No items to plot."}), 404
    return png_response(fig_to_png_bytes(fig))


# ── Training history ─────────────────────────────────────────────────────────
@app.get("/api/training-runs")
def api_training_runs():
    if not HISTORY_CSV.exists():
        return jsonify({"error": f"{HISTORY_CSV} not found — run training first.",
                        "runs": []}), 404

    runs = []
    for idx, (label, df) in enumerate(load_history(HISTORY_CSV)):
        loss_fig = training_plot(df, "train_loss", "val_loss",
                                 "Cross-entropy loss", "Loss")
        acc_fig  = training_plot(df, "train_acc", "val_acc",
                                 "Accuracy", "Accuracy (%)", pct=True)
        runs.append({
            "id":         idx,
            "label":      label,
            "epochs":     int(len(df)),
            "final_val_acc": float(df["val_acc"].iloc[-1]) if "val_acc" in df else None,
            "loss_chart": to_base64(fig_to_png_bytes(loss_fig)),
            "acc_chart":  to_base64(fig_to_png_bytes(acc_fig)),
        })
    return jsonify({"runs": runs})


@app.get("/api/confusion-matrix")
def api_confusion_matrix():
    if not CONFUSION_PNG.exists():
        return jsonify({"error": f"Confusion matrix not found at {CONFUSION_PNG}."}), 404
    return send_file(CONFUSION_PNG.resolve(), mimetype="image/png")


@app.get("/api/per-class-accuracy")
def api_per_class_accuracy():
    """The per-class table as JSON (used for the chart caption)."""
    try:
        data = per_class_data(refresh=wants_refresh())
    except Exception as exc:
        return jsonify({"error": f"Evaluation failed: {exc}"}), 500
    return jsonify(data)


@app.get("/api/chart/per-class-accuracy")
def api_per_class_accuracy_chart():
    try:
        data = per_class_data(refresh=wants_refresh())
    except Exception as exc:
        return jsonify({"error": f"Evaluation failed: {exc}"}), 500

    fig = make_per_class_accuracy_bar(data["classes"], data["overall_accuracy"])
    if fig is None:
        return jsonify({"error": "No per-class data to plot."}), 404
    return png_response(fig_to_png_bytes(fig))


# ── Dataset analysis — nutrition across the 20 classes, independent of a photo ─
# Twenty USDA lookups take ~50s and eat into the rate limit, so the result is
# cached; ?refresh=1 on any of these endpoints re-fetches it.
_dataset_lock = threading.Lock()
_dataset_refreshed_at = 0.0
REFRESH_COALESCE_S = 15     # a page load fires four requests; rebuild once


def dataset_data(refresh=False):
    """Cached nutrition for every class, built by src/dataset_nutrition.py.

    One page view asks for the JSON plus three charts, so without coalescing a
    single Refresh click would be 80 live USDA calls instead of 20.
    """
    global _dataset_refreshed_at
    with _dataset_lock:                 # one build at a time, not one per request
        if refresh and time.time() - _dataset_refreshed_at < REFRESH_COALESCE_S:
            refresh = False             # another request rebuilt it a moment ago
        if refresh:
            print("[foodlens] refreshing dataset nutrition from USDA ...", flush=True)
        data = dataset_nutrition.load(refresh=refresh, verbose=refresh)
        if refresh:
            _dataset_refreshed_at = time.time()
        return data


def _dataset_chart(make_fig, **kwargs):
    """Shared plumbing for the three dataset charts."""
    try:
        data = dataset_data(refresh=wants_refresh())
    except Exception as exc:
        return jsonify({"error": f"Nutrition lookup failed: {exc}"}), 500
    fig = make_fig(data["classes"], **kwargs)
    if fig is None:
        return jsonify({"error": "No nutrition data to plot."}), 404
    return png_response(fig_to_png_bytes(fig))


@app.get("/api/dataset-nutrition")
def api_dataset_nutrition():
    """The per-class nutrition table as JSON (used for the chart captions)."""
    try:
        return jsonify(dataset_data(refresh=wants_refresh()))
    except Exception as exc:
        return jsonify({"error": f"Nutrition lookup failed: {exc}"}), 500


@app.get("/api/chart/dataset/calorie-protein")
def api_dataset_calorie_protein():
    return _dataset_chart(make_calorie_protein_scatter)


@app.get("/api/chart/dataset/macro-composition")
def api_dataset_macro_composition():
    sort = request.args.get("sort", "carbs")
    if sort not in ("protein", "carbs", "fat"):
        sort = "carbs"
    return _dataset_chart(make_macro_composition_bar, sort_key=sort)


@app.get("/api/chart/dataset/calorie-density")
def api_dataset_calorie_density():
    return _dataset_chart(make_calorie_density_bar)


# ── Gemini chat — dietary questions grounded in the meal just analysed ───────
@app.get("/api/chat/status")
@require_user
def api_chat_status():
    """Whether a Gemini key is configured, so the UI can say so up front."""
    return jsonify({
        "available": read_api_key() is not None,
        "key_name":  GEMINI_KEY_NAME,
        "model":     GEMINI_MODEL,
    })


@app.post("/api/chat")
@require_user
def api_chat():
    """One chat turn.

    The client sends the meal context with every message; the server keeps no
    session state, so a restart or a second tab never gets a borrowed meal.
    """
    data = request.get_json(silent=True) or {}
    message = str(data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message sent."}), 400
    if len(message) > MAX_CHAT_CHARS:
        return jsonify({"error": f"Message too long "
                                 f"(max {MAX_CHAT_CHARS} characters)."}), 400

    context = data.get("context") or {}
    if not isinstance(context, dict):
        context = {}
    history = data.get("history")
    history = history[-MAX_HISTORY_TURNS:] if isinstance(history, list) else []

    try:
        reply = ask_gemini(
            message,
            recognized=context.get("recognized"),
            totals=context.get("totals"),
            unrecognized=context.get("unrecognized"),
            history=history,
        )
    except GeminiError as exc:
        # Upstream failures arrive here already phrased for a user.
        return jsonify({"error": str(exc)}), 502

    return jsonify({"reply": reply, "model": GEMINI_MODEL})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"Image too large (max {MAX_UPLOAD_MB} MB)."}), 413


if __name__ == "__main__":
    init_model()
    # debug=False: the reloader would load the model twice.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
