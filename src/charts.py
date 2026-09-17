"""Shared presentation helpers — palette, matplotlib figures, box drawing.

Used by both front ends (app.py and server.py) so they render identical charts
from one implementation. Nothing here imports Streamlit or Flask: every function
returns a plain matplotlib Figure or PIL Image.
"""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import pandas as pd
from PIL import ImageDraw

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
NAVY   = "#0F172A"
SLATE  = "#64748B"
BORDER = "#E2E8F0"
BG     = "#F8FAFC"
GREEN  = "#16A34A"
AMBER  = "#D97706"
RED    = "#DC2626"

BOX_COLORS = [GREEN, BLUE, AMBER, "#7C3AED", "#0EA5E9", RED]

HISTORY_CSV = Path("outputs/history.csv")


# ── Image annotation ──────────────────────────────────────────────────────────
def annotate_image(pil_image, regions, region_labels):
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    for i, (region, label) in enumerate(zip(regions, region_labels)):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = region["box"]
        for w in range(4):
            draw.rectangle([x1 + w, y1 + w, x2 - w, y2 - w], outline=color)
        text = label[:24]
        tbbox = draw.textbbox((x1 + 8, y1 + 6), text)
        pad = 5
        draw.rectangle(
            [tbbox[0] - pad, tbbox[1] - pad, tbbox[2] + pad, tbbox[3] + pad],
            fill=color,
        )
        draw.text((x1 + 8, y1 + 6), text, fill="white")
    return img


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _base_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.tick_params(colors=SLATE, labelsize=10)
    return fig, ax


def make_macro_pie(pro, carb, fat):
    total = pro + carb + fat
    if total < 0.1:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    wedges, texts, autotexts = ax.pie(
        [pro, carb, fat],
        labels=[f"Protein\n{pro:.1f}g", f"Carbs\n{carb:.1f}g", f"Fat\n{fat:.1f}g"],
        colors=[GREEN, AMBER, RED],
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 3, "width": 0.65},
    )
    for t in texts:
        t.set_fontsize(10); t.set_color(NAVY); t.set_fontweight("600")
    for at in autotexts:
        at.set_fontsize(11); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Macro split", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    plt.tight_layout(pad=1.0)
    return fig


def make_calorie_bar(merged):
    if not merged:
        return None
    names, kcals, colors = [], [], []
    for i, entry in enumerate(merged):
        n = entry["nutrition"]
        count = entry["count"]
        name = entry["name"].replace("_", " ").title()
        if count > 1:
            name += f" ×{count}"
        names.append(name)
        kcals.append((n["calories"] or 0) * count)
        colors.append(BOX_COLORS[i % len(BOX_COLORS)])

    fig, ax = _base_fig(5.2, max(2.4, len(names) * 0.78 + 1.0))
    bars = ax.barh(names, kcals, color=colors, height=0.48,
                   edgecolor="white", linewidth=0)
    max_k = max(kcals) if kcals else 1
    for bar, val in zip(bars, kcals):
        ax.text(bar.get_width() + max_k * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} kcal",
                va="center", ha="left", fontsize=10, color=NAVY, fontweight="600")
    ax.set_xlabel("kcal", fontsize=10, color=SLATE, labelpad=6)
    ax.set_title("Calories per item", fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max_k * 1.30 + 1)
    ax.xaxis.label.set_color(SLATE)
    plt.tight_layout(pad=1.0)
    return fig


# ── Training history helpers ──────────────────────────────────────────────────
# One label per run in outputs/history.csv, oldest first. Runs 1-3 validated on
# Food-101's test split, so their "val acc" is really a test score the checkpoint
# was also selected on; run 4 is the first with a held-out validation set.
RUN_LABELS = [
    "15-class capped (200/class — 84.7%, val = test split)",
    "15-class full-data (750/class — 89.5%, val = test split)",
    "20-class full-data (750/class — 90.7%, val = test split)",
    "20-class, held-out val (650/class — 85.0% val, 89.7% test)",
]


def load_history(history_csv=HISTORY_CSV):
    """Split history.csv into runs and label them, oldest first."""
    df = pd.read_csv(history_csv)
    runs, current = [], []
    for _, row in df.iterrows():
        if row["phase"] == "smoke":
            if current:
                runs.append(pd.DataFrame(current).reset_index(drop=True))
                current = []
            continue
        # A run always opens with head epoch 1, so that row starts a new one
        # whatever came before it — a finished run or one that was interrupted.
        if row["phase"] == "head" and int(row["epoch"]) == 1 and current:
            runs.append(pd.DataFrame(current).reset_index(drop=True))
            current = []
        current.append(row.to_dict())
    if current:
        runs.append(pd.DataFrame(current).reset_index(drop=True))

    # Any run past the end of RUN_LABELS still gets a name, so a new run shows
    # up in the picker instead of being silently dropped by a short label list.
    labels = list(RUN_LABELS[:len(runs)])
    for i in range(len(labels), len(runs)):
        best = runs[i]["val_acc"].max() * 100
        labels.append(f"run {i + 1} ({len(runs[i])} epochs — {best:.1f}% val acc)")
    return list(zip(labels, runs))


def training_plot(df, y_train, y_val, title, ylabel, pct=False):
    fig, ax = _base_fig(5.6, 3.8)
    n_head = int((df["phase"] == "head").sum())
    xs = list(range(1, len(df) + 1))
    if n_head:
        ax.axvspan(0.5, n_head + 0.5, alpha=0.06, color=GREEN)
        ax.axvspan(n_head + 0.5, len(df) + 0.5, alpha=0.05, color=AMBER)
    scale = 100 if pct else 1
    ax.plot(xs, df[y_train] * scale, color=BLUE, lw=2.2,
            marker="o", markersize=4.5, label="Train")
    ax.plot(xs, df[y_val] * scale, color=SLATE, lw=2.2,
            marker="s", markersize=4.5, linestyle="--", label="Val")
    if n_head and n_head < len(df):
        ax.axvline(n_head + 0.5, color="#CBD5E1", lw=1, linestyle=":", alpha=0.9)
        ylo, yhi = ax.get_ylim()
        mid_y = ylo + (yhi - ylo) * 0.04
        ax.text(n_head / 2 + 0.5, mid_y, "head phase", fontsize=8, color=SLATE, ha="center")
        ax.text(n_head + (len(df) - n_head) / 2 + 0.5, mid_y, "full fine-tune",
                fontsize=8, color=SLATE, ha="center")
    ax.set_xlabel("Epoch", fontsize=10, color=SLATE)
    ax.set_ylabel(ylabel, fontsize=10, color=SLATE)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10, framealpha=0.7, edgecolor="#E2E8F0")
    ax.set_xlim(0.3, len(df) + 0.7)
    plt.tight_layout(pad=1.0)
    return fig


def make_per_class_accuracy_bar(rows, overall_acc):
    """Horizontal per-class accuracy bars, weakest class at the top.

    `rows` is evaluate.per_class_report()'s "classes" list, already sorted
    worst-first; nothing is recomputed here. Bars below the overall accuracy are
    amber, at or above it blue, and the dashed line marks the overall figure.
    """
    if not rows:
        return None

    names = [r["name"].replace("_", " ") for r in rows]
    accs = [r["accuracy"] * 100 for r in rows]
    overall_pct = overall_acc * 100
    colors = [AMBER if a < overall_pct else BLUE for a in accs]

    fig, ax = _base_fig(7.4, max(3.2, len(rows) * 0.34 + 1.5))
    bars = ax.barh(names, accs, color=colors, height=0.62,
                   edgecolor="white", linewidth=0)
    for bar, acc in zip(bars, accs):
        label = ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                        f"{acc:.1f}%", va="center", ha="left",
                        fontsize=9, color=NAVY, fontweight="600", zorder=4)
        # keeps the dashed overall line from cutting through the value label
        label.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.axvline(overall_pct, color=NAVY, lw=1.4, linestyle="--", alpha=0.85)
    ax.annotate(f"overall {overall_pct:.1f}%",
                xy=(overall_pct, 1.0), xycoords=("data", "axes fraction"),
                xytext=(5, -11), textcoords="offset points",
                fontsize=9, color=NAVY, fontweight="600")

    ax.invert_yaxis()                       # worst class at the top
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Test accuracy (%)", fontsize=10, color=SLATE, labelpad=6)
    ax.set_title("Per-class accuracy — worst to best",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout(pad=1.0)
    return fig


# ── Dataset-level nutrition charts ───────────────────────────────────────────
# These describe the 20 classes themselves; rows come from dataset_nutrition.py.
_MACRO_COLORS = {"protein": GREEN, "carbs": AMBER, "fat": RED}


def _place_labels(ax, xs, ys, labels, fontsize=8):
    """Label each scatter point, nudged to the first slot that stays clear.
    matplotlib has no label-placement pass, so each label tries a ring of candidate
    offsets and keeps the first that hits neither a marker nor a placed label; where
    every candidate collides, the least-overlapping one wins.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    taken = []
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        taken.append((px - 5, py - 5, px + 5, py + 5))       # the marker itself

    candidates = [(8, 4), (8, -9), (-8, 4), (-8, -9), (0, 10), (0, -14),
                  (16, 11), (-16, -15), (13, -18), (-13, 14)]

    def overlap(a, b):
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        return dx * dy if dx > 0 and dy > 0 else 0.0

    def padded(ann):
        bb = ann.get_window_extent(renderer)
        return (bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2)

    frame = ax.get_window_extent(renderer)
    bounds = (frame.x0, frame.y0, frame.x1, frame.y1)

    def spill(box):
        """How far a label pokes outside the plotting area, as an area."""
        inside = overlap(box, bounds)
        return (box[2] - box[0]) * (box[3] - box[1]) - inside

    for x, y, text in zip(xs, ys, labels):
        ann = ax.annotate(text, xy=(x, y), xytext=candidates[0],
                          textcoords="offset points", fontsize=fontsize,
                          color=NAVY, zorder=5)
        best, best_cost = candidates[0], None
        for cand in candidates:
            ann.set_ha("right" if cand[0] < 0 else "left")
            ann.xyann = cand
            box = padded(ann)
            cost = sum(overlap(box, t) for t in taken) + spill(box) * 2
            if cost == 0:
                best, best_cost = cand, 0.0
                break
            if best_cost is None or cost < best_cost:
                best, best_cost = cand, cost
        ann.set_ha("right" if best[0] < 0 else "left")
        ann.xyann = best
        taken.append(padded(ann))


def make_calorie_protein_scatter(rows):
    """Calories vs protein per 100g, one labelled point per class.

    Per 100g rather than per serving so portion size cannot distort the
    picture: this is about what each dish *is*, not how much of it is served.
    """
    pts = [r for r in rows
           if r.get("calories_per_100g") and r.get("protein_per_100g") is not None]
    if not pts:
        return None

    xs = [r["calories_per_100g"] for r in pts]
    ys = [r["protein_per_100g"] for r in pts]
    colors = []
    for r in pts:
        macro = r.get("macro_pct") or {}
        dominant = max(macro, key=macro.get) if macro else "carbs"
        colors.append(_MACRO_COLORS.get(dominant, BLUE))

    fig, ax = _base_fig(8.6, 5.6)
    ax.grid(True, color=BORDER, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.scatter(xs, ys, s=80, c=colors, edgecolor="white", linewidth=1.3, zorder=3)

    ax.set_xlim(0, max(xs) * 1.20)
    ax.set_ylim(0, max(ys) * 1.28)
    ax.set_xlabel("Calories per 100g (kcal)", fontsize=10, color=SLATE, labelpad=6)
    ax.set_ylabel("Protein per 100g (g)", fontsize=10, color=SLATE, labelpad=6)
    ax.set_title("Calories vs protein — all 20 classes",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=8,
                      markerfacecolor=_MACRO_COLORS[k], markeredgecolor="white",
                      label=f"mostly {k}")
               for k in ("protein", "carbs", "fat")
               if _MACRO_COLORS[k] in colors]
    legend = ax.legend(handles=handles, loc="upper left", frameon=True,
                       fontsize=9, labelcolor=NAVY, borderpad=0.6)
    legend.get_frame().set_edgecolor(BORDER)
    legend.get_frame().set_facecolor("white")

    _place_labels(ax, xs, ys, [r["label"] for r in pts])
    plt.tight_layout(pad=1.0)
    return fig


def make_macro_composition_bar(rows, sort_key="carbs"):
    """Protein / carbs / fat as a share of each class's calories.

    Percentages, not grams: a 60g samosa and a 250g curry are not comparable by
    weight, but their composition is. 4/4/9 kcal per gram, so every bar totals 100%.
    """
    pts = [r for r in rows if r.get("macro_pct")]
    if not pts:
        return None
    pts = sorted(pts, key=lambda r: r["macro_pct"].get(sort_key, 0))
    names = [r["label"] for r in pts]

    fig, ax = _base_fig(8.6, max(4.0, len(pts) * 0.36 + 1.8))
    left = [0.0] * len(pts)
    for key, title in (("protein", "Protein"), ("carbs", "Carbs"), ("fat", "Fat")):
        vals = [r["macro_pct"].get(key, 0.0) for r in pts]
        ax.barh(names, vals, left=left, height=0.64, label=title,
                color=_MACRO_COLORS[key], edgecolor="white", linewidth=0.8)
        for i, (val, start) in enumerate(zip(vals, left)):
            if val >= 9:                       # skip slivers with no room
                ax.text(start + val / 2, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold", zorder=4)
        left = [s + v for s, v in zip(left, vals)]

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Share of calories", fontsize=10, color=SLATE, labelpad=6)
    ax.set_title(f"Macro composition — sorted by {sort_key} share",
                 fontsize=13, fontweight="bold", color=NAVY, pad=26)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3,
                       frameon=False, fontsize=9, labelcolor=NAVY)
    for text in legend.get_texts():
        text.set_color(NAVY)
    plt.tight_layout(pad=1.0)
    return fig


def make_calorie_density_bar(rows):
    """Calories per 100g for every class, densest first.

    Amber marks classes above the average (dashed line), blue those at or below —
    the same convention as the per-class accuracy chart.
    """
    pts = [r for r in rows if r.get("calories_per_100g")]
    if not pts:
        return None
    pts = sorted(pts, key=lambda r: r["calories_per_100g"], reverse=True)
    names = [r["label"] for r in pts]
    vals = [r["calories_per_100g"] for r in pts]
    mean = sum(vals) / len(vals)
    colors = [AMBER if v > mean else BLUE for v in vals]

    fig, ax = _base_fig(8.6, max(3.4, len(pts) * 0.34 + 1.5))
    bars = ax.barh(names, vals, color=colors, height=0.62,
                   edgecolor="white", linewidth=0)
    span = max(vals)
    for bar, val in zip(bars, vals):
        label = ax.text(bar.get_width() + span * 0.015,
                        bar.get_y() + bar.get_height() / 2, f"{val:.0f}",
                        va="center", ha="left", fontsize=9, color=NAVY,
                        fontweight="600", zorder=4)
        # keeps the dashed average line from cutting through the value label
        label.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.axvline(mean, color=NAVY, lw=1.4, linestyle="--", alpha=0.85)
    ax.annotate(f"average {mean:.0f}", xy=(mean, 1.0),
                xycoords=("data", "axes fraction"), xytext=(5, -11),
                textcoords="offset points", fontsize=9, color=NAVY,
                fontweight="600")
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.12)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Calories per 100g (kcal)", fontsize=10, color=SLATE, labelpad=6)
    ax.set_title("Calorie density — most to least",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    plt.tight_layout(pad=1.0)
    return fig


def fig_to_png_bytes(fig, dpi=140):
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def pil_to_png_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def to_base64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")
