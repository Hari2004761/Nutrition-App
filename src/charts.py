"""Shared presentation helpers — palette, matplotlib figures, box drawing.

Extracted verbatim from the original Streamlit app.py so that both the
Streamlit front end (app.py) and the Flask back end (server.py) render the
exact same charts and annotated images from one implementation.

Nothing here knows about Streamlit or Flask: every function returns a plain
matplotlib Figure or PIL Image, and fig_to_png_bytes / fig_to_base64 turn
those into bytes for whichever transport the caller needs.
"""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from PIL import ImageDraw

# ── Palette (strictly limited per spec) ──────────────────────────────────────
BLUE   = "#2563EB"
NAVY   = "#0F172A"
SLATE  = "#64748B"
BORDER = "#E2E8F0"
BG     = "#F8FAFC"
GREEN  = "#16A34A"
AMBER  = "#D97706"
RED    = "#DC2626"

# Chart box-color cycle
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
def load_history(history_csv=HISTORY_CSV):
    df = pd.read_csv(history_csv)
    runs, current = [], []
    for _, row in df.iterrows():
        if row["phase"] == "smoke":
            if current:
                runs.append(pd.DataFrame(current).reset_index(drop=True))
                current = []
            continue
        if (row["phase"] == "head" and int(row["epoch"]) == 1
                and current and current[-1]["phase"] == "full"):
            runs.append(pd.DataFrame(current).reset_index(drop=True))
            current = []
        current.append(row.to_dict())
    if current:
        runs.append(pd.DataFrame(current).reset_index(drop=True))
    labels = [
        "15-class capped (200 imgs/class — 84.7% val acc)",
        "15-class full-data (750 imgs/class — 89.5% val acc)",
        "20-class full-data (750 imgs/class — 90.7% val acc)",
    ]
    return list(zip(labels[: len(runs)], runs))


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

    `rows` is exactly what evaluate.per_class_report() returns in its
    "classes" key -- {name, accuracy, confused_with, confused_count},
    already sorted worst-to-best. Nothing is recomputed here; the numbers
    come straight from the evaluation.

    Bars below the overall accuracy are amber, at-or-above are blue, and a
    dashed line marks the overall figure.
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
        # bars near the overall line would otherwise have the dashed line
        # running straight through their value label
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


# ── Transport helpers (used by the Flask back end) ───────────────────────────
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
