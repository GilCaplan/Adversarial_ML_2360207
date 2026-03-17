"""
plot_unsteered_baseline.py
--------------------------
Generates ONE figure showing ONLY the unsteered baseline responses across
both experiment types, using tighter v2 keyword lists.

Hallucination baseline  → all 0% with tight keywords (blank image never
  produces concept words without steering). Reported as a single text note;
  not plotted since a flat-zero bar chart is uninformative.

Transfer baseline (the interesting data):
  For each source→target pair and model, we show:
    ■  Source recognition %  — how often the model correctly identifies the
                               SOURCE concept in the source image (no steering).
                               High = model understands the image correctly.
    □  Target false-positive % — how often the model mentions the TARGET
                                  concept in the source image (no steering).
                                  High = experiment is confounded even without
                                  any steering vector.

Output (same directory as this script):
    unsteered_baseline_figure.png
    unsteered_baseline_figure.pdf

Run from the project root:
    python baseline_analysis/plot_unsteered_baseline.py
"""

import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
_here    = os.path.dirname(os.path.abspath(__file__))
_project = os.path.dirname(_here)

RESULT_DIRS = {
    "emotions": os.path.join(_project, "json_results_emotions"),
    "weather":  os.path.join(_project, "json_results"),
}

# ── Tighter v2 keyword lists ─────────────────────────────────────────────────
EMOTION_KW = {
    "happy":   ["happy","smile","smiling","cheerful","laughing","laughter",
                "joyful","joy","grinning","grin","gleeful","elated",
                "celebrating","celebration","enjoying"],
    "sad":     ["sad","sadness","crying","cry","tears","tear","weeping",
                "sorrowful","sorrow","melancholy","gloomy","grief","unhappy",
                "downcast","despondent","desolate","lonely"],
    "angry":   ["angry","anger","mad","rage","furious","fury","frown","frowning",
                "shouting","shout","aggressive","aggression","fierce","hostile",
                "clashing","chaos","flaming","fiery","stormy","jagged"],
    "neutral": ["neutral expression","neutral face","expressionless",
                "blank expression","no expression","no emotion",
                "flat expression","impassive"],
}
WEATHER_KW = {
    "rain":    ["rain","raining","rainfall","rainy","drizzle","pouring",
                "thunderstorm","storm","shower","raindrops","puddle","soaked","wet","splash"],
    "shine":   ["sunshine","sun","sunny","sunlight","sunbeam","solar",
                "blue sky","rays","beam","glare","radiant","summer"],
    "cloudy":  ["cloud","clouds","cloudy","overcast","fog","foggy",
                "mist","misty","haze","hazy","cumulus","stratus","gloom","covered","dim"],
}

def kw(concept, domain):
    return (WEATHER_KW if domain == "weather" else EMOTION_KW).get(concept, [])

def match(text, kws):
    t = text.lower()
    return any(k in t for k in kws)

# ── Load transfer baselines ───────────────────────────────────────────────────
MODEL_ORDER = ["llama_11B", "qwen_7B", "qwen_2B"]
MODEL_STYLE = {
    "llama_11B": {"color": "#8e44ad", "label": "Llama-3.2-11B"},
    "qwen_7B":   {"color": "#2980b9", "label": "Qwen2-VL-7B"},
    "qwen_2B":   {"color": "#e67e22", "label": "Qwen2-VL-2B"},
}

# Store: data[domain][pair][model] = {src_recog: float, tgt_fp: float}
data = {"emotions": defaultdict(dict), "weather": defaultdict(dict)}

for domain, rdir in RESULT_DIRS.items():
    for fpath in sorted(glob.glob(os.path.join(rdir, "exp_emotion_transfer_*.json"))):
        model = (os.path.basename(fpath)
                 .replace("exp_emotion_transfer_", "").replace(".json", ""))
        with open(fpath) as f:
            entries = json.load(f)

        by_pair = defaultdict(lambda: {"src": [], "tgt": []})
        for e in entries:
            src = e.get("source", ""); tgt = e.get("target", "")
            src_kws = kw(src, domain); tgt_kws = kw(tgt, domain)
            if not src_kws or not tgt_kws:
                continue
            pair = f"{src}→{tgt}"
            for r in e.get("results", []):
                by_pair[pair]["src"].append(match(r["baseline"], src_kws))
                by_pair[pair]["tgt"].append(match(r["baseline"], tgt_kws))

        for pair, d in by_pair.items():
            src_pct = sum(d["src"]) / len(d["src"]) * 100
            tgt_pct = sum(d["tgt"]) / len(d["tgt"]) * 100
            data[domain][pair][model] = {"src_recog": src_pct, "tgt_fp": tgt_pct}

# ── Pair ordering (matches paper section order) ───────────────────────────────
EMOTION_PAIRS  = ["happy→sad", "sad→happy", "neutral→angry"]
WEATHER_PAIRS  = ["cloudy→shine", "rain→shine", "shine→rain"]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                         gridspec_kw={"hspace": 0.45, "wspace": 0.30})

DOMAIN_CONFIG = [
    ("emotions", EMOTION_PAIRS,  axes[0][0], axes[1][0], "Emotion Domain"),
    ("weather",  WEATHER_PAIRS,  axes[0][1], axes[1][1], "Weather Domain"),
]

n_models = len(MODEL_ORDER)
bar_w    = 0.22
group_w  = n_models * bar_w + 0.15

for domain, pairs, ax_src, ax_tgt, title in DOMAIN_CONFIG:
    group_centers = np.arange(len(pairs)) * group_w
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bar_w

    for m_idx, model in enumerate(MODEL_ORDER):
        color = MODEL_STYLE[model]["color"]
        x_pos = group_centers + offsets[m_idx]

        src_vals = [data[domain].get(p, {}).get(model, {}).get("src_recog", 0) for p in pairs]
        tgt_vals = [data[domain].get(p, {}).get(model, {}).get("tgt_fp",   0) for p in pairs]

        ax_src.bar(x_pos, src_vals, width=bar_w, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax_tgt.bar(x_pos, tgt_vals, width=bar_w, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   hatch="///")

    tick_labels = [p.replace("→", " →\n") for p in pairs]

    for ax, row_title in [(ax_src, "Source recognition %\n(model correctly labels source image)"),
                           (ax_tgt, "Target false-positive %\n(target keywords in source image, no steering)")]:
        ax.set_xticks(group_centers)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel(row_title, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# ── Row labels ────────────────────────────────────────────────────────────────
fig.text(0.01, 0.73, "Source\nrecognition", fontsize=10, fontweight="bold",
         ha="left", va="center", rotation=0, color="#2c3e50")
fig.text(0.01, 0.27, "Target\nfalse-positive", fontsize=10, fontweight="bold",
         ha="left", va="center", rotation=0, color="#2c3e50")

# ── Legend ────────────────────────────────────────────────────────────────────
model_patches = [
    mpatches.Patch(facecolor=MODEL_STYLE[m]["color"], edgecolor="white",
                   alpha=0.85, label=MODEL_STYLE[m]["label"])
    for m in MODEL_ORDER
]
fig.legend(handles=model_patches, loc="lower center", ncol=3,
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Unsteered Baseline — Semantic Transfer Experiment\n"
    "Top: source concept recognition (higher = model understands input)  "
    "Bottom: target keyword false-positive rate (higher = result is confounded)",
    fontsize=11, y=1.01,
)

# ── Note on hallucination baseline ───────────────────────────────────────────
fig.text(0.5, -0.07,
         "Hallucination baseline (model sees blank noise image without steering): "
         "0% target-keyword match across all concepts, models, and α — "
         "confirming that the evaluation metric has no false-positive inflation in that setting.",
         ha="center", va="top", fontsize=8, color="gray", style="italic",
         wrap=True)

plt.tight_layout()

out_png = os.path.join(_here, "unsteered_baseline_figure.png")
out_pdf = os.path.join(_here, "unsteered_baseline_figure.pdf")
fig.savefig(out_png, dpi=180, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
