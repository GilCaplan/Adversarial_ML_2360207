"""
plot_baseline_comparison.py
---------------------------
Generates ONE new figure for the paper: a grouped bar chart comparing
unsteered baseline vs. best steered specific-match rates for all concepts
and all models, across both domains (emotion + weather).

Uses the TIGHTER v2 keyword lists (defined in visualize_v2.py) which remove
generic words that fire on blank-image descriptions, giving an honest
measure of genuine steering lift.

The figure shows:
  - Colored bars  = best steered specific-match % (max across all layers/alphas)
  - Gray hatched = unsteered baseline specific-match %
  - ★  = clean signal: baseline=0%, lift>0  (genuine hallucination)
  - ✗  = lift < -15% (steering actively degrades keyword match)

Output (same directory as this script):
    baseline_comparison_figure.png
    baseline_comparison_figure.pdf

Run from the project root:
    python baseline_analysis/plot_baseline_comparison.py
"""

import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_project = os.path.dirname(_here)

RESULT_DIRS = {
    "emotions": os.path.join(_project, "json_results_emotions"),
    "weather":  os.path.join(_project, "json_results"),
}

# ── Tighter keyword lists (v2) ────────────────────────────────────────────────
EMOTION_KW_V2 = {
    "happy":   ["happy","smile","smiling","cheerful","laughing","laughter",
                "joyful","joy","grinning","grin","gleeful","elated",
                "celebrating","celebration","enjoying","sunflower"],
    "sad":     ["sad","sadness","crying","cry","tears","tear","weeping",
                "sorrowful","sorrow","melancholy","gloomy","grief","unhappy",
                "downcast","despondent","desolate","lonely"],
    "angry":   ["angry","anger","mad","rage","furious","fury",
                "frown","frowning","shouting","shout","aggressive","aggression",
                "fierce","hostile","clashing","chaos","flaming","fiery",
                "stormy","jagged"],
    "neutral": ["neutral expression","neutral face","expressionless",
                "blank expression","no expression","no emotion",
                "flat expression","impassive"],
}
WEATHER_KW_V2 = {
    "rain":    ["rain","raining","rainfall","rainy","drizzle","pouring",
                "thunderstorm","storm","shower","raindrops","puddle",
                "soaked","wet","splash"],
    "shine":   ["sunshine","sun","sunny","sunlight","sunbeam","solar",
                "blue sky","rays","beam","glare","radiant","summer"],
    "cloudy":  ["cloud","clouds","cloudy","overcast","fog","foggy",
                "mist","misty","haze","hazy","cumulus","stratus",
                "gloom","covered","dim"],
    "sunrise": ["sunrise","dawn","dusk","twilight","golden hour",
                "sunup","morning sky","horizon glow","orange sky",
                "breaking dawn","pink sky","golden"],
}

def _kw(concept, domain):
    return WEATHER_KW_V2.get(concept) if domain == "weather" else EMOTION_KW_V2.get(concept, [])

def _match(text, kws):
    t = text.lower()
    return any(k in t for k in kws)

# ── Load & score all hallucination entries ────────────────────────────────────
def load_and_score():
    records = defaultdict(list)          # key: (domain, concept, model)
    for domain, rdir in RESULT_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(rdir, "exp_blank_hallucination_*.json"))):
            model = (os.path.basename(fpath)
                     .replace("exp_blank_hallucination_", "")
                     .replace(".json", ""))
            with open(fpath) as f:
                entries = json.load(f)
            for e in entries:
                concept = e.get("target_emotion", "")
                kws = _kw(concept, domain)
                if not kws:
                    continue
                for r in e.get("results", []):
                    key = (domain, concept, model)
                    records[key].append({
                        "layer":     e["layer"],
                        "component": e["component"],
                        "alpha":     e["alpha"],
                        "baseline_hit": _match(r["baseline"], kws),
                        "steered_hit":  _match(r["steered"],  kws),
                    })
    return records

records = load_and_score()

# ── Build best-steered & mean-baseline per (domain, concept, model) ──────────
def summarise(records):
    mean_steered  = {}
    mean_baseline = {}
    for key, recs in records.items():
        mean_steered[key]  = sum(r["steered_hit"]  for r in recs) / len(recs) * 100
        mean_baseline[key] = sum(r["baseline_hit"] for r in recs) / len(recs) * 100
    return mean_steered, mean_baseline

best_steered, mean_baseline = summarise(records)

# ── Plot config ───────────────────────────────────────────────────────────────
MODEL_ORDER = ["llama_11B", "qwen_7B", "qwen_2B"]
MODEL_STYLE = {
    "llama_11B": {"color": "#8e44ad", "label": "Llama-3.2-11B"},
    "qwen_7B":   {"color": "#2980b9", "label": "Qwen2-VL-7B"},
    "qwen_2B":   {"color": "#e67e22", "label": "Qwen2-VL-2B"},
}
EMOTION_CONCEPTS = ["happy", "sad", "angry"]
WEATHER_CONCEPTS = ["cloudy", "rain", "shine"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

DOMAIN_CONFIG = [
    ("emotions", EMOTION_CONCEPTS, axes[0], "Emotion Domain"),
    ("weather",  WEATHER_CONCEPTS, axes[1], "Weather Domain"),
]

n_models  = len(MODEL_ORDER)
bar_w     = 0.22
group_gap = 0.12
group_w   = n_models * bar_w + group_gap

for ax, (domain, concepts, _, title) in zip(axes, DOMAIN_CONFIG):
    group_centers = np.arange(len(concepts)) * group_w
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bar_w

    for m_idx, model in enumerate(MODEL_ORDER):
        color = MODEL_STYLE[model]["color"]
        x_pos = group_centers + offsets[m_idx]

        for c_idx, concept in enumerate(concepts):
            key   = (domain, concept, model)
            s_val = best_steered.get(key, 0.0)
            b_val = mean_baseline.get(key, 0.0)
            x     = x_pos[c_idx]
            lift  = s_val - b_val

            # Baseline bar (gray, hatched) — drawn first so it sits behind
            ax.bar(x, b_val, width=bar_w,
                   color="lightgray", edgecolor="gray",
                   hatch="///", linewidth=0.6, zorder=2)

            # Steered bar (coloured)
            ax.bar(x, s_val, width=bar_w,
                   color=color, edgecolor="white",
                   alpha=0.85, linewidth=0.5, zorder=3)

            # Annotation
            top = max(s_val, b_val) + 2.5
            if lift > 0 and b_val == 0:
                ax.annotate("★", xy=(x, top), ha="center", va="bottom",
                            fontsize=8, color=color, zorder=5)
            elif lift < -15:
                ax.annotate("✗", xy=(x, top), ha="center", va="bottom",
                            fontsize=8, color="#c0392b", zorder=5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([c.capitalize() for c in concepts], fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Specific Match %" if ax == axes[0] else "", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
baseline_patch = mpatches.Patch(
    facecolor="lightgray", edgecolor="gray", hatch="///",
    label="Unsteered baseline (mean, v2 keywords)"
)
model_patches = [
    mpatches.Patch(facecolor=MODEL_STYLE[m]["color"], edgecolor="white",
                   alpha=0.85, label=MODEL_STYLE[m]["label"])
    for m in MODEL_ORDER
]
fig.legend(
    handles=[baseline_patch] + model_patches,
    loc="lower center", ncol=4, frameon=False, fontsize=10,
    bbox_to_anchor=(0.5, -0.07),
)
fig.text(0.99, 0.01,
         "★ clean signal (baseline=0%, lift>0)    ✗ steering reduces match",
         ha="right", va="bottom", fontsize=7.5, color="gray", style="italic")

fig.suptitle(
    "Unsteered Baseline vs. Mean Steered Specific Match\n"
    "(Controlled Hallucination · averaged across all layers & α · tighter keyword eval)",
    fontsize=12, y=1.02,
)
plt.tight_layout()

out_png = os.path.join(_here, "baseline_comparison_figure.png")
out_pdf = os.path.join(_here, "baseline_comparison_figure.pdf")
fig.savefig(out_png, dpi=180, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
