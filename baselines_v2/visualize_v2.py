"""
visualize_v2.py
---------------
Drop-in replacement for visualize.py that addresses two problems found in the
original analysis:

  1. TIGHTER KEYWORD LISTS — removes generic words that fire on blank/noise
     image descriptions and inflate both steered AND baseline rates, making
     steering success look artificially high.

  2. LIFT-OVER-BASELINE — every summary graph now includes a shaded "baseline"
     band and plots the NET LIFT (steered − baseline) so that results which are
     already present in unsteered responses are clearly distinguished.

Output directories:
  graphs_emotions_v2/    (parallel to graphs_emotions/)
  graphs_weather_v2/     (parallel to graphs_weather/)

Both JSON result directories (json_results/ and json_results_emotions/) are
read automatically — the same detection logic as the original is used.

Usage:
    python visualize_v2.py
"""

import os, json, glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# 1. TIGHTER KEYWORD DEFINITIONS
# ---------------------------------------------------------------------------
# Rationale for each removal is noted inline.

EMOTION_KEYWORD_MAP_V2 = {
    # Kept: unambiguous facial / bodily expressions of happiness
    # Removed: "bright", "vibrant", "warm", "sunny", "vivid", "tranquility",
    #          "serene", "pleasant" — all appear in routine blank-image descriptions
    "happy": [
        "happy", "smile", "smiling", "cheerful", "laughing", "laughter",
        "joyful", "joy", "grinning", "grin", "gleeful", "elated",
        "celebrating", "celebration", "enjoying", "sunflower",
    ],

    # Kept: direct sadness markers or body-language words
    # Removed: "dark", "gray", "muted", "obscured", "blurred", "pixelated",
    #          "neutral", "arid", "dry", "sparse", "distorted" — all common in
    #          blank/noise-image responses; "neutral" appears in the neutral list too
    "sad": [
        "sad", "sadness", "crying", "cry", "tears", "tear", "weeping",
        "sorrowful", "sorrow", "melancholy", "gloomy", "grief", "unhappy",
        "downcast", "despondent", "desolate", "lonely",
    ],

    # Kept: direct anger markers
    # Removed: "red", "orange", "vivid", "dynamic", "energy", "bold",
    #          "movement" — all appear in general image descriptions
    "angry": [
        "angry", "anger", "mad", "rage", "furious", "fury",
        "frown", "frowning", "shouting", "shout", "aggressive", "aggression",
        "fierce", "hostile", "clashing", "chaos", "flaming", "fiery",
        "stormy", "jagged",
    ],

    # Kept: only multi-word or unambiguous neutral-expression phrases
    # Removed ALL single common words: "calm", "plain", "blank", "flat",
    #          "simple", "minimalist", "monochrome", "beige", "gray",
    #          "balanced", "still", "undisturbed", "serene", "standard" —
    #          every one of these fires on ANY white/noise-image response.
    "neutral": [
        "neutral expression", "neutral face", "expressionless",
        "blank expression", "no expression", "no emotion",
        "flat expression", "impassive",
    ],
}

WEATHER_KEYWORD_MAP_V2 = {
    # Kept: precipitation-specific words
    # Removed: "gray", "dark", "water" — too generic
    "rain": [
        "rain", "raining", "rainfall", "rainy", "drizzle", "pouring",
        "thunderstorm", "storm", "shower", "raindrops", "puddle",
        "soaked", "wet", "splash",
    ],

    # Kept: unambiguous solar / clear-sky words
    # Removed: "bright", "light", "warm", "clear", "illuminated" — all
    #          appear in any description of a white or well-lit image
    "shine": [
        "sunshine", "sun", "sunny", "sunlight", "sunbeam", "solar",
        "blue sky", "rays", "beam", "glare", "radiant", "summer",
    ],

    # Kept: cloud-specific words
    # Removed: "white", "sky", "gray" — far too generic
    "cloudy": [
        "cloud", "clouds", "cloudy", "overcast", "fog", "foggy",
        "mist", "misty", "haze", "hazy", "cumulus", "stratus",
        "gloom", "covered", "dim",
    ],

    "sunrise": [
        "sunrise", "dawn", "dusk", "twilight", "golden hour",
        "sunup", "morning sky", "horizon glow", "orange sky",
        "breaking dawn", "pink sky", "golden",
    ],
}

# General-domain keywords (unchanged — these are broad by design)
FACE_KEYWORDS = [
    "face", "person", "human", "expression", "eyes", "mouth", "facial",
    "features", "head", "portrait", "individual", "skin", "nose",
    "appearance", "man", "woman", "child", "girl", "boy",
]
SCENE_KEYWORDS = [
    "outdoor", "sky", "landscape", "nature", "view", "scene",
    "environment", "outside", "horizon", "mountain", "grass", "tree",
    "world", "ground", "field", "water", "sea", "ocean", "weather",
    "atmosphere", "day", "night",
]

# ---------------------------------------------------------------------------
# 2. HELPERS (match originals where unchanged)
# ---------------------------------------------------------------------------

def clean_model_label(raw_name):
    name = raw_name.replace("weather_","").replace("emotion_","")
    name = name.replace("_Instruct","").replace("-"," ")
    if "llama" in name.lower():
        name = name.lower().replace("llama_","Llama ").replace("meta_llama_","Llama ")
    elif "qwen" in name.lower():
        name = name.lower().replace("qwen_","Qwen ")
    return name.title()

def get_model_style(model_name):
    name = model_name.lower()
    if "llama" in name:
        return "#8e44ad", "D"
    elif "qwen" in name:
        if "7b" in name: return "#2980b9", "s"
        if "2b" in name: return "#e67e22", "^"
    return "#7f8c8d", "x"

def get_domain_config(entry):
    target_val = entry.get("target_emotion") or entry.get("target")
    if target_val in WEATHER_KEYWORD_MAP_V2:
        return "weather", WEATHER_KEYWORD_MAP_V2, SCENE_KEYWORDS, "General Scene"
    return "emotion", EMOTION_KEYWORD_MAP_V2, FACE_KEYWORDS, "General Face"

def load_json_files(experiment_prefix):
    """Load from BOTH result directories."""
    datasets = {"emotion": {}, "weather": {}}
    for rdir in ("json_results", "json_results_emotions"):
        for fpath in sorted(glob.glob(os.path.join(rdir, f"{experiment_prefix}*.json"))):
            filename  = os.path.basename(fpath)
            model_name = filename.replace(f"{experiment_prefix}_","").replace(".json","")
            try:
                with open(fpath) as f:
                    content = json.load(f)
                if not content:
                    continue
                domain, *_ = get_domain_config(content[0])
                datasets[domain][model_name] = content
                print(f"  [{domain.upper()}] {model_name}: {len(content)} entries")
            except Exception as e:
                print(f"  [Skip] {filename}: {e}")
    return datasets

# ---------------------------------------------------------------------------
# 3. SCORING
# ---------------------------------------------------------------------------

def _hit(text, keywords):
    t = text.lower()
    return any(kw in t for kw in keywords)

def score_entry(entry, kw_map, gen_kw, field):
    """Return (specific_rate, general_only_rate, failure_rate) for one field."""
    results = entry.get("results", [])
    target  = entry.get("target_emotion", entry.get("target","")).lower()
    keywords = kw_map.get(target, [target])
    total    = len(results)
    if total == 0:
        return 0.0, 0.0, 100.0

    spec  = sum(1 for r in results if _hit(r[field], keywords))
    gen   = sum(1 for r in results if _hit(r[field], gen_kw))

    spec_rate      = spec / total * 100
    gen_only_rate  = max(0, gen - spec) / total * 100
    fail_rate      = max(0, total - gen) / total * 100
    return spec_rate, gen_only_rate, fail_rate

def analyze_steered(entry, kw_map, gen_kw):
    return score_entry(entry, kw_map, gen_kw, "steered")

def analyze_baseline(entry, kw_map, gen_kw):
    return score_entry(entry, kw_map, gen_kw, "baseline")

def get_success_rate(entry, target_key, kw_map, field="steered"):
    results  = entry.get("results", [])
    keywords = kw_map.get(target_key, [target_key])
    if not results:
        return 0.0
    return sum(1 for r in results if _hit(r[field], keywords)) / len(results) * 100

# ---------------------------------------------------------------------------
# 4. INDIVIDUAL HALLUCINATION PLOTS (per model, per alpha)
#    Same layout as original but adds a dashed baseline line on each subplot
# ---------------------------------------------------------------------------

def plot_hallucination_experiment():
    print("\n--- [v2] Hallucination graphs ---")
    all_datasets = load_json_files("exp_blank_hallucination")

    for domain, data_map in all_datasets.items():
        if not data_map:
            continue
        output_dir = f"graphs_{domain}_v2"
        os.makedirs(output_dir, exist_ok=True)

        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, gen_kw, gen_label = get_domain_config(sample_entry)

        for model_name, entries in data_map.items():
            pretty_name = clean_model_label(model_name)
            targets     = sorted(set(e["target_emotion"] for e in entries))
            alphas      = sorted(set(e["alpha"]          for e in entries))

            for alpha in alphas:
                fig, axes = plt.subplots(
                    len(targets), 2, figsize=(16, 4 * len(targets)), squeeze=False
                )
                fig.suptitle(
                    f"[{domain.title()} | v2 keywords] Hallucination: {pretty_name} (α={alpha})",
                    fontsize=14,
                )

                for row_idx, tgt in enumerate(targets):
                    for col_idx, comp in enumerate(["LLM", "Vision"]):
                        ax = axes[row_idx, col_idx]
                        subset = sorted(
                            [e for e in entries
                             if e["target_emotion"]==tgt and e["component"]==comp
                             and e["alpha"]==alpha],
                            key=lambda x: x["layer"],
                        )
                        if not subset:
                            ax.text(0.5, 0.5, "No Data", ha="center",
                                    transform=ax.transAxes)
                            continue

                        layers = [d["layer"] for d in subset]
                        s_metrics = [analyze_steered(d,  kw_map, gen_kw) for d in subset]
                        b_metrics = [analyze_baseline(d, kw_map, gen_kw) for d in subset]

                        match_r   = [m[0] for m in s_metrics]
                        gen_r     = [m[1] for m in s_metrics]
                        fail_r    = [m[2] for m in s_metrics]
                        base_spec = [m[0] for m in b_metrics]

                        ax.stackplot(
                            layers, match_r, gen_r, fail_r,
                            labels=[f"Matched {tgt.capitalize()}", gen_label, "Noise/Failure"],
                            colors=["#2ecc71", "#f1c40f", "#e74c3c"], alpha=0.8,
                        )
                        # Overlay unsteered baseline
                        ax.plot(layers, base_spec, color="black",
                                linestyle="--", linewidth=1.8,
                                label="Unsteered baseline", zorder=5)

                        ax.set_title(f"{comp} · '{tgt}'")
                        ax.set_ylim(0, 110)
                        ax.set_ylabel("Response %")
                        ax.set_xlabel("Layer Index")

                axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.0),
                                  fontsize="small")
                plt.tight_layout(rect=[0, 0.03, 0.88, 0.95])
                out_path = os.path.join(output_dir,
                                        f"hallucination_{model_name}_alpha{alpha}.png")
                plt.savefig(out_path, dpi=120)
                plt.close()
                print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 5. INDIVIDUAL TRANSFORMATION PLOTS
#    Adds a dashed "baseline success" series alongside the steered series
# ---------------------------------------------------------------------------

def plot_transformation_experiment():
    print("\n--- [v2] Transformation graphs ---")
    all_datasets = load_json_files("exp_emotion_transfer")

    for domain, data_map in all_datasets.items():
        if not data_map:
            continue
        output_dir = f"graphs_{domain}_v2"
        os.makedirs(output_dir, exist_ok=True)

        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, _, _ = get_domain_config(sample_entry)

        for model_name, entries in data_map.items():
            pretty_name = clean_model_label(model_name)
            pairs = sorted(set(
                f"{e['source']}->{e['target']}" for e in entries if "source" in e
            ))
            if not pairs:
                continue

            fig, axes = plt.subplots(
                len(pairs), 2, figsize=(16, 4 * len(pairs)), squeeze=False
            )
            fig.suptitle(
                f"[{domain.title()} | v2 keywords] Transfer: {pretty_name}", fontsize=14
            )

            for row_idx, pair in enumerate(pairs):
                src, tgt = pair.split("->")
                for col_idx, comp in enumerate(["LLM", "Vision"]):
                    ax = axes[row_idx, col_idx]
                    subset = sorted(
                        [e for e in entries
                         if e["source"]==src and e["target"]==tgt
                         and e["component"]==comp],
                        key=lambda x: x["layer"],
                    )
                    if not subset:
                        ax.text(0.5, 0.5, "No Data", ha="center",
                                transform=ax.transAxes)
                        continue

                    layers       = [d["layer"] for d in subset]
                    steered_r    = [get_success_rate(d, tgt, kw_map, "steered")   for d in subset]
                    baseline_r   = [get_success_rate(d, tgt, kw_map, "baseline")  for d in subset]
                    lift_r       = [s - b for s, b in zip(steered_r, baseline_r)]
                    norms        = [d["vector_norm"] for d in subset]

                    l1, = ax.plot(layers, steered_r,  color="#3498db",
                                  marker="o", label="Steered %")
                    l2, = ax.plot(layers, baseline_r, color="#7f8c8d",
                                  marker="x", linestyle="--", label="Baseline %")
                    l3, = ax.plot(layers, lift_r,     color="#27ae60",
                                  marker="s", linestyle="-.",
                                  label="Lift (steered − baseline)")
                    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
                    ax.set_ylabel("Rate (%)")
                    ax.set_ylim(-105, 105)

                    ax2 = ax.twinx()
                    l4, = ax2.plot(layers, norms, color="#e67e22",
                                   linestyle=":", marker="^",
                                   alpha=0.6, label="Vector Norm")
                    ax2.set_ylabel("Vector Norm", color="#e67e22")

                    ax.set_title(f"{src} → {tgt}  ({comp})")
                    lines = [l1, l2, l3, l4]
                    ax.legend(lines, [l.get_label() for l in lines],
                              loc="best", fontsize="small")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = os.path.join(output_dir, f"transformation_{model_name}.png")
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 6. SUMMARY PLOTS — hallucination & transformation
#    Key additions:
#      * Shaded baseline band per model
#      * Net lift plotted as a separate panel below the main panel
# ---------------------------------------------------------------------------

def plot_comparative_summary():
    print("\n--- [v2] Summary comparison plots ---")

    # ---- A. Hallucination summary ----------------------------------------
    all_datasets = load_json_files("exp_blank_hallucination")

    for domain, data_map in all_datasets.items():
        if not data_map:
            continue
        output_dir = f"graphs_{domain}_v2"
        os.makedirs(output_dir, exist_ok=True)

        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, gen_kw, _ = get_domain_config(sample_entry)

        targets = sorted(set().union(*[
            {e["target_emotion"] for e in entries}
            for entries in data_map.values()
        ]))

        # Two rows: top = raw steered + baseline, bottom = lift
        fig, axes = plt.subplots(
            2, len(targets),
            figsize=(max(6, 6*len(targets)), 10),
            sharey="row", squeeze=False,
        )
        fig.suptitle(
            f"[{domain.title()} | v2 keywords] Hallucination — Steered vs Baseline & Lift",
            fontsize=15,
        )

        for i, tgt in enumerate(targets):
            ax_top = axes[0][i]
            ax_bot = axes[1][i]

            for model_name, entries in data_map.items():
                pretty = clean_model_label(model_name)
                color, _ = get_model_style(model_name)
                tgt_entries = [e for e in entries if e.get("target_emotion")==tgt]
                if not tgt_entries:
                    continue

                unique_layers = sorted(set(e["layer"] for e in tgt_entries))
                avg_steered, avg_base, avg_lift = [], [], []

                for layer in unique_layers:
                    le = [e for e in tgt_entries if e["layer"]==layer]
                    s_vals = [score_entry(e, kw_map, gen_kw, "steered")[0]  for e in le]
                    b_vals = [score_entry(e, kw_map, gen_kw, "baseline")[0] for e in le]
                    avg_steered.append(np.mean(s_vals))
                    avg_base.append(   np.mean(b_vals))
                    avg_lift.append(   np.mean(s_vals) - np.mean(b_vals))

                # Top panel: steered (solid) + baseline (dashed, same colour)
                ax_top.plot(unique_layers, avg_steered, color=color,
                            linestyle="-", marker="o", linewidth=2,
                            label=f"{pretty} steered")
                ax_top.plot(unique_layers, avg_base, color=color,
                            linestyle="--", linewidth=1.2, alpha=0.55,
                            label=f"{pretty} baseline")
                ax_top.fill_between(unique_layers, avg_base, avg_steered,
                                    color=color, alpha=0.10)

                # Bottom panel: lift
                ax_bot.plot(unique_layers, avg_lift, color=color,
                            linestyle="-", marker="o", linewidth=1.8,
                            label=pretty)

            ax_top.set_title(f"Target: {tgt.upper()}")
            ax_top.set_ylim(-5, 105)
            ax_top.grid(True, linestyle="--", alpha=0.4)
            ax_top.legend(fontsize="x-small", loc="upper left")
            if i==0: ax_top.set_ylabel("Specific Match %")

            ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax_bot.set_ylim(-55, 105)
            ax_bot.set_xlabel("Layer")
            ax_bot.grid(True, linestyle="--", alpha=0.4)
            ax_bot.legend(fontsize="x-small", loc="upper left")
            if i==0: ax_bot.set_ylabel("Lift (steered − baseline) %")

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        out = os.path.join(output_dir, "summary_hallucination_v2.png")
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Saved: {out}")

    # ---- B. Transformation summary ---------------------------------------
    all_datasets = load_json_files("exp_emotion_transfer")

    for domain, data_map in all_datasets.items():
        if not data_map:
            continue
        output_dir = f"graphs_{domain}_v2"
        os.makedirs(output_dir, exist_ok=True)

        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, _, _ = get_domain_config(sample_entry)

        all_pairs = sorted(set(
            f"{e['source']}->{e['target']}"
            for entries in data_map.values()
            for e in entries if "source" in e
        ))
        if not all_pairs:
            continue

        fig, axes = plt.subplots(
            2, len(all_pairs),
            figsize=(max(6, 6*len(all_pairs)), 10),
            sharey="row", squeeze=False,
        )
        fig.suptitle(
            f"[{domain.title()} | v2 keywords] Transfer — Steered vs Baseline & Lift",
            fontsize=15,
        )

        for i, pair in enumerate(all_pairs):
            src, tgt = pair.split("->")
            ax_top = axes[0][i]
            ax_bot = axes[1][i]

            for model_name, entries in data_map.items():
                pretty = clean_model_label(model_name)
                color, marker = get_model_style(model_name)
                pe = [e for e in entries
                      if e.get("source")==src and e.get("target")==tgt]
                if not pe:
                    continue

                unique_layers = sorted(set(e["layer"] for e in pe))
                avg_s, avg_b, avg_l = [], [], []
                for layer in unique_layers:
                    le  = [e for e in pe if e["layer"]==layer]
                    s   = np.mean([get_success_rate(e, tgt, kw_map, "steered")  for e in le])
                    b   = np.mean([get_success_rate(e, tgt, kw_map, "baseline") for e in le])
                    avg_s.append(s); avg_b.append(b); avg_l.append(s-b)

                ax_top.plot(unique_layers, avg_s, color=color,
                            marker=marker, linestyle="-", linewidth=2,
                            label=f"{pretty} steered")
                ax_top.plot(unique_layers, avg_b, color=color,
                            linestyle="--", linewidth=1.2, alpha=0.55,
                            label=f"{pretty} baseline")
                ax_top.fill_between(unique_layers, avg_b, avg_s,
                                    color=color, alpha=0.10)
                ax_bot.plot(unique_layers, avg_l, color=color,
                            marker=marker, linewidth=1.8, label=pretty)

            ax_top.set_title(f"{src} → {tgt}")
            ax_top.set_ylim(-5, 105)
            ax_top.grid(True, linestyle="--", alpha=0.4)
            ax_top.legend(fontsize="x-small")
            if i==0: ax_top.set_ylabel("Specific Match %")

            ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax_bot.set_ylim(-105, 105)
            ax_bot.set_xlabel("Layer")
            ax_bot.grid(True, linestyle="--", alpha=0.4)
            ax_bot.legend(fontsize="x-small")
            if i==0: ax_bot.set_ylabel("Lift (steered − baseline) %")

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        out = os.path.join(output_dir, "summary_transformation_v2.png")
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 7. KEYWORD DIFF — print exactly what changed vs original
# ---------------------------------------------------------------------------

def print_keyword_diff():
    from visualize import EMOTION_KEYWORD_MAP, WEATHER_KEYWORD_MAP  # original

    print("\n" + "="*60)
    print("KEYWORD CHANGES (v2 vs original)")
    print("="*60)
    for name, orig, new in [
        ("EMOTION", EMOTION_KEYWORD_MAP,  EMOTION_KEYWORD_MAP_V2),
        ("WEATHER", WEATHER_KEYWORD_MAP,  WEATHER_KEYWORD_MAP_V2),
    ]:
        print(f"\n  {name}")
        for concept in sorted(set(list(orig)+list(new))):
            o_set = set(orig.get(concept, []))
            n_set = set(new.get(concept,  []))
            removed = sorted(o_set - n_set)
            added   = sorted(n_set - o_set)
            if removed or added:
                print(f"    {concept}:")
                if removed: print(f"      removed: {removed}")
                if added:   print(f"      added:   {added}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print_keyword_diff()
    except ImportError:
        print("[Info] Could not import visualize.py for diff — skipping")

    plot_hallucination_experiment()
    plot_transformation_experiment()
    plot_comparative_summary()
    print("\n[Done] All v2 visualizations generated.")
