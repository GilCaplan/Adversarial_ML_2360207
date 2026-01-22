import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np

# --- Configuration ---
RESULTS_DIR = "json_results"

# ==========================================
# 1. KEYWORD DEFINITIONS
# ==========================================

# --- EMOTION DOMAIN ---
EMOTION_KEYWORD_MAP = {
    "happy": [
        "happy", "smile", "smiling", "cheerful", "vibrant", "sunny", "bright", 
        "joy", "lively", "energetic", "warm", "bloom", "sunflower", "pleasant", 
        "tranquility", "serene", "celebration", "enjoying", "vivid", "laughing"
    ],
    "sad": [
        "sad", "crying", "sorrow", "muted", "obscured", "distorted", "blurred", 
        "melancholy", "somber", "gloomy", "dark", "gray", "arid", "dry", 
        "sparse", "lonely", "desolate", "obscure", "pixelated", "neutral", "tear"
    ],
    "angry": [
        "angry", "mad", "rage", "fiery", "intense", "sharp", "jagged", "red", 
        "orange", "vivid", "dynamic", "movement", "energy", "stormy", 
        "aggressive", "bold", "clashing", "chaos", "flaming", "frown", "shouting"
    ],
    "neutral": [
        "neutral", "expressionless", "calm", "flat", "blank", "plain", 
        "standard", "simple", "minimalist", "monochrome", "beige", "gray", 
        "balanced", "still", "undisturbed", "serene"
    ]
}

# Used to check for "General Hallucination" (Is it a person even if the emotion is wrong?)
FACE_KEYWORDS = [
    "face", "person", "human", "expression", "eyes", "mouth", "facial", 
    "features", "head", "portrait", "individual", "skin", "nose", "appearance",
    "man", "woman", "child", "girl", "boy"
]

# --- WEATHER DOMAIN ---
WEATHER_KEYWORD_MAP = {
    "rain": [
        "rain", "rainy", "storm", "wet", "drizzle", "pouring", "thunderstorm", 
        "shower", "droplets", "puddle", "soaked", "gray", "dark", "water", "splash"
    ],
    "shine": [
        "sun", "sunny", "shine", "bright", "clear", "blue sky", "light", 
        "rays", "beam", "glare", "radiant", "warm", "summer", "illuminated"
    ],
    "cloudy": [
        "cloud", "cloudy", "overcast", "gray", "gloom", "fog", "mist", 
        "haze", "white", "fluffy", "cumulus", "sky", "covered", "dim"
    ],
    "sunrise": [
        "sunrise", "dawn", "morning", "breaking", "sunup", "orange", "glow", 
        "horizon", "early", "start", "rising", "pink", "golden", "dusk"
    ]
}

# Used to check for "General Hallucination" (Is it a landscape even if the weather is wrong?)
SCENE_KEYWORDS = [
    "outdoor", "sky", "landscape", "nature", "view", "scene", "environment", 
    "outside", "horizon", "mountain", "grass", "tree", "world", "ground", 
    "field", "water", "sea", "ocean", "weather", "atmosphere", "day", "night"
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_model_label(raw_name):
    """Converts raw filenames (e.g., 'weather_llama_11B') into pretty labels."""
    # Strip common prefixes to avoid "Weather Weather Llama"
    name = raw_name.replace("weather_", "").replace("emotion_", "")
    name = name.replace("_Instruct", "").replace("-", " ")
    
    if "llama" in name.lower():
        name = name.lower().replace("llama_", "Llama ").replace("meta_llama_", "Llama ")
    elif "qwen" in name.lower():
        name = name.lower().replace("qwen_", "Qwen ")
    
    return name.title().replace("B", "B")

def get_model_style(model_name):
    """Returns (color, marker) based on model type/size."""
    name = model_name.lower()
    if "llama" in name:
        if "11b" in name: return "#8e44ad", "D" # Purple
        if "3b" in name: return "#9b59b6", "d"
        return "#e056fd", "o"
    elif "qwen" in name:
        if "7b" in name: return "#2980b9", "s" # Blue
        if "2b" in name: return "#e67e22", "^" # Orange
        return "#3498db", "o"
    return "#7f8c8d", "x"

def get_domain_config(data_entry):
    """
    Automatic Detection Logic.
    Inspects a JSON entry to determine if it belongs to 'Weather' or 'Emotion'.
    
    Returns: (domain_name, keyword_map, general_keywords, general_label_text)
    """
    # 1. Try to find the target key (Experiment 1 uses 'target_emotion', Exp 2 uses 'target')
    # We check the VALUE of this key against our known maps.
    target_val = data_entry.get("target_emotion") or data_entry.get("target")
    
    if target_val in WEATHER_KEYWORD_MAP:
        return "weather", WEATHER_KEYWORD_MAP, SCENE_KEYWORDS, "General Scene"
    else:
        # Default to emotion if not found in weather map
        return "emotion", EMOTION_KEYWORD_MAP, FACE_KEYWORDS, "General Face"

def load_json_files(experiment_prefix):
    """
    Loads JSON files and automatically sorts them into 'emotion' or 'weather' buckets.
    """
    search_path = os.path.join(RESULTS_DIR, f"{experiment_prefix}*.json")
    files = glob.glob(search_path)
    
    # Storage structure: datasets['weather']['llama_11b'] = [data...]
    datasets = {"emotion": {}, "weather": {}}
    
    if not files:
        print(f"[Warning] No files found for prefix '{experiment_prefix}'")
        return datasets

    print(f"Found {len(files)} files for {experiment_prefix}")

    for fpath in files:
        filename = os.path.basename(fpath)
        # Clean the filename to get a model identifier
        model_name = filename.replace(f"{experiment_prefix}_", "").replace(".json", "")
        
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
                if not content: continue
                
                # INSPECT CONTENT: Check the first entry to decide domain
                first_entry = content[0]
                domain, _, _, _ = get_domain_config(first_entry)
                
                datasets[domain][model_name] = content
                print(f"  > Loaded as [{domain.upper()}]: {model_name} ({len(content)} entries)")
                
        except (json.JSONDecodeError, FileNotFoundError, IndexError):
            print(f"  > [Skip] Corrupt or empty file: {filename}")
            
    return datasets

def analyze_hallucination_results(entry, keyword_map, general_keywords):
    """
    Calculates 3 metrics for a single data point:
    1. Specific Match % (e.g. "It is raining")
    2. General Domain % (e.g. "It is a landscape" but missed the rain)
    3. Failure % (Noise/Other)
    """
    results = entry.get("results", [])
    target = entry.get("target_emotion", "").lower() # Note: 'target_emotion' key is used in exp 1
    total = len(results)
    
    if total == 0: return 0, 0, 100

    match_count = 0
    general_count = 0
    
    keywords = keyword_map.get(target, [target])

    for res in results:
        text = res["steered"].lower()
        
        # 1. Rigorous Specific Match
        if any(kw in text for kw in keywords):
            match_count += 1
            general_count += 1 # If it matched specific, it is also valid general domain
        
        # 2. General Domain Match (if specific didn't match, check if it's at least the right type of image)
        elif any(kw in text for kw in general_keywords):
            general_count += 1
                
    match_rate = (match_count / total) * 100
    general_only_rate = ((general_count - match_count) / total) * 100
    failure_rate = ((total - general_count) / total) * 100
    
    return match_rate, general_only_rate, failure_rate

def get_success_rate(entry, target_key, keyword_map):
    """Simple success rate calculation for Transformation experiment."""
    results = entry.get("results", [])
    keywords = keyword_map.get(target_key, [target_key])
    if not results: return 0
    
    match_count = sum(1 for res in results if any(kw in res["steered"].lower() for kw in keywords))
    return (match_count / len(results)) * 100

# ============================================================
# 3. PLOTTING FUNCTIONS
# ============================================================

def plot_hallucination_experiment():
    print("\n--- Generating Individual Hallucination Graphs ---")
    all_datasets = load_json_files("exp_blank_hallucination")

    # Iterate over domains found (e.g., 'weather', 'emotion')
    for domain, data_map in all_datasets.items():
        if not data_map: continue
        
        # 1. Setup Output Directory
        output_dir = f"graphs_{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Get Domain Specific Config (Keywords)
        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, gen_kw, gen_label = get_domain_config(sample_entry)

        # 3. Process each Model
        for model_name, entries in data_map.items():
            pretty_name = clean_model_label(model_name)
            targets = sorted(list(set(e["target_emotion"] for e in entries)))
            if not targets: continue

            components = ["LLM", "Vision"]
            alphas = sorted(list(set(e["alpha"] for e in entries)))
            
            # Create one plot per Alpha value
            for alpha in alphas:
                fig, axes = plt.subplots(len(targets), 2, figsize=(16, 4 * len(targets)), squeeze=False)
                fig.suptitle(f"[{domain.title()}] Hallucination: {pretty_name} (Alpha {alpha})", fontsize=16)

                for row_idx, tgt in enumerate(targets):
                    for col_idx, comp in enumerate(components):
                        ax = axes[row_idx, col_idx]
                        
                        # Filter Data
                        subset = sorted([
                            e for e in entries 
                            if e["target_emotion"] == tgt and e["component"] == comp and e["alpha"] == alpha
                        ], key=lambda x: x["layer"])
                        
                        if not subset:
                            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                            continue

                        # Analyze
                        layers = [d["layer"] for d in subset]
                        metrics = [analyze_hallucination_results(d, kw_map, gen_kw) for d in subset]
                        
                        match_r = [m[0] for m in metrics]
                        gen_r = [m[1] for m in metrics]
                        fail_r = [m[2] for m in metrics]

                        # Plot Stacked Area
                        ax.stackplot(layers, match_r, gen_r, fail_r, 
                                     labels=[f'Matched {tgt.capitalize()}', gen_label, 'Noise/Failure'],
                                     colors=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.8)

                        ax.set_title(f"{comp} Layer Injection: '{tgt}'")
                        ax.set_ylim(0, 100)
                        ax.set_ylabel("Response %")
                        ax.set_xlabel("Layer Index")

                # Legend on top-right plot only
                axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                
                plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
                out_path = os.path.join(output_dir, f"hallucination_{model_name}_alpha{alpha}.png")
                plt.savefig(out_path)
                plt.close()
                print(f"Saved: {out_path}")

def plot_transformation_experiment():
    print("\n--- Generating Individual Transformation Graphs ---")
    all_datasets = load_json_files("exp_emotion_transfer")

    for domain, data_map in all_datasets.items():
        if not data_map: continue
        
        output_dir = f"graphs_{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, _, _ = get_domain_config(sample_entry)

        for model_name, entries in data_map.items():
            pretty_name = clean_model_label(model_name)
            # Find all unique pairs
            pairs = sorted(list(set(f"{e['source']}->{e['target']}" for e in entries if 'source' in e)))
            if not pairs: continue

            components = ["LLM", "Vision"]
            
            fig, axes = plt.subplots(len(pairs), 2, figsize=(16, 4 * len(pairs)), squeeze=False)
            fig.suptitle(f"[{domain.title()}] Transfer: {pretty_name}", fontsize=16)
            
            for row_idx, pair in enumerate(pairs):
                src, tgt = pair.split("->")
                
                for col_idx, comp in enumerate(components):
                    ax = axes[row_idx, col_idx]
                    subset = sorted([
                        e for e in entries 
                        if e["source"] == src and e["target"] == tgt and e["component"] == comp
                    ], key=lambda x: x["layer"])
                    
                    if not subset: 
                        ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                        continue

                    layers = [d["layer"] for d in subset]
                    success_rates = [get_success_rate(d, tgt, kw_map) for d in subset]
                    norms = [d["vector_norm"] for d in subset]

                    # Plot Success Rate (Left Axis)
                    l1, = ax.plot(layers, success_rates, color='#3498db', marker='o', label='Success %')
                    ax.set_ylabel("Success Rate (%)", color='#3498db')
                    ax.set_ylim(-5, 105)

                    # Plot Vector Norm (Right Axis)
                    ax2 = ax.twinx()
                    l2, = ax2.plot(layers, norms, color='#e67e22', linestyle=':', marker='x', label='Vector Norm')
                    ax2.set_ylabel("Vector Norm", color='#e67e22')

                    ax.set_title(f"{src} -> {tgt} ({comp})")
                    
                    # Combined Legend
                    lines = [l1, l2]
                    ax.legend(lines, [l.get_label() for l in lines], loc='best')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = os.path.join(output_dir, f"transformation_{model_name}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved: {out_path}")

# ============================================================
# 4. SUMMARY PLOTS (COMPARISON)
# ============================================================

def plot_comparative_summary():
    print("\n--- Generating Comparative Summaries ---")
    
    # ---------------------------
    # A. Hallucination Summary
    # ---------------------------
    all_datasets = load_json_files("exp_blank_hallucination")
    
    for domain, data_map in all_datasets.items():
        if not data_map: continue
        
        output_dir = f"graphs_{domain}"
        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, gen_kw, gen_label = get_domain_config(sample_entry)
        
        # Collect all targets present in this domain
        targets = sorted(list(set().union(*[set(e["target_emotion"] for e in entries) for entries in data_map.values()])))
        
        fig, axes = plt.subplots(1, len(targets), figsize=(max(6, 6 * len(targets)), 7), sharey=True, squeeze=False)
        axes = axes[0]
        fig.suptitle(f"Comparison: [{domain.title()}] Hallucination Sensitivity\nSolid: Specific Match | Dashed: {gen_label}", fontsize=16)

        for i, tgt in enumerate(targets):
            ax = axes[i]
            has_data = False
            for model_name, entries in data_map.items():
                pretty_name = clean_model_label(model_name)
                color, _ = get_model_style(model_name)
                
                tgt_entries = [e for e in entries if e.get("target_emotion") == tgt]
                if not tgt_entries: continue
                has_data = True
                
                unique_layers = sorted(list(set(e["layer"] for e in tgt_entries)))
                avg_match, avg_gen = [], []
                
                for layer in unique_layers:
                    layer_entries = [e for e in tgt_entries if e["layer"] == layer]
                    m_rates = []
                    g_rates = []
                    for le in layer_entries:
                        m, g_only, _ = analyze_hallucination_results(le, kw_map, gen_kw)
                        m_rates.append(m)
                        g_rates.append(m + g_only) # "Any" valid domain response
                    
                    avg_match.append(np.mean(m_rates))
                    avg_gen.append(np.mean(g_rates))
                
                ax.plot(unique_layers, avg_match, marker='o', label=f"{pretty_name}", color=color, linestyle='-', linewidth=2)
                ax.plot(unique_layers, avg_gen, marker='', color=color, linestyle='--', linewidth=1, alpha=0.5)

            ax.set_title(f"Targeting: {tgt.upper()}")
            ax.set_xlabel("Layer")
            if i == 0: ax.set_ylabel("Avg Rate (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_ylim(-5, 105)
            if has_data: ax.legend(fontsize='small', loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "summary_hallucination_comparison.png"))
        plt.close()

    # ---------------------------
    # B. Transformation Summary
    # ---------------------------
    all_datasets = load_json_files("exp_emotion_transfer")
    
    for domain, data_map in all_datasets.items():
        if not data_map: continue
        
        output_dir = f"graphs_{domain}"
        sample_entry = next(iter(data_map.values()))[0]
        _, kw_map, _, _ = get_domain_config(sample_entry)
        
        all_pairs = []
        for m_entries in data_map.values():
            all_pairs.extend([f"{e['source']}->{e['target']}" for e in m_entries if 'source' in e])
        pairs = sorted(list(set(all_pairs)))
        if not pairs: continue

        fig, axes = plt.subplots(1, len(pairs), figsize=(max(6, 6 * len(pairs)), 6), sharey=True, squeeze=False)
        axes = axes[0]
        fig.suptitle(f"Comparison: [{domain.title()}] Transfer Success Rate", fontsize=16)

        for i, pair in enumerate(pairs):
            ax = axes[i]
            src, tgt = pair.split("->")
            has_data = False
            for model_name, entries in data_map.items():
                pretty_name = clean_model_label(model_name)
                color, marker = get_model_style(model_name)
                
                pair_entries = [e for e in entries if e.get("source") == src and e.get("target") == tgt]
                if not pair_entries: continue
                has_data = True
                
                unique_layers = sorted(list(set(e["layer"] for e in pair_entries)))
                avg_success = []
                for layer in unique_layers:
                    l_points = [get_success_rate(e, tgt, kw_map) for e in pair_entries if e["layer"] == layer]
                    avg_success.append(np.mean(l_points))
                
                ax.plot(unique_layers, avg_success, marker=marker, label=pretty_name, color=color, linewidth=2, alpha=0.8)

            ax.set_title(f"{pair.upper()}")
            ax.set_xlabel("Layer")
            if i == 0: ax.set_ylabel("Success Rate (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            if has_data: ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "summary_transformation_comparison.png"))
        plt.close()

if __name__ == "__main__":
    plot_hallucination_experiment()
    plot_transformation_experiment()
    plot_comparative_summary()
    print("\n[Done] All visualizations generated.")