import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np

# --- Configuration ---
RESULTS_DIR = "json_results"
OUTPUT_DIR = "graphs"
# This ensures the folder is created but NEVER deleted/cleared
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rigorous Keyword Mapping for Emotion and Hallucination Detection
EMOTION_KEYWORD_MAP = {
    "happy": [
        "happy", "smile", "smiling", "cheerful", "vibrant", "sunny", "bright", 
        "joy", "lively", "energetic", "warm", "bloom", "sunflower", "pleasant", 
        "tranquility", "serene", "celebration", "enjoying", "vivid"
    ],
    "sad": [
        "sad", "crying", "sorrow", "muted", "obscured", "distorted", "blurred", 
        "melancholy", "somber", "gloomy", "dark", "gray", "arid", "dry", 
        "sparse", "lonely", "desolate", "obscure", "pixelated", "neutral"
    ],
    "angry": [
        "angry", "mad", "rage", "fiery", "intense", "sharp", "jagged", "red", 
        "orange", "vivid", "dynamic", "movement", "energy", "stormy", 
        "aggressive", "bold", "clashing", "chaos", "flaming"
    ],
    "neutral": [
        "neutral", "expressionless", "calm", "flat", "blank", "plain", 
        "standard", "simple", "minimalist", "monochrome", "beige", "gray", 
        "balanced", "still", "undisturbed", "serene"
    ]
}

FACE_KEYWORDS = [
    "face", "person", "human", "expression", "eyes", "mouth", "facial", 
    "features", "head", "portrait", "individual", "skin", "nose", "appearance"
]

FAILURE_KEYWORDS = [
    "noise", "scattered dots", "pattern", "discernible", "no discernible", 
    "recognizable", "no recognizable", "texture", "blank", "placeholder"
]

def load_json_files(experiment_prefix):
    files = glob.glob(os.path.join(RESULTS_DIR, f"{experiment_prefix}*.json"))
    data_map = {}
    if not files:
        print(f"[Warning] No JSON files found for prefix '{experiment_prefix}' in {RESULTS_DIR}")
        return {}

    for fpath in files:
        filename = os.path.basename(fpath)
        # Extract model name safely
        model_name = filename.replace(experiment_prefix + "_", "").replace(".json", "")
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
                if content: # Ensure file is not empty list
                    data_map[model_name] = content
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[Skip] Could not read {filename}")
            continue
    return data_map

def analyze_hallucination_results(entry):
    """Categorizes results into [Emotion Match %, General Face %, Failure %]"""
    results = entry.get("results", [])
    target_emo = entry.get("target_emotion", "").lower()
    total = len(results)
    
    if total == 0: return 0, 0, 100

    match_count = 0
    face_count = 0
    
    # Get keywords for the target emotion
    keywords = EMOTION_KEYWORD_MAP.get(target_emo, [target_emo])

    for res in results:
        text = res["steered"].lower()
        
        # 1. Rigorous Emotion Match
        if any(kw in text for kw in keywords):
            match_count += 1
            face_count += 1 
        # 2. General Face Hallucination (if emotion didn't match)
        elif any(kw in text for kw in FACE_KEYWORDS):
            face_count += 1
                
    match_rate = (match_count / total) * 100
    face_only_rate = ((face_count - match_count) / total) * 100
    failure_rate = ((total - face_count) / total) * 100
    
    return match_rate, face_only_rate, failure_rate

def plot_hallucination_experiment():
    print("--- Generating Individual Hallucination Graphs ---")
    data_map = load_json_files("exp_blank_hallucination")
    
    if not data_map:
        return

    for model_name, entries in data_map.items():
        emotions = sorted(list(set(e["target_emotion"] for e in entries)))
        if not emotions: continue

        components = ["LLM", "Vision"]
        alphas = sorted(list(set(e["alpha"] for e in entries)))
        
        for alpha in alphas:
            # Squeeze=False ensures axes is always a 2D array even if 1 row
            fig, axes = plt.subplots(len(emotions), 2, figsize=(18, 5 * len(emotions)), squeeze=False)
            fig.suptitle(f"Hallucination Analysis: {model_name} (Alpha {alpha})\n"
                         f"Keywords: Emotion-specific, Face-detection, and Noise-failure", fontsize=16)

            for row_idx, emo in enumerate(emotions):
                for col_idx, comp in enumerate(components):
                    ax = axes[row_idx, col_idx]
                    subset = sorted([e for e in entries if e["target_emotion"] == emo 
                                     and e["component"] == comp and e["alpha"] == alpha], 
                                    key=lambda x: x["layer"])
                    
                    if not subset:
                        ax.text(0.5, 0.5, "No Data", ha='center')
                        continue

                    layers = [d["layer"] for d in subset]
                    metrics = [analyze_hallucination_results(d) for d in subset]
                    
                    match_r = [m[0] for m in metrics]
                    face_r = [m[1] for m in metrics]
                    fail_r = [m[2] for m in metrics]

                    ax.stackplot(layers, match_r, face_r, fail_r, 
                                 labels=[f'Matched {emo.capitalize()}', 'General Face', 'Failure (Noise)'],
                                 colors=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.8)

                    ax.set_title(f"Injecting '{emo}' into {comp}")
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Response %")

            # Only add legend to the first row's second column to save space
            if len(emotions) > 0:
                axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
            
            plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
            out_path = os.path.join(OUTPUT_DIR, f"hallucination_{model_name}_alpha{alpha}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved: {out_path}")

def plot_transformation_experiment():
    print("--- Generating Individual Transformation Graphs ---")
    data_map = load_json_files("exp_emotion_transfer")
    
    if not data_map:
        return

    for model_name, entries in data_map.items():
        pairs = sorted(list(set(f"{e['source']}->{e['target']}" for e in entries)))
        if not pairs: continue

        components = ["LLM", "Vision"]
        
        fig, axes = plt.subplots(len(pairs), 2, figsize=(18, 5 * len(pairs)), squeeze=False)
        fig.suptitle(f"Emotion Transfer Performance: {model_name}", fontsize=16)
        
        for row_idx, pair in enumerate(pairs):
            src, tgt = pair.split("->")
            keywords = EMOTION_KEYWORD_MAP.get(tgt, [tgt])

            for col_idx, comp in enumerate(components):
                ax = axes[row_idx, col_idx]
                subset = sorted([e for e in entries if e["source"] == src 
                                 and e["target"] == tgt and e["component"] == comp], 
                                key=lambda x: x["layer"])
                
                if not subset: 
                    ax.text(0.5, 0.5, "No Data", ha='center')
                    continue

                layers = [d["layer"] for d in subset]
                success_rates = []
                for d in subset:
                    res_list = d.get("results", [])
                    if not res_list:
                        success_rates.append(0)
                        continue
                    
                    s_count = sum(1 for r in res_list if any(kw in r["steered"].lower() for kw in keywords))
                    success_rates.append((s_count / len(res_list)) * 100)

                norms = [d["vector_norm"] for d in subset]

                ax.plot(layers, success_rates, color='#3498db', marker='o', label='Success %')
                ax.set_ylabel("Success Rate (%)", color='#3498db')
                ax.set_ylim(-5, 105)

                ax2 = ax.twinx()
                ax2.plot(layers, norms, color='#e67e22', linestyle=':', marker='x', label='Vector Norm')
                ax2.set_ylabel("Steering Vector Norm", color='#e67e22')

                ax.set_title(f"Transfer: {src} to {tgt} ({comp})")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(OUTPUT_DIR, f"transformation_{model_name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")
    
def get_success_rate(entry, target_key):
    results = entry.get("results", [])
    keywords = EMOTION_KEYWORD_MAP.get(target_key, [target_key])
    if not results: return 0
    match_count = sum(1 for res in results if any(kw in res["steered"].lower() for kw in keywords))
    return (match_count / len(results)) * 100

# ============================================================
# FIXED SUMMARY 1: Blank Hallucination (Face vs Emotion Match)
# ============================================================
def plot_hallucination_comparison_summary():
    print("--- Generating Model Comparison: Face vs Emotion Hallucination ---")
    data_map = load_json_files("exp_blank_hallucination")
    if not data_map: return

    emotions = ["happy", "sad", "angry"]
    
    fig, axes = plt.subplots(1, len(emotions), figsize=(20, 7), sharey=True)
    if len(emotions) == 1: axes = [axes] # Handle single emotion case
    
    fig.suptitle("Hallucination Sensitivity Comparison\n(Solid: Emotion Match | Dashed: Any Face Detection)", fontsize=16)

    for i, emo in enumerate(emotions):
        ax = axes[i]
        has_data = False
        
        for model_name, entries in data_map.items():
            # 1. Filter entries for this specific emotion
            emo_entries = [e for e in entries if e.get("target_emotion") == emo]
            if not emo_entries: continue
            
            has_data = True
            
            # 2. Get unique layers
            unique_layers = sorted(list(set(e["layer"] for e in emo_entries)))
            
            avg_emotion_match = []
            avg_any_face = []
            
            for layer in unique_layers:
                # 3. Aggregate all Alphas for this specific layer
                layer_entries = [e for e in emo_entries if e["layer"] == layer]
                
                em_rates = [] 
                af_rates = [] 
                
                for le in layer_entries:
                    m_rate, f_only_rate, _ = analyze_hallucination_results(le)
                    em_rates.append(m_rate)
                    af_rates.append(m_rate + f_only_rate) 
                
                avg_emotion_match.append(np.mean(em_rates))
                avg_any_face.append(np.mean(af_rates))
            
            # Dynamic Labeling (includes model name to differentiate multiple 2B/7B models)
            model_clean_name = model_name.replace("_Instruct", "").replace("Qwen_", "").replace("2-VL-", "")
            
            # Color logic
            if "7B" in model_name:
                base_color = "#1f77b4" # Blue
            elif "2B" in model_name:
                base_color = "#ff7f0e" # Orange
            else:
                base_color = "#2ca02c" # Green
            
            ax.plot(unique_layers, avg_emotion_match, marker='o', label=f"{model_clean_name} - Emo", 
                    color=base_color, linestyle='-', linewidth=2)
            
            ax.plot(unique_layers, avg_any_face, marker='x', label=f"{model_clean_name} - Face", 
                    color=base_color, linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_title(f"Targeting: {emo.upper()}")
        ax.set_xlabel("Layer Index")
        if i == 0: ax.set_ylabel("Avg Rate (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-5, 105)
        if has_data:
            ax.legend(fontsize='small', loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "comparison_hallucination_face_vs_emotion.png")
    plt.savefig(out_path)
    print(f"Saved Summary: {out_path}")
    plt.close()

# ============================================================
# SUMMARY 2: Emotion Transformation 
# ============================================================
def plot_transformation_comparison_summary():
    print("--- Generating Model Comparison: Emotion Transfer ---")
    data_map = load_json_files("exp_emotion_transfer")
    if not data_map: return

    # Identify all transfer pairs present in the data
    all_pairs = []
    for m_entries in data_map.values():
        all_pairs.extend([f"{e['source']}->{e['target']}" for e in m_entries if 'source' in e])
    pairs = sorted(list(set(all_pairs)))
    
    if not pairs: return

    fig, axes = plt.subplots(1, len(pairs), figsize=(20, 6), sharey=True)
    if len(pairs) == 1: axes = [axes]

    fig.suptitle("Emotion Transfer Performance Comparison", fontsize=16)

    for i, pair in enumerate(pairs):
        ax = axes[i]
        src, tgt = pair.split("->")
        has_data = False

        for model_name, entries in data_map.items():
            pair_entries = [e for e in entries if e.get("source") == src and e.get("target") == tgt]
            if not pair_entries: continue
            
            has_data = True
            unique_layers = sorted(list(set(e["layer"] for e in pair_entries)))
            avg_success = []
            
            for layer in unique_layers:
                layer_points = [get_success_rate(e, tgt) for e in pair_entries if e["layer"] == layer]
                avg_success.append(np.mean(layer_points))
            
            # Dynamic Labeling
            model_clean_name = model_name.replace("_Instruct", "").replace("Qwen_", "").replace("2-VL-", "")
            ax.plot(unique_layers, avg_success, marker='s', label=model_clean_name, linewidth=2)

        ax.set_title(f"Transfer: {pair.upper()}")
        ax.set_xlabel("Layer Index")
        if i == 0: ax.set_ylabel("Avg Success Rate (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if has_data:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "comparison_transformation_summary.png")
    plt.savefig(out_path)
    print(f"Saved Summary: {out_path}")
    plt.close()

if __name__ == "__main__":
    # 1. Run Individual Model Analyses
    plot_hallucination_experiment()
    plot_transformation_experiment()

    # 2. Run Comparative Summaries
    plot_hallucination_comparison_summary()
    plot_transformation_comparison_summary()
    
    print("\n[Done] All visualizations generated in 'graphs/' folder.")