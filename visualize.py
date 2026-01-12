import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np

RESULTS_DIR = "json_results"
OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json_files(experiment_prefix):
    """
    Finds all JSON files matching the experiment prefix (e.g., 'exp_blank_hallucination_*')
    Returns a dictionary: { "Model_Name": [data_entries] }
    """
    files = glob.glob(os.path.join(RESULTS_DIR, f"{experiment_prefix}*.json"))
    data_map = {}
    
    for fpath in files:
        # Extract model name from filename (e.g., exp_blank_hallucination_qwen_2B.json -> qwen_2B)
        filename = os.path.basename(fpath)
        model_name = filename.replace(experiment_prefix + "_", "").replace(".json", "")
        
        try:
            with open(fpath, "r") as f:
                data_map[model_name] = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping corrupted file: {fpath}")
            
    return data_map

def calculate_success_rate(entry, keyword_check=True):
    """
    Calculates success rate for a single data entry.
    If keyword_check is True, checks if target emotion is in the text.
    """
    results = entry["results"]
    target = entry.get("target_emotion") or entry.get("target") # Handle key difference
    count = 0
    total = len(results)
    
    if total == 0: return 0

    for res in results:
        text = res["steered"].lower()
        if keyword_check and target:
            # Success if the target emotion appears in the description
            if target.lower() in text:
                count += 1
        else:
            # Fallback: Success if it changed significantly from baseline (simple length check or exact match)
            if res["steered"] != res["baseline"]:
                count += 1
                
    return (count / total) * 100

# ==========================================
# 1. Visualize Hallucination (Blank Images)
# ==========================================
def plot_hallucination_experiment():
    print("--- Generating Hallucination Graphs ---")
    data_map = load_json_files("exp_blank_hallucination")
    
    if not data_map:
        print("No hallucination results found.")
        return

    for model_name, entries in data_map.items():
        # Organization: entries -> [ {layer, component, target_emotion, alpha, results...} ]
        
        # We want to plot: X=Layer, Y=SuccessRate, Hue=Emotion (One plot per Component & Alpha?)
        # Let's simplify: One Figure per Model. 
        # Rows = Emotions, Cols = Component (LLM vs Vision). Lines = Alphas.
        
        emotions = sorted(list(set(e["target_emotion"] for e in entries)))
        components = ["LLM", "Vision"]
        alphas = sorted(list(set(e["alpha"] for e in entries)))
        
        fig, axes = plt.subplots(len(emotions), 2, figsize=(15, 5 * len(emotions)), squeeze=False)
        fig.suptitle(f"Hallucination Success Rate - {model_name}", fontsize=16)

        for row_idx, emo in enumerate(emotions):
            for col_idx, comp in enumerate(components):
                ax = axes[row_idx, col_idx]
                
                # Filter data for this subplot
                subset = [e for e in entries if e["target_emotion"] == emo and e["component"] == comp]
                
                if not subset:
                    ax.text(0.5, 0.5, "No Data", ha='center')
                    continue

                # Plot a line for each Alpha value
                for alpha in alphas:
                    alpha_data = sorted([s for s in subset if s["alpha"] == alpha], key=lambda x: x["layer"])
                    if not alpha_data: continue
                    
                    x = [d["layer"] for d in alpha_data]
                    y = [calculate_success_rate(d, keyword_check=True) for d in alpha_data]
                    
                    ax.plot(x, y, marker='o', label=f"Alpha {alpha}")

                ax.set_title(f"Injecting '{emo}' into {comp}")
                ax.set_xlabel("Layer Index")
                ax.set_ylabel("Success Rate (%)")
                ax.set_ylim(-5, 105)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(OUTPUT_DIR, f"hallucination_{model_name}.png")
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close()

# ==========================================
# 2. Visualize Transformation (Emotion Transfer)
# ==========================================
def plot_transformation_experiment():
    print("--- Generating Transformation Graphs ---")
    data_map = load_json_files("exp_emotion_transfer")
    
    if not data_map:
        print("No transformation results found.")
        return

    for model_name, entries in data_map.items():
        # Get unique source->target pairs
        pairs = sorted(list(set(f"{e['source']}->{e['target']}" for e in entries)))
        components = ["LLM", "Vision"]
        
        fig, axes = plt.subplots(len(pairs), 2, figsize=(15, 5 * len(pairs)), squeeze=False)
        fig.suptitle(f"Emotion Transfer Success Rate - {model_name}", fontsize=16)
        
        for row_idx, pair in enumerate(pairs):
            src, tgt = pair.split("->")
            
            for col_idx, comp in enumerate(components):
                ax = axes[row_idx, col_idx]
                
                # Filter data
                subset = [
                    e for e in entries 
                    if e["source"] == src and e["target"] == tgt and e["component"] == comp
                ]
                subset.sort(key=lambda x: x["layer"])
                
                if not subset:
                    ax.text(0.5, 0.5, "No Data", ha='center')
                    continue

                # Prepare Plot Data
                layers = [d["layer"] for d in subset]
                success_rates = [calculate_success_rate(d, keyword_check=True) for d in subset]
                norms = [d["vector_norm"] for d in subset]

                # Double axis: Left=Success Rate, Right=Vector Norm
                ax.plot(layers, success_rates, color='tab:blue', marker='o', label='Success %', linewidth=2)
                ax.set_xlabel("Layer Index")
                ax.set_ylabel("Success Rate (%)", color='tab:blue')
                ax.tick_params(axis='y', labelcolor='tab:blue')
                ax.set_ylim(-5, 105)
                ax.grid(True, linestyle="--", alpha=0.5)

                # Twin axis for Vector Norm (to see if strength correlates with success)
                ax2 = ax.twinx()
                ax2.plot(layers, norms, color='tab:red', linestyle=':', marker='x', label='Vector Norm', alpha=0.7)
                ax2.set_ylabel("Steering Vector Norm", color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')

                ax.set_title(f"{pair} ({comp})")
                
                # Combined legend
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(OUTPUT_DIR, f"transformation_{model_name}.png")
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close()

if __name__ == "__main__":
    plot_hallucination_experiment()
    plot_transformation_experiment()
    print("\nDone! Check the 'graphs' folder.")