import json
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
JSON_PATH = "/Users/USER/Desktop/University/Semester 7/CS Adverserial ML/Project/json_results/emotion_steering_results_blank.json"
OUTPUT_DIR = "/Users/USER/Desktop/University/Semester 7/CS Adverserial ML/Project/json_results/plots"

SUCCESS_KEYWORDS = [
    # Positive
    "face", "smile", "smiling", "grin", "happy", "joy", "laugh", "laughter",
    "cheerful", "positive", "excited", "toothy",
    
    # Negative / Sad
    "sad", "crying", "tear", "unhappy", "grief", "depressed", "sobbing",
    "distressed", "contemplation", "introspection", "concerned",
    
    # Aggressive / Angry
    "angry", "frown", "upset", "mad", "annoyed", "frustration", "stern",
    "serious", "screaming", "shouting", "yelling", "shocked", "pain",
    
    # General Human/Portrait terms (Strong indicators of steering success on noise)
    "person", "man", "woman", "human", "portrait", "close-up",
    "expression", "eyes", "mouth", "teeth", "looking"
]

FAILURE_KEYWORDS = [
    # Direct denials
    "pure noise", "no discernible", "nothing", "blank canvas", "empty", "void",
    
    # Abstract/Blur descriptions (Common hallucinations on noise)
    "abstract", "blur", "blurred", "blurry", "indistinct", "ethereal",
    "monochromatic", "grayscale", "gradient", "faint", "imperceptible",
    
    # Texture descriptions
    "texture", "pattern", "dots", "mesh", "grain", "grainy", 
    "dust", "powder", "metallic", "silver", "paper", "fabric", "surface",
    
    # Geometric/Simple descriptions
    "white background", "black background", "square", "rectangle", 
    "line", "lines", "shape", "geometric", "simple design", "minimalist"
]

def check_success(text):
    """
    Returns True if the text describes a face/emotion, False if it describes noise/background.
    """
    text = text.lower()
    
    # Check if any success keyword is present
    for keyword in SUCCESS_KEYWORDS:
        if keyword in text:
            return True
            
    return False

def generate_graphs():
    # 1. Load Data
    if not os.path.exists(JSON_PATH):
        print(f"Error: File not found at {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    # 2. Process Data
    # Structure: processed_data[emotion][model_type][layer] = count
    processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for entry in data:
        key = entry[0]        # e.g., "blank_happy_1"
        model_type = entry[1] # "LLM" or "Vision"
        responses = entry[2]  # List of [baseline, steered] lists
        
        # Parse Key
        match = re.search(r"blank_([a-z]+)_(\d+)", key)
        if not match:
            continue
            
        emotion = match.group(1)
        layer = int(match.group(2))
        
        # Calculate Score (Count of successful steerings)
        # We look at the "steered" response (index 1 in the inner lists)
        steered_texts = [pair[1] for pair in responses]
        success_count = sum(1 for text in steered_texts if check_success(text))
        
        processed_data[emotion][model_type][layer] = success_count

    # 3. Generate Plots
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for emotion, models in processed_data.items():
        plt.figure(figsize=(10, 5))
        
        # Plot LLM
        if "LLM" in models:
            # Sort layers numerically
            layers = sorted(models["LLM"].keys())
            counts = [models["LLM"][l] for l in layers]
            plt.plot(layers, counts, marker='o', label='LLM Layers', color='blue', linewidth=2)

        # Plot Vision
        if "Vision" in models:
            # Sort layers numerically
            layers = sorted(models["Vision"].keys())
            counts = [models["Vision"][l] for l in layers]
            plt.plot(layers, counts, marker='s', linestyle='--', label='Vision Layers', color='orange', linewidth=2)

        # Formatting
        plt.title(f"Steering Effectiveness: '{emotion.capitalize()}' Injection on Noise", fontsize=14)
        plt.xlabel("Layer Number", fontsize=12)
        plt.ylabel("Count of Responses with Hallucinated Face/Emotion", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Ensure integer ticks for Y axis
        plt.yticks(range(0, max(max(counts if 'Vision' in models else [0]), max(counts if 'LLM' in models else [0])) + 2))

        # Save
        output_path = os.path.join(OUTPUT_DIR, f"steering_graph_{emotion}.png")
        plt.savefig(output_path)
        print(f"Graph saved: {output_path}")
        plt.close()

if __name__ == "__main__":
    generate_graphs()