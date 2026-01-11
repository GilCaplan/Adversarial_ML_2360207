
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import re

def generate_steering_graphs(filename="emotion_steering_results.json"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    with open(filename, "r") as f:
        try:
            full_data = json.load(f)
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            return

    # --- 1. SEPARATE DATA STREAMS ---
    # Standard: Key is an int (e.g., 1, 3, 5)
    standard_data = [row for row in full_data if isinstance(row[0], int)]
    
    # Blank/New: Key is a string (e.g., "blank_happy_1")
    blank_data = [row for row in full_data if isinstance(row[0], str) and row[0].startswith("blank_")]

    # --- 2. PLOT STANDARD EXPERIMENTS (If data exists) ---
    if standard_data:
        print(f"Found {len(standard_data)} standard data points. Generating standard graph...")
        plot_standard_data(standard_data)
    else:
        print("No standard integer-key data found.")

    # --- 3. PLOT BLANK IMAGE EXPERIMENTS (If data exists) ---
    if blank_data:
        print(f"Found {len(blank_data)} blank image data points. Generating hallucination graph...")
        plot_blank_data(blank_data)
    else:
        print("No blank image data found.")

def plot_standard_data(data):
    """Handles the original plotting logic for standard FER images."""
    llm_data = [row for row in data if row[1] == 'LLM']
    vision_data = [row for row in data if row[1] == 'Vision']

    # Extract Data
    llm_layers = sorted([row[0] for row in llm_data])
    llm_steering_cnt = [row[2][3] for row in sorted(llm_data, key=lambda x: x[0])]
    
    vis_layers = sorted([row[0] for row in vision_data])
    vis_steering_cnt = [row[2][3] for row in sorted(vision_data, key=lambda x: x[0])]

    # Calculate Baselines
    llm_happy_counts = [row[2][2] for row in llm_data]
    vis_happy_counts = [row[2][2] for row in vision_data]
    
    # Safety check for empty lists
    llm_mean = np.mean(llm_happy_counts) if llm_happy_counts else 0
    vis_mean = np.mean(vis_happy_counts) if vis_happy_counts else 0
    llm_std = np.std(llm_happy_counts) if llm_happy_counts else 0
    vis_std = np.std(vis_happy_counts) if vis_happy_counts else 0

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)

    ax1.plot(llm_layers, llm_steering_cnt, marker='o', label='LLM Layers', color='blue')
    ax1.plot(vis_layers, vis_steering_cnt, marker='s', linestyle='--', label='Vision Layers', color='orange')
    ax1.set_title('Standard Steering: Response Changes on Real Images')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Count of Changed Responses')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    labels = ['LLM Backbone', 'Vision Encoder']
    means = [llm_mean, vis_mean]
    stds = [llm_std, vis_std]
    colors = ['green', 'red']

    bars = ax2.bar(labels, means, yerr=stds, color=colors, capsize=10, alpha=0.8)
    ax2.set_title('Baseline Model Performance (No Steering)')
    ax2.set_ylabel('Average Happy Count')
    
    output_img = "emotion_steering_standard.png"
    plt.savefig(output_img)
    print(f"-> Saved: {output_img}")
    plt.close(fig)

def plot_blank_data(data):
    """Handles the new logic for Blank/Hallucination experiments."""
    # Structure to hold parsed data: processed[emotion][type] = [(layer, count), ...]
    processed = {} 
    emotions_found = set()

    for row in data:
        key, type_, result_tuple = row
        # Parse key: "blank_happy_5" -> emotion="happy", layer=5
        match = re.search(r"blank_([a-zA-Z]+)_(\d+)", key)
        if match:
            emotion = match.group(1)
            layer = int(match.group(2))
            count = result_tuple[3] # Index 3 is cnt_steering (changes/hallucinations)
            
            if emotion not in processed: processed[emotion] = {'LLM': [], 'Vision': []}
            processed[emotion][type_].append((layer, count))
            emotions_found.add(emotion)

    # Create subplots: One subplot per Emotion found
    emotions_list = sorted(list(emotions_found))
    num_plots = len(emotions_list)
    
    if num_plots == 0: return

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))
    if num_plots == 1: axes = [axes] # Ensure iterable if only 1 plot
    
    plt.subplots_adjust(hspace=0.4)

    for i, emotion in enumerate(emotions_list):
        ax = axes[i]
        
        # Get LLM data
        llm_points = sorted(processed[emotion]['LLM'], key=lambda x: x[0])
        llm_x = [p[0] for p in llm_points]
        llm_y = [p[1] for p in llm_points]
        
        # Get Vision data
        vis_points = sorted(processed[emotion]['Vision'], key=lambda x: x[0])
        vis_x = [p[0] for p in vis_points]
        vis_y = [p[1] for p in vis_points]

        # Plot
        ax.plot(llm_x, llm_y, marker='o', label='LLM Injection', color='purple')
        ax.plot(vis_x, vis_y, marker='s', linestyle='--', label='Vision Injection', color='teal')
        
        ax.set_title(f'Hallucination Rate: Injecting "{emotion.upper()}" into Blank Images')
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Hallucinations (Changed Captions)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    output_img = "emotion_steering_hallucination.png"
    plt.savefig(output_img)
    print(f"-> Saved: {output_img}")
    plt.close(fig)