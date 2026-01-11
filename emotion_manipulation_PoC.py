from Load_fer_dataset import load_fer_data
from Load_VLM import load_qwen_model, get_vlm_response
from PIL import Image
from VLM_manipulation import get_layer_representation, generate_with_vector_insertion
import re
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import torch
import numpy as np

def generate_steering_graphs(filename="emotion_steering_results.json"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    with open(filename, "r") as f:
        data = json.load(f)

    # Filter data
    llm_data = [row for row in data if row[1] == 'LLM']
    vision_data = [row for row in data if row[1] == 'Vision']

    # --- TOP GRAPH DATA (Steering) ---
    llm_layers = sorted([row[0] for row in llm_data])
    # Re-sort steering counts to match sorted layers
    llm_steering_cnt = [row[2][3] for row in sorted(llm_data, key=lambda x: x[0])]
    
    vis_layers = sorted([row[0] for row in vision_data])
    vis_steering_cnt = [row[2][3] for row in sorted(vision_data, key=lambda x: x[0])]

    # --- BOTTOM GRAPH DATA (Baseline Averages) ---
    llm_happy_counts = [row[2][2] for row in llm_data]
    vis_happy_counts = [row[2][2] for row in vision_data]

    llm_mean, llm_std = np.mean(llm_happy_counts), np.std(llm_happy_counts)
    vis_mean, vis_std = np.mean(vis_happy_counts), np.std(vis_happy_counts)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)

    # Subplot 1: Steering Effectiveness (Line Plot)
    ax1.plot(llm_layers, llm_steering_cnt, marker='o', label='LLM Layers', color='blue')
    ax1.plot(vis_layers, vis_steering_cnt, marker='s', linestyle='--', label='Vision Layers', color='orange')
    ax1.set_title('Steering Effectiveness: Number of Responses Changed')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Count of Changed Responses')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Baseline Comparison (Bar Plot)
    labels = ['LLM Backbone', 'Vision Encoder']
    means = [llm_mean, vis_mean]
    stds = [llm_std, vis_std]
    colors = ['green', 'red']

    bars = ax2.bar(labels, means, yerr=stds, color=colors, capsize=10, alpha=0.8)
    ax2.set_title('Baseline Model Performance: Average "Happy" Labels (No Steering)')
    ax2.set_ylabel('Average Count')
    
    # Add text labels on top of bars for clarity
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    # Save and Show
    output_img = "emotion_steering_analysis_refined.png"
    plt.savefig(output_img)
    print(f"Refined graphs saved to {output_img}")
    plt.show()


def load_file_image(info):
    filename = "emotion_steering_results.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(info)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def get_label(text):
    text = text.lower()
    
    # 1. HAPPY (Check first)
    happy_keywords = [
        "happy", "happiness", "joy", "joyful", "joyous", "delighted", "pleased",
        "content", "contented", "satisfied", "elated", "cheery",
        "smile", "smiling", "grin", "grinning", "beaming","bright expression", "warm expression",
        "laugh", "laughing", "amused", "amusement","cheerful", "playful", "lighthearted",
        "excited", "excitement", "enthusiastic", "energetic", "lively","friendly", "approachable", "positive mood", "in good spirits"
    ]
    if any(word in text for word in happy_keywords):
        return "happy"

    sad_keywords = [
        # affect
        "sad", "unhappy", "sorrowful", "melancholy", "depressed",
        "heartbroken", "downhearted",

        # facial / expression
        "frown", "frowning", "tearful", "crying", "tears", "pained", "troubled", "anguished", "woeful", "forlorn",
        "distressed", "upset", "dejected", "downcast","mournful", "gloomy", "somber", "despondent",
        "slumped", "drooping", "lowered gaze", "averted gaze","sad", "looks upset", "appears unhappy"
    ]
    if any(word in text for word in sad_keywords):
        return "sad"

    # 3. NEUTRAL (Check last)
    neutral_keywords = [
        "neutral", "emotionless", "impassive", "blank", "flat expression", "unemotional", "indifferent", "detached",
        "unaffected", "unmoved", "dispassionate", "apathetic","calm", "composed", "stoic", "reserved","controlled", "matter-of-fact",
        "straight face", "no visible emotion","expressionless", "still expression",
        "serious", "focused", "contemplative","observant", "attentive","appears neutral", "neither happy nor sad"
    ]
    if any(word in text for word in neutral_keywords):
        return "neutral"

    return None

def steering_images_by_layer(model, processor, images, steering_images, alpha=1, emotion="happy", llm_use=True, layer_n=10, batch=1):
    responses, steered_responses = [], []
    cnt_steering, cnt_emotion = 0, 0
    system_prompt = "Caption the image, focusing on the facial expression" 

    for i in range(0, min(len(images), 20 * batch), batch):

        test_img = Image.open(images[i]).convert("RGB").resize((336, 336))
        
        vectors = []
        for j in range(batch):
            idx = (i + j) % len(steering_images)
            s_img = Image.open(steering_images[idx]).convert("RGB").resize((336, 336))
            
            vec = get_layer_representation(model, processor, s_img, layer_n, LLM_use=llm_use)
            vectors.append(vec)
        
        steering_vector = torch.stack(vectors).mean(dim=0)

        response_wo = get_vlm_response(model, processor, test_img, system_prompt)
        response_w = generate_with_vector_insertion(
            model, processor, test_img, layer_n, steering_vector, 
            alpha=alpha, prompt=system_prompt, LLM_use=llm_use
        )
        responses.append(response_wo)
        steered_responses.append(response_w)
        
        if get_label(response_w) != get_label(response_wo):
            cnt_steering += 1
        if get_label(response_w) == emotion:
            cnt_emotion += 1
        # print(f"Image {i}: Without Steering: '{response_wo}' | With Steering: '{response_w}'")
    return responses, steered_responses, cnt_emotion, cnt_steering
            
def steering_images():
    print("--- Loading Data ---")
    happy, sad, neutral, _, _, _, _ = load_fer_data()
    print(f"Loaded {len(happy)} happy images.")

    print("\n--- Loading Model ---")
    model, processor = load_qwen_model()
    # print(f"LLM Layers: {model.config.num_hidden_layers}") #28
    # print(f"Vision Layers: {model.config.vision_config.depth}") #32
    print(f"\n--- Running Emotion Manipulation PoC - LLM ---")
    for layer in tqdm(range(1,model.config.num_hidden_layers,2)):
        result = steering_images_by_layer(model, processor, happy, sad, alpha=1.0, layer_n=layer)
        load_file_image([layer, 'LLM', result])
    print(f"\n--- Running Emotion Manipulation PoC - Vision ---")
    for layer in tqdm(range(1,model.config.vision_config.depth,2)):
        result = steering_images_by_layer(model, processor, happy, sad, alpha=1.0, llm_use=False, layer_n=layer)
        load_file_image([layer, 'Vision', result])

if __name__ == "__main__":
    steering_images()
    generate_steering_graphs("emotion_steering_results.json")