import os
import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from Load_fer_dataset import load_fer_data
from Load_VLM import load_qwen_model, get_vlm_response
from VLM_manipulation import get_layer_representation, generate_with_vector_insertion

def load_file_image(info, experiment_name="blank"):
    filename = f"json_results/emotion_steering_results_{experiment_name}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(info)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def steering_images_by_layer(model, processor, images, steering_images, neutral, alpha=1, prompt="Caption the image, if pure noise say so as the caption, otherwise if you see an expression, provide a caption", emotion="happy", llm_use=True, layer_n=10, batch=10, num=5):
    responses, steered_responses = [], []
    system_prompt = prompt

    # Iterate through images in batches
    for i in range(0, min(len(steering_images), num * batch), batch):
        # --- Handle Test Image (Path vs Object) ---
        if isinstance(images, list):
            img_input = images[i]
        else:
            img_input = images
        if isinstance(img_input, str):
            test_img = Image.open(img_input).convert("RGB").resize((336, 336))
        else:
            test_img = img_input.convert("RGB").resize((336, 336))
        
        # --- Create Steering Vector ---
        vectors = []
        for j in range(batch):
            idx = (i + j) % len(steering_images)
            s_input = steering_images[idx]
            
            # Handle Steering Image (Path vs Object)
            if isinstance(s_input, str):
                s_img = Image.open(s_input).convert("RGB").resize((336, 336))
            else:
                s_img = s_input.convert("RGB").resize((336, 336))
            
            # Get representation
            n_img = Image.open(neutral[idx]).convert("RGB").resize((336, 336))
            vec = get_layer_representation(model, processor, s_img, layer_n, LLM_use=llm_use) - get_layer_representation(model, processor, n_img, layer_n, LLM_use=llm_use)
            vectors.append(vec)
        
        # Stack and Average (Fixes the list error)
        steering_vector = torch.stack(vectors).mean(dim=0)

        # --- Generate Responses ---
        # 1. Without Steering
        response_wo = get_vlm_response(model, processor, test_img, system_prompt)
        
        # 2. With Steering
        response_w = generate_with_vector_insertion(
            model, processor, test_img, layer_n, steering_vector, 
            alpha=alpha, prompt=system_prompt, LLM_use=llm_use
        )
        
        responses.append(response_wo)
        steered_responses.append(response_w)
            
    return responses, steered_responses

def steering_blank_images_emotion():
    print("--- Loading Data ---")
    happy, sad, neutral, angry, _, _, _ = load_fer_data()
    print(f"Loaded {len(happy)} happy images.")

    print("\n--- Loading Model ---")
    model, processor = load_qwen_model()
    
    print(f"\n--- Running Emotion Manipulation on Blank images ---")
    results = {}
    
    # Define emotions to steer with
    emotions_map = {'happy': happy, 'sad': sad, 'angry': angry}

    for emotion, steer_imgs in emotions_map.items():
        results[emotion] = []
        
        width, height = 640, 640
        base_array = np.full((height, width, 3), 255, dtype=np.float32)
        noise = np.random.normal(0, 2, (height, width, 3))
        noisy_array = base_array - np.abs(noise)
        noisy_img = np.clip(noisy_array, 0, 255).astype(np.uint8)
        blank_image_pil = Image.fromarray(noisy_img)
        print(f"\nGenerated blank image for steering with '{emotion}'.")

        # prompt = "Describe the image. If you see any specific emotion or object, describe it clearly."

        # --- LOOP 1: LLM LAYERS ---
        print(f"  -> Running LLM steering...")
        for layer in tqdm(range(1, model.config.num_hidden_layers, 2), desc=f"LLM {emotion}"):
            res = steering_images_by_layer(
                model, processor, blank_image_pil, steer_imgs, neutral,
                alpha=5.0, emotion=emotion, 
                llm_use=True,   # <--- Correct: True for LLM
                layer_n=layer, batch=20
            )
            # Save with unique key for blanks
            load_file_image([f"blank_{emotion}_{layer}", 'LLM', res])
            results[emotion].append(res)
        
        # --- LOOP 2: VISION LAYERS ---
        print(f"  -> Running Vision steering...")
        for layer in tqdm(range(1, model.config.vision_config.depth, 2), desc=f"Vision {emotion}"):
            res = steering_images_by_layer(
                model, processor, blank_image_pil, steer_imgs, neutral,
                alpha=5.0, emotion=emotion, 
                llm_use=False,  # <--- Correct: False for Vision
                layer_n=layer, batch=20
            )
            load_file_image([f"blank_{emotion}_{layer}", 'Vision', res])
            results[emotion].append(res)

def steering_images_emotion():
    print("--- Loading Data ---")
    happy, sad, _, _, _, _, _ = load_fer_data()
    print(f"Loaded {len(happy)} happy images.")

    print("\n--- Loading Model ---")
    model, processor = load_qwen_model()
    # print(f"LLM Layers: {model.config.num_hidden_layers}") #28
    # print(f"Vision Layers: {model.config.vision_config.depth}") #32
    print(f"\n--- Running Emotion Manipulation PoC - LLM ---")
    for layer in tqdm(range(1,model.config.num_hidden_layers,2)):
        result = steering_images_by_layer(model, processor, happy, sad, alpha=5.0, layer_n=layer)
        load_file_image([layer, 'LLM', result])
    print(f"\n--- Running Emotion Manipulation PoC - Vision ---")
    for layer in tqdm(range(1,model.config.vision_config.depth,2)):
        result = steering_images_by_layer(model, processor, happy, sad, alpha=5.0, llm_use=False, layer_n=layer)
        load_file_image([layer, 'Vision', result])


if __name__ == "__main__":
    steering_blank_images_emotion()
