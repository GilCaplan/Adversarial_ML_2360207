from Load_fer_dataset import load_fer_data
from Load_VLM import load_qwen_model, get_vlm_response
from PIL import Image
from VLM_manipulation import get_layer_representation, generate_with_vector_insertion
import re
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

def generate_steering_graphs(filename="emotion_steering_results.json"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    with open(filename, "r") as f:
        data = json.load(f)

    # JSON structure: [layer_num, type_str, [responses, steered_responses, cnt_emotion, cnt_steering]]
    llm_data = sorted([row for row in data if row[1] == 'LLM'], key=lambda x: x[0])
    vision_data = sorted([row for row in data if row[1] == 'Vision'], key=lambda x: x[0])

    # Extract X (Layers) and Y (Counts)
    llm_layers = [row[0] for row in llm_data]
    
    # FIX: Access integers directly (removed len())
    # row[2][2] is cnt_emotion (Happy count in your setup)
    # row[2][3] is cnt_steering (Changed count)
    
    llm_steering_cnt = [row[2][3] for row in llm_data]
    llm_happy_cnt = [row[2][2] for row in llm_data]

    vis_layers = [row[0] for row in vision_data]
    vis_steering_cnt = [row[2][3] for row in vision_data]
    vis_happy_cnt = [row[2][2] for row in vision_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    ax1.plot(llm_layers, llm_steering_cnt, marker='o', linestyle='-', label='LLM Layers', color='blue')
    ax1.plot(vis_layers, vis_steering_cnt, marker='s', linestyle='--', label='Vision Layers', color='orange')
    ax1.set_title('Steering Effectiveness: Number of Responses Changed')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Count of Changed Responses')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(llm_layers, llm_happy_cnt, marker='o', linestyle='-', label='LLM Layers', color='green')
    ax2.plot(vis_layers, vis_happy_cnt, marker='s', linestyle='--', label='Vision Layers', color='red')
    ax2.set_title('Number of Responses "Happy" - No Steering')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Count of "Happy" Labels')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Save and Show
    output_img = "emotion_steering_analysis.png"
    plt.savefig(output_img)
    print(f"Graphs saved to {output_img}")
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
        "happiness", "joy", "joyful", "smile", "smiling", "laugh", 
        "laughing", "laughter", "amused", "amusement", "cheerful", 
        "excited", "excitement", "enthusiastic"
    ]
    if any(word in text for word in happy_keywords):
        return "happy"

    # 2. SAD (Check second)
    sad_keywords = ["sad", "distressed", "upset", "unhappy", "sorrowful", "dejected",
                     "downcast", "mournful", "gloomy", "somber", "crying", "tears"]
    if any(word in text for word in sad_keywords):
        return "sad"

    # 3. NEUTRAL (Check last)
    neutral_keywords = [
        "neutral", "calm", "composed", "serious", "contemplative", 
        "stern", "serene", "straight"
    ]
    if any(word in text for word in neutral_keywords):
        return "neutral"

    return None

def steering_images_by_layer(model, processor, images, steering_images, alpha=1, emotion="happy", llm_use=True, layer_n=10):
    responses, steered_responses = [], []
    cnt_steering, cnt_emotion = 0,0
    system_prompt = "Caption the image, focusing on the facial expression" 
    for i in range(1, min(len(images), 100),2):
            img = Image.open(images[i]).convert("RGB").resize((336, 336))
            # system_prompt = "Classify the facial expression given the image input as 'happy' or 'neutral' or 'sad' or 'unsure', response should be exactly one word " 
            response_wo = get_vlm_response(model, processor, img, system_prompt)

            steer_img = Image.open(steering_images[i]).convert("RGB").resize((336, 336))
            steering_vector = get_layer_representation(model, processor, steer_img, layer_n, LLM_use=llm_use) #- get_layer_representation(model, processor, neutral_img, 20) 
            response_w = generate_with_vector_insertion(model, processor, img, layer_n, steering_vector, alpha=alpha, prompt=system_prompt, LLM_use=llm_use)

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