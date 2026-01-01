from Load_fer_dataset import load_fer_data
from Load_VLM import load_qwen_model, get_vlm_response
from PIL import Image
from VLM_manipulation import get_layer_representation, generate_with_vector_insertion
import re


def get_label(text):
    match = re.search(r"(happy|neutral|sad)", text.lower())
    return match.group(0) if match else None

if __name__ == "__main__":
    print("--- 1. Loading Data ---")
    happy, sad, neutral, _, _, _, _ = load_fer_data()
    print(f"Loaded {len(happy)} happy images.")

    print("\n--- 2. Loading Model ---")
    model, processor = load_qwen_model()

    print("\n--- 3. Running Emotion Manipulation PoC ---")
    cnt_steering, cnt_happy = 0,0
    for i in range(1, min(len(happy), 100), 5):
            img = Image.open(happy[i]).convert("RGB").resize((336, 336))
        
            system_prompt = "Classify the facial expression given the image input as 'happy' or 'neutral' or 'sad' or 'unsure', response should be exactly one word." #+ ", first check if the image is happy, verify and respond accordingly"
            
            print(f"\n--- 4. Testing on Image Index: {i} ---")
            response_wo = get_vlm_response(model, processor, img, system_prompt)

            sad_img = Image.open(sad[i]).convert("RGB").resize((336, 336))
            neutral_img = Image.open(neutral[i]).convert("RGB").resize((336, 336))
            print(f"VLM Response w/o steering: {response_wo}")
            steering_vector = get_layer_representation(model, processor, sad_img, 10) #- get_layer_representation(model, processor, neutral_img, 20) 
            response_w = generate_with_vector_insertion(model, processor, img, 10, steering_vector, alpha=1, prompt=system_prompt)

            print(f"Image {i} VLM Response w steering: {response_w}")
            
            if get_label(response_w) != get_label(response_wo):
                cnt_steering += 1
            if get_label(response_w) == "happy":
                cnt_happy += 1

    print(f"number of changed responses: {cnt_steering} out of {min(len(happy), 100)//5}")
    print(f"number of 'happy' responses after steering: {cnt_happy} out of {min(len(happy), 100)//5}")
            
