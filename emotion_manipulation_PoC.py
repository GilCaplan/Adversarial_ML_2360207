from Load_fer_dataset import load_fer_data
from Load_VLM import load_qwen_model, get_vlm_response
from PIL import Image
from VLM_manipulation import get_layer_representation, generate_with_vector_insertion
if __name__ == "__main__":
    print("--- 1. Loading Data ---")
    happy, sad, neutral, _, _, _, _ = load_fer_data()
    print(f"Loaded {len(happy)} happy images.")

    print("\n--- 2. Loading Model ---")
    model, processor = load_qwen_model()

    if happy:
        happy_img = Image.open(happy[1]).convert("RGB")
        sad_img = Image.open(sad[1]).convert("RGB")
        neutral_img = Image.open(neutral[1]).convert("RGB")
        print(f"\n--- 3. Testing on: {happy_img} ---")
        response = get_vlm_response(model, processor, happy_img, "Describe the facial expression in the image provided in the prompt.")
        print(f"VLM Response w/o steering: {response}")

        response = generate_with_vector_insertion(model, processor, happy_img, 3, get_layer_representation(model, processor, sad_img, 3), alpha=0.5)
        print(f"VLM Response w steering: {response}")