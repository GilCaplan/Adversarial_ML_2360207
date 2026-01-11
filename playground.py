count = 20
import numpy as np
from PIL import Image
from tqdm import tqdm
from Load_VLM import load_qwen_model, get_vlm_response
import matplotlib.pyplot as plt

def run_baseline_blank_test():
    # 1. Load Model

    print("\n--- Generating Blank Image ---")
    width, height = 640, 640
    color = (255, 255, 255)  # White
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.uint8) # mean 0, std 5
    base_array = np.full((height, width, 3), 255, dtype=np.uint8)
    noisy_img = np.clip(base_array - noise, 0, 255).astype(np.uint8) # Subtract noise to keep it mostly white
    blank_img = Image.fromarray(noisy_img)
    plt.figure(figsize=(6, 6))
    plt.imshow(blank_img)
    plt.title(f"Generated Image ({width}x{height})")
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        
    # Remove axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    

    print("--- Loading Qwen-VL Model ---")
    model, processor = load_qwen_model()

    # 3. Define Prompts to Test
    prompts = [
        "Describe the image in detail.",
        "What objects or emotions do you see in the image?",
        "Provide a creative description of the scene.",
        "Is there any specific emotion conveyed by the image?",
        "Describe the colors and shapes present in the image."
        ]

    # 4. Run Inference
    print(f"\n--- Running Baseline Inference on Blank Image ---")
    for i, prompt in enumerate(prompts):
        response = get_vlm_response(model, processor, blank_img, prompt)
        print(f"\n[Prompt {i+1}]: {prompt}")
        print(f"[Response]: {response}")

if __name__ == "__main__":
    run_baseline_blank_test()