import os
import json
import torch
import numpy as np
import tempfile
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from Load_fer_dataset import load_fer_data
from Load_VLM import load_vlm_model, get_vlm_response
from VLM_manipulation import get_layer_representation, generate_with_vector_insertion

# --- Configuration ---
RESULTS_DIR = "json_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEFAULT_PROMPT = "Describe the image. If you see any specific emotion or object, describe it clearly."

# Fixed alpha for the transformation experiment (since we only vary it for blank images)
TRANSFORMATION_FIXED_ALPHA = 1.0 

class TempImageHandler:
    """
    Context manager to handle PIL images for functions that strictly require file paths.
    """
    def __init__(self, image_input):
        self.image_input = image_input
        self.temp_path = None

    def __enter__(self):
        if isinstance(self.image_input, str):
            return self.image_input
        
        # It is a PIL Image, save to temp
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            self.image_input.save(tmp.name)
            self.temp_path = tmp.name
        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_path and os.path.exists(self.temp_path):
            os.remove(self.temp_path)

def save_result_entry(experiment_name, model_name, entry):
    """
    Appends results to a single JSON file per Model + Experiment combo.
    File format: json_results/exp_name_model_name.json
    """
    # Clean model name for filename (e.g., "qwen_2B")
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    filename = f"{RESULTS_DIR}/{experiment_name}_{safe_model_name}.json"
    
    data = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    
    data.append(entry)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def create_noise_image(width=640, height=640):
    """Generates a white noise image."""
    base_array = np.full((height, width, 3), 255, dtype=np.float32)
    noise = np.random.normal(0, 25, (height, width, 3))
    noisy_array = base_array - np.abs(noise)
    return Image.fromarray(np.clip(noisy_array, 0, 255).astype(np.uint8))

def get_averaged_steering_vector(model, processor, pos_images, neg_images, layer_idx, use_llm, batch_size=30):
    """Calculates Mean(Positive) - Mean(Neutral) vector."""
    vectors = []
    limit = min(len(pos_images), batch_size)
    
    for i in range(limit):
        p_img = pos_images[i] if not isinstance(pos_images[i], str) else Image.open(pos_images[i]).convert("RGB")
        n_img = neg_images[i] if not isinstance(neg_images[i], str) else Image.open(neg_images[i]).convert("RGB")
        
        p_img = p_img.resize((336, 336))
        n_img = n_img.resize((336, 336))

        vec = get_layer_representation(model, processor, p_img, layer_idx, LLM_use=use_llm) - \
              get_layer_representation(model, processor, n_img, layer_idx, LLM_use=use_llm)
        vectors.append(vec)
    
    # Return the average vector
    return torch.stack(vectors).mean(dim=0)

def process_steering_batch(model, processor, provider, target_images, steer_pos_imgs, steer_neg_imgs, 
                           layer_idx, alpha, use_llm, prompt):
    """
    Computes steering vector -> Runs Baseline -> Runs Steered Generation.
    Returns: results (list), vector_norm (float)
    """
    # 1. Compute Steering Vector
    steering_vector = get_averaged_steering_vector(
        model, processor, steer_pos_imgs, steer_neg_imgs, layer_idx, use_llm
    )
    
    # 2. Calculate Norm (Strength of the steering direction)
    vector_norm = steering_vector.norm().item()

    results = []
    
    # 3. Process Target Images
    for idx, img_input in enumerate(target_images[:5]): 
        
        if isinstance(img_input, str):
            pil_img = Image.open(img_input).convert("RGB").resize((336, 336))
        else:
            pil_img = img_input.convert("RGB").resize((336, 336))

        # --- Baseline Generation ---
        with TempImageHandler(pil_img) as img_path:
            baseline_resp = get_vlm_response(model, processor, provider, img_path, prompt)

        # --- Steered Generation ---
        steered_resp = generate_with_vector_insertion(
            model, processor, pil_img, layer_idx, steering_vector, 
            alpha=alpha, prompt=prompt, LLM_use=use_llm
        )
        
        results.append({
            "image_index": idx,
            "baseline": baseline_resp,
            "steered": steered_resp
        })
        
    return results, vector_norm

# ==========================================
# Experiment 1: Blank Images (Hallucination)
# ==========================================
def run_blank_image_experiment(model, processor, provider, model_name, emotions_dict, neutral_imgs):
    experiment_name = "exp_blank_hallucination"
    print(f"\n[Experiment] Running {experiment_name} on {model_name}...")

    # We use a set of alphas here as requested
    alpha_values = [1.0, 3.0, 5.0, 10.0]
    
    blank_img = create_noise_image()
    target_inputs = [blank_img]

    for emotion_name, emotion_imgs in emotions_dict.items():
        print(f" > Steering with '{emotion_name}' vectors...")
        
        max_llm = model.config.num_hidden_layers
        max_vis = model.config.vision_config.num_hidden_layers if hasattr(model.config.vision_config, "num_hidden_layers") else model.config.vision_config.depth
        
        layer_configs = [
            ("LLM", True, range(1, max_llm, 4)), 
            ("Vision", False, range(1, max_vis, 4))
        ]

        for comp_name, is_llm, layer_range in layer_configs:
            for layer in tqdm(layer_range, desc=f"{emotion_name} ({comp_name})"):
                
                # Iterate over alpha values for the blank experiment
                for alpha in alpha_values:
                    outputs, vec_norm = process_steering_batch(
                        model, processor, provider, target_inputs, 
                        steer_pos_imgs=emotion_imgs, 
                        steer_neg_imgs=neutral_imgs,
                        layer_idx=layer, 
                        alpha=alpha, 
                        use_llm=is_llm, 
                        prompt="Describe the image. If it is just noise, say so. If you see a face, describe the expression."
                    )

                    # Save to JSON
                    save_result_entry(experiment_name, model_name, {
                        "layer": layer,
                        "component": comp_name,
                        "target_emotion": emotion_name,
                        "alpha": alpha,       # Log the alpha
                        "vector_norm": vec_norm, # Log the norm
                        "results": outputs
                    })

# ==========================================
# Experiment 2: Emotion Transformation
# ==========================================
def run_emotion_transformation_experiment(model, processor, provider, model_name, emotions_dict, neutral_imgs):
    experiment_name = "exp_emotion_transfer"
    print(f"\n[Experiment] Running {experiment_name} on {model_name}...")

    pairs = [("happy", "sad"), ("sad", "happy"), ("neutral", "angry")]

    for src_emo, target_emo in pairs:
        print(f" > Turning {src_emo} -> {target_emo}...")
        
        target_inputs = emotions_dict[src_emo]
        steer_pos = emotions_dict[target_emo]
        
        max_llm = model.config.num_hidden_layers
        max_vis = model.config.vision_config.num_hidden_layers if hasattr(model.config.vision_config, "num_hidden_layers") else model.config.vision_config.depth

        layer_configs = [
            ("LLM", True, range(5, max_llm, 5)), 
            ("Vision", False, range(5, max_vis, 5))
        ]

        for comp_name, is_llm, layer_range in layer_configs:
            for layer in tqdm(layer_range, desc=f"{src_emo}->{target_emo}"):
                
                # Single fixed alpha for transformation experiment
                outputs, vec_norm = process_steering_batch(
                    model, processor, provider, target_inputs, 
                    steer_pos_imgs=steer_pos, 
                    steer_neg_imgs=neutral_imgs,
                    layer_idx=layer, 
                    alpha=TRANSFORMATION_FIXED_ALPHA, 
                    use_llm=is_llm, 
                    prompt=DEFAULT_PROMPT
                )

                save_result_entry(experiment_name, model_name, {
                    "layer": layer,
                    "component": comp_name,
                    "source": src_emo,
                    "target": target_emo,
                    "alpha": TRANSFORMATION_FIXED_ALPHA,
                    "vector_norm": vec_norm,
                    "results": outputs
                })

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("--- Loading FER Dataset ---")
    happy, sad, neutral, angry, _, _, _ = load_fer_data()
    emotions_dict = {"happy": happy, "sad": sad, "neutral": neutral, "angry": angry}

    models_to_run = [
        ("qwen", "2B", True),   # <--- Valid for Qwen2-VL
        ("qwen", "7B", True),
        # ("llama", "11B", True) 
    ]

    for provider, size, load_4bit in models_to_run:
        full_name = f"{provider}_{size}"
        print(f"\n{'='*40}\nProcessing Model: {full_name}\n{'='*40}")

        try:
            # UNPACK 3 VALUES
            model, processor, provider_str = load_vlm_model(provider, size, load_in_4bit=load_4bit)
            
            # Experiment A: Hallucination (Iterates Alphas)
            run_blank_image_experiment(model, processor, provider_str, full_name, emotions_dict, neutral)

            # Experiment B: Transfer (Fixed Alpha)
            run_emotion_transformation_experiment(model, processor, provider_str, full_name, emotions_dict, neutral)
            
            # Clean up VRAM
            del model, processor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"!!! Error processing {full_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll experiments completed.")