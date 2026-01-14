import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

# --- HELPER 1: UNIFIED LAYER ACCESS ---
def get_target_layers(model, is_llm):
    """
    Dynamically finds the correct layer list for Qwen2-VL and Llama 3.2.
    """
    if is_llm:
        # --- LLM LAYERS ---
        # Case A: Llama 3.2 (Composite model structure)
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            return model.language_model.model.layers
        # Case B: Qwen2-VL (Standard structure)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        # Case C: Fallback (some older HF models)
        elif hasattr(model, "layers"):
            return model.layers
        else:
            raise AttributeError("Could not find LLM layers (checked 'language_model', 'model.model', and 'layers')")
            
    else:
        # --- VISION LAYERS ---
        # Case A: Qwen2-VL
        if hasattr(model, "visual") and hasattr(model.visual, "blocks"):
            return model.visual.blocks
        # Case B: Llama 3.2 Vision
        elif hasattr(model, "vision_model") and hasattr(model.vision_model, "transformer"):
            return model.vision_model.transformer.layers
        # Case C: Generic ViT fallback
        elif hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder"):
            return model.vision_model.encoder.layers
        else:
            raise AttributeError("Could not find Vision layers (checked 'visual.blocks' and 'vision_model')")

# --- HELPER 2: UNIFIED INPUT PROCESSING ---
def prepare_inputs(model, processor, image_input, prompt):
    """
    Handles input processing differences between Qwen2-VL and Llama 3.2.
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_input},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    # Check model type string to decide processing path
    model_type = str(type(model)).lower()
    
    if "qwen" in model_type:
        # Qwen-Specific Processing
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    else:
        # Llama 3.2 / Standard HF Processing
        # Llama processor handles PIL images directly in the 'images' argument
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image_input, return_tensors="pt")

    return inputs.to(model.device)


def get_layer_representation(model, processor, image_input, layer_idx, prompt="Describe this image.", LLM_use=True):
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations['value'] = output[0].detach()
        else:
            activations['value'] = output.detach()

    # 1. Use the new robust layer finder
    try:
        layers = get_target_layers(model, LLM_use)
        target_layer = layers[layer_idx]
    except Exception as e:
        return f"Layer Error: {e}"
    
    handle = target_layer.register_forward_hook(hook_fn)

    # 2. Use the new robust input preparer
    try:
        inputs = prepare_inputs(model, processor, image_input, prompt)
        
        with torch.no_grad():
            model(**inputs) 
    finally:
        handle.remove()
    
    if 'value' in activations:
        return activations['value']
    else:
        return "Error: Hook did not capture any activations."

def generate_with_vector_insertion(model, processor, image_input, layer_idx, injection_vector, alpha=1.0, prompt="Describe the expression.", LLM_use=True):
    
    # Validation
    if not isinstance(injection_vector, torch.Tensor):
        return "Steering Error: Invalid injection vector."

    def insertion_hook(module, input, output):
        if isinstance(output, tuple):
            original_hs = output[0] 
        else:
            original_hs = output

        # Only steer LLM during prefill
        if LLM_use and original_hs.shape[1] == 1:
            return output

        # Shape Matching & Interpolation
        if injection_vector.shape[1] != original_hs.shape[1]:
            temp_v = injection_vector.transpose(1, 2).to(original_hs.dtype)
            resized_v = F.interpolate(temp_v, size=original_hs.shape[1], mode='linear')
            target_v = resized_v.transpose(1, 2)
        else:
            target_v = injection_vector.to(original_hs.dtype)

        # Steering Math
        modified_hs = original_hs + (alpha * target_v)
        
        if isinstance(output, tuple):
            return (modified_hs,) + output[1:]
        else:
            return modified_hs

    # 1. Use the new robust layer finder
    try:
        layers = get_target_layers(model, LLM_use)
        target_layer = layers[layer_idx]
    except Exception as e:
        return f"Layer Access Error: {e}"

    handle = target_layer.register_forward_hook(insertion_hook)

    # 2. Use the new robust input preparer
    try:
        inputs = prepare_inputs(model, processor, image_input, prompt)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            
        # Decode response
        input_len = inputs.input_ids.shape[1]
        response = processor.batch_decode(
            generated_ids[:, input_len:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response.strip()

    finally:
        handle.remove()