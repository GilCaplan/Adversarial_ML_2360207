import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

def get_layer_representation(model, processor, image_input, layer_idx, prompt="Describe this image.", LLM_use=True):
    activations = {}

    def hook_fn(module, input, output):
        # Outputs are usually (hidden_states, attention_weights, ...)
        if isinstance(output, tuple):
            activations['value'] = output[0].detach()
        else:
            activations['value'] = output.detach()

    # --- VERSION-AWARE LAYER ACCESS ---
    if LLM_use:
        # Target Language Model Layers
        if hasattr(model, "language_model"):
            if hasattr(model.language_model, "layers"):
                target_layer = model.language_model.layers[layer_idx]
            elif hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
                target_layer = model.language_model.model.layers[layer_idx]
            else:
                raise AttributeError("Could not find layers in model.language_model.")
        else:
            # Fallback for models where the base is the LM (e.g. Qwen2.5 pure text, though unlikely here)
            target_layer = model.model.layers[layer_idx]
    else:
        # Target Vision Model Layers
        # Qwen2-VL usually stores the vision encoder in model.visual
        if hasattr(model, "visual") and hasattr(model.visual, "blocks"):
             target_layer = model.visual.blocks[layer_idx]
        elif hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "blocks"):
             target_layer = model.model.visual.blocks[layer_idx]
        else:
             raise AttributeError("Could not find vision blocks in model.visual.")
    
    handle = target_layer.register_forward_hook(hook_fn)
    # ----------------------------------

    # Prepare inputs
    messages = [{"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    # Run forward pass to trigger the hook
    with torch.no_grad():
        model(**inputs) 

    handle.remove()
    
    if 'value' in activations:
        return activations['value']
    else:
        return "Error: Hook did not capture any activations."

def generate_with_vector_insertion(model, processor, image_input, layer_idx, injection_vector, alpha=1.0, prompt="Describe the expression.", LLM_use=True):
    """
    Adds a steering vector to the model's hidden state (Activation Addition).
    Math: New_State = Original_State + (alpha * Injection_Vector)
    """
    
    # 1. VALIDATION
    if not isinstance(injection_vector, torch.Tensor):
        print(f"Error: injection_vector is {type(injection_vector)}, expected torch.Tensor.")
        return "Steering Error: Invalid injection vector."

    def insertion_hook(module, input, output):
        # Handle tuple output (common in transformers)
        if isinstance(output, tuple):
            original_hs = output[0] 
        else:
            original_hs = output

        # 2. KV-CACHE CHECK (Only relevant for LLM)
        # If we are steering the LLM, we only want to steer during prefill (seq_len > 1).
        # If steering Vision, this check is usually irrelevant as vision runs once per image, 
        # but the check 'original_hs.shape[1] == 1' is still safe to keep.
        if LLM_use and original_hs.shape[1] == 1:
            return output

        # 3. SHAPE MATCHING: Resize injection_vector if token counts differ
        # Vision tokens (spatial) vs LLM tokens (text) will have very different lengths.
        # F.interpolate handles this dynamic resizing.
        if injection_vector.shape[1] != original_hs.shape[1]:
            # Transpose to [batch, hidden_dim, seq_len] for interpolation
            temp_v = injection_vector.transpose(1, 2).to(original_hs.dtype)
            resized_v = F.interpolate(temp_v, size=original_hs.shape[1], mode='linear')
            target_v = resized_v.transpose(1, 2)
        else:
            target_v = injection_vector.to(original_hs.dtype)

        # 4. THE MATH: ADDITION (Steering)
        modified_hs = original_hs + (alpha * target_v)
        
        # Return tuple if original was tuple, else return tensor
        if isinstance(output, tuple):
            return (modified_hs,) + output[1:]
        else:
            return modified_hs

    # 5. ROBUST LAYER ACCESS
    try:
        if LLM_use:
            # Target Language Model
            if hasattr(model, "language_model"):
                if hasattr(model.language_model, "layers"):
                    target_layer = model.language_model.layers[layer_idx]
                else:
                    target_layer = model.language_model.model.layers[layer_idx]
            else:
                target_layer = model.model.layers[layer_idx]
        else:
            # Target Vision Model
            if hasattr(model, "visual") and hasattr(model.visual, "blocks"):
                 target_layer = model.visual.blocks[layer_idx]
            elif hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "blocks"):
                 target_layer = model.model.visual.blocks[layer_idx]
            else:
                 raise AttributeError("Could not find vision blocks in model.visual.")
                 
    except Exception as e:
        return f"Layer Access Error: {e}"

    handle = target_layer.register_forward_hook(insertion_hook)

    # 6. RUN GENERATION
    messages = [{"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    # 7. TRIM PROMPT FROM OUTPUT
    input_len = inputs.input_ids.shape[1]
    response = processor.batch_decode(
        generated_ids[:, input_len:], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    handle.remove()
    return response.strip()