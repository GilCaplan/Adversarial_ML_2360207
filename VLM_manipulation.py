import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

def get_layer_representation(model, processor, image_input, layer_idx, prompt="Describe this image."):
    activations = {}

    def hook_fn(module, input, output):
        # Outputs are usually (hidden_states, attention_weights, ...)
        if isinstance(output, tuple):
            activations['value'] = output[0].detach()
        else:
            activations['value'] = output.detach()

    # --- VERSION-AWARE LAYER ACCESS ---
    if hasattr(model, "language_model"):
        if hasattr(model.language_model, "layers"):
            target_layer = model.language_model.layers[layer_idx]
        elif hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            target_layer = model.language_model.model.layers[layer_idx]
        else:
            raise AttributeError("Could not find layers in model.language_model.")
    else:
        target_layer = model.model.layers[layer_idx]
    
    handle = target_layer.register_forward_hook(hook_fn)
    # ----------------------------------

    # Prepare inputs
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    # Run forward pass to trigger the hook
    with torch.no_grad():
        # We don't need to generate a full response, just one token is enough to trigger the hook
        model(**inputs) 

    handle.remove()
    
    if 'value' in activations:
        return activations['value']
    else:
        return "Error: Hook did not capture any activations."

def generate_with_vector_insertion(model, processor, image_input, layer_idx, injection_vector, alpha=1.0, prompt="Describe the expression."):
    """
    Adds a steering vector to the model's hidden state (Activation Addition).
    Math: New_State = Original_State + (alpha * Injection_Vector)
    """
    
    # 1. VALIDATION
    if not isinstance(injection_vector, torch.Tensor):
        print(f"Error: injection_vector is {type(injection_vector)}, expected torch.Tensor.")
        return "Steering Error: Invalid injection vector."

    def insertion_hook(module, input, output):
        original_hs = output[0] # Shape: [batch, seq_len, hidden_dim]
        
        # 2. KV-CACHE CHECK: Only steer during the initial 'prefill' pass (full image processing).
        # When seq_len is 1, it's just generating the next text token; we skip steering to be safe.
        if original_hs.shape[1] == 1:
            return output

        # 3. SHAPE MATCHING: Resize injection_vector if token counts differ
        if injection_vector.shape[1] != original_hs.shape[1]:
            # Transpose to [batch, hidden_dim, seq_len] for interpolation
            temp_v = injection_vector.transpose(1, 2).to(original_hs.dtype)
            resized_v = F.interpolate(temp_v, size=original_hs.shape[1], mode='linear')
            target_v = resized_v.transpose(1, 2)
        else:
            target_v = injection_vector.to(original_hs.dtype)

        # 4. THE MATH FIX: ADDITION (Steering)
        # Old way (Swap): modified_hs = (1 - alpha) * original_hs + (alpha * target_v)
        # New way (Add):  modified_hs = original_hs + (alpha * target_v)
        modified_hs = original_hs + (alpha * target_v)
        
        return (modified_hs,) + output[1:]

    # 5. ROBUST LAYER ACCESS
    try:
        if hasattr(model, "language_model"):
            if hasattr(model.language_model, "layers"):
                target_layer = model.language_model.layers[layer_idx]
            else:
                target_layer = model.language_model.model.layers[layer_idx]
        else:
            target_layer = model.model.layers[layer_idx]
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