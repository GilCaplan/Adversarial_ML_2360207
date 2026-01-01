import torch

def get_layer_representation(model, processor, image_input, layer_idx, prompt="Describe this image."):
    activations = {}

    def hook_fn(module, input, output):
        # Outputs are usually (hidden_states, attention_weights, ...)
        if isinstance(output, tuple):
            activations['value'] = output[0].detach()
        else:
            activations['value'] = output.detach()

    # --- VERSION-AWARE LAYER ACCESS ---
    # We check the attributes of model.language_model directly
    if hasattr(model, "language_model"):
        # This handles the 'Qwen2VLTextModel' structure you have
        if hasattr(model.language_model, "layers"):
            target_layer = model.language_model.layers[layer_idx]
        elif hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            target_layer = model.language_model.model.layers[layer_idx]
        else:
            raise AttributeError("Could not find layers in model.language_model.")
    else:
        # Fallback for other versions
        target_layer = model.model.layers[layer_idx]
    
    handle = target_layer.register_forward_hook(hook_fn)
    # ----------------------------------

    # Prepare inputs
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    # Run forward pass (don't need .generate for simple vector extraction)
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return activations.get('value')

def generate_with_vector_insertion(model, processor, image_input, layer_idx, injection_vector, alpha=1.0, prompt="Describe the expression."):
    """
    Inserts or adds a custom vector at a specific layer during generation.
    alpha: 1.0 = replace entirely, 0.5 = mix 50/50, etc.
    """
    
    def insertion_hook(module, input, output):
        # hidden_states is the first element of the output tuple
        original_hs = output[0]
        
        # Ensure injection_vector matches shape [batch, seq_len, hidden_dim]
        # For simplicity, we assume injection_vector is already shaped or we broadcast it
        modified_hs = (1 - alpha) * original_hs + (alpha * injection_vector)
        
        return (modified_hs,) + output[1:]

    # Register the "Intervention" hook
    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(insertion_hook)

    # Prepare inputs
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    # Generate with the injected "thought"
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    # Clean up
    handle.remove()
    
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]