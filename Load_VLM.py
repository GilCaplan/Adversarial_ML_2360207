import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_qwen_model():
    # 1. Force MPS explicitly instead of relying on auto-map
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # bfloat16 is often better supported on newer M-series chips
        dtype = torch.bfloat16 
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Loading Qwen2-VL-2B on {device}...")

    # 2. Optimized Loading Parameters
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=dtype,
        # Avoid device_map="auto" for MPS to prevent CPU-offloading slowness
        device_map=None, 
        low_cpu_mem_usage=True
    ).to(device) # Move the whole model to MPS at once
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        
    print("Model loaded successfully.")
    return model, processor

def get_vlm_response(model, processor, image_path, prompt="Describe this image."):
    """
    Generates a response for a single image and prompt.
    """
    # Create the conversation structure
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs using Qwen's specific utilities
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to the correct device
    inputs = inputs.to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Decode and trim (Qwen returns the input prompt + new text, we trim the prompt)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # The output usually contains the prompt text; this split isolates the assistant's reply
    # Note: Adjust logic if the model behavior changes, but usually the last part is the answer.
    response = output_text[0]
    
    # Simple cleanup to remove the "system" or "user" prompt artifacts if they appear in raw decode
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1]
        
    return response.strip()