import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,  # Back to the specific, stable class
    MllamaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

def get_optimal_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_vlm_model(provider="qwen", size="7B", load_in_4bit=False):
    device = get_optimal_device()
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major < 8:  # Ampere (8.0) is when native bfloat16 started
            dtype = torch.float16
            print(f"Detected older GPU (Pascal/Volta). Forcing float16 instead of bfloat16.")
        else:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # ... rest of your setup ...
    if provider.lower() == "qwen":
        model_id = f"Qwen/Qwen2-VL-{size}-Instruct" 
        model_class = Qwen2VLForConditionalGeneration
    elif provider.lower() == "llama":
        model_id = f"meta-llama/Llama-3.2-{size}-Vision-Instruct"
        model_class = MllamaForConditionalGeneration
    else:
        raise ValueError("Provider must be 'qwen' or 'llama'")

    print(f"Loading {model_id}...")
    
    bnb_config = None
    if load_in_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Standard manual move if not auto-mapped
    if device != "cuda" and not load_in_4bit and device_map is None:
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, provider.lower()

def get_vlm_response(model, processor, provider, image_path, prompt):
    device = model.device
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}
    ]

    # Manual image loading to avoid external dependencies
    if isinstance(image_path, str):
        image_obj = Image.open(image_path).convert("RGB")
    else:
        image_obj = image_path.convert("RGB")

    if provider == "qwen":
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image_obj],
            padding=True,
            return_tensors="pt",
        )
    else:
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image_obj, text_prompt, return_tensors="pt")

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False,
            # Add these 3 lines to silence warnings:
            temperature=None,
            top_p=None,
            top_k=None
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
    return response.strip()