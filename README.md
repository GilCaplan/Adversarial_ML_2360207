# Steer to Feel: Steering Abstract Concepts into Vision-Language Models

Activation steering experiments on Qwen2-VL (2B/7B) and Llama-3.2-Vision (11B) across emotion and weather concept domains. Batch-averaged steering vectors are injected at inference time to control abstract latent concepts without modifying model weights or input pixels.

## Prerequisites

- **Python** 3.10+
- **GPU strongly recommended** — models require significant VRAM:
  - Qwen2-VL-2B: ~6 GB VRAM
  - Qwen2-VL-7B: ~16 GB VRAM (or 4-bit: ~8 GB)
  - Llama-3.2-Vision-11B: ~24 GB VRAM (or 4-bit: ~12 GB)
- CUDA 11.8+ (for GPU inference); MPS (Apple Silicon) and CPU are supported but very slow

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install Pillow tqdm numpy matplotlib kagglehub
pip install qwen-vl-utils
```

> **Llama access:** Llama-3.2-Vision is a gated model. You need a Hugging Face account with access granted at [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), then log in:
> ```bash
> huggingface-cli login
> ```

## Datasets

Datasets are downloaded automatically via `kagglehub` on first run. You need a Kaggle account and API token configured (`~/.kaggle/kaggle.json`).

- **Emotion (FER):** [`ananthu017/emotion-detection-fer`](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) → saved to `fer_dataset/`
- **Weather:** [`pratik2901/multiclass-weather-dataset`](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) → saved to `weather_dataset/`

## Running Experiments

All experiments are run from the project root.

### Controlled Hallucination + Semantic Transfer

```bash
python VLM_experiments.py
```

This runs both experiments across all configured models and saves JSON results to:
- `json_results/` — weather domain
- `json_results_emotions/` — emotion domain

### Visualisation

```bash
# Per-model hallucination and transfer figures
python visualize.py

# Baseline comparison (steered vs unsteered, v2 keywords)
python baseline_analysis/plot_baseline_comparison.py

# Unsteered baseline figure (transfer experiment)
python baseline_analysis/plot_unsteered_baseline.py
```

Figures are saved to `graphs_emotions/`, `graphs_weather/`, and `baseline_analysis/`.

## Project Structure

```
├── VLM_experiments.py        # Main experiment runner
├── VLM_manipulation.py       # Steering vector injection hooks
├── Load_VLM.py               # Model loading (Qwen / Llama)
├── Load_dataset.py           # Dataset loading via kagglehub
├── visualize.py              # Hallucination & transfer figures
├── baseline_analysis/        # Baseline extraction and plotting scripts
│   ├── extract_baselines.py
│   ├── plot_baseline_comparison.py
│   ├── plot_unsteered_baseline.py
│   └── *.png / *.pdf         # Generated figures
├── json_results/             # Weather experiment JSON outputs
├── json_results_emotions/    # Emotion experiment JSON outputs
├── graphs_emotions/          # Emotion visualisation figures
└── graphs_weather/           # Weather visualisation figures
```

## Configuration

Key parameters in `VLM_experiments.py`:

| Parameter | Default | Description |
|---|---|---|
| `N` | 10 | Images per concept for steering vector |
| `alpha` | [1, 3, 5, 10] | Steering strength values swept |
| Layer stride | every 4th from L1 | Layers sampled for injection |
| `load_in_4bit` | False | Enable 4-bit quantisation to reduce VRAM |

To enable 4-bit quantisation (requires `bitsandbytes` and CUDA):
```python
model, processor, provider = load_vlm_model(provider="llama", size="11B", load_in_4bit=True)
```
