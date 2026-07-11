# C-EmoEditor: Affective Image Editing Based on Valence-Arousal Model

This repository is the official implementation of "C-EmoEditor: Affective Image Editing Based on Valence-Arousal Model" 

## Requirements

### Environment Setup

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate C-EmoEditor
```

### Pre-trained Models

| Model | Source | Used By |
|-------|--------|---------|
| [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | HuggingFace | EmotionMapper training & inference |
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | HuggingFace | VAPredictor & evaluation |
| [instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) | HuggingFace | EmotionMapper training & inference |
| [dinov2-base](https://huggingface.co/facebook/dinov2-base) | HuggingFace | Evaluation (DINO-I metric) |

## Execution Steps

### 1. Preprocess: Extract CLIP Embeddings

Before training EmotionMapper, pre-extract CLIP features for original images and instructions:

```bash
python script/preprocess.py
```

> Modify the `config` dict in `script/preprocess.py` to set your dataset paths and CLIP model path.

### 2. Train EmotionMapper

Train the EmotionMapper model that maps (valence, arousal, image_embeds) to prompt embeddings:

```bash
python script/train_emotion_mapper.py
```

> Modify the `config_dic` in `script/train_emotion_mapper.py` to configure dataset paths, learning rate, batch size, etc.

### 3. Train VAPredictor (Optional)

Train the multimodal valence-arousal predictor used for emotion evaluation:

```bash
python script/train_mm_vap.py
```

> Modify the argument defaults in `script/train_mm_vap.py` to configure paths and hyperparameters.

### 4. Inference

Edit a single image with target emotion values:

```bash
python script/inference.py \
    --image_path example/1.png \
    --checkpoint /path/to/emotion_mapper_latest.pth \
    --valence 1.5 \
    --arousal 3.0 \
    --pix2pix_path /path/to/instruct-pix2pix \
    --clip_path /path/to/clip-vit-large-patch14 \
    --save_dir ./inference_results
```

Parameters:
- `--valence`: Target valence value (range: -3.0 ~ 3.0)
- `--arousal`: Target arousal value (range: -3.0 ~ 3.0)
- `--guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `--image_guidance_scale`: Image guidance scale (default: 1.5)
- `--num_steps`: Number of diffusion steps (default: 100)

### 5. Evaluation

**Emotion-related metrics** (Valence Error, Arousal Error, EGS, Edit Diversity):

```bash
python metrics/emotion-related_metric.py
```

**Structure & semantic preserving metrics** (SSIM, PSNR, LPIPS, MSE, CLIP-I, DINO-I):

```bash
python metrics/structure-semantic_preserving_metric.py
```

> Modify the paths at the bottom of each script to point to your original images, edited results, and model checkpoints.

