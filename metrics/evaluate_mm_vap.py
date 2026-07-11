import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from model.mm_vap import *


def inverse_scale_from_0_1(v, a):
    valence = v * 6.0 - 3.0
    arousal = a * 6.0 - 3.0
    return valence, arousal


def predict_va(va_predictor, image_dir, image_name, prompt, device, clip_processor, clip_model):

    with torch.no_grad():
        image_path = os.path.join(image_dir, f"{image_name}.jpg")

        image = Image.open(image_path).convert("RGB")

        data = clip_processor(images=image,  text=[prompt], return_tensors="pt", padding="max_length", truncation=True)
        pixel_values = data['pixel_values'].to(device)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)

        vision_outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        text_outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

        image_hidden_states = list(vision_outputs.hidden_states)
        text_hidden_states = list(text_outputs.hidden_states)

        out = va_predictor(image_hidden_states, text_hidden_states)

        valence_pred_scale, arousal_pred_scale = inverse_scale_from_0_1(out[0, 0].item(), out[0, 1].item())
        
        return (valence_pred_scale, arousal_pred_scale)


def evaluate(truth_csv_path, device):

    try:
        truth_df = pd.read_csv(truth_csv_path)
    except FileNotFoundError:
        print(f"\n[Error] Ground truth file not found: {truth_csv_path}")
        return
    except Exception as e:
        print(f"\n[Error] Error loading CSV: {e}")
        return

    required_cols = ['Id', 'Valence', 'Arousal']
    if not all(col in truth_df.columns for col in required_cols):
        print(f"Need: {required_cols}")
        return

    clip_processor = CLIPProcessor.from_pretrained("/pretrained_models/clip-vit-base-patch32")      # TODO
    clip_model = CLIPModel.from_pretrained("/pretrained_models/clip-vit-base-patch32").to(device)   # TODO
    clip_model.requires_grad_(False).eval()

    vision_config = clip_model.config.vision_config
    text_config = clip_model.config.text_config

    model = VAPredictor(vision_config, text_config)
    state = torch.load("runs/mm_vap/va_predictor_best.pth", map_location=device)   # TODO
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    all_true_valence = []
    all_pred_valence = []
    all_true_arousal = []
    all_pred_arousal = []
    
    processed_count = 0
    skipped_count = 0

    for row in tqdm(truth_df.itertuples(), total=len(truth_df), desc="val"):
        try:
            image_id = row.Id
            true_valence = row.Valence
            true_arousal = row.Arousal
            prompt = row.Emotional_Prompt
        

            # TODO
            (pred_valence, pred_arousal) = predict_va(model, "/datasets/EmotiCrafter/val/images",
                                                      image_id, prompt, device, clip_processor, clip_model)
            
            all_true_valence.append(true_valence)
            all_pred_valence.append(pred_valence)
            all_true_arousal.append(true_arousal)
            all_pred_arousal.append(pred_arousal)
            processed_count += 1

        except Exception as e:
            print(f"\n[Error] An unexpected error occurred while processing image {image_id}: {e}")
            skipped_count += 1
            continue

    if processed_count == 0:
        print("[Error] No images processed.")
        return
    
    y_true_v = np.array(all_true_valence)
    y_pred_v = np.array(all_pred_valence)
    y_true_a = np.array(all_true_arousal)
    y_pred_a = np.array(all_pred_arousal)

    print("\n[Valence]")
    valence_mae = mean_absolute_error(y_true_v, y_pred_v)
    print(f"  Mean Absolute Error (MAE): {valence_mae:.3f}")
    valence_spearman = spearmanr(y_true_v, y_pred_v)
    print(f"  Spearman Rank Corr (S.R): {valence_spearman.correlation:.3f}")
    print(f"  Spearman p-value:         {valence_spearman.pvalue:.4g}")

    print("\n[Arousal]")
    arousal_mae = mean_absolute_error(y_true_a, y_pred_a)
    print(f"  Mean Absolute Error (MAE): {arousal_mae:.3f}")
    arousal_spearman = spearmanr(y_true_a, y_pred_a)
    print(f"  Spearman Rank Corr (S.R): {arousal_spearman.correlation:.3f}")
    print(f"  Spearman p-value:         {arousal_spearman.pvalue:.4g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MM_VAP",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--truth_csv",type=str, default='/private/ljh/datasets/EmotiCrafter_split2/val/labels.csv')
    
    parser.add_argument("--image_dir", type=str, default='/private/ljh/datasets/EmotiCrafter_split2/val/images')

    parser.add_argument("--device", type=str, default=torch.device('cuda:0'))

    parser.add_argument("--suffix", type=str, default=".jpg")
    args = parser.parse_args()

    evaluate(args.truth_csv, args.image_dir, args.device)

