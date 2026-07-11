import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from tqdm import tqdm
import torch
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
import argparse
from PIL import Image
from model.mm_vap import *

def inverse_scale_from_0_1(x):
    out = x * 6.0 - 3.0
    return out

def get_va(va_predictor, processor, clip_model, image_path, caption, device):

    image = Image.open(image_path).convert("RGB")
    input_data = processor(images=image,  text=[caption], return_tensors="pt", padding="max_length", truncation=True).to(device)

    vision_outputs = clip_model.vision_model(pixel_values=input_data["pixel_values"], output_hidden_states=True, return_dict=True)
    text_outputs = clip_model.text_model(input_ids=input_data["input_ids"], attention_mask=input_data["attention_mask"], output_hidden_states=True, return_dict=True)

    image_hidden_states = list(vision_outputs.hidden_states)
    text_hidden_states = list(text_outputs.hidden_states)

    preds = va_predictor(image_hidden_states=image_hidden_states, text_hidden_states=text_hidden_states)


    v = inverse_scale_from_0_1(preds[0, 0].item())
    a = inverse_scale_from_0_1(preds[0, 1].item())

    return v, a


def main(args):

    valence = []
    arousal = []

    clip_model = CLIPModel.from_pretrained(args.clip_model_path).eval().to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_path)

    vision_config = clip_model.config.vision_config
    text_config = clip_model.config.text_config
    
    va_predictor = VAPredictor(vision_config=vision_config, text_config=text_config)
    state = torch.load(args.checkpoint_path, map_location=args.device)
    va_predictor.load_state_dict(state, strict=True)
    va_predictor.eval().to(args.device)

    df = pd.read_csv(args.input_csv_path)
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="construction"):
            image_path = row['image_path']
            caption = row["caption"]
        
            image_full_path = os.path.join(args.dataset_root, image_path)

            v, a = get_va(va_predictor=va_predictor, processor=processor, clip_model=clip_model, image_path=image_full_path, caption=caption, device=args.device)
            valence.append(v)
            arousal.append(a)
            
    df['valence'] = valence
    df['arousal'] = arousal

    try:
        df.to_csv(args.output_csv_path, index=False)
        print(f"\nDone!")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="construct dataset", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset_root", type=str, default='/datasets/EmoEditSet')
    parser.add_argument("--input_csv_path", type=str, default="/datasets/EmoEditSet/image_caption.csv")
    parser.add_argument("--output_csv_path", type=str, default="data/C-EmoEditSet.csv")
    parser.add_argument("--checkpoint_path", type=str, default="runs/mm_vap/mm_vap_latest.pth")
    parser.add_argument("--clip_model_path", type=str, default="/pretrained_models/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=torch.device("cuda:0"))
    args = parser.parse_args()

    main(args)