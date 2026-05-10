import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def preprocess_and_save(args):

    device = torch.device(args["device"])
    edited_data_csv_path = args['edited_data_root']
    data_root = args['data_root']
    
    embed_save_root = os.path.join(data_root, args['embed_save_name'])
    os.makedirs(embed_save_root, exist_ok=True)

    model_path = args['model_path']
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).to(device)
    model.eval()

    df_edited = pd.read_csv(edited_data_csv_path)

    processed_paths = set()

    for index, row in tqdm(df_edited.iterrows(), total=len(df_edited)):
        edited_image_rel_path = row["image_path"]

        path_parts = edited_image_rel_path.split('/')
        dataset_name = '_'.join(path_parts[-3].split('_')[:-1])
        file_name = path_parts[-1]
        
        name_clean = '_'.join(file_name.split('_')[:-1])
        summary = file_name.split('.')[0].split('_')[-1]
        
        origin_image_rel_path = f"origin_image/{dataset_name}_crop/{name_clean}.jpg"
        origin_image_abs_path = os.path.join(data_root, origin_image_rel_path)
        
        instruction = 'add ' + summary
        save_name = f"{name_clean}_{summary}.pt"
        save_dir = os.path.join(embed_save_root, f"origin_image/{dataset_name}_crop")
        save_path = os.path.join(save_dir, save_name)

        if save_path in processed_paths:
            continue
        
        if os.path.exists(save_path):
            processed_paths.add(save_path)
            continue

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        image = Image.open(origin_image_abs_path).convert("RGB")

        inputs = processor(
            images=image, 
            text=[instruction], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        data_to_save = {
            'origin_image_embeds': outputs.image_embeds.cpu(),
            'instruction_embeds': outputs.text_model_output.last_hidden_state.cpu()
        }
        
        torch.save(data_to_save, save_path)
        processed_paths.add(save_path)

    print("Complete processing all data!")

if __name__ == "__main__":
    config = {
        'edited_data_root': "/private/ljh/datasets/EmoEditSet/edited_image_captions_weighted.csv",
        'data_root': "/private/ljh/datasets/EmoEditSet",
        'device': "cuda:0",
        'embed_save_name': "clip_cache",
        'model_path': "/private/ljh/pretrained_models/clip-vit-large-patch14"
    }
    preprocess_and_save(config)