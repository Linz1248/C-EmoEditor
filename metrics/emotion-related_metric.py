
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import re
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from model.mm_vap import VAPredictor
from tqdm import tqdm
import math
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from collections import defaultdict
from itertools import combinations
import lpips


def crop_img(img, output_size=(512,512)):
    width, height = img.size

    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2

    img_cropped = img.crop((left, top, right, bottom))

    img_resized = img_cropped.resize(output_size)
    return img_resized


def DistanceOfSemantic(img0, img1, model, processor):
    data_pro = processor(images=[img0, img1], return_tensors="pt", padding=True).to(model.device)
    data_pro = model.get_image_features(**data_pro)
    d = 1 - F.cosine_similarity(data_pro[0,:].unsqueeze(0), data_pro[1,:].unsqueeze(0))
    mse = F.mse_loss(data_pro[0,:].unsqueeze(0), data_pro[1,:].unsqueeze(0))
    return d.item(), mse.item()



def semantic_diversity(edited_dir, model, processor):
    device = model.device
    
    image_groups = defaultdict(list)
    all_files = [f for f in os.listdir(edited_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in all_files:
        orig_name = filename.split('|')[0]
        image_groups[orig_name].append(os.path.join(edited_dir, filename))

    total_cos_dist = []
    total_mse_dist = []

    with torch.no_grad():
        for orig_name, paths in tqdm(image_groups.items()):
            if len(paths) < 2:
                continue
                
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            
            features = model.get_image_features(**inputs)
            features_norm = F.normalize(features, p=2, dim=-1)
            
            group_cos = []
            group_mse = []
            
            num_imgs = len(paths)
            for i, j in combinations(range(num_imgs), 2):
                cos_sim = torch.sum(features_norm[i] * features_norm[j])
                group_cos.append(1 - cos_sim.item())
                
                mse = F.mse_loss(features[i], features[j])
                group_mse.append(mse.item())
            
            if group_cos:
                total_cos_dist.append(sum(group_cos) / len(group_cos))
                total_mse_dist.append(sum(group_mse) / len(group_mse))

    avg_cos = sum(total_cos_dist) / len(total_cos_dist) if total_cos_dist else 0
    avg_mse = sum(total_mse_dist) / len(total_mse_dist) if total_mse_dist else 0
    
    return avg_cos, avg_mse


def main(origin_img_dir, target_img_dir, device):
    test_img_paths = []
    for root, _, file_path in os.walk(target_img_dir):
        for file in file_path:
            if file.endswith("png") or file.endswith("jpg"):
                test_img_paths.append(os.path.join(root, file))

    valence_errors = []
    arousal_errors = []
    egs_scores = []

    clip_processor = CLIPProcessor.from_pretrained("/pretrained_models/clip-vit-base-patch32")   # TODO
    clip_model = CLIPModel.from_pretrained("/pretrained_models/clip-vit-base-patch32").to(device)    # TODO
    clip_model.requires_grad_(False).eval()

    vision_config = clip_model.config.vision_config
    text_config = clip_model.config.text_config
    va_model = VAPredictor(vision_config=vision_config, text_config=text_config).to(device)
    state = torch.load("runs/mm_vap/model_latest.pth", map_location=device)   # TODO
    va_model.load_state_dict(state, strict=True)
    va_model.to(device).eval()

    lpips_alex = lpips.LPIPS(net='alex')
    lpips_alex.eval()


    
    with torch.no_grad():
        for test_img_path in tqdm(test_img_paths):
            test_img_name = test_img_path.split('/')[-1].split('_')[0]

            try:
                valence_part = test_img_name.split('|')[1]
                valence_target = float(valence_part.split('=')[-1])

                arousal_part = test_img_name.split('|')[2]
                arousal_target = float(arousal_part.split('=')[-1].split('.')[0])

                test_img_name = test_img_name.split('|')[0]

                for suffix in ['jpg','png']:
                    origin_img_path = os.path.join(origin_img_dir, f"{test_img_name}.{suffix}")

                    if os.path.isfile(origin_img_path) is True:
                        break
                
                with Image.open(test_img_path) as test_img, Image.open(origin_img_path) as origin_img:

                    origin_data = clip_processor(images=origin_img, text=[""], return_tensors="pt", padding="max_length", truncation=True)
                    test_data = clip_processor(images=test_img, text=[""], return_tensors="pt", padding="max_length", truncation=True)

                    origin_vision_out = clip_model.vision_model(pixel_values=origin_data['pixel_values'].to(device), output_hidden_states=True, return_dict=True)
                    origin_text_out = clip_model.text_model(input_ids=origin_data['input_ids'].to(device), attention_mask=origin_data['attention_mask'].to(device), output_hidden_states=True, return_dict=True)
                    origin_image_hidden_states = list(origin_vision_out.hidden_states)
                    origin_text_hidden_states = list(origin_text_out.hidden_states)

                    origin_preds = va_model(image_hidden_states=origin_image_hidden_states, text_hidden_states=origin_text_hidden_states, attention_mask=origin_data['attention_mask'].to(device))
                    origin_valence_pred = origin_preds[0, 0].item()
                    origin_arousal_pred = origin_preds[0, 1].item()

                    test_vision_out = clip_model.vision_model(pixel_values=test_data['pixel_values'].to(device), output_hidden_states=True, return_dict=True)
                    test_text_out = clip_model.text_model(input_ids=test_data['input_ids'].to(device), attention_mask=test_data['attention_mask'].to(device), output_hidden_states=True, return_dict=True)
                    test_image_hidden_states = list(test_vision_out.hidden_states)
                    test_text_hidden_states = list(test_text_out.hidden_states)

                    test_preds = va_model(image_hidden_states=test_image_hidden_states, text_hidden_states=test_text_hidden_states, attention_mask=test_data['attention_mask'].to(device))
                    test_valence_pred = test_preds[0, 0].item()
                    test_arousal_pred = test_preds[0, 1].item()

                    v_err = math.fabs(valence_target - test_valence_pred)
                    a_err = math.fabs(arousal_target - test_arousal_pred)
 
                    dist_orig = math.sqrt((origin_valence_pred - valence_target)**2 + 
                                          (origin_arousal_pred - arousal_target)**2)

                    dist_edit = math.sqrt((test_valence_pred - valence_target)**2 + 
                                          (test_arousal_pred - arousal_target)**2)
                    
                    if dist_orig > 1e-8:
                        egs = max(0.0, (dist_orig - dist_edit) / dist_orig)
                        egs_scores.append(egs)

                    valence_errors.append(v_err)
                    arousal_errors.append(a_err)

            except Exception as e:
                print(f"Error processing {test_img_path}: {e}")
                continue

    v_err_array = np.array(valence_errors)
    a_err_array = np.array(arousal_errors)
    egs_array = np.array(egs_scores)

    avg_v_err = np.mean(v_err_array)
    avg_a_err = np.mean(a_err_array)
    avg_egs = np.mean(egs_array)

    std_v_err = np.std(v_err_array)
    std_a_err = np.std(a_err_array)
    std_egs = np.std(egs_array)

    print(f"Valence Error: Mean = {avg_v_err:.3f}, Std = {std_v_err:.3f}")
    print(f"Arousal Error: Mean = {avg_a_err:.3f}, Std = {std_a_err:.3f}")
    print(f"Emotion Gain Score (EGS): Mean = {avg_egs:.3f}, Std = {std_egs:.3f}")

    avg_cos, _ = semantic_diversity(target_img_dir, clip_model, clip_processor)
    print(f"Edit Diversity (Edit-D): {avg_cos:.3f}")



if __name__ == "__main__":
    origin_img_dir="/datasets/EmoEditSet/test_original_405"   #TODO
    target_img_dir = 'validate_images/'    # TODO
    device = torch.device("cuda:0")
    main(origin_img_dir=origin_img_dir, target_img_dir=target_img_dir, device=device)