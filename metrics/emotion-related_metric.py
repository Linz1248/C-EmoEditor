
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import re
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from models.va_predictor_only_image import *
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


def DistanceOfSemantic(img0, img1, model, processor):  # pair_image should be two image, RBG, range(0~1)
    data_pro = processor(images=[img0, img1], return_tensors="pt", padding=True).to(model.device)
    data_pro = model.get_image_features(**data_pro)
    d = 1 - F.cosine_similarity(data_pro[0,:].unsqueeze(0), data_pro[1,:].unsqueeze(0))
    mse = F.mse_loss(data_pro[0,:].unsqueeze(0), data_pro[1,:].unsqueeze(0))
    return d.item(), mse.item()



def semantic_diversity(edited_dir, model, processor):
    """
    计算整个测试集的多样性指标 (Cosine Distance 和 MSE)
    """
    device = model.device
    
    # 1. 扫描编辑目录并按原始图像分组
    # key: 原始文件名, value: 编辑后的图像路径列表
    image_groups = defaultdict(list)
    all_files = [f for f in os.listdir(edited_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in all_files:
        # 提取原始文件名，例如 "2|valence=1.0|arousal=1.5.jpg" -> "2"
        orig_name = filename.split('|')[0]
        image_groups[orig_name].append(os.path.join(edited_dir, filename))
    
    print(f"找到 {len(image_groups)} 组图像。正在计算多样性...")

    total_cos_dist = []
    total_mse_dist = []

    with torch.no_grad():
        for orig_name, paths in tqdm(image_groups.items()):
            if len(paths) < 2:
                continue
                
            # 加载并处理这一组的所有图像
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            
            # 提取特征并归一化 (用于 Cosine)
            features = model.get_image_features(**inputs) # [N, Dim]
            features_norm = F.normalize(features, p=2, dim=-1)
            
            # 计算组内两两距离
            group_cos = []
            group_mse = []
            
            # 使用 combinations 生成两两组合的索引
            num_imgs = len(paths)
            for i, j in combinations(range(num_imgs), 2):
                # Cosine Distance = 1 - Cosine Similarity
                cos_sim = torch.sum(features_norm[i] * features_norm[j])
                group_cos.append(1 - cos_sim.item())
                
                # MSE Loss
                mse = F.mse_loss(features[i], features[j])
                group_mse.append(mse.item())
            
            # 计算该组的平均值
            if group_cos:
                total_cos_dist.append(sum(group_cos) / len(group_cos))
                total_mse_dist.append(sum(group_mse) / len(group_mse))

    avg_cos = sum(total_cos_dist) / len(total_cos_dist) if total_cos_dist else 0
    avg_mse = sum(total_mse_dist) / len(total_mse_dist) if total_mse_dist else 0
    
    return avg_cos, avg_mse


def semantic_operation_diversity(original_dir, edited_dir, model, processor):
    device = model.device
    
    image_groups = defaultdict(list)
    all_edited_files = [f for f in os.listdir(edited_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in all_edited_files:
        orig_name = filename.split('|')[0]  # 提取原文件名，如 "2"
        image_groups[orig_name].append(os.path.join(edited_dir, filename))
    
    # 建立原图文件的映射表 (处理扩展名不同的情况)
    # 比如 orig_name 是 "2"，对应的原图可能是 "2.jpg" 或 "2.png"
    all_orig_files = os.listdir(original_dir)
    orig_file_map = {os.path.splitext(f)[0]: os.path.join(original_dir, f) for f in all_orig_files}

    print(f"找到 {len(image_groups)} 组编辑结果。正在计算编辑操作多样性...")

    total_op_cos_dist = []
    total_op_mse_dist = []

    model.eval()
    with torch.no_grad():
        for orig_name, edit_paths in tqdm(image_groups.items()):
            if len(edit_paths) < 2 or orig_name not in orig_file_map:
                continue
            
            # --- Step A: 提取原图特征 ---
            orig_img = Image.open(orig_file_map[orig_name]).convert("RGB")
            orig_input = processor(images=orig_img, return_tensors="pt").to(device)
            orig_feat = model.get_image_features(**orig_input) # [1, Dim]
            
            # --- Step B: 提取所有编辑后的图像特征 ---
            edit_images = [Image.open(p).convert("RGB") for p in edit_paths]
            edit_inputs = processor(images=edit_images, return_tensors="pt", padding=True).to(device)
            edit_feats = model.get_image_features(**edit_inputs) # [10, Dim]
            
            # --- Step C: 计算编辑残差 (Delta Features) ---
            # deltas 表示从原图到编辑图的“语义位移”
            deltas = edit_feats - orig_feat  # 广播机制: [10, Dim] - [1, Dim] = [10, Dim]
            
            # 为计算余弦距离，对 delta 进行归一化
            deltas_norm = F.normalize(deltas, p=2, dim=-1)
            
            # --- Step D: 计算 Delta 之间的两两距离 ---
            group_cos = []
            group_mse = []
            num_edits = len(edit_paths)
            
            for i, j in combinations(range(num_edits), 2):
                # 衡量编辑方向的多样性
                cos_sim = torch.sum(deltas_norm[i] * deltas_norm[j])
                group_cos.append(1 - cos_sim.item())
                
                # 衡量编辑强度和内容差异的多样性
                mse = F.mse_loss(deltas[i], deltas[j])
                group_mse.append(mse.item())
            
            if group_cos:
                total_op_cos_dist.append(sum(group_cos) / len(group_cos))
                total_op_mse_dist.append(sum(group_mse) / len(group_mse))

    # 5. 计算全数据集平均值
    avg_cos = sum(total_op_cos_dist) / len(total_op_cos_dist) if total_op_cos_dist else 0
    avg_mse = sum(total_op_mse_dist) / len(total_op_mse_dist) if total_op_mse_dist else 0
    
    return avg_cos, avg_mse



def main(origin_img_dir, target_img_dir, device):
    test_img_paths = []
    for root, _, file_path in os.walk(target_img_dir):
        for file in file_path:
            if file.endswith("png") or file.endswith("jpg"):
                test_img_paths.append(os.path.join(root, file))

    # 初始化用于存储所有误差的列表
    valence_errors = []
    arousal_errors = []
    eas_scores = []  # 存储归一化情感增幅
    
    # Emotional Amplification Score(EAS)

    model = VA_Predictor().to(device)
    state = torch.load("experiment_results_old/runs_va_predictor_only_image/model_latest.pth", map_location=device)   # TODO
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    clip_processor = CLIPProcessor.from_pretrained("/private/ljh/pretrained_models/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("/private/ljh/pretrained_models/clip-vit-large-patch14").to(device)
    clip_model.requires_grad_(False).eval()

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
                    # origin_img = crop_img(origin_img)
                    # test_img = test_img.resize((512, 512))

                    origin_data = clip_processor(images=origin_img,  text=[""], return_tensors="pt", padding="max_length", truncation=True)
                    test_data = clip_processor(images=test_img,  text=[""], return_tensors="pt", padding="max_length", truncation=True)

                    origin_image_embeds = clip_model(pixel_values=origin_data['pixel_values'].to(device), input_ids=origin_data['input_ids'].to(device))["image_embeds"].squeeze(0)
                    test_image_embeds = clip_model(pixel_values=test_data['pixel_values'].to(device), input_ids=test_data['input_ids'].to(device))["image_embeds"].squeeze(0)

                    origin_valence_pred, origin_arousal_pred = model(origin_image_embeds)
                    test_valence_pred, test_arousal_pred = model(test_image_embeds)

                    origin_arousal_pred = origin_arousal_pred.detach().cpu().item()
                    origin_valence_pred = origin_valence_pred.detach().cpu().item()
                    test_arousal_pred = test_arousal_pred.detach().cpu().item()
                    test_valence_pred = test_valence_pred.detach().cpu().item()

                    # 计算绝对误差
                    v_err = math.fabs(valence_target - test_valence_pred)
                    a_err = math.fabs(arousal_target - test_arousal_pred)
 
                    # 1. 计算原图到目标的 L2 距离 (初始距离)
                    dist_orig = math.sqrt((origin_valence_pred - valence_target)**2 + 
                                          (origin_arousal_pred - arousal_target)**2)
                    # 2. 计算编辑图到目标的 L2 距离 (最终距离)
                    dist_edit = math.sqrt((test_valence_pred - valence_target)**2 + 
                                          (test_arousal_pred - arousal_target)**2)
                    
                    # 3. 计算 EAS (防止原图已在目标点导致分母为 0)
                    if dist_orig > 1e-8:
                        # eas = (dist_orig - dist_edit) / dist_orig
                        eas = max(0.0, (dist_orig - dist_edit) / dist_orig)
                        eas_scores.append(eas)
                    # -------------------------------

                    # 将误差添加到列表中
                    valence_errors.append(v_err)
                    arousal_errors.append(a_err)

            except Exception as e:
                print(f"Error processing {test_img_path}: {e}")
                continue

    # 将列表转换为 numpy 数组进行统计计算
    v_err_array = np.array(valence_errors)
    a_err_array = np.array(arousal_errors)
    eas_array = np.array(eas_scores)

    # 计算平均值
    avg_v_err = np.mean(v_err_array)
    avg_a_err = np.mean(a_err_array)
    avg_eas = np.mean(eas_array)

    # 计算标准差
    std_v_err = np.std(v_err_array)
    std_a_err = np.std(a_err_array)
    std_eas = np.std(eas_array)

    print(f"Valence Error: Mean = {avg_v_err:.3f}, Std = {std_v_err:.3f}")
    print(f"Arousal Error: Mean = {avg_a_err:.3f}, Std = {std_a_err:.3f}")
    print(f"Normalized Emotional Gain (NEG): Mean = {avg_eas:.3f}, Std = {std_eas:.3f}")

    avg_cos, avg_mse = semantic_diversity(target_img_dir, clip_model, clip_processor)
    print("==============================")
    print(f"Image Diversity (cos): {avg_cos:.3f}")
    print(f"Image Diversity (mse): {avg_mse:.3f}")

    avg_cos, avg_mse = semantic_operation_diversity(origin_img_dir, target_img_dir, clip_model, clip_processor)
    print("==============================")
    print(f"Operation Diversity (cos): {avg_cos:.3f}")
    print(f"Operation Diversity (mse): {avg_mse:.3f}")



if __name__ == "__main__":
    origin_img_dir="/private/ljh/datasets/EmoEditSet/test_original_405"
    target_img_dir = 'edited_results/validate_images/emotion_mapper_v17/'
    device = torch.device("cuda:1")
    main(origin_img_dir=origin_img_dir, target_img_dir=target_img_dir, device=device)