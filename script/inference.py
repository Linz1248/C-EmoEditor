import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from model.emotion_mapper import EmotionMapper
import random


def get_all_paths(dir_root, suffix: list):
    txt_file_list = []
    for root, _, file_path in os.walk(dir_root):
        for file in file_path:
            for suffix_name in suffix:
                if file.endswith(suffix_name):
                    tmp = os.path.join(root, file)
                    txt_file_list.append(tmp)
    txt_file_list.sort()
    return txt_file_list

def resize_image_to_512(img):
    img = img.convert('RGB')
    width, height = img.size

    if width < height:
        new_width = 512
        new_height = int((height / width) * 512)
    else:
        new_height = 512
        new_width = int((width / height) * 512)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

def generate_va_pairs(count=10, min_val=-3.0, max_val=3.0):
    va_data = []
    for _ in range(count):
        # 使用 random.uniform 获取指定范围内的浮点数
        valence = round(random.uniform(min_val, max_val), 2)
        arousal = round(random.uniform(min_val, max_val), 2)
        va_data.append((valence, arousal))
    return va_data


def main(checkpoint_path, image_dir, save_dir):
    device = torch.device("cuda:0")
    os.makedirs(save_dir, exist_ok=True)

    validation_images_path = get_all_paths(image_dir, ['jpg','png'])

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/private/ljh/pretrained_models/instruct-pix2pix", requires_safety_checker=False, safety_checker=None,
        torch_dtype=torch.float16)
    pipeline.set_progress_bar_config(disable=False)
    pipeline.to(device)

    clip_model = CLIPModel.from_pretrained("/private/ljh/pretrained_models/clip-vit-large-patch14").eval().to(device)
    processor = CLIPProcessor.from_pretrained("/private/ljh/pretrained_models/clip-vit-large-patch14")

    model = EmotionMapper()
    trained_para = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(trained_para)
    model.eval().to(device)

    intervals = [(-3.0, -1.0), (-1.0, 1.0), (1.0, 3.0)]

    for image_path in validation_images_path:
        image_name = os.path.basename(image_path).split('.')[0]
        with Image.open(image_path).convert("RGB") as validation_image:
            validation_image = resize_image_to_512(validation_image)

            with torch.no_grad():
                input_data = processor(images=validation_image,text=[''], return_tensors="pt", truncation=True, padding="max_length")
                output = clip_model(pixel_values=input_data['pixel_values'].to(clip_model.device), input_ids=input_data['input_ids'].to(clip_model.device))
                image_embeds = output["image_embeds"].unsqueeze(0)

                for v_range in intervals:
                    for a_range in intervals:

                        valence = round(random.uniform(v_range[0], v_range[1]), 2)
                        arousal = round(random.uniform(a_range[0], a_range[1]), 2)
                        
                        valence_tensor = torch.FloatTensor([[valence]]).to(device)
                        arousal_tensor = torch.FloatTensor([[arousal]]).to(device)

                        save_path = os.path.join(save_dir, f"{image_name}|valence={valence}|arousal={arousal}.jpg")
                        
                        pred_prompt_embeds = model(valence=valence_tensor, arousal=arousal_tensor, image_embeds=image_embeds)

                        image = pipeline(prompt_embeds=pred_prompt_embeds, image=validation_image,
                                        guidance_scale=7.5, image_guidance_scale=1.5, num_inference_steps=100).images[0]
                        image.save(os.path.join(save_path))


if __name__ == "__main__":

    checkpoint_path = '/private/ljh/backup_data/lanyun4090/C-EmoEdit/experiment_results_old/runs_emotion_mapper_v17_mseloss_0.2ldm/model_latest.pth'
    image_dir = "/private/ljh/datasets/EmoEditSet/test_original_405"


    save_dir = './inference_results'
    os.makedirs(save_dir, exist_ok=True)

    main(checkpoint_path=checkpoint_path, image_dir=image_dir, save_dir=save_dir)
