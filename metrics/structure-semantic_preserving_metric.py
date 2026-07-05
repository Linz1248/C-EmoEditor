import argparse
import os
import re
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import lpips
from tqdm import tqdm

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

def count_ssim(test_img_array, origin_img_array):
    return ssim(test_img_array, origin_img_array, channel_axis=2)


def count_psnr(test_img_array, origin_img_array):
    return psnr(test_img_array, origin_img_array, data_range=origin_img_array.max() - origin_img_array.min())

def count_mse(test_img_array, origin_img_array):
    return np.mean((test_img_array - origin_img_array) ** 2)

def count_lpips(test_img_array, origin_img_array, lpips_alex, device):

    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])

    in0 = transform(test_img_array)
    in1 = transform(origin_img_array)

    in0, in1 = in0.to(device), in1.to(device)

    lpips_alex = lpips_alex.to(device)
    lpips_score = lpips_alex(in0, in1)
    return lpips_score.item()


def count_CLIP_I(test_img, origin_img, model, processor):

    data_pro = processor(images=[test_img, origin_img], return_tensors="pt", padding=True).to(model.device)
    data_pro = model.get_image_features(**data_pro)
    d = F.cosine_similarity(data_pro[0, :].unsqueeze(0), data_pro[1, :].unsqueeze(0))
    return d.item()


@torch.no_grad()
class DINO_extractor():
    def __init__(self, device):
        self.processor = AutoImageProcessor.from_pretrained("/private/ljh/pretrained_models/dinov2-base") # TODO
        self.model = AutoModel.from_pretrained("/private/ljh/pretrained_models/dinov2-base").to(device) # TODO
        self.origin_image_dino = {}

    @torch.no_grad()
    def extract_dino(self, img):
        inputs = self.processor(images=[img], return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        feature = outputs.last_hidden_state.mean(dim=1)
        feature = feature.unsqueeze(1)
        return feature[0]

    @torch.no_grad()
    def count_dino_i(self,test_img, origin_img, origin_img_name):
        if origin_img_name not in self.origin_image_dino:
            self.origin_image_dino[origin_img_name] = self.extract_dino(origin_img).cpu()
        dino_sim = F.cosine_similarity(self.origin_image_dino[origin_img_name], self.extract_dino(test_img).cpu()).item()
        return dino_sim

def main(origin_img_dir, target_img_dir):
    device = torch.device("cuda:0")
    dino_Ex = DINO_extractor(device=device)
    lpips_alex = lpips.LPIPS(net='alex')
    model = CLIPModel.from_pretrained("/private/ljh/pretrained_models/clip-vit-base-patch32").to(device) # TODO
    processor = CLIPProcessor.from_pretrained("/private/ljh/pretrained_models/clip-vit-base-patch32") # TODO
    test_img_paths = []
    for root, _, file_path in os.walk(target_img_dir):
        for file in file_path:
            if file.endswith("png") or file.endswith("jpg"):
                test_img_paths.append(os.path.join(root, file))
    img_metrics = {
        'SSIM': [],
        'PSNR': [],
        'LPIPS': [],
        'MSE': [],
        'CLIP-I': [],
        'DINO-I': [],
    }
    for test_img_path in tqdm(test_img_paths):
        test_img_name = test_img_path.split('/')[-1].split('_')[0]
        test_img_name = test_img_name.split('|')[0]
        for suffix in ['jpg','png']:
            origin_img_path = os.path.join(origin_img_dir, f"{test_img_name}.{suffix}")

            if os.path.isfile(origin_img_path) is True:
                break
        with Image.open(test_img_path) as test_img, Image.open(origin_img_path) as origin_img:
            origin_img = crop_img(origin_img)
            test_img = test_img.resize((512, 512))
            test_img_array = np.array(test_img)
            origin_img_array = np.array(origin_img)

            ssim = count_ssim(test_img_array, origin_img_array)
            img_metrics['SSIM'].append(ssim)

            psnr = count_psnr(test_img_array, origin_img_array)
            img_metrics['PSNR'].append(psnr)

            current_mse = count_mse(test_img_array, origin_img_array)
            img_metrics['MSE'].append(current_mse)

            current_lpips = count_lpips(test_img_array, origin_img_array, lpips_alex, device=device)
            img_metrics['LPIPS'].append(current_lpips)

            current_clip_i = count_CLIP_I(test_img, origin_img, model, processor)
            img_metrics['CLIP-I'].append(current_clip_i)

            current_DINO = dino_Ex.count_dino_i(test_img_array, origin_img_array, test_img_name)
            img_metrics['DINO-I'].append(current_DINO)


    averages = {key: sum(values) / len(values) if values else 0 for key, values in img_metrics.items()}
    with open(os.path.join(target_img_dir,'img_metrics_averages.txt'), 'a') as file:
        for key, avg in averages.items():
            file.write(f"{key}: {avg:.6f}\n")


if __name__ == "__main__":
    origin_img_dir = "/private/ljh/datasets/EmoEditSet/test_original_405/" # TODO
    target_img_dir = 'edited_results/validate_images/'         # TODO
    main(origin_img_dir=origin_img_dir, target_img_dir=target_img_dir)