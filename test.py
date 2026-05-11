import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import CLIPModel, CLIPProcessor
from model.emotion_mapper import EmotionMapper


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


def main(args):
    device = torch.device("cuda:0")
    os.makedirs(args.save_dir, exist_ok=True)

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pix2pix_path, requires_safety_checker=False, safety_checker=None,
        torch_dtype=torch.float16)
    pipeline.set_progress_bar_config(disable=False)
    pipeline.to(device)

    clip_model = CLIPModel.from_pretrained(args.clip_path).eval().to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_path)

    model = EmotionMapper()
    trained_para = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(trained_para)
    model.eval().to(device)

    image_name = os.path.basename(args.image_path).split('.')[0]
    with Image.open(args.image_path).convert("RGB") as input_image:
        input_image = resize_image_to_512(input_image)

        with torch.no_grad():
            input_data = processor(images=input_image, text=[''], return_tensors="pt", truncation=True, padding="max_length")
            output = clip_model(pixel_values=input_data['pixel_values'].to(clip_model.device), input_ids=input_data['input_ids'].to(clip_model.device))
            image_embeds = output["image_embeds"].unsqueeze(0)

            valence_tensor = torch.FloatTensor([[args.valence]]).to(device)
            arousal_tensor = torch.FloatTensor([[args.arousal]]).to(device)

            save_path = os.path.join(args.save_dir, f"{image_name}_v{args.valence}_a{args.arousal}.jpg")

            pred_prompt_embeds = model(valence=valence_tensor, arousal=arousal_tensor, image_embeds=image_embeds)

            image = pipeline(prompt_embeds=pred_prompt_embeds, image=input_image,
                             guidance_scale=args.guidance_scale, image_guidance_scale=args.image_guidance_scale,
                             num_inference_steps=args.num_steps).images[0]
            image.save(save_path)
            print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image emotion editing inference")
    parser.add_argument("--image_path", type=str, default="example/1.png", help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="/private/ljh/backup_data/lanyun4090/C-EmoEdit/experiment_results_old/runs_emotion_mapper_v17_mseloss_0.2ldm/model_latest.pth", help="Path to the EmotionMapper checkpoint")
    parser.add_argument("--valence", type=float, default=1.5, help="Target valence value (-3.0 ~ 3.0)")
    parser.add_argument("--arousal", type=float, default=3.0, help="Target arousal value (-3.0 ~ 3.0)")
    parser.add_argument("--save_dir", type=str, default="./test_results", help="Directory to save results")
    parser.add_argument("--pix2pix_path", type=str, default="/private/ljh/pretrained_models/instruct-pix2pix")
    parser.add_argument("--clip_path", type=str, default="/private/ljh/pretrained_models/clip-vit-large-patch14")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_steps", type=int, default=100)
    args = parser.parse_args()

    main(args)
