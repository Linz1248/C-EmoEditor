import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import math
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import argparse
import math
import numpy as np
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from model.mm_vap import VAPredictor
    

def scale_targets_to_0_1(valence, arousal):
    v = (valence + 3.0) / 6.0
    a = (arousal + 3.0) / 6.0
    return v, a

def inverse_scale_from_0_1(v, a):
    valence = v * 6.0 - 3.0
    arousal = a * 6.0 - 3.0
    return valence, arousal

def get_density(alist, vlist):
    import numpy as np
    from sklearn.neighbors import KernelDensity
    data = np.vstack([alist.T[0], vlist.T[0]]).T  
    kde = KernelDensity(kernel='gaussian', bandwidth="silverman")
    kde.fit(data)
    log_density = kde.score_samples(data)
    density = np.exp(log_density)
    return density

class TrainDataset(Dataset):
    def __init__(self, data_root: str, clip_processor):

        self.samples = []
        image_dir = os.path.join(data_root, 'images')
        df = pd.read_csv(os.path.join(data_root, 'labels.csv'))

        temp_samples_info = []
        val_list = []
        aro_list = []

        for index, row in df.iterrows():
            image_path = os.path.join(image_dir, f"{row['Id']}.jpg")
            if os.path.exists(image_path):
                v = row['Valence']
                a = row['Arousal']
                prompt = row['Emotional_Prompt']
                
                temp_samples_info.append((image_path, v, a, prompt))
                val_list.append(v)
                aro_list.append(a)

        v_np = np.array(val_list).astype(float).reshape(-1, 1)
        a_np = np.array(aro_list).astype(float).reshape(-1, 1)

        densities = get_density(a_np, v_np) 

        for i, (image_path, v, a, prompt) in enumerate(temp_samples_info):
            self.samples.append((image_path, v, a, prompt, densities[i]))

        self.clip_processor = clip_processor
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, v, a, prompt, density = self.samples[idx]
        example = {}

        image = Image.open(image_path).convert("RGB")
        data = self.clip_processor(images=image,  text=[prompt], return_tensors="pt", padding="max_length", truncation=True)
        example["pixel_values"] = data['pixel_values'].squeeze(0)
        example["input_ids"] = data['input_ids'].squeeze(0)
        example["attention_mask"] = data['attention_mask'].squeeze(0)

        v = float(v)
        a = float(a)
        v_scaled, a_scaled = scale_targets_to_0_1(v, a)
        valence_arousal = torch.tensor([v_scaled, a_scaled], dtype=torch.float32)

        example["valence_arousal"] = valence_arousal
        example["density"] = torch.tensor(density, dtype=torch.float32)
        return example

class ValDataset(Dataset):
    def __init__(self, data_root: str, clip_processor):
       
        self.samples = []
        image_dir = os.path.join(data_root, 'images')
        df = pd.read_csv(os.path.join(data_root, 'labels.csv'))

        temp_samples_info = []
        val_list = []
        aro_list = []

        for index, row in df.iterrows():
            image_path = os.path.join(image_dir, f"{row['Id']}.jpg")
            if os.path.exists(image_path):
                v = row['Valence']
                a = row['Arousal']
                prompt = row['Emotional_Prompt']
                
                temp_samples_info.append((image_path, v, a, prompt))
                val_list.append(v)
                aro_list.append(a)

        v_np = np.array(val_list).astype(float).reshape(-1, 1)
        a_np = np.array(aro_list).astype(float).reshape(-1, 1)

        densities = get_density(a_np, v_np) 

        for i, (image_path, v, a, prompt) in enumerate(temp_samples_info):
            self.samples.append((image_path, v, a, prompt, densities[i]))

        self.clip_processor = clip_processor
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, v, a, prompt, density = self.samples[idx]
        example = {}


        image = Image.open(image_path).convert("RGB")
        data = self.clip_processor(images=image,  text=[prompt], return_tensors="pt", padding="max_length", truncation=True)
        example["pixel_values"] = data['pixel_values'].squeeze(0)
        example["input_ids"] = data['input_ids'].squeeze(0)
        example["attention_mask"] = data['attention_mask'].squeeze(0)


        v = float(v)
        a = float(a)
        v_scaled, a_scaled = scale_targets_to_0_1(v, a)
        valence_arousal = torch.tensor([v_scaled, a_scaled], dtype=torch.float32)

        example["valence_arousal"] = valence_arousal
        example["density"] = torch.tensor(density, dtype=torch.float32)
        return example

    


def train(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(accelerator.device)
    clip_model.requires_grad_(False).eval()


    train_dataset = TrainDataset(data_root=os.path.join(args.data_root, 'train'), clip_processor=clip_processor)
    val_dataset = ValDataset(data_root=os.path.join(args.data_root, 'val'), clip_processor=clip_processor)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    vision_config = clip_model.config.vision_config
    text_config = clip_model.config.text_config
    model = VAPredictor(vision_config=vision_config, text_config=text_config)
    

    params = list(model.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)



    # prepare model, optimizer and dataloaders with accelerator
    model, optim, train_dataloader, val_dataloader  = accelerator.prepare(
        model, optim, train_dataloader, val_dataloader
    )


    best_loss = float('inf')
    global_step = 0
    if accelerator.is_main_process:
        accelerator.init_trackers("train_logs")

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Training_epoch {epoch}/{args.epochs}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                pixel_values = batch['pixel_values'].to(accelerator.device)
                input_ids = batch['input_ids'].to(accelerator.device)
                attention_mask = batch['attention_mask'].to(accelerator.device)
                valence_arousal = batch['valence_arousal'].to(accelerator.device)
                density = batch['density'].to(accelerator.device)

                vision_outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
                text_outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

                image_hidden_states = list(vision_outputs.hidden_states)
                text_hidden_states = list(text_outputs.hidden_states)

                preds = model(image_hidden_states=image_hidden_states, text_hidden_states=text_hidden_states)


                loss = F.mse_loss(preds, valence_arousal)

                avg_loss = accelerator.gather_for_metrics(loss.detach()).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                total_loss += train_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                # pbar.update(1)
                global_step += 1
                for tracker in accelerator.trackers:
                    tracker.writer.add_scalar("train_step_loss", train_loss, global_step)

                train_loss = 0.0

            pbar.set_postfix({"loss": loss.detach().item()})

        
        # End of epoch: run validation on validation set
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            pbar.close()
            epoch_avg_loss = total_loss / steps_per_epoch
            print(f"Epoch {epoch}/{args.epochs} | Training avg loss: {epoch_avg_loss:.4f}")

            # log to trackers
            for tracker in accelerator.trackers:
                tracker.writer.add_scalar("train_epoch_loss", epoch_avg_loss, epoch)

            os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
            if epoch % args.save_per_epochs == 0:
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, f"checkpoints/model_{epoch}.pth"))
        
        model.eval()
        total_loss_val = torch.tensor(0.0, device=accelerator.device, dtype=torch.float64)
        total_samples = 0

        pbar_val = tqdm(val_dataloader, desc=f"Evaluating_epoch {epoch}/{args.epochs}", disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch in pbar_val:
                pixel_values = batch['pixel_values'].to(accelerator.device)
                input_ids = batch['input_ids'].to(accelerator.device)
                attention_mask = batch['attention_mask'].to(accelerator.device)
                valence_arousal = batch['valence_arousal'].to(accelerator.device)
                density = batch['density'].to(accelerator.device)

                vision_outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
                text_outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

                image_hidden_states = list(vision_outputs.hidden_states)
                text_hidden_states = list(text_outputs.hidden_states)

                preds = model(image_hidden_states=image_hidden_states, text_hidden_states=text_hidden_states)

                local_batch_loss = F.mse_loss(preds, valence_arousal)



                gathered_losses_val = accelerator.gather_for_metrics(local_batch_loss.detach())
                total_loss_val += torch.sum(gathered_losses_val)
                total_samples += gathered_losses_val.numel()

        if accelerator.is_main_process:
            if total_samples == 0:
                val_loss = float('inf')
            else:
                val_loss = total_loss_val.item() / total_samples
        
        if accelerator.is_main_process:
            pbar_val.close()
            print(f"Epoch {epoch}/{args.epochs} | Validation loss: {val_loss:.4f}")

            for tracker in accelerator.trackers:
                tracker.writer.add_scalar("val_loss", val_loss, epoch)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, "mm_vap_best.pth"))


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, "mm_vap_latest.pth"))
        print("complete!")
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/datasets/EmotiCrafter', help='Root folder prepended to relative image paths')
    parser.add_argument('--output_dir', type=str, default='runs/mm_vap', help='Where to save checkpoints')
    parser.add_argument('--clip_model', type=str, default='/pretrained_models/clip-vit-base-patch32', help='HuggingFace CLIP model identifier')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_per_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )


    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    train(args)
