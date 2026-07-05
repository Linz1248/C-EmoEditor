import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import pandas as pd
import argparse
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
import math
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import shutil
from PIL import Image
from torch.utils.data import Dataset
from model.emotion_mapper import *



def parse_args(edited_data_root, data_root, max_train_steps, pretrained_model_name_or_path, embed_save_name,
               num_train_epochs, seed, output_dir,diffusion_rate,instruction_rate,learning_rate,
               conditioning_dropout_prob=None, batch_size=64, gradient_accumulation_steps=1):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=conditioning_dropout_prob,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=pretrained_model_name_or_path,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--edited_data_root", type=str, default=edited_data_root, help="A folder containing the training data after edited."
    )
    parser.add_argument(
        "--data_root", type=str, default=data_root, help="A folder containing the training data."
    )
    parser.add_argument(
        "--embed_save_name", type=str, default=embed_save_name, help="The name of the folder where the embeddings will be saved."
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=seed, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=batch_size, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=max_train_steps,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=gradient_accumulation_steps,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--instruction_rate",
        type=float,
        default=instruction_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--diffusion_rate",
        type=float,
        default=diffusion_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def preprocess_images(image, resolution):
    original_images = convert_to_np(image, resolution)
    # We need to ensure that the original and the edited images undergo the same
    # augmentation transforms.
    images = torch.tensor(original_images)
    images = 2 * (images / 255) - 1
    final_image = images.reshape(-1, 3, resolution, resolution)
    return final_image


class EmoEditDataset_MultiData(Dataset):
    def __init__(
            self,
            edited_data_csv,
            data_root,
            embed_save_name,
            mixed_precision,
            size=256,
            repeats=1
    ):
        precision_mapping = {
            'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16
        }
        self.mixed_precision = precision_mapping[mixed_precision]
        self.edited_image_data = []
        self.data_root = data_root
        self.embeds_root = os.path.join(data_root, embed_save_name)

        df_edited = pd.read_csv(edited_data_csv)
        for index, row in df_edited.iterrows():
            edited_image_path = os.path.join(data_root, row["image_path"])
            self.edited_image_data.append((edited_image_path, row["valence"], row["arousal"], row["caption"]))

        self.size = size
        self.num_images = len(self.edited_image_data)
        self._length = self.num_images * repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        path, valence, arousal, edited_caption = self.edited_image_data[i % self.num_images]

        dataset_name = '_'.join(path.split('/')[-3].split('_')[:-1])
        img_name = '_'.join(path.split('/')[-1].split('_')[:-1])
        summary = path.split('/')[-1].split('.')[0].split('_')[-1]
        
        edited_image = Image.open(path).convert("RGB")
        origin_image_path = os.path.join(self.data_root, f"origin_image/{dataset_name}_crop/{img_name}.jpg")
        origin_image = Image.open(origin_image_path).convert("RGB")

        example["origin_image"] = preprocess_images(origin_image, self.size).squeeze(0)
        example["edited_image"] = preprocess_images(edited_image, self.size).squeeze(0)
        example["valence"] = torch.tensor([valence], dtype=torch.float32)
        example["arousal"] = torch.tensor([arousal], dtype=torch.float32)


         # 加载预提取的 CLIP 特征
        embed_path = os.path.join(self.embeds_root, f"origin_image/{dataset_name}_crop/{img_name}_{summary}.pt")
        try:
            precomputed = torch.load(embed_path, map_location="cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f"The precomputed feature file does not exist: {embed_path}")

        example['origin_image_embeds'] = precomputed['origin_image_embeds']
        example['instruction_embeds'] = precomputed['instruction_embeds']

        return example


def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )

    model = EmotionMapper()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    unet.requires_grad_(False)
    unet.eval()

    train_dataset = EmoEditDataset_MultiData(
        edited_data_csv=args.edited_data_root,
        data_root=args.data_root,
        embed_save_name=args.embed_save_name,
        mixed_precision=args.mixed_precision,
        size=args.resolution,
        repeats=args.repeats,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=args.dataloader_num_workers)

    optimizer_cls = torch.optim.AdamW

    parameters = list(model.parameters())

    optimizer = optimizer_cls(
        parameters,  # TODO need to change
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.validation_epochs is not None:
        args.validation_steps = args.validation_epochs * len(train_dataset) // accelerator.num_processes

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )


    text_encoder, optimizer, train_dataloader, lr_scheduler, model = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler, model
    )
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("train_logs")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            args.resume_from_checkpoint = None
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True, desc="Steps")
    rank = accelerator.process_index
    print(f"Process {rank}, DataLoader length: {len(train_dataloader)}")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model, unet):
                origin_image_embeds = batch['origin_image_embeds'].to(accelerator.device)
                instruction_embeds = batch['instruction_embeds'].to(accelerator.device)
                valence = batch['valence'].to(accelerator.device)
                arousal = batch['arousal'].to(accelerator.device)

                pred_instruction_embeds = model(valence=valence, arousal=arousal, image_embeds=origin_image_embeds)


                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_image"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["origin_image"].to(weight_dtype)).latent_dist.mode()

                encoder_hidden_states = pred_instruction_embeds.to(weight_dtype)

                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""], tokenizer).to(accelerator.device))[0].to(weight_dtype)
                    new_encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds
                else:
                    new_encoder_hidden_states = encoder_hidden_states
                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(concatenated_noisy_latents, timesteps, new_encoder_hidden_states, return_dict=False)[0]
    
                ldm_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                cosine_similarity = F.cosine_similarity(pred_instruction_embeds.float().view(bsz, -1), 
                                                        instruction_embeds.float().view(bsz, -1), dim=1)
                instruction_loss = torch.mean(1 - cosine_similarity)

                loss = args.instruction_rate * instruction_loss + args.diffusion_rate*ldm_loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                for tracker in accelerator.trackers:
                    tracker.writer.add_scalar("Total_Loss", loss, global_step)
                    tracker.writer.add_scalar("Diffusion_Loss", ldm_loss, global_step)
                    tracker.writer.add_scalar("Instruction_Loss", instruction_loss, global_step)


                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        save_dir = os.path.join(args.output_dir, f"emotion_mapper")
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                   os.path.join(args.output_dir, f"emotion_mapper/emotion_mapper_{global_step}.pth"))
                    

            logs = {"ldm_loss": ldm_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, "emotion_mapper_latest.pth"))
    accelerator.end_training()


if __name__ == "__main__":
    config_dic = {
        'edited_data_root': "/private/ljh/datasets/EmoEditSet/edited_image_captions_weighted.csv",
        'data_root': "/private/ljh/datasets/EmoEditSet",
        'pretrained_model_name_or_path': "/private/ljh/pretrained_models/instruct-pix2pix",
        'embed_save_name': "clip_cache",
        'max_train_steps': 30000,
        'num_train_epochs': 1000,
        'seed': 47500,
        'diffusion_rate': 0.2,
        'instruction_rate': 1.0,
        'learning_rate': 0.0001,
        'output_dir': 'experiment_results/runs_emotion_mapper',
        'batch_size': 64,
        'gradient_accumulation_steps': 2
    } #TODO
    os.makedirs(config_dic['output_dir'], exist_ok=True)
    args = parse_args(**config_dic)
    main(args)
