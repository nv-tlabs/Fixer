# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gc
import lpips
import clip
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo, load_ckpt_from_state_dict, save_ckpt
from utils.training_utils import parse_args_paired_training, PairedDatasetV2
from utils.style_loss import style_loss


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,

    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pix2pix = Pix2Pix_Turbo(
        freeze_vae_encoder=args.freeze_vae_encoder, 
        freeze_vae=args.freeze_vae, 
        train_full_unet=args.train_full_unet, 
        timestep=args.timestep,
        use_sched=args.use_sched,
        vae_skip_connection=args.vae_skip_connection,
        pretrained_path=args.pretrained_path
    )
    net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if not args.swinir: 
            if is_xformers_available():
                net_pix2pix.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)
    
    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    if args.train_full_unet:
        print("="*50)
        print('adding unet parameters')
        print("="*50)
        layers_to_opt += list(net_pix2pix.unet.parameters())
    #if not args.freeze_vae_encoder:
    if not args.freeze_vae:
        if args.freeze_vae_encoder:
            print("="*50)
            print('adding vae decoder parameters')
            print("="*50)
            layers_to_opt += list(net_pix2pix.vae.decoder.parameters())            
        else:
            print("="*50)
            print('adding whole vae parameters')
            print("="*50)
            layers_to_opt += list(net_pix2pix.vae.parameters())
    
    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedDatasetV2(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train")
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDatasetV2(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test")
    random.Random(42).shuffle(dataset_val.img_names)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_pix2pix, net_disc, optimizer, optimizer_disc = load_ckpt_from_state_dict(
                net_pix2pix, net_disc, optimizer, optimizer_disc, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_pix2pix, net_disc, optimizer, optimizer_disc = load_ckpt_from_state_dict(
                net_pix2pix, net_disc, optimizer, optimizer_disc, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips, net_vgg = accelerator.prepare(net_clip, net_lpips, net_vgg)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                assert len(x_src.shape) == 4
                assert len(x_tgt.shape) == 4
                
                # forward pass
                x_tgt_pred = net_pix2pix(x_src)       
                         
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                
                crop_h = crop_w = random.randint(128, 512)
                top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        
                loss_lpips = net_lpips(crop(x_tgt_pred, top, left, crop_h, crop_w),
                                       crop(x_tgt, top, left, crop_h, crop_w)).mean() * args.lambda_lpips
                
                loss = loss_l2 + loss_lpips
                
                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        # x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (400, 400), mode="bilinear", align_corners=False)
                        # crop_h, crop_w = 400, 400
                        crop_h, crop_w = 512, 512
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        # x_tgt_renorm = F.interpolate(x_tgt_renorm, (400, 400), mode="bilinear", align_corners=False)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        
                        loss_gram = style_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                    
                
                # CLIP similarity loss
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    if args.lambda_gan > 0:
                        logs["lossG"] = lossG.detach().item()
                        logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_pix2pix).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_pix2pix), net_disc, optimizer, optimizer_disc, outf, train_full_unet=args.train_full_unet, freeze_vae=args.freeze_vae)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src)
                                
                                if step % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(x_src[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(x_tgt[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(x_tgt_pred[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                

                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                # compute clip similarity loss
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_clipsim.append(clipsim.item())
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].float().cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
