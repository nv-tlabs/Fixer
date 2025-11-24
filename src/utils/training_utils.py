# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
from io import BytesIO
from packaging import version as pver
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_reg", default=10, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--lambda_tv", default=1.0, type=float)
    parser.add_argument("--N_resize", default=2.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--freeze_vae_encoder", action="store_true")
    parser.add_argument("--freeze_vae", action="store_true")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--train_full_unet", action="store_true")
    parser.add_argument("--unet_in_channels", default=4, type=int)
    parser.add_argument("--timestep", default=999, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--hf_path", type=str, default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    
    # cosmos parameters
    parser.add_argument("--experiment_name", type=str, default="official_runs_t2i_fast_205_stage3_0p6b_1024res_synthrealmix_1_1_filtered_with_alpamayo")
    parser.add_argument("--s3_checkpoint_dir", type=str, default="None")
    
    parser.add_argument("--av_model_path", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--swinir", default=False, action="store_true")
    parser.add_argument("--unetscratch", default=False, action="store_true")
    parser.add_argument("--use_sched", action="store_true")
    parser.add_argument("--use_large_postnet", action="store_true")
    parser.add_argument("--vae_skip_connection", action="store_true")
    parser.add_argument("--fix_deconv", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_paired_l2", default=1.0, type=float)
    parser.add_argument("--lambda_paired_lpips", default=5.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--paired_ratio", type=float, default=0.5)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--tracker_run_name", type=str, required=True)
    parser.add_argument("--tracker_run_id", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    # resume
    parser.add_argument("--resume", default=None, type=str)
    
    args = parser.parse_args()
    return args


def build_transform(image_prep, interpolation = Image.LANCZOS):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resized_random_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop((512, 512)),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=interpolation),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=interpolation)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=interpolation)
        ])
    elif image_prep == "resize_576x1024":
        T = transforms.Compose([
            transforms.Resize((576, 1024), interpolation=interpolation),
        ])
    elif image_prep == "resize_288x512":
        T = transforms.Compose([
            transforms.Resize((288, 512), interpolation=interpolation),
        ])
    elif image_prep == "resize_384x704":
        T = transforms.Compose([
            transforms.Resize((384, 704), interpolation=interpolation),
        ])
    elif image_prep == "resize_448x832":
        T = transforms.Compose([
            transforms.Resize((448, 832), interpolation=interpolation),
        ])        
    elif image_prep == "resize_200x360":
        T = transforms.Compose([
            transforms.Resize((200, 360), interpolation=interpolation),
        ])
    elif image_prep == "resize_544x960":
        T = transforms.Compose([
            transforms.Resize((544, 960), interpolation=interpolation),
        ])
    elif image_prep == "resize_384x672":
        T = transforms.Compose([
            transforms.Resize((384, 672), interpolation=interpolation),
        ])
    elif image_prep == "resize_768x1360":
        T = transforms.Compose([
            transforms.Resize((768, 1360), interpolation=interpolation),
        ])
    elif image_prep == "resize_416x736":
        T = transforms.Compose([
            transforms.Resize((416, 736), interpolation=interpolation),
        ])
    elif image_prep == "resize_1088x1920":
        T = transforms.Compose([
            transforms.Resize((1088, 1920), interpolation=interpolation),
        ])
    elif image_prep == "resize_2176x3840":
        T = transforms.Compose([
            transforms.Resize((2176, 3840), interpolation=interpolation),
        ])
    elif image_prep == "resize_576x1024_cropcar":
        T = transforms.Compose([
            transforms.Resize((576, 1024), interpolation=interpolation),
            transforms.Lambda(lambda img: F.crop(img, 0, 0, 416, 1024)),    # crop out the car
        ])
    elif image_prep == "resize_576x1024_cropcar_randomcrop_400x400":
        T = transforms.Compose([
            transforms.Resize((576, 1024), interpolation=interpolation),
            transforms.Lambda(lambda img: F.crop(img, 0, 0, 416, 1024)),    # crop out the car
            transforms.RandomCrop((400, 400)),
        ])
    elif image_prep == "resize_1024x576":
        T = transforms.Compose([
            transforms.Resize((1024, 576), interpolation=interpolation),
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T



class PairedDatasetV2(torch.utils.data.Dataset):

    @staticmethod
    def _load_from_json(json_path, split):
        """Load data from JSON file"""
        with open(json_path, "r") as f:
            json_data = json.load(f)[split]
        return json_data, list(json_data.keys())
        
    @staticmethod    
    def _load_from_directory(dir_path, split):
        """Load data from directory structure"""
        input_dir = os.path.join(dir_path, f'{split}_A')
        output_dir = os.path.join(dir_path, f'{split}_B')
        caption_file = os.path.join(dir_path, f'{split}_prompts.json')

        with open(caption_file, "r") as f:
            captions = json.load(f)
        input_files = list(captions.keys())

        new_data = {}
        for img_file in tqdm(input_files, desc=f"Loading {split} from {os.path.basename(dir_path)}"):
            new_data[img_file] = {
                "image": os.path.join(input_dir, img_file),
                "target_image": os.path.join(output_dir, img_file),
                "prompt": captions[img_file] # Empty string if no caption exists
            }
        
        return new_data, input_files
            
    def __init__(self, dataset_folder, split, image_prep, tokenizer=None):
        
        super().__init__()
        self.data = {}
        self.img_names = []
        
        # Split the dataset_folder string into individual paths
        data_sources = [source.strip() for source in dataset_folder.split(',')]
        
        for source in data_sources:
            if source.endswith('.json'):
                data, file_names = self._load_from_json(source, split)
            else:
                data, file_names = self._load_from_directory(source, split)

            self.data.update(data)
            self.img_names.extend(file_names)

        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer
        
    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.img_names)

    def preprocess_image(self, input_img):
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t) 
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        return img_t

    
    def __getitem__(self, idx):

        img_name = self.img_names[idx]
        
        input_img = self.data[img_name]["image"]
        output_img = self.data[img_name]["target_image"]
        
        ref_img = None
        
        caption = self.data[img_name]["prompt"]
        
                  
        input_img = Image.open(input_img)
        output_img = Image.open(output_img)
        
        if ref_img is not None:
            ref_img = [ref_img] if type(ref_img) != list else ref_img
            
            ref_img = [Image.open(path) for path in ref_img]


        # input images scaled to -1,1
        img_t = self.preprocess_image(input_img)
                
        # output images scaled to -1,1
        output_t = self.preprocess_image(output_img)
        
        ref_t = None
        if ref_img is not None:
            ref_t_all = []
            for tmp_img in ref_img:
                ref_t = self.preprocess_image(tmp_img)
                ref_t_all.append(ref_t)
            
            img_t = torch.stack([img_t] + ref_t_all, dim=0)
            output_t = torch.stack([output_t] + ref_t_all, dim=0)
        

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption
        }
        
            
        return out

