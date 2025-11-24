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
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
from einops import rearrange, repeat
import time
import torch.nn.functional as F

import sys


from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.tokenizers.tokenizer import ResidualBlock, CausalConv3d
from cosmos_predict2.configs.base.config_text2image import (
    get_cosmos_predict2_text2image_pipeline,
)

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline


from imaginaire.lazy_config import LazyDict, instantiate

config = get_cosmos_predict2_text2image_pipeline(model_size="0.6B", fast_tokenizer=True)

### MiniTrainDIT
config.dit_path = '/work/models/base/model_fast_tokenizer.pt'
config.tokenizer["vae_pth"] = '/work/models/base/tokenizer_fast.pth'
config.guardrail_config.enabled=False

from model import make_1step_sched_base as make_1step_sched  # , my_vae_encoder_fwd, my_vae_decoder_fwd


def load_ckpt_from_state_dict(net_pix2pix, net_disc, optimizer, optimizer_disc, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
        
    net_pix2pix.unet.load_state_dict(sd["state_dict_unet"])
    net_pix2pix.vae.load_state_dict(sd["state_dict_vae"])
    
    net_disc.load_state_dict({k[7:]: v for k, v in sd["net_disc"].items()})
    
    optimizer.load_state_dict(sd["optimizer"])
    optimizer_disc.load_state_dict(sd["optimizer_disc"])
    
    print()
    print('!!!! loading, load pretrained weight from', pretrained_path)
    print()
    return net_pix2pix, net_disc, optimizer, optimizer_disc


def save_ckpt(net_pix2pix, net_disc, optimizer, optimizer_disc, outf, train_full_unet=False, freeze_vae=False):
    sd = {}

    sd["state_dict_unet"] = net_pix2pix.unet.state_dict()
    sd["state_dict_vae"] = net_pix2pix.vae.state_dict()
        
    sd["net_disc"] = net_disc.state_dict()
    sd["optimizer"] = optimizer.state_dict()
    sd["optimizer_disc"] = optimizer_disc.state_dict()
    
    torch.save(sd, outf)


CACHE_T = 2


def my_vae_encoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    l_blocks = []
    # downsamples
    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
            
        l_blocks.append(x) 

    # middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    # head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
            
    self.current_down_blocks = l_blocks
    return x


def my_vae_decoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    # conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

        
    skip_convs = [
        self.skip_conv_0, 
        self.skip_conv_1, 
        self.skip_conv_2, 
        self.skip_conv_3, 
        self.skip_conv_4, 
        self.skip_conv_5,
        self.skip_conv_6,
        self.skip_conv_7,
        self.skip_conv_8
    ]
    skip_acts = self.incoming_skip_acts
    
    
    # middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    enc_dec_mapping = {0:0, 1:1, 2:2, 5:3, 6:4, 8:6, 10:7, 12:9, 14:10}
    
    for dec_idx, layer in enumerate(self.upsamples):
        
        if dec_idx in enc_dec_mapping:
            enc_indx = enc_dec_mapping[dec_idx]
            layer_index = list(enc_dec_mapping.keys()).index(dec_idx)
            
            skip_input = skip_acts[::-1][enc_indx]
            skip_in = skip_convs[layer_index](skip_input)  # 1x1 conv
            x = x + skip_in  # add skip
        
        
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
            

    # head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


import time


class Pix2Pix_Turbo(torch.nn.Module):
    
    def __init__(self, experiment_name = None, s3_checkpoint_dir = None, pretrained_path = None,
                 ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4, hf_path = None,
                 unet_in_channels=4, freeze_vae_encoder=False, 
                 freeze_vae=False, train_full_unet=True, timestep=999, 
                 use_sched = False, vae_skip_connection = False, batch_size = 1):
        super().__init__()
        
        self.experiment_name = experiment_name
        self.s3_checkpoint_dir = s3_checkpoint_dir
        self.batch_size = batch_size

        
        self.timesteps = torch.tensor([timestep], device="cuda")#.half()#.long()
        self.timesteps = torch.cat([self.timesteps] * batch_size, 0)
        
        self.timesteps_int =  timestep
        
        self.train_full_unet = train_full_unet
        self.freeze_vae = freeze_vae
        self.freeze_vae_encoder = freeze_vae_encoder
        self.vae_skip_connection = vae_skip_connection    
        
        self.use_sched = use_sched
        if self.use_sched:
            self.sched = None #make_1step_sched()
         

        self.initialize_cosmos_model()
        self.set_train()

        if pretrained_path is not None:
            print('loading from', pretrained_path)
            sd = torch.load(pretrained_path, map_location="cpu")

            self.unet.load_state_dict(sd["state_dict_unet"], strict=False)
            self.vae.load_state_dict(sd["state_dict_vae"], strict=False)

            
        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def sample_batch_image(self):
        #h, w = 384, 640
        #h, w = 768, 1360
        h, w = 544, 960
        
        batch_size = self.batch_size
        data_batch = {
            "dataset_name": "image_data",
            "images": torch.zeros(batch_size, 3, h, w).cuda(),
            "t5_text_embeddings": torch.zeros(batch_size, 512, 1024).cuda(),
            "fps": torch.ones((batch_size,)).cuda() * 24,
            "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
        }
        return data_batch


    def initialize_cosmos_model(self):
        
        model = Text2ImagePipeline.from_config(
            config,
            dit_path=config.dit_path,
            use_text_encoder = False
        )
        
        
        ##### conditioning
        conditioner = model.conditioner
        data_batch = self.sample_batch_image()
        is_image_batch = True
        
        condition, uncondition = conditioner.get_condition_uncondition(data_batch)
        del condition
        
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        self.condition = uncondition

        self.unet = model#.net # MiniTrainDIT
        vae = model.tokenizer #model.tokenizer <projects.cosmos.diffusion.v2.tokenizers.wan2pt1.Wan2pt1VAEInterface object at 0x155401d12c20>
        
        
        self.sigma_data = model.sigma_data  
        self.vae = vae
              

        print('=' * 50)
        print('SUCCESS in initializing Cosmos Model')
        print(f"Number of parameters in UNet: {sum(p.numel() for p in self.unet.parameters() ) / 1e6:.2f}M")
        print(f"Number of parameters in VAE: {sum(p.numel() for p in self.vae.parameters()) / 1e6:.2f}M")
        print('=' * 50)

        self.unet.to("cuda")
        self.vae.to("cuda")
        
    def vae_encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state) * self.sigma_data        

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.sigma_data)
        
    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        
        if self.train_full_unet:
            self.unet.requires_grad_(True)
            #self.unet.net_ema.requires_grad_(False)
        else:
            raise ValueError('!! train partial Unet not implemented')
            
            
        self.vae.train()
        self.vae.requires_grad_(True)
        
        for name, param in self.vae.named_parameters():
            if "time_conv" in name:
                param.requires_grad = False
                print("set ", name, "grad to be false")
                
            
        if self.freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

        if self.freeze_vae_encoder:
            self.vae.encoder.eval()
            self.vae.encoder.requires_grad_(False)

        
    def forward(self, x, timesteps=None):

        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        assert len(x.shape) == 4 
        
        x = x[:, :, None, :, :]

        unet_input = self.vae_encode(x)
        
        sigma_B_T = self.timesteps.to(dtype=unet_input.dtype) / 1000
        
        model_pred = self.unet.denoise(xt_B_C_T_H_W = unet_input, 
                               sigma = sigma_B_T,
                               condition = self.condition ).x0
       
        z_denoised = model_pred
        
        if self.vae_skip_connection:
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            
        
        output_image = self.vae_decode(z_denoised)
        
        output_image = output_image[:, :, 0]

        return output_image

    def save_model(self, outf, net_disc, optimizer, optimizer_disc):
        sd = {}
        
        sd["state_dict_unet"] = self.unet.state_dict()
        sd["state_dict_vae"] = self.vae.state_dict()
        
        sd["net_disc"] = net_disc.state_dict()
        sd["optimizer"] = optimizer.state_dict()
        sd["optimizer_disc"] = optimizer_disc.state_dict()
        
        torch.save(sd, outf)
