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

import sys
import warnings
from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo

from tqdm import tqdm
import time

import torch
import argparse
import transformer_engine as te #important
from glob import glob
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from natsort import natsorted

import imageio

# Suppress warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)

def save_folder2video(save_dir, remove_key = 0):
    if remove_key==0:
        video_file = os.path.join(save_dir,  save_dir.split('/')[-1] + "_video.mp4")
    else:
        video_file = os.path.join(save_dir,  "video_removekey_" + str(remove_key) + ".mp4")
    im_files = natsorted(glob(os.path.join(save_dir, "*.png")) + glob(os.path.join(save_dir, "*.jpg")))
    # Loads all images at once - can run out of memory if too many frames
    ims: list[np.ndarray] = [imageio.v2.imread(f) for f in im_files]

    # chop image if dimension not divisible by 2 to fix the ffmpeg error
    for i in range(len(ims)):
        if ims[i].shape[0] % 2 == 1:
            ims[i] = ims[i][:-1, :, :]
        if ims[i].shape[1] % 2 == 1:
            ims[i] = ims[i][:, :-1, :]

    # Results in an Array is not the same as ArrayLike annotation error, so we suppress it
    if remove_key!=0:
        ims = [ims[i] for i in range(len(ims)) if i % remove_key != 0]
        
    
    imageio.v2.mimwrite(video_file, ims, fps=30, macro_block_size=1)  # type: ignore
    imageio.v2.mimwrite(video_file.replace("video.mp4", "video_10fps.mp4"), 
                        ims[::3], fps=10, macro_block_size=1)  # type: ignore
    imageio.v2.mimwrite(video_file.replace("video.mp4", "video_15fps.mp4"), 
                        ims[::2], fps=15, macro_block_size=1)  # type: ignore


def encode_step(vae, batch_size: int, h: int, w: int, dtype: torch.dtype, device: torch.device):
    x = torch.randn(batch_size, 3, 1, h, w, dtype=dtype, device=device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        #print('x.shape', x.shape)
        latent = vae.encode(x)
        

def diffuse_step(model, condition, sigma_B_T, batch_size: int, c: int, h: int, w: int, dtype: torch.dtype, device: torch.device):
    compression = 8
    x = torch.randn(batch_size, 16, 1, 
                    h // compression, w // compression,
                    dtype=dtype, device=device)
    
    
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        output = model.denoise(xt_B_C_T_H_W = x, 
                               sigma = sigma_B_T,
                               condition = condition).x0


def decode_step(vae, batch_size: int, h: int, w: int, dtype: torch.dtype, device: torch.device):
    compression = 8
    latent = torch.randn(batch_size, 16, 1, h // compression, w // compression, dtype=dtype, device=device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        y = vae.decode(latent)


def model_inference(model, batch_size: int, h: int, w: int, dtype: torch.dtype, device: torch.device,
                    x = None):
    if x is None:
        x = torch.randn(batch_size, 3,  h, w, dtype=dtype, device=device)
        
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        output = model(x)
        return output


def warmup_model(model: torch.nn.Module, batch_size: int, h: int, w: int, 
                 dtype: torch.dtype, device: torch.device, n: int = 10) -> None:
    """Warmup the model with dummy inference runs.
    
    Args:
        model: The compiled model to warmup
        batch_size: Batch size for warmup
        h: Height dimension
        w: Width dimension  
        dtype: Data type
        device: Device to run on
        n: Number of warmup iterations (default: 10)
    """
    print(f"Warming up model with {n} iterations...")
    for i in tqdm(range(n), desc="Warmup", leave=False):
        model_inference(model, batch_size, h, w, dtype, device)


def speed_measure(
    model_path: str,
    timestep: int,
    vae_skip_connection: bool,
    batch_size: int,
    h: int,
    w: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int = 50,
    test_iters: int = 50
) -> float:
    """Measure inference speed by loading model and running benchmarks.
    
    Args:
        model_path: Path to model checkpoint
        timestep: Diffusion timestep
        vae_skip_connection: Whether to use VAE skip connections
        batch_size: Batch size for testing
        h: Height dimension
        w: Width dimension
        dtype: Data type
        device: Device to run on
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations
        
    Returns:
        Average latency per sample in seconds
    """
    print("\n" + "=" * 70)
    print("⚡ SPEED MEASUREMENT")
    print("=" * 70)
    
    # Load and compile model
    print("Loading model for speed test...")
    model = load_and_compile_model(
        model_path=model_path,
        timestep=timestep,
        vae_skip_connection=vae_skip_connection,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        compile=True
    )
    
    # Warmup
    warmup_model(model, batch_size, h, w, dtype, device, n=warmup_iters)
    
    # Speed test
    print(f"Running speed test with {test_iters} iterations...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        start_time = time.time()
    
        for i in tqdm(range(test_iters), desc="Speed test", leave=False):
            model_inference(model, batch_size, h, w, dtype, device)
    
        torch.cuda.synchronize()
        end_time = time.time()
    
    latency = (end_time - start_time) / test_iters / batch_size
    
    print()
    print("=" * 70)
    print("🚀 SPEED TEST RESULTS")
    print("=" * 70)
    print(f"  Batch Size:        {batch_size}")
    print(f"  Warmup Iterations: {warmup_iters}")
    print(f"  Test Iterations:   {test_iters}")
    print(f"  Latency:           {latency:.4f} s/sample")
    print(f"  Throughput:        {1/latency:.2f} samples/s")
    print("=" * 70)
    print()
    
    return latency


def get_resolution_size(resolution: int) -> tuple[int, int]:
    """Map resolution to (width, height) size tuple."""
    resolution_map = {
        960: (960, 544),
        1360: (1360, 768),
        704: (704, 384),
        512: (512, 288),
        256: (256, 144),
        1024: (1024, 576)
    }
    assert resolution in resolution_map, f"Resolution {resolution} not supported. Choose from {list(resolution_map.keys())}"
    return resolution_map[resolution]


def preprocess_image(img: Image.Image, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    c_t = transforms.ToTensor()(img)    
    c_t = transforms.Normalize([0.5], [0.5])(c_t).unsqueeze(0)
    return c_t.to(device=device, dtype=dtype)


def postprocess_output(output_tensor: torch.Tensor, target_size: tuple[int, int]) -> Image.Image:
    """Convert model output tensor to PIL image."""
    output_image = output_tensor.float()
    output_image = output_image[0].cpu() * 0.5 + 0.5
    output_image = torch.clamp(output_image, 0.0, 1.0)
    output_pil = transforms.ToPILImage()(output_image)
    output_pil = output_pil.resize(target_size, Image.BILINEAR)
    return output_pil


def load_and_compile_model(
    model_path: str, 
    timestep: int, 
    vae_skip_connection: bool, 
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    compile: bool = True
) -> torch.nn.Module:
    """Initialize and optionally compile the model."""
    model = Pix2Pix_Turbo(
        pretrained_path=model_path, 
        timestep=timestep, 
        vae_skip_connection=vae_skip_connection, 
        batch_size=batch_size
    ).to(device=device, dtype=dtype)
    
    if compile:
        model = torch.compile(model)
    
    return model


def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    resolution: int,
    batch_size: int,
    h: int,
    w: int,
    dtype: torch.dtype,
    device: torch.device
) -> tuple[Image.Image, str]:
    """Process a single image through the model.
    
    Returns:
        tuple: (output_pil_image, basename)
    """
    size = get_resolution_size(resolution)
    
    input_image = Image.open(image_path).convert('RGB')
    original_shape = input_image.size
    input_image = input_image.resize(size, Image.BILINEAR)
    
    bname = os.path.basename(image_path)
    
    with torch.no_grad():
        c_t = preprocess_image(input_image, device, dtype)
        output_tensor = model_inference(model, batch_size, h, w, dtype, device, x=c_t)
        output_pil = postprocess_output(output_tensor, original_shape)
    
    return output_pil, bname


def get_image_paths(input_dir: str, max_frames: int = None, skip_frames: int = 1) -> list[str]:
    """Get sorted list of image paths from directory."""
    all_img_paths = glob(input_dir + '/*.png') + glob(input_dir + '/*.jpg') + glob(input_dir + '/*.jpeg')
    all_img_paths.sort()
    
    if max_frames is None:
        max_frames = len(all_img_paths)
    
    return all_img_paths[:max_frames][::skip_frames]


def inference(
    model_path: str,
    timestep: int,
    vae_skip_connection: bool,
    input_dir: str,
    output_dir: str,
    resolution: int,
    batch_size: int,
    h: int,
    w: int,
    dtype: torch.dtype,
    device: torch.device,
    max_frames: int = None,
    skip_frames: int = 1,
    save_video: bool = False,
    warmup_iters: int = 10
):
    """Run inference on a directory of images.
    
    Args:
        model_path: Path to model checkpoint
        timestep: Diffusion timestep
        vae_skip_connection: Whether to use VAE skip connections
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        resolution: Target resolution
        batch_size: Batch size for inference
        h: Height dimension
        w: Width dimension
        dtype: Data type
        device: Device to run on
        max_frames: Maximum number of frames to process
        skip_frames: Frame skip interval
        save_video: Whether to save output as video
        warmup_iters: Number of warmup iterations
    """
    print("\n" + "=" * 70)
    print("🎨 INFERENCE MODE")
    print("=" * 70)
    
    # Load and compile model
    print("Loading model for inference...")
    model = load_and_compile_model(
        model_path=model_path,
        timestep=timestep,
        vae_skip_connection=vae_skip_connection,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        compile=True
    )
    
    # Warmup the model
    warmup_model(model, batch_size, h, w, dtype, device, n=warmup_iters)
    
    # Get image paths
    image_paths = get_image_paths(input_dir, max_frames=max_frames, skip_frames=skip_frames)
    
    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Batch size: {batch_size}, Resolution: {resolution}\n")
    
    # Process images
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        output_pil, bname = process_single_image(
            model, img_path, resolution, batch_size, h, w, dtype, device
        )
        
        sv_path = os.path.join(output_dir, bname)
        output_pil.save(sv_path)
    
    print(f'\n✓ Processed {len(image_paths)} images -> {output_dir}')
    
    if save_video:
        save_folder2video(output_dir)
        print(f'✓ Video saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Run inference on an image.')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=None, help='path to a model state dict to be used')
    parser.add_argument('--timestep', type=int, default=400, help='Diffusion timestep')
    parser.add_argument('--vae_skip_connection', action='store_true')
    
    # Inference arguments
    parser.add_argument('--input', type=str, required=True, help='input_image')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--output', type=str, default='output', help='output_dir')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--max_frames', type=int, default=3000000, help='max_frames')
    parser.add_argument('--skip_frames', type=int, default=1, help='skip_frames')     
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    
    # Speed test arguments
    parser.add_argument('--test-speed', action='store_true', help='Run speed benchmark before inference')
    parser.add_argument('--speed-test-iters', type=int, default=50, help='Number of iterations for speed test')
    parser.add_argument('--warmup-iters', type=int, default=50, help='Number of warmup iterations')
    
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    print('dtype', dtype)
    
    args = parser.parse_args()

    h: int = 1024 
    w: int = 576 
    
    # Optional speed test (uses args.batch_size)
    if args.test_speed:
        speed_measure(
            model_path=args.model,
            timestep=args.timestep,
            vae_skip_connection=args.vae_skip_connection,
            batch_size=args.batch_size,
            h=h,
            w=w,
            dtype=dtype,
            device=device,
            warmup_iters=args.warmup_iters,
            test_iters=args.speed_test_iters
        )
    
    # Run inference (uses args.batch_size)
    inference(
        model_path=args.model,
        timestep=args.timestep,
        vae_skip_connection=args.vae_skip_connection,
        input_dir=args.input,
        output_dir=args.output,
        resolution=args.resolution,
        batch_size=1,
        h=h,
        w=w,
        dtype=dtype,
        device=device,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        save_video=args.save_video,
        warmup_iters=args.warmup_iters
    )


if __name__ == "__main__":
    main()
