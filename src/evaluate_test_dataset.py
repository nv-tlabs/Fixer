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

import argparse
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
from glob import glob
from tqdm import tqdm
import transformer_engine as te
import time
import torchvision.transforms.functional as F
import numpy as np
import yaml
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
import pynvml

# Import modular functions from inference_pretrained_model
from inference_pretrained_model import (
    load_and_compile_model,
    warmup_model,
    speed_measure,
    preprocess_image,
    postprocess_output,
    get_resolution_size,
    model_inference
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test dataset with PSNR and LPIPS metrics.')
    parser.add_argument('--input', type=str, help='Path to test dataset directory', 
                        default='test_dataset/')
    parser.add_argument('--output', type=str, help='Output directory', 
                        default='evaluation/')
    parser.add_argument('--model', type=str, help='Model path', required=True)
    parser.add_argument('--calculate-for-input', action='store_true', 
                        help='Calculate metrics between input and GT (in addition to output and GT)')

    args = parser.parse_args()

    # Hardcoded parameters
    RESOLUTION = 1024
    TIMESTEP = 250
    VAE_SKIP_CONNECTION = False
    BATCH_SIZE = 1  # For evaluation
    SPEED_TEST_BATCH_SIZE = 8  # For speed benchmarking
    H = 1024
    W = 576
    DTYPE = torch.bfloat16
    DEVICE = torch.device("cuda")
    WARMUP_ITERS = 50
    SPEED_TEST_ITERS = 50

    print("\n" + "="*80)
    print("RUNNING SPEED BENCHMARK")
    print("="*80)

    # Run speed benchmark with batch_size=8 (loads its own model)
    assert te, "WARNING: Make sure to import transformer_engine as te"
    torch.set_grad_enabled(False)
    speedLatency = speed_measure(
        model_path=args.model,
        timestep=TIMESTEP,
        vae_skip_connection=VAE_SKIP_CONNECTION,
        batch_size=SPEED_TEST_BATCH_SIZE,
        h=H,
        w=W,
        dtype=DTYPE,
        device=DEVICE,
        warmup_iters=WARMUP_ITERS,
        test_iters=SPEED_TEST_ITERS
    )

    print("\n" + "="*80)
    print("LOADING MODEL FOR EVALUATION")
    print("="*80 + "\n")

    # Reset pytorch memory statistics
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Initialize NVML for GPU monitoring
    pynvml.nvmlInit()
    gpuHandle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
    
    # Record baseline memory usage (before loading anything)
    memInfo = pynvml.nvmlDeviceGetMemoryInfo(gpuHandle)
    baselineMemoryGb = memInfo.used / (1024 ** 3)
    
    # Initialize peak tracking
    peakMemoryUsedGb = 0.0  # Peak memory used by this process only

    # Load model using the modular function (batch_size=1 for evaluation)
    print(f"Loading model from {args.model}...")
    model = load_and_compile_model(
        model_path=args.model,
        timestep=TIMESTEP,
        vae_skip_connection=VAE_SKIP_CONNECTION,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        dtype=DTYPE,
        compile=True
    )
    print("Model loaded successfully\n")
    
    # Initialize metrics using torchmetrics
    print("Initializing PSNR, LPIPS, and FID metrics...")
    psnrMetric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    lpipsMetric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
    fidMetric = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    print("Metrics initialized\n")
    
    # Find all scene directories
    testDatasetPath = args.input
    sceneIds = [d for d in os.listdir(testDatasetPath) 
                if os.path.isdir(os.path.join(testDatasetPath, d))]
    sceneIds.sort()
    
    print(f"Found {len(sceneIds)} scenes to process\n")
    
    # Metrics storage
    allMetrics = {}
    overallPsnr = []
    overallLpips = []
    totalImagesProcessed = 0
    
    # Storage for input metrics if requested
    if args.calculate_for_input:
        overallInputPsnr = []
        overallInputLpips = []
        overallInputFidMetric = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    
    # FID metric for overall computation (accumulates all images)
    overallFidMetric = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    
    # Process each scene
    for sceneIdx, sceneId in enumerate(sceneIds):
        print(f"\n{'='*80}")
        print(f"Processing scene {sceneIdx+1}/{len(sceneIds)}: {sceneId}")
        print(f"{'='*80}\n")
        
        scenePath = os.path.join(testDatasetPath, sceneId)
        renderPath = os.path.join(scenePath, 'render')
        gtPath = os.path.join(scenePath, 'gt')
        
        # Check if render and gt paths exist
        if not os.path.exists(renderPath) or not os.path.exists(gtPath):
            print(f"Warning: Missing render or gt folder for scene {sceneId}, skipping...")
            continue
        
        # Find camera folders
        cameraFolders = [d for d in os.listdir(renderPath) 
                        if os.path.isdir(os.path.join(renderPath, d))]
        
        scenePsnr = []
        sceneLpips = []
        
        # FID metric for this scene
        sceneFidMetric = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        
        # Input metrics for this scene if requested
        if args.calculate_for_input:
            sceneInputPsnr = []
            sceneInputLpips = []
            sceneInputFidMetric = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        
        # Process each camera folder
        for cameraFolder in cameraFolders:
            renderCameraPath = os.path.join(renderPath, cameraFolder)
            gtCameraPath = os.path.join(gtPath, cameraFolder)
            
            if not os.path.exists(gtCameraPath):
                print(f"Warning: No ground truth for camera {cameraFolder}, skipping...")
                continue
            
            # Get all render images
            renderImgPaths = glob(os.path.join(renderCameraPath, '*.png')) + \
                           glob(os.path.join(renderCameraPath, '*.jpeg')) + \
                           glob(os.path.join(renderCameraPath, '*.jpg'))
            renderImgPaths.sort()
            
            if len(renderImgPaths) == 0:
                print(f"Warning: No images found in {renderCameraPath}, skipping...")
                continue
            
            # Create output directory
            outputScenePath = os.path.join(args.output, sceneId, cameraFolder)
            os.makedirs(outputScenePath, exist_ok=True)
            
            print(f"Processing {len(renderImgPaths)} images from {cameraFolder}...")
            
            # Process each image
            with torch.no_grad():
                for imgIdx, imgPath in enumerate(tqdm(renderImgPaths, desc=f"  {cameraFolder}")):
                    imgName = os.path.basename(imgPath)
                    gtImgPath = os.path.join(gtCameraPath, imgName)
                    
                    # Check if ground truth exists
                    if not os.path.exists(gtImgPath):
                        print(f"Warning: Ground truth not found for {imgName}, skipping...")
                        continue
                    
                    # Load input image
                    inputImage = Image.open(imgPath).convert('RGB')
                    originalShape = inputImage.size
                    
                    # Resize input to processing resolution
                    size = get_resolution_size(RESOLUTION)
                    resizedInput = inputImage.resize(size, Image.BILINEAR)
                    
                    # Preprocess using modular function
                    inputTensor = preprocess_image(resizedInput, DEVICE, DTYPE)
                    
                    # Run inference
                    outputTensor = model_inference(model, BATCH_SIZE, H, W, DTYPE, DEVICE, x=inputTensor)
                    
                    # Post-process output using modular function
                    outputPil = postprocess_output(outputTensor, originalShape)
                    
                    # Save output image
                    outputPath = os.path.join(outputScenePath, imgName)
                    outputPil.save(outputPath)
                    
                    # Load ground truth
                    gtImage = Image.open(gtImgPath).convert('RGB')
                    gtTensor = F.to_tensor(gtImage)
                    
                    # Convert output back to tensor for metrics
                    outputTensor = F.to_tensor(outputPil)
                    
                    # Prepare tensors for metrics (add batch dimension and move to GPU)
                    outputBatch = outputTensor.unsqueeze(0).cuda()
                    gtBatch = gtTensor.unsqueeze(0).cuda()
                    
                    # Calculate PSNR using torchmetrics (expects range [0, 1])
                    psnr = psnrMetric(outputBatch, gtBatch).item()
                    scenePsnr.append(psnr)
                    
                    # Calculate LPIPS using torchmetrics (expects range [0, 1])
                    lpipsValue = lpipsMetric(outputBatch, gtBatch).item()
                    sceneLpips.append(lpipsValue)
                    
                    # Update FID metrics (expects uint8 images in range [0, 255])
                    outputBatchUint8 = (outputBatch * 255).clamp(0, 255).to(torch.uint8)
                    gtBatchUint8 = (gtBatch * 255).clamp(0, 255).to(torch.uint8)
                    sceneFidMetric.update(gtBatchUint8, real=True)
                    sceneFidMetric.update(outputBatchUint8, real=False)
                    overallFidMetric.update(gtBatchUint8, real=True)
                    overallFidMetric.update(outputBatchUint8, real=False)
                    
                    # Calculate metrics for input image if requested
                    if args.calculate_for_input:
                        # Resize input to match GT dimensions for fair comparison
                        inputForMetrics = inputImage.resize(gtImage.size, Image.BILINEAR)
                        inputTensorForMetrics = F.to_tensor(inputForMetrics).unsqueeze(0).cuda()
                        
                        # Calculate PSNR and LPIPS for input
                        inputPsnr = psnrMetric(inputTensorForMetrics, gtBatch).item()
                        sceneInputPsnr.append(inputPsnr)
                        
                        inputLpipsValue = lpipsMetric(inputTensorForMetrics, gtBatch).item()
                        sceneInputLpips.append(inputLpipsValue)
                        
                        # Update FID metric for input
                        inputBatchUint8 = (inputTensorForMetrics * 255).clamp(0, 255).to(torch.uint8)
                        sceneInputFidMetric.update(gtBatchUint8, real=True)
                        sceneInputFidMetric.update(inputBatchUint8, real=False)
                        overallInputFidMetric.update(gtBatchUint8, real=True)
                        overallInputFidMetric.update(inputBatchUint8, real=False)
                    
                    # Track peak GPU memory usage (delta from baseline)
                    memInfo = pynvml.nvmlDeviceGetMemoryInfo(gpuHandle)
                    currentTotalMemoryGb = memInfo.used / (1024 ** 3)
                    currentMemoryUsedGb = currentTotalMemoryGb - baselineMemoryGb
                    peakMemoryUsedGb = max(peakMemoryUsedGb, currentMemoryUsedGb)
                    
                    totalImagesProcessed += 1
        
        # Calculate scene statistics
        if len(scenePsnr) > 0:
            avgScenePsnr = np.mean(scenePsnr)
            avgSceneLpips = np.mean(sceneLpips)
            
            # Compute FID for this scene
            sceneFid = sceneFidMetric.compute().item()
            
            allMetrics[sceneId] = {
                'psnr': float(avgScenePsnr),
                'lpips': float(avgSceneLpips),
                'fid': float(sceneFid),
                'num_images': len(scenePsnr)
            }
            
            if args.calculate_for_input:
                avgSceneInputPsnr = np.mean(sceneInputPsnr)
                avgSceneInputLpips = np.mean(sceneInputLpips)
                sceneInputFid = sceneInputFidMetric.compute().item()
                
                allMetrics[sceneId]['input_psnr'] = float(avgSceneInputPsnr)
                allMetrics[sceneId]['input_lpips'] = float(avgSceneInputLpips)
                allMetrics[sceneId]['input_fid'] = float(sceneInputFid)
                
                overallInputPsnr.extend(sceneInputPsnr)
                overallInputLpips.extend(sceneInputLpips)
            
            overallPsnr.extend(scenePsnr)
            overallLpips.extend(sceneLpips)
            
            print(f"\nScene {sceneId} metrics:")
            print(f"  Images processed: {len(scenePsnr)}")
            print(f"  Average PSNR: {avgScenePsnr:.4f} dB")
            print(f"  Average LPIPS: {avgSceneLpips:.4f}")
            print(f"  FID: {sceneFid:.4f}")
        else:
            print(f"\nWarning: No valid images processed for scene {sceneId}")
    
    # Calculate overall statistics
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}\n")
    
    if len(overallPsnr) > 0:
        avgPsnr = np.mean(overallPsnr)
        stdPsnr = np.std(overallPsnr)
        avgLpips = np.mean(overallLpips)
        stdLpips = np.std(overallLpips)
        
        # Compute overall FID
        overallFid = overallFidMetric.compute().item()
        
        # Compute overall input metrics if requested
        if args.calculate_for_input:
            avgInputPsnr = np.mean(overallInputPsnr)
            stdInputPsnr = np.std(overallInputPsnr)
            avgInputLpips = np.mean(overallInputLpips)
            stdInputLpips = np.std(overallInputLpips)
            overallInputFid = overallInputFidMetric.compute().item()
        
        # Get final memory statistics from NVML
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(gpuHandle)
        currentTotalMemoryGb = memInfo.used / (1024 ** 3)
        currentMemoryUsedGb = currentTotalMemoryGb - baselineMemoryGb
        totalMemoryGb = memInfo.total / (1024 ** 3)
        processMemoryUtilization = (peakMemoryUsedGb / totalMemoryGb) * 100
        
        print(f"Total scenes processed: {len(allMetrics)}")
        print(f"Total images processed: {totalImagesProcessed}")
        print(f"\nMetrics:")
        print(f"  PSNR: {avgPsnr:.4f} ± {stdPsnr:.4f} dB")
        print(f"  LPIPS: {avgLpips:.4f} ± {stdLpips:.4f}")
        print(f"  FID: {overallFid:.4f}")
        print(f"\nSpeed Benchmark Results (Batch Size {SPEED_TEST_BATCH_SIZE}):")
        print(f"  Latency: {speedLatency:.4f}s per sample")
        print(f"  Throughput: {1/speedLatency:.2f} samples/second")
        
        if args.calculate_for_input:
            print(f"\nInput vs GT Metrics:")
            print(f"  Input PSNR: {avgInputPsnr:.4f} ± {stdInputPsnr:.4f} dB")
            print(f"  Input LPIPS: {avgInputLpips:.4f} ± {stdInputLpips:.4f}")
            print(f"  Input FID: {overallInputFid:.4f}")
        
        print(f"\nGPU Memory Usage (This Process Only):")
        print(f"  Baseline (other processes): {baselineMemoryGb:.4f} GB")
        print(f"  Peak memory used: {peakMemoryUsedGb:.4f} GB")
        print(f"  Current memory used: {currentMemoryUsedGb:.4f} GB")
        print(f"  Total GPU memory: {totalMemoryGb:.4f} GB")
        print(f"  Peak utilization: {processMemoryUtilization:.2f}%")
        
        # Save metrics to YAML
        metricsYamlPath = os.path.join(args.output, 'metrics.yaml')
        metricsData = {
            'overall': {
                'psnr_mean': float(avgPsnr),
                'psnr_std': float(stdPsnr),
                'lpips_mean': float(avgLpips),
                'lpips_std': float(stdLpips),
                'fid': float(overallFid),
                'speed_benchmark_batch_size': SPEED_TEST_BATCH_SIZE,
                'speed_benchmark_latency_per_sample': float(speedLatency),
                'speed_benchmark_throughput_samples_per_sec': float(1/speedLatency),
                'baseline_memory_gb': float(baselineMemoryGb),
                'peak_memory_used_gb': float(peakMemoryUsedGb),
                'current_memory_used_gb': float(currentMemoryUsedGb),
                'total_gpu_memory_gb': float(totalMemoryGb),
                'peak_memory_utilization_percent': float(processMemoryUtilization),
                'total_images': totalImagesProcessed
            },
            'per_scene': allMetrics
        }
        
        if args.calculate_for_input:
            metricsData['overall']['input_psnr_mean'] = float(avgInputPsnr)
            metricsData['overall']['input_psnr_std'] = float(stdInputPsnr)
            metricsData['overall']['input_lpips_mean'] = float(avgInputLpips)
            metricsData['overall']['input_lpips_std'] = float(stdInputLpips)
            metricsData['overall']['input_fid'] = float(overallInputFid)
        
        with open(metricsYamlPath, 'w') as f:
            yaml.dump(metricsData, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Metrics saved to {metricsYamlPath}")
        print(f"✓ Output images saved to {args.output}")
    else:
        print("Error: No images were processed successfully")
    
    # Cleanup NVML
    pynvml.nvmlShutdown()

if __name__ == '__main__':
    main()
