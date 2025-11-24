# Fixer Test Dataset Preparation Tutorial

This tutorial explains how to prepare test datasets for evaluating the Fixer model. The test dataset is used to quantitatively measure model performance using metrics like PSNR and LPIPS.

## Dataset Structure

The test dataset should be organized in the following directory structure:

```
test_dataset/
├── {scene_id_1}/
│   ├── render/
│   │   ├── {camera_id_1}/
│   │   │   ├── {timestamp_1}.png
│   │   │   ├── {timestamp_2}.png
│   │   │   └── ...
│   │   ├── {camera_id_2}/
│   │   │   └── ...
│   │   └── ...
│   └── gt/
│       ├── {camera_id_1}/
│       │   ├── {timestamp_1}.png
│       │   ├── {timestamp_2}.png
│       │   └── ...
│       ├── {camera_id_2}/
│       │   └── ...
│       └── ...
├── {scene_id_2}/
│   ├── render/
│   └── gt/
└── ...
```

### Directory Explanation

- **`{scene_id}`**: A unique identifier for each scene (e.g., UUID like `05443ac1-8125-4ace-9daa-6ecdc0df43ff`)
- **`render/`**: Contains the degraded/rendered images that will be fed to the Fixer model as input
- **`gt/`**: Contains the ground truth images that will be used to evaluate the model's output
- **`{camera_id}`**: Camera identifier (e.g., `camera_front_tele_30fov`, `camera_front_wide_120fov`)
- **`{timestamp}.png`**: Image files with timestamp-based filenames (e.g., `116930318359.png`)

### Important Notes

1. **Matching Structure**: The `render/` and `gt/` directories must have identical subdirectory structures and matching filenames
2. **Image Format**: Images should be in PNG, JPEG, or JPG format
3. **Filename Matching**: For each image in `render/{camera_id}/{timestamp}.png`, there must be a corresponding `gt/{camera_id}/{timestamp}.png`
4. **Resolution**: Images can be of any resolution; they will be automatically resized during evaluation

## Generating Test Dataset using NuRec

You can generate test datasets using the same methods described in the [dataset preparation tutorial](./dataset_preparation_tutorial.md) (sparse reconstruction, model underfitting, or cross reference), but with specific configurations for test dataset generation.

## Running Evaluation

Once your test dataset is prepared, run the evaluation script:

```bash
# Start the PyTorch container
docker run --gpus=all --ipc=host -it \
  -v $(pwd):/work \
  fixer-cosmos-env

  # Run evaluation
  python /work/src/evaluate_test_dataset.py \
  --model /work/models/pretrained/pretrained_fixer.pkl \
  --input /work/test_dataset
```

### Evaluation Output

The evaluation script will:

1. Process each scene and camera in the test dataset
2. Apply the Fixer model to all rendered images
3. Compare the model outputs with ground truth images
4. Calculate PSNR and LPIPS metrics
5. Save enhanced images to the `evaluation/` directory with the same structure as `test_dataset/`
6. Generate a `metrics.yaml` file with:
   - Overall metrics (mean and standard deviation)
   - Per-scene metrics
   - Inference time statistics
   - GPU memory usage statistics

### Example Metrics Output

```yaml
overall:
  psnr_mean: 29.682592587593273
  psnr_std: 2.202454623879068
  lpips_mean: 0.1018559524200411
  lpips_std: 0.04364989604340509
  fid: 57.642303466796875
  speed_benchmark_batch_size: 8
  speed_benchmark_latency_per_sample: 0.0548715090751648
  speed_benchmark_throughput_samples_per_sec: 18.224393986142555
  baseline_memory_gb: 1.28570556640625
  peak_memory_used_gb: 4.59375
  current_memory_used_gb: 4.59375
  total_gpu_memory_gb: 80.0
  peak_memory_utilization_percent: 5.7421875
  total_images: 117
per_scene:
  05443ac1-8125-4ace-9daa-6ecdc0df43ff:
    psnr: 28.67823600769043
    lpips: 0.18728190287947655
    fid: 144.3624725341797
    num_images: 12
  ...
```
