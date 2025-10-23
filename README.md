# Fixer - Improving 3D Reconstructions with Single-Step Diffusion Models
<p align="center">
  <img src="assets/demo.gif" alt="Fixer Demo">
</p>

Fixer is a single-step image diffusion model trained to enhance and remove artifacts in rendered novel views caused by underconstrained regions of three-dimensional (3D) representation. 

Fixer is based on Difix3d+. See **[Paper](https://arxiv.org/abs/2503.01774) | [Code](https://github.com/nv-tlabs/Difix3D/tree/main) | [Project Page](https://research.nvidia.com/labs/toronto-ai/difix3d/)**

> **Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models**  
> [Jay Zhangjie Wu*](https://zhangjiewu.github.io/), [Yuxuan Zhang*](https://scholar.google.com/citations?user=Jt5VvNgAAAAJ&hl=en), [Haithem Turki](https://haithemturki.com/), [Xuanchi Ren](https://xuanchiren.com/), [Jun Gao](https://www.cs.toronto.edu/~jungao/),  
[Mike Zheng Shou](https://sites.google.com/view/showlab/home?authuser=0), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Zan Gojcic†](https://zgojcic.github.io/), [Huan Ling†](https://www.cs.toronto.edu/~linghuan/) _(*/† equal contribution/advising)_  
CVPR 2025 (Oral) 

## Setup

- **Environment**: 
  - We use `nvcr.io/nvidia/pytorch:24.10-py3` as the environment for inference with the pretrained JIT model
  - We use `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` as the base environment for training and inference with the pretrained model.

- **Build Docker Image**:
  ```sh
  # For training and standard inference
  docker build -t fixer-cosmos-env -f Dockerfile.cosmos .
  ```
  
- **Download Pretrained Model Weights**:
  ```sh
  # Install Hugging Face CLI if not already installed
  pip install huggingface_hub[cli]
  
  # Download all model weights to models/
  huggingface-cli download nvidia/Fixer \
    --local-dir models \
    --local-dir-use-symlinks False
  ```
  
  This downloads:
  ```
  models/base/cosmos_ablation_2B_0502_t2i_309_1024res_multidataset_v4_synthetic_ratio_1_3.pth
  models/base/tokenizer.pth
  models/pretrained/pretrained_fixer.pkl
  models/pretrained/pretrained_fixer_jit.pt
  ```

## Fixer Inference and Training

### Model Inference
> First image processing includes one-time initialization overhead: Model weight loading from disk to GPU, CUDA kernel compilation. Expect the first image inference to take ~20-30s
#### 1. Inference with pretrained JIT model

```bash
# Run inference with the PyTorch container
docker run --gpus=all \
  -v $(pwd):/work \
  nvcr.io/nvidia/pytorch:24.10-py3 \
  python /work/src/inference_pretrained_model_jit.py \
  --model /work/models/pretrained/pretrained_fixer_jit.pt \
  --input /work/examples \
  --output /work/output \
  --resolution 1024
```
#### 2. Inference with pretrained model 
```bash
# Run the Cosmos Container:
docker run --gpus=all -it \
  -v $(pwd):/work \
  fixer-cosmos-env

# Inside the container, run inference:
python src/inference_pretrained_model.py \
  --model /work/models/pretrained/pretrained_fixer.pkl \
  --input /work/examples \
  --output /work/output \
  --vae_skip_connection --timestep 250 --resolution 1024
```

### Training

#### 1. Data Preparation

Prepare your dataset in the following JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```


#### 2. Multiple GPUs Training Command

```bash
export NUM_NODES=1
export NUM_GPUS=8
export OUTPUT_DIR="/path/to/checkpointing directory" 
export DATASET_FOLDER="/data/data.json" # Set to your data path
export WANDB_MODE=offline

accelerate launch --mixed_precision=bf16 --main_process_port 29501 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_pix2pix_turbo_nocond_cosmos_base_2x_smallimg_skip_datav2.py \
    --output_dir=${OUTPUT_DIR} \
    --dataset_folder=${DATASET_FOLDER} \
    --max_train_steps 10000 \
    --learning_rate 2e-5 \
    --train_batch_size=1 --gradient_accumulation_steps 1 --dataloader_num_workers 8 \
    --checkpointing_steps=2000 --eval_freq 1000 --viz_freq 1000 \
    --train_image_prep "resize_576x1024" --test_image_prep "resize_576x1024" \
    --lambda_clipsim 0.0 --lambda_lpips 0.3 --lambda_gan 0.0 --lambda_l2 1.0 --lambda_gram 0.0 \
    --use_sched --vae_skip_connection --report_to "wandb" --tracker_project_name "cosmos_fixer" --tracker_run_name "train" --train_full_unet --timestep 250 --track_val_fid --num_samples_eval 20
```

**Resume training:** use ```--resume ${OUTPUT_DIR}/checkpoints``` if you want to resume the model training

**Best practice:** We set the hyperparameters from our best practice explicitly in the command above. Specifically, we used a learning rate of ```2e-5```, timesteps of ```250```, on resolution of ```576×1024```, and a perceptual loss weight of ```0.3```, etc. We encourage users to start training with these defaults parameters first and adjust them to their dataset as needed.

## Citation

```bibtex
@inproceedings{wu2025difix3d+,
  title={DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models},
  author={Wu, Jay Zhangjie and Zhang, Yuxuan and Turki, Haithem and Ren, Xuanchi and Gao, Jun and Shou, Mike Zheng and Fidler, Sanja and Gojcic, Zan and Ling, Huan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={26024--26035},
  year={2025}
}
```