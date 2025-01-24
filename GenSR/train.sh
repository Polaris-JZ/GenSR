#!/bin/bash
#SBATCH -J v62
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --time=5-00:00:00
#SBATCH --output=train.txt 

# 激活 conda 环境
source ~/.bashrc
conda activate llama

srun --gres=gpu:1 \
    python train.py \
    --base_train_batch_size 200 \
    --base_valid_batch_size 1400 \
    --base_model_name google/flan-t5-base \
    --item_emb_path ./dataset/lgn-gowalla-2-64.pth.tar \
    --base_lr 1e-4 \
    --filter_threshold 0 \
    --filter_item_num 1 \
    --contrastive_weight 0.01 \