#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/dp/results/beta_CNN.out
#SBATCH --error=/cluster/project/sachan/jiaxie/dp/results/beta_CNN.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=1:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache

source activate /cluster/scratch/jiaxie/deepprotein

cd /cluster/project/sachan/jiaxie/DeepProtein

method="prot_bert"
SEED=7
wandb_proj="DeepProtein"
LR=0.0001
EPOCH=100

python -u train/beta.py \
      --target_encoding ${method} \
      --seed ${SEED} \
      --wandb_proj ${wandb_proj} \
      --lr ${LR} \
      --epochs ${EPOCH} \