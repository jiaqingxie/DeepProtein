#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/dp/results/stability_LlaSMol.out
#SBATCH --error=/cluster/project/sachan/jiaxie/dp/results/stability_LlaSMol.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=04:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache

source activate /cluster/scratch/jiaxie/deepprotein

cd /cluster/project/sachan/jiaxie/DeepProtein

method="LlaSMol"
SEED=7
wandb_proj="DeepPurposePP"
LR=1e-4
EPOCH=100
BATCH_SIZE=16

python -u train/llm/llm_stability.py \
      --target_encoding ${method} \
      --seed ${SEED} \
      --wandb_proj ${wandb_proj} \
      --lr ${LR} \
      --epochs ${EPOCH} \
      --batch_size ${BATCH_SIZE} \

