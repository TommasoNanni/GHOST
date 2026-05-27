#!/bin/bash -l
#SBATCH --job-name=eval_placer
#SBATCH --output=logs/eval_placer_%j.out
#SBATCH --error=logs/eval_placer_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

cd /users/tnanni/ghost

pixi run python scripts/eval_placer_trans.py \
    --scene_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
    --rich_root   /capstor/scratch/cscs/tnanni/datasets/rich \
    --smplx_model body_models/SMPLX_NEUTRAL.pkl \
    --max_scenes  2 \
    --checkpoint  checkpoints/fusion_module_latest/best.pt \
    --device      cuda \
    > results/eval_placer_10scenes.txt 2>&1
