#!/bin/bash -l
#SBATCH --job-name=eval_rich_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

pixi run python evaluation/evaluate_on_rich_test.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
    --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich \
    --checkpoint        checkpoints/fusion_module_latest/best.pt \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --device            cuda \
    --gt_split          train \
    --modalities        1 \
    --skip_scenes       "Pavallion_013_plankjack,Pavallion_013_phonesiteat" \
    --skip_cameras      "Pavallion_003_018_tossball:cam_06;ParkingLot2_008_pushup2:cam_03;ParkingLot2_014_takingphotos2:cam_01"

echo ""
echo "Done: $(date)"
