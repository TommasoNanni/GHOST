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
    --modalities        3 \
    --scenes            "BBQ_001_juggle,ParkingLot1_002_overfence1,ParkingLot1_002_overfence2,ParkingLot1_002_pushup1,ParkingLot1_004_pushup2,ParkingLot1_004_takingphotos1,ParkingLot1_005_overfence1,ParkingLot2_015_pushup1,Pavallion_000_plankjack,Pavallion_006_sidebalancerun"

echo ""
echo "Done: $(date)"
