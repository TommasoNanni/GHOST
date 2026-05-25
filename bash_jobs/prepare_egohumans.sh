#!/bin/bash -l
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --account=a144
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=prepare_egohumans
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

cd /users/tnanni/ghost

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo ""

pixi run python scripts/prepare_egohumans.py \
    --data_root   /capstor/scratch/cscs/tnanni/datasets/egohumans \
    --smpl_model  /users/tnanni/ghost/body_models/SMPL_NEUTRAL.npz \
    --smplx_model /users/tnanni/ghost/body_models/SMPLX_NEUTRAL.npz \
    --jobs 32 \
    --skip_existing \

echo ""
echo "Done: $(date)"
