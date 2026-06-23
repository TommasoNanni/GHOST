#!/bin/bash -l
#SBATCH --job-name=eval_sapiens_dlt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

SCENES=(
    "BBQ_001_guitar"
)

for scene in "${SCENES[@]}"; do
    echo "========================================"
    echo "Scene: $scene"
    echo "========================================"

    pixi run python scripts/test_sapiens_dlt.py \
        --scene_dir   /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/${scene} \
        --rich_root   /capstor/scratch/cscs/tnanni/datasets/rich \
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \
        --checkpoint  checkpoints/fusion_module/best.pt \
        --body_split  train_body \
        --device      cuda

    echo ""
done

echo "Done: $(date)"
