#!/bin/bash -l
#SBATCH --job-name=test_mf_vggt
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

# Mount RICH train squashfs on compute node (frames at /tmp/rich_train)
mkdir -p /tmp/rich_train
squashfuse /capstor/scratch/cscs/tnanni/datasets/rich/train_dataset.sqsh /tmp/rich_train

cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

# Regenerate VGGT outputs for both scenes overwritten by the sliding-window experiment.
for SCENE in BBQ_001_guitar Pavallion_013_yoga2; do
    SCENE_OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/${SCENE}
    echo "Removing stale VGGT outputs for ${SCENE} ..."
    rm -f "${SCENE_OUT}/vggt_cameras.npz" "${SCENE_OUT}/vggt_depth.npz"
done

echo ""
echo "--- Running per-frame VGGT on both scenes ---"
pixi run python scripts/rerun_vggt_only.py \
    --vggt-weights checkpoints/vggt_omega/vggt_omega_1b_512.pt \
    --scenes BBQ_001_guitar Pavallion_013_yoga2

echo ""
echo "Done: $(date)"
