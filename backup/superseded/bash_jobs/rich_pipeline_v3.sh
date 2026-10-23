#!/bin/bash -l
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --job-name=rich_pipeline_v3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=debug

set -euo pipefail

cd /users/tnanni/ghost
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_OFFLINE=1
ulimit -c 0  # disable core dumps — they fill up home quota

# Mount the PP-centered train archive node-locally. It must land on the path
# CONFIG.data.rich_data_root points at: capstor is Lustre and refuses a FUSE mount,
# and the surviving vggt_cameras_centered.npz were computed from CENTERED images, so
# segmentation/body estimation must run on the same image set to stay consistent.
SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_train.sqsh
MOUNT=/tmp/centered_train
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
echo "Mounted $SQSH → $MOUNT"
N_SCENES=$(ls "$MOUNT" | wc -l)
echo "[sqsh] scenes visible: $N_SCENES"
if [ "$N_SCENES" -eq 0 ]; then echo "ERROR: empty mount, aborting"; exit 1; fi

# Unmount on exit (success or failure).
trap "fusermount -u '$MOUNT' && echo 'Unmounted $MOUNT'" EXIT

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo ""

# NOT `-m scripts.rich_pipeline_v3`: a dependency installs a top-level `scripts`
# package into the pixi env which shadows this repo's scripts/ directory. Invoke by
# file path and put the repo root on PYTHONPATH so `configuration` etc. still import.
# PIPELINE_ARGS lets a driver pass through e.g. --skip-mapanything or --scene.
PYTHONPATH=/users/tnanni/ghost pixi run python scripts/rich_pipeline_v3.py ${PIPELINE_ARGS:-}

echo ""
echo "Done: $(date)"
