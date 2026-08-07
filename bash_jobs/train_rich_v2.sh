#!/bin/bash -l
# debug is the only partition with GPU hours left, so training must run as resumable
# 1:30 chunks driven by bash_jobs/train_rich_v2_driver.sh (last.pt is written every epoch).
#SBATCH --time=01:30:00
#SBATCH --account=a144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --job-name=train_rich_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=debug

set -euo pipefail

cd /users/tnanni/ghost
ulimit -c 0  # disable core dumps — they fill up home quota

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29500

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_NNODES  ($SLURM_JOB_NODELIST)"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "Start:        $(date)"
echo ""

# Training needs RICH GT + calibration, and rich_data_root now points at a node-local
# mount (centered_train exists only as a .sqsh; capstor is Lustre and refuses FUSE).
SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_train.sqsh
MOUNT=/tmp/centered_train
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
trap "fusermount -u '$MOUNT' 2>/dev/null || true" EXIT
echo "Mounted $SQSH → $MOUNT ($(ls "$MOUNT" | wc -l) scenes)"

# NOT `-m scripts.train_rich_v2`: a dependency installs a top-level `scripts` package
# into the pixi env which shadows this repo's scripts/ directory. Invoke by file path
# with the repo root on PYTHONPATH. TRAIN_ARGS passes e.g. --resume.
srun env PYTHONPATH=/users/tnanni/ghost pixi run torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    scripts/train_rich_v2.py ${TRAIN_ARGS:-}

echo ""
echo "Done: $(date)"
