#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH --account=a144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --job-name=train_rich
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

# torchrun with c10d rendezvous works for both single- and multi-node.
# --nnodes=$SLURM_NNODES ensures it scales automatically with allocation.
srun pixi run torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    -m scripts.train_rich 

echo ""
echo "Done: $(date)"
