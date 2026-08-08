#!/bin/bash -l
#SBATCH --time=1:30:00
#SBATCH --account=a0185
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --job-name=alignment_experiment_multi_egohumans
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
#SBATCH --partition=debug

set -euo pipefail

nvidia-smi

cd /users/tnanni/ghost
ulimit -c 0  # disable core dumps — they fill up home quota

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo ""

pixi run python -m evaluation.alignment_experiments_multi_egohumans

echo ""
echo "Done: $(date)"
