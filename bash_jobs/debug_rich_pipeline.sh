#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --account=a144
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40G 
#SBATCH --job-name=debug_rich_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=tnanni@ethz.ch


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd users/tnanni/ghost

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $SLURM_GPUS"
echo "Start:        $(date)"
echo "Working dir:  $SCRIPT_DIR"
echo ""


# Run via pixi (activates the correct conda env automatically)
pixi run python -m scripts.rich_pipeline

echo ""
echo "Done: $(date)"