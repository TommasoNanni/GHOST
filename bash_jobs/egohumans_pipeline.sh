#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --job-name=egohumans_pipeline_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /users/tnanni/ghost

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo ""


export HF_TOKEN=$(cat ~/.hf_token)

# Run via pixi (activates the correct conda env automatically)
pixi run python -m scripts.egohumans_pipeline --scene-start 14 --scene-end 15

echo ""
echo "Done: $(date)"
