#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:40G
#SBATCH --job-name=test_sam3d
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=tnanni@ethz.ch

# ── User config ────────────────────────────────────────────────────────────────
DATA_ROOT="/cluster/project/cvg/data/EgoExo_georgiatech/raw/takes"   # adjust
OUTPUT_DIR="/cluster/project/cvg/students/tnanni/ghost/test_outputs/segmentation_test" # adjust
SAM3D_STEP=1          # run SAM3D every N frames (1 = every frame)
SLICE=1              # set to "--slice N" to process only the first N scenes
SMOOTH=""             # set to "--smooth" to enable temporal smoothing
# ── End config ─────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /cluster/project/cvg/students/tnanni/ghost

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
pixi run python -m test.test_sam3d_reid

echo ""
echo "Done: $(date)"