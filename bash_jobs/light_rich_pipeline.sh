#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38G
#SBATCH --job-name=light_rich_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# Usage: sbatch bash_jobs/light_rich_pipeline.sh <scene_start> <scene_end>
# Example (two halves):
#   sbatch bash_jobs/light_rich_pipeline.sh 0 50
#   sbatch bash_jobs/light_rich_pipeline.sh 50 100

SCENE_START=${1:-}
SCENE_END=${2:-}

set -euo pipefail

cd /cluster/project/cvg/students/tnanni/ghost

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $SLURM_GPUS"
echo "Scene slice:  [$SCENE_START, $SCENE_END)"
echo "Start:        $(date)"
echo ""

EXTRA_ARGS=""
[ -n "$SCENE_START" ] && EXTRA_ARGS="$EXTRA_ARGS --scene-start $SCENE_START"
[ -n "$SCENE_END"   ] && EXTRA_ARGS="$EXTRA_ARGS --scene-end $SCENE_END"

pixi run python -m scripts.rich_pipeline $EXTRA_ARGS

echo ""
echo "Done: $(date)"
