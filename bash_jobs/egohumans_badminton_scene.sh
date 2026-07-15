#!/bin/bash -l
#SBATCH --job-name=badminton_scene
#SBATCH --output=logs/badminton_%x_%j.out
#SBATCH --error=logs/badminton_%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Scene: ${SCENE}  |  Start: $(date)"

DATA_ROOT="/iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans"
OUTPUT_DIR="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans"

mkdir -p logs "$OUTPUT_DIR"

pixi run python scripts/egohumans_pipeline.py \
    --data-root  "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --activity   06_badminton \
    --seq        "${SCENE}"

echo "Job finished: $(date)"
