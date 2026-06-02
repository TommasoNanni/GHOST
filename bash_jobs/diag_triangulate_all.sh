#!/bin/bash -l
#SBATCH --job-name=diag_triangulate
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

cd /users/tnanni/ghost
ulimit -c 0

echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

echo "=== MHR70 ===" | tee results/diag_triangulate_mhr70.txt
pixi run python -u -m scripts.diag_triangulate_joints --all --modality mhr70 \
    2>/dev/null | tee -a results/diag_triangulate_mhr70.txt

echo ""
echo "=== ViTPose ===" | tee results/diag_triangulate_vitpose.txt
pixi run python -u -m scripts.diag_triangulate_joints --all --modality vitpose \
    2>/dev/null | tee -a results/diag_triangulate_vitpose.txt

echo ""
echo "Done: $(date)"
