#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=alignment_experiment_multi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /cluster/project/cvg/students/tnanni/ghost

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo "Working dir:  $SCRIPT_DIR"
echo ""


# Run via pixi (activates the correct conda env automatically)
CONDA_OVERRIDE_CUDA=12.6 pixi run python -m evaluation.alignment_experiments_multi

echo ""
echo "Done: $(date)"
