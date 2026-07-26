#!/bin/bash -l
#SBATCH --job-name=eval_egoexo
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

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo ""

# JOINT_CONF=1 feeds the per-joint confidence channel (pred_joint_confidence) to
# the fusion model as joint_mask. The model was trained with it but no evaluation
# script has ever passed it. Unset => legacy behaviour, numbers unchanged.
#   sbatch --export=ALL,JOINT_CONF=1 bash_jobs/eval_egoexo.sh
JOINT_CONF_FLAG=""
if [ "${JOINT_CONF:-0}" = "1" ]; then
    JOINT_CONF_FLAG="--joint_conf"
fi

pixi run python evaluation/evaluate_egoexo.py \
    --ghost_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \
    --gt_root     /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \
    --smplx_model body_models/SMPLX_NEUTRAL.pkl \
    --checkpoint  checkpoints/fusion_module/best.pt \
    --scale       "${SCALE_MODE:-baseline}" \
    --reid_map    manual_reid.json \
    ${JOINT_CONF_FLAG}

echo ""
echo "Done: $(date)"
