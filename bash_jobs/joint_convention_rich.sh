#!/bin/bash -l
#SBATCH --job-name=joint_conv
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

# Is the chordal mean's PA-MPJPE win a joint-convention artifact?
#
# JointPositionLoss  : 55 SMPL-X joints, root-relative,   GT betas
# PA-MPJPE           : 24 SMPL joints,   Procrustes Sim3, predicted betas
#
# Three factors differ at once. This holds betas fixed (GT, as the loss does)
# and sweeps joint set x alignment as a 2x2, so the flip (if any) can be
# attributed to one factor.
#
# SCENES restricts to a comma-separated subset. Pass it through the ENVIRONMENT,
# not --export=ALL,SCENES=... — sbatch splits --export on commas and would keep
# only the first scene.
#
#   SCENES="a,b,c" sbatch --export=ALL bash_jobs/joint_convention_rich.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

echo "=== GPU STATUS ==="
nvidia-smi -L
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Checkpoint: ${CKPT:-checkpoints/fusion_module/best.pt}"
echo "Scenes: ${SCENES:-<all>}"
echo ""

EXTRA=()
[ -n "${SCENES:-}" ]     && EXTRA+=(--scenes     "$SCENES")
[ -n "${MAX_SCENES:-}" ] && EXTRA+=(--max_scenes "$MAX_SCENES")

pixi run python evaluation/joint_convention_rich.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_data_root    /capstor/scratch/cscs/tnanni/datasets/rich/centered_test \
    --rich_gt_dir       /capstor/scratch/cscs/tnanni/datasets/rich \
    --checkpoint        "${CKPT:-checkpoints/fusion_module/best.pt}" \
    --body_split        "${BODY_SPLIT:-test_body}" \
    --device            cuda \
    --min_views         "${MIN_VIEWS:-1}" \
    --chunk             "${CHUNK:-32}" \
    --out               "${OUT:-eval_explainability/joint_convention_rich.json}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "Done: $(date)"
