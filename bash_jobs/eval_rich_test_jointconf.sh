#!/bin/bash -l
#SBATCH --job-name=eval_rich_jc
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

# RICH test evaluation WITH the per-joint confidence channel (--joint_conf).
#
# Uses evaluation/evaluate_rich.py rather than evaluate_on_rich_test.py, because
# only this one exposes --joint_conf. The model was TRAINED with
# `use_joint_confidence: true`, so feeding the channel is the matched condition;
# omitting it is a train/eval mismatch.
#
# centered_test.sqsh must be mounted node-local and passed via --centered_root:
# Lustre forbids FUSE mounts under rich_root, and without crop_meta.json the
# script silently falls back to offsets=0 (the bug behind every historical RICH
# number).
#
#   sbatch --export=ALL,CKPT=checkpoints/fusion_module_new/best.pt \
#          bash_jobs/eval_rich_test_jointconf.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_test.sqsh
MOUNT=/tmp/centered_test
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
trap "fusermount -u '$MOUNT' 2>/dev/null || true" EXIT
echo "Mounted $SQSH -> $MOUNT ($(ls "$MOUNT" | wc -l) scenes)"

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "Checkpoint: ${CKPT:-checkpoints/fusion_module_new/best.pt}"
echo ""

# JC=1 (default) feeds the confidence channel; JC=0 is the legacy protocol, which is
# what the published numbers (PA 30.4 / WA-100 50.4 / W-100 70.4 / RTE 1.00) used —
# those ran WITH crop offsets and WITHOUT joint_conf.
JC_FLAG=""
[ "${JC:-1}" = "1" ] && JC_FLAG="--joint_conf"
echo "joint_conf: ${JC:-1}"

pixi run python evaluation/evaluate_rich.py \
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
    --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich \
    --checkpoint        "${CKPT:-checkpoints/fusion_module_new/best.pt}" \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --centered_root     "$MOUNT" \
    --device            cuda \
    --gt_split          test \
    --max_scenes        "${MAX_SCENES:-52}" \
    ${JC_FLAG} \
    --scale             "${SCALE:-baseline}" \
    --scale_smooth      "${SCALE_SMOOTH:-none}"

echo ""
echo "Done: $(date)"
