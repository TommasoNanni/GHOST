#!/bin/bash -l
#SBATCH --job-name=train_r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

# Retrain the fusion module under the R0 / R1 / R2 ablation.
#
#   R0  baseline           : both changes OFF (clean reference; the existing
#                            checkpoints were trained with a different setup)
#   R1  CHANGE 1 only      : L_joint restricted to root + 21 body joints,
#                            L_pose reweighted (hands 3x, mean-normalised)
#   R2  CHANGE 1 + CHANGE 2: adds the kinematic-tree hard mask (k=2) on the
#                            joint-attention axis
#
# Everything else is held fixed: Adam, lr 1e-4 cosine, batch 1 scene, 400 epochs,
# 4 SST layers, 8 heads, temporal window 128, RICH train split,
# lambda_joint=1 / lambda_pose=0.05, joint-confidence bias OFF.
#
# Config carries the R2 values; R0/R1 switch pieces off on the command line, so
# all three runs execute the same code.
#
#   RUN=R0 sbatch --export=ALL bash_jobs/train_rich_r012.sh
#   RUN=R1 sbatch --export=ALL bash_jobs/train_rich_r012.sh
#   RUN=R2 sbatch --export=ALL bash_jobs/train_rich_r012.sh
#
# The debug partition caps a job at 1:30, so a full run must be chained. Use the
# driver rather than submitting this directly:
#   nohup bash bash_jobs/train_rich_r2_driver.sh > logs/train_r2_driver.log 2>&1 </dev/null &
# RESUME=1 makes this pass --resume; the driver sets it once a checkpoint exists.

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

RUN="${RUN:-R0}"
case "$RUN" in
    R0) FLAGS=(--joint_body_only 0 --pose_hand_weight 1.0 --kintree_k -1) ;;
    R1) FLAGS=(--joint_body_only 1 --pose_hand_weight 3.0 --kintree_k -1) ;;
    R2) FLAGS=(--joint_body_only 1 --pose_hand_weight 3.0 --kintree_k  2) ;;
    *)  echo "RUN must be R0, R1 or R2 (got '$RUN')" >&2; exit 1 ;;
esac

CKPT_DIR="${CKPT_DIR:-checkpoints/fusion_${RUN,,}}"

echo "=== GPU STATUS ==="
nvidia-smi -L
echo "========================="
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
echo "RUN=$RUN  flags: ${FLAGS[*]}"
echo "checkpoint_dir=$CKPT_DIR"
echo ""

# Layout guard: the loss weights and the attention mask both index the packed
# joint layout, and the hardcoded kintree must still match the SMPL-X model.
# Cheap, and it fails loudly rather than training on a silent off-by-one.
pixi run python scripts/verify_joint_layout.py > "logs/layout_${RUN}_${SLURM_JOB_ID}.txt" 2>&1
echo "layout verification written to logs/layout_${RUN}_${SLURM_JOB_ID}.txt"
echo ""

pixi run python scripts/train_rich_v2.py \
    --run_tag "$RUN" \
    --checkpoint_dir "$CKPT_DIR" \
    "${FLAGS[@]}" \
    ${RESUME:+--resume}

echo ""
echo "Done: $(date)"
