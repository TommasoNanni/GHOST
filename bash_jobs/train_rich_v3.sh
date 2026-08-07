#!/bin/bash -l
#SBATCH --job-name=train_v3
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

# Train the V3 RESIDUAL fusion module (fusion/fusion_module_v3.py).
#
# V3 predicts a CORRECTION to the visibility-weighted chordal mean of the
# per-camera estimates, from tokens carrying each camera's DEVIATION from that
# mean. Zero-weight + identity-bias init means an untrained model reproduces the
# mean EXACTLY, so the run starts at the baseline (RICH test, median smoothing:
# WA-100 47.6 / W-100 67.7 / PA 26.5) instead of ~1 mm behind it, which is where
# the best direct-prediction run (R2 ep217: 48.3 / 68.2 / 27.7) plateaued.
#
# Parameter shapes are IDENTICAL to v2, so this differs from the R2 run only in
# the residual formulation. The loss flags below are R2's, unchanged, so the
# V3-vs-R2 comparison isolates the architecture.
#
#   sbatch --export=ALL bash_jobs/train_rich_v3.sh
#   MAX_EPOCHS=300 sbatch --export=ALL bash_jobs/train_rich_v3.sh
#
# The debug partition caps a job at 1:30, so a full run must be chained. Use the
# driver rather than submitting this directly:
#   nohup bash bash_jobs/train_rich_r2_driver.sh > logs/train_r2_driver.log 2>&1 </dev/null &
# RESUME=1 makes this pass --resume; the driver sets it once a checkpoint exists.

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

RUN="${RUN:-V3}"
# R2's loss configuration, held fixed so the only difference is the architecture.
FLAGS=(--joint_body_only 1 --pose_hand_weight 3.0 --kintree_k 2)
# Previous runs stopped near epoch 270-296 of a 400-epoch cosine schedule and so
# never reached the low-LR refinement phase. Set the schedule to the length the
# run will actually reach.
FLAGS+=(--max_epochs "${MAX_EPOCHS:-300}")

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

pixi run python scripts/train_rich_v3.py \
    --run_tag "$RUN" \
    --checkpoint_dir "$CKPT_DIR" \
    "${FLAGS[@]}" \
    ${RESUME:+--resume}

echo ""
echo "Done: $(date)"
