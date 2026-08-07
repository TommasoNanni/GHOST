#!/bin/bash -l
#SBATCH --job-name=metric_ladder
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

# All fusion rules scored on ONE metric, ONE set of slots: RR-MPJPE (mm) over
# root + 21 body joints, with GT betas and GT root supplied, so only the fused
# body pose differs.
#
#   deterministic : chordal mean (shipped), Karcher mean, geodesic median,
#                   Tukey biweight, trim-worst
#   oracles       : per-frame and per-joint weighting/selection — upper bounds
#                   that use GT, so no method can exceed them
#   trained       : any checkpoints passed as NAME=path (v2 vs v3 class is read
#                   from the checkpoint's model_config)
#
# WHY THIS RUN: on the 10 VAL scenes R2 scored 37.5mm against the chordal mean's
# 47.5 — beating even the oracle view-weighting bound, which is possible because a
# learned prior acts outside the space of convex combinations of the inputs. But
# `best.pt` was SELECTED on those val scenes, and RR-MPJPE with GT betas/root is
# literally the training objective, so the number is doubly optimistic. The
# earlier controlled study on TEST scenes found the opposite sign. This job runs
# the same comparison on the 52-scene TEST split to settle it.
#
# centered_test.sqsh is mounted node-local: Lustre forbids FUSE mounts under
# rich_root, and without crop_meta.json the pipeline silently falls back to
# offsets=0.
#
#   sbatch bash_jobs/fusion_methods_compare.sh
#   MODELS="R2=checkpoints/fusion_r2/best.pt" sbatch bash_jobs/fusion_methods_compare.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_test.sqsh
MOUNT=/tmp/centered_test_cmp
mkdir -p "$MOUNT"
squashfuse "$SQSH" "$MOUNT"
trap "fusermount -u '$MOUNT' 2>/dev/null || true" EXIT
echo "Mounted $SQSH -> $MOUNT ($(ls "$MOUNT" | wc -l) scenes)"

echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"
nvidia-smi -L
echo ""

MODELS="${MODELS:-R2=checkpoints/fusion_r2/best.pt}"
echo "models: $MODELS"
echo "split: ${SPLIT:-test}  max_scenes: ${MAX_SCENES:-52}"
echo ""

pixi run python debug/pose_metric_ladder_rich.py \
    --max_scenes     "${MAX_SCENES:-52}" \
    --device         cuda \
    --rich_data_root "$MOUNT" \
    --fk_stride      "${FK_STRIDE:-8}" \
    --models         $MODELS \
    --out_json       "${OUT:-eval_explainability/pose_metric_ladder_TEST52.json}"

echo ""
echo "Done: $(date)"
