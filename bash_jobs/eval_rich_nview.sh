#!/bin/bash -l
# N-view RICH ablation, stage 2: evaluate the shadow root.
#
# Runs evaluation/evaluate_rich_median.py UNMODIFIED against
# ghost_outputs/rich_${SPLIT}_nview${N}, so the protocol is byte-identical to the
# published 4-view run: M10 (neutral pred vs gendered GT), geodesic-median
# fusion, Procrustes-DLT placement, MapAnything BASELINE scale, always
# median-smoothed (both are hardcoded in fusion/placer.py:274-310 — there is no
# flag to get them wrong). The only difference is the number of cameras VGGT,
# MapAnything and the fusion saw.
#
# crop_meta.json lives in the EXTRACTED centered_${SPLIT}/ directory, so no sqsh
# mount is needed here — this runs directly on a free login-node GPU:
#
#   N=2 bash bash_jobs/eval_rich_nview.sh
#   N=3 CUDA_VISIBLE_DEVICES=1 bash bash_jobs/eval_rich_nview.sh

set -euo pipefail
cd /users/tnanni/ghost

N="${N:?set N=2 or N=3}"
SPLIT="${SPLIT:-test}"

GHOST_ROOT="/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_${SPLIT}_nview${N}"
RICH_ROOT="/capstor/scratch/cscs/tnanni/datasets/rich"
CENTERED="${RICH_ROOT}/centered_${SPLIT}"
LOG="paper_results/eval_rich_nview${N}_${SPLIT}.log"

echo "N=$N  SPLIT=$SPLIT  start=$(date)"
echo "ghost_output_root: $GHOST_ROOT  ($(ls "$GHOST_ROOT" 2>/dev/null | wc -l) scenes)"
echo "log: $LOG"
echo ""

pixi run python evaluation/evaluate_rich_median.py \
    --ghost_output_root "$GHOST_ROOT" \
    --rich_root         "$RICH_ROOT" \
    --smplx_model       body_models/SMPLX_NEUTRAL.pkl \
    --centered_root     "$CENTERED" \
    --device            "${DEVICE:-cuda}" \
    --gt_split          "$SPLIT" \
    --max_scenes        "${MAX_SCENES:-52}" \
    2>&1 | tee "$LOG"

echo ""
echo "Done: $(date)"
