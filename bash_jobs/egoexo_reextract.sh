#!/bin/bash
# Re-extract the CORRECT frame for the wrong-frame EgoExo4D takes.
#
# Background: utilities/extract_egoexo4d_frames.py used to `seek()` then take the
# FIRST decoded frame = the keyframe at-or-before the target, not the target
# frame itself. On dynamic takes the subject had moved (up to ~5 m) by the true
# frame, so the reconstruction was at the wrong instant. Fixed 2026-07-15
# (decode forward to the exact pts). This script redoes only the affected takes.
#
# Run on a LOGIN node (needs S3 + internet; no GPU). SAFE BY DEFAULT: prints what
# it would delete/extract. Set GO=1 to actually do it.
#
#   aws sts get-caller-identity     # must succeed first (creds live)
#   bash bash_jobs/egoexo_reextract.sh          # dry-run
#   GO=1 bash bash_jobs/egoexo_reextract.sh     # arm
#
# After this completes, submit the pipeline rerun: sbatch bash_jobs/egoexo_rerun_pipeline.sh
set -euo pipefail
cd /users/tnanni/ghost

LIST=bash_jobs/egoexo_redo_takes.txt
MANIFEST=/capstor/scratch/cscs/tnanni/datasets/egoexo4d/job_manifest.json
FRAMES=/capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames
GT_DIR=/capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt
OUT=/iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d
GO=${GO:-0}

[ "$GO" = 1 ] && echo ">>> GO=1 : ARMED (will delete + re-extract)" || echo ">>> DRY-RUN (set GO=1 to arm)"
echo ">>> redo list: $LIST ($(wc -l < "$LIST") takes)"
echo

# 1) clean stale frames + undistort sentinel + stale pipeline outputs, then
#    re-extract the correct frame per take (per-manifest-index; running the
#    extractor over ALL takes is unsafe because migrated good takes lose their
#    top-level jpg and would be re-extracted un-undistorted).
while IFS=$'\t' read -r take idx; do
    [ -z "$take" ] && continue
    echo "=== $take (manifest idx $idx) ==="
    for camdir in "$FRAMES/$take"/*/; do
        [ -d "$camdir" ] || continue
        echo "    purge $camdir{frames/*.jpg, frame_*.jpg, .undistorted}"
        if [ "$GO" = 1 ]; then
            rm -f "$camdir"frames/*.jpg "$camdir"frame_*.jpg "$camdir".undistorted
        fi
    done
    echo "    purge output $OUT/$take"
    if [ "$GO" = 1 ]; then rm -rf "${OUT:?}/$take"; fi
    if [ "$GO" = 1 ]; then
        pixi run python utilities/extract_egoexo4d_frames.py \
            --manifest "$MANIFEST" --out_dir "$FRAMES" --take_idx "$idx"
    fi
done < "$LIST"

echo
echo "=== 2) undistort (all takes; skips any with a .undistorted sentinel, so only"
echo "        the re-extracted takes are processed) ==="
if [ "$GO" = 1 ]; then
    pixi run python utilities/undistort_egoexo4d.py \
        --frames-root "$FRAMES" --gt-dir "$GT_DIR" --workers 16
fi

echo
echo "Done. Next:"
echo "  1) confirm the fix on one take:  pixi run python scripts/egoexo_confirm_frame_fix.py cmu_soccer06_3"
echo "     (needs its pipeline rerun first — or run the rerun then confirm)"
echo "  2) rerun the pipeline:           sbatch bash_jobs/egoexo_rerun_pipeline.sh"
