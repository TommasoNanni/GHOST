#!/bin/bash -l
# Paper figure: reconstruction with and without temporal synchronisation.
#
# Injects random per-camera delays into one RICH test scene, renders the fused
# body plus the VGGT depth cloud at one instant, then lets the Synchronizer
# estimate the delays and renders the same instant again.  Two PNGs, no viewer.
#
# Runs on a login node: one VGGT forward pass per condition (the fusion model
# takes no cameras and the placer reads them per frame, so only the rendered
# frame needs VGGT).  No sbatch, no SLURM allocation.
#
#   bash bash_jobs/sync_demo.sh
#   SCENE=LectureHall_010_plankjack1 bash bash_jobs/sync_demo.sh
#   FRAME=430 AZIM=120 DIST=6 bash bash_jobs/sync_demo.sh
#
# Env: SCENE, FRAME, WINDOW, MAX_SHIFT, SEED, AZIM, ELEV, DIST, RADIUS,
#      COLOUR_BY_CAM (=1), OUT_DIR

set -euo pipefail

cd /users/tnanni/ghost

echo "start=$(date)"
nvidia-smi -L || echo "(no GPU visible — VGGT will fall back to CPU and be slow)"
echo ""

# The mount point MUST be node-local (/tmp): capstor is Lustre and FUSE refuses
# to mount over it.  The centered frames live only inside the squashfs; the
# extracted centered_test/ tree on capstor holds crop_meta.json but no images.
RICH_ROOT=/capstor/scratch/cscs/tnanni/datasets/rich
CENTERED_SQSH="${RICH_ROOT}/centered_test.sqsh"
CENTERED_MNT="/tmp/centered_test_$$"

mkdir -p "$CENTERED_MNT"
squashfuse "$CENTERED_SQSH" "$CENTERED_MNT"
trap 'fusermount -u "$CENTERED_MNT" 2>/dev/null || true; rmdir "$CENTERED_MNT" 2>/dev/null || true; rm -rf "/tmp/sync_demo_$$"' EXIT
echo "[sqsh] mounted $CENTERED_SQSH -> $CENTERED_MNT"

N_META=$(find "$CENTERED_MNT" -maxdepth 2 -name crop_meta.json | wc -l)
echo "[sqsh] scenes with crop_meta.json: $N_META"
if [ "$N_META" -eq 0 ]; then
    echo "ERROR: no crop_meta.json under $CENTERED_MNT — aborting"
    exit 1
fi
echo ""

ARGS=(
    --scene         "${SCENE:-LectureHall_010_sidebalancerun1}"
    --scenes-root   /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test
    --centered-root "$CENTERED_MNT"
    --body-split    test_body
    --checkpoint    checkpoints/fusion_module/best.pt
    --out-dir       "${OUT_DIR:-figures/sync_demo}"
    --work-dir      "/tmp/sync_demo_$$"
    --window        "${WINDOW:-64}"
    --max-shift     "${MAX_SHIFT:-30}"
    --seed          "${SEED:-0}"
    --azim          "${AZIM:-45}"
    --elev          "${ELEV:-25}"
    --dist          "${DIST:--1}"
    --depth-voxel   "${DEPTH_VOXEL:-0.02}"
)
[ -n "${FRAME:-}" ] && ARGS+=(--frame "$FRAME")
[ -n "${ZOOM:-}" ] && ARGS+=(--zoom "$ZOOM")
[ "${COLOUR_BY_CAM:-0}" = "1" ] && ARGS+=(--colour-by-cam)
[ "${KEEP_PEOPLE:-0}" = "1" ] && ARGS+=(--no-mask-people)
[ "${NO_FRUSTA:-0}" = "1" ] && ARGS+=(--no-show-frusta)
[ "${INSET:-0}" = "1" ] && ARGS+=(--inset)
# Replay the cached VGGT / fusion / placer outputs — seconds, not minutes.
[ "${RENDER_ONLY:-0}" = "1" ] && ARGS+=(--render-only)

pixi run python scripts/sync_demo.py "${ARGS[@]}"

echo ""
echo "done=$(date)"
