#!/bin/bash -l
#SBATCH --job-name=flyover_rich
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

# Flyover render of RICH test scenes (visualize/render_flyover.py --dataset rich).
#
# Runs as a batch job for ONE reason: the RICH frames live only inside
# centered_test.sqsh, and the house rule is squashfuse -> node-local /tmp inside
# a job, never anything else pointed at the archive. Without the mount the
# renderer finds no images and the point cloud comes out grey — silently, which
# is why the mount contents are echoed below before anything renders.
#
# crop_meta.json also lives in that mount. Passing it matters: without it the
# placer runs offsets=0, which is the bug behind the historical RICH numbers.
#
#   sbatch bash_jobs/render_flyover_rich.sh
#   sbatch --export=ALL,SCENES="Gym_010_cooking1" bash_jobs/render_flyover_rich.sh

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

SQSH=/capstor/scratch/cscs/tnanni/datasets/rich/centered_test.sqsh
MOUNT=/tmp/centered_test
OUTDIR=/capstor/scratch/cscs/tnanni/ghost_renders
GHOST=/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test

# Best-scoring RICH scenes by camera-centre residual + scale error, read off
# logs/eval_rich_m9_median_2745572.err:
#   Gym_010_cooking1            7.6cm  scale +0.5%
#   ParkingLot2_009_overfence2 10.1cm  scale +0.3%   (outdoor, venue variety)
SCENES="${SCENES:-Gym_010_cooking1 ParkingLot2_009_overfence2}"

mkdir -p "$MOUNT" "$OUTDIR"
squashfuse "$SQSH" "$MOUNT"
trap "fusermount -u '$MOUNT' 2>/dev/null || true" EXIT
echo "Mounted $SQSH -> $MOUNT ($(ls "$MOUNT" | wc -l) scenes)"

echo "Scenes to render: $SCENES"

nvidia-smi
echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)"

# Keep the thread count low: RLIMIT_CPU is 14400 CPU-seconds and torch otherwise
# grabs every core, which killed a login-node render at 6 minutes wall.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

for SCENE in $SCENES; do
    echo "=== $SCENE ==="
    # Frame naming differs between venues (Gym: 00001_01.jpg). Print it per scene:
    # a frames-root that resolves to nothing is the one failure that would
    # otherwise render a plausible-looking grey cloud. `|| true` — head SIGPIPEs
    # find, and pipefail would abort the job.
    find "$MOUNT/$SCENE" -maxdepth 2 -name "*.jpg" | head -3 || true
    CROP="$MOUNT/$SCENE/crop_meta.json"
    [ -f "$CROP" ] || { echo "WARNING: $CROP missing — placer will run offsets=0"; CROP=""; }

    pixi run python -m visualize.render_flyover \
        --dataset rich \
        --scene "$GHOST/$SCENE" \
        --frames-root "$MOUNT/$SCENE" \
        ${CROP:+--crop-meta "$CROP"} \
        --size 1920 1080 --ssaa 2 --depth-stride 1 --voxel 0.012 \
        --cull-3d 0.12 --conf-thr 5 ${EXTRA:-} \
        --out "$OUTDIR/rich_${SCENE}${SUFFIX:-}.mp4"
done

echo "Done: $(date)"
ls -la "$OUTDIR"/rich_*.mp4
