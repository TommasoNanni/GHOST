#!/bin/bash -l
# Stage EgoHumans GT (processed_data only) for all 7 activities, then pack it
# into a single squashfs. Idempotent: finished scenes carry a .gt_done sentinel
# and are skipped, so this is safe to re-run after any interruption.
#
# Run fully detached (survives ssh/session teardown):
#   setsid nohup bash bash_jobs/stage_egohumans_gt.sh > logs/stage_gt.log 2>&1 < /dev/null &
set -uo pipefail
cd /users/tnanni/ghost

STAGE=/iopsstor/scratch/cscs/tnanni/egohumans_gt_staging
SQSH_OUT=/capstor/scratch/cscs/tnanni/datasets/egohumans_gt.sqsh
SRC=/iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans/06_badminton/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/06_badminton
ACTS="01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis"

echo "START $(date)  host=$(hostname)"

echo
echo "=== [1] seed badminton 001-030 from local GT (rsync, idempotent) ==="
for i in $(seq 1 30); do
  s=$(printf "%03d" $i)
  if [ -d "$SRC/${s}_badminton/processed_data" ]; then
    mkdir -p "$STAGE/06_badminton/${s}_badminton"
    rsync -a "$SRC/${s}_badminton/processed_data" "$STAGE/06_badminton/${s}_badminton/"
    touch "$STAGE/06_badminton/${s}_badminton/.gt_done"
  fi
done
echo "seeded. staged: $(du -sh "$STAGE" 2>/dev/null | awk '{print $1}')"

echo
echo "=== [2] fetch GT for all activities (rclone backend, resumable) ==="
for act in $ACTS; do
  echo "--------- $act  $(date +%T) ---------"
  mkdir -p "$STAGE/$act"
  pixi run python scripts/download_egohumans_gt.py \
      --activity "$act" \
      --dest-root "$STAGE/$act" \
      --backend rclone
done

echo
echo "=== [3] staged inventory ==="
du -sh "$STAGE"
for a in "$STAGE"/*/; do
  echo "  $(basename "$a"): $(ls -d "$a"*/ 2>/dev/null | wc -l) scenes"
done

echo
echo "=== [4] pack squashfs -> $SQSH_OUT  $(date +%T) ==="
# mksquashfs writes its 96-byte superblock LAST. A killed run leaves intact data
# behind a zeroed header -- exactly how egohumans_06_badminton_a.sqsh died.
# Build to a .tmp and only move it into place after the superblock verifies.
TMP_SQSH="${SQSH_OUT}.tmp"
rm -f "$TMP_SQSH"
mksquashfs "$STAGE" "$TMP_SQSH" -comp zstd -Xcompression-level 15 -noappend -no-progress

echo
echo "=== [5] verify superblock before publishing ==="
if unsquashfs -s "$TMP_SQSH" > /dev/null 2>&1; then
    mv -f "$TMP_SQSH" "$SQSH_OUT"
    echo "OK superblock valid -> $SQSH_OUT"
    ls -la "$SQSH_OUT"
    unsquashfs -s "$SQSH_OUT" | head -8
else
    echo "FAIL: no valid superblock in $TMP_SQSH -- NOT publishing."
    echo "      staging tree is intact at $STAGE; re-run to retry the pack."
    exit 1
fi

echo
echo "=== STAGING DONE $(date) ==="
