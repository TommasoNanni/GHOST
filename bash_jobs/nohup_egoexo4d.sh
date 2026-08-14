#!/bin/bash
# Keeps submitting egoexo4d_pipeline SLURM jobs until every scene has all outputs.
# A scene is "done" when mapanything_scale_baseline.npy exists (last file-producing step).
#
# Usage:
#   nohup bash bash_jobs/nohup_egoexo4d.sh &
#   tail -f logs/nohup_egoexo4d.log

cd /users/tnanni/ghost

FRAMES_ROOT=/capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames
OUTPUT_DIR=/iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d
SLURM_SCRIPT=bash_jobs/egoexo4d_pipeline.sh
LOG=logs/nohup_egoexo4d.log
POLL=300   # seconds between checks

mkdir -p logs
exec >> "$LOG" 2>&1
echo "=== nohup_egoexo4d started $(date) ==="

n_done() {
    local count=0
    for d in "$FRAMES_ROOT"/*/; do
        scene=$(basename "$d")
        [[ -f "$OUTPUT_DIR/$scene/mapanything_scale_baseline.npy" ]] && count=$((count + 1))
    done
    echo "$count"
}

TOTAL=$(ls -d "$FRAMES_ROOT"/*/ 2>/dev/null | wc -l)
echo "Total scenes: $TOTAL"

while true; do
    DONE=$(n_done)
    echo "$(date '+%Y-%m-%d %H:%M:%S')  done=$DONE/$TOTAL"

    if [[ "$DONE" -ge "$TOTAL" ]]; then
        echo "=== All $TOTAL scenes complete. Exiting. ==="
        exit 0
    fi

    RUNNING=$(squeue -u "$USER" -n egoexo4d_pipeline --noheader 2>/dev/null | grep -c . || true)
    if [[ "$RUNNING" -eq 0 ]]; then
        JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT")
        echo "  Submitted job $JOB_ID  (${DONE}/${TOTAL} done)"
    else
        echo "  Job running — waiting ${POLL}s..."
    fi

    sleep "$POLL"
done
