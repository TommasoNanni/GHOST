#!/bin/bash -l
#SBATCH --job-name=explain_q1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144

# Explainability Q1: does the fusion module select views, or just average?
# Corrupt one camera by an exact geodesic angle, sweep the angle, measure how far
# the fused output moves. Compare against a chordal L2 mean and a geodesic L1
# median on the same inputs. Inference only — no placer, no DLT, no GT.
#
#   sbatch bash_jobs/explainability_q1.sh
#   MAX_SCENES=10 sbatch --export=ALL,MAX_SCENES bash_jobs/explainability_q1.sh
#
# Env: MAX_SCENES, MAX_FRAMES, DELTAS, OUT

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

echo "Job $SLURM_JOB_ID  start=$(date)"
nvidia-smi -L
echo ""

MODE="${MODE:-sensitivity}"

ARGS=(
    --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test
    --checkpoint        checkpoints/fusion_module/best.pt
    --device            cuda
    --mode              "$MODE"
    --out               "${OUT:-eval_explainability/q1_${MODE}_rich_test.json}"
)
if [ "$MODE" = "sensitivity" ]; then
    ARGS+=(--deltas "${DELTAS:-0,5,10,20,40,60,80}")
elif [ "$MODE" = "joint" ]; then
    # Q2: causal joint-to-joint influence. No GT needed (displacement, not error).
    ARGS+=(--joint_delta "${JOINT_DELTA:-40}")
else
    # accuracy and temporal both score against GT rotations.
    ARGS+=(--rich_data_root /capstor/scratch/cscs/tnanni/datasets/rich/centered_test
           --rich_gt_dir    /capstor/scratch/cscs/tnanni/datasets/rich
           --body_split     test_body)
    [ "$MODE" = "temporal" ] && ARGS+=(--windows "${WINDOWS:-0,1,2,4,8,16,32,64,128}")
fi
[ -n "${MAX_SCENES:-}" ] && ARGS+=(--max_scenes "$MAX_SCENES")
[ -n "${MAX_FRAMES:-}" ] && ARGS+=(--max_frames "$MAX_FRAMES")

pixi run python evaluation/explainability_q1.py "${ARGS[@]}"

echo ""
echo "Done: $(date)"
