#!/bin/bash -l
#SBATCH --job-name=sqsh_ego
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Usage: sbatch mksquashfs_egohumans_activity.sh <activity>
# e.g.:  sbatch mksquashfs_egohumans_activity.sh 01_tagging

set -euo pipefail
ACTIVITY="${1:?Usage: sbatch mksquashfs_egohumans_activity.sh <activity>}"

SRC="/capstor/scratch/cscs/tnanni/datasets/egohumans/${ACTIVITY}"
DST="/capstor/scratch/cscs/tnanni/datasets/egohumans_${ACTIVITY}.sqsh"

echo "Start: $(date)  Node: $SLURMD_NODENAME"
echo "Squashing ${ACTIVITY} ..."

mksquashfs "$SRC" "$DST" -comp zstd -mem 4G -noappend -processors 16

echo "Done: $(date)"
ls -lh "$DST"
