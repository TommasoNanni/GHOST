#!/bin/bash -l
#SBATCH --job-name=sqsh_badminton_a
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

SRC="/capstor/scratch/cscs/tnanni/datasets/egohumans/06_badminton"
DST="/capstor/scratch/cscs/tnanni/datasets/egohumans_06_badminton_a.sqsh"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/06_badminton"

echo "Start: $(date)  Node: $SLURMD_NODENAME"

# Delete corrupted file if exists
rm -f "$DST"

EXCLUDES=""
for i in $(seq -w 31 61); do EXCLUDES="$EXCLUDES -e ${INNER}/${i}_badminton"; done

mksquashfs "$SRC" "$DST" -comp zstd -mem 4G -noappend -processors 4 -wildcards -e "*.tar.gz" $EXCLUDES

echo "Done: $(date)"
ls -lh "$DST"
