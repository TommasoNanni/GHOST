#!/bin/bash
#SBATCH --job-name=rich_bmp2jpg
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/tnanni/rich_bmp2jpg_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=tnanni@ethz.ch

set -e

echo "Starting BMP → JPEG conversion  (job $SLURM_JOB_ID)"
date

conda run -n ghost python -m utilities.convert_rich_bmp_to_jpeg \
    --root /cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train \
    --quality 92 \
    --workers 16

echo "Conversion finished"
date
