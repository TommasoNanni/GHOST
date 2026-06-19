#!/bin/bash
# Extract specific frames from all 182 EgoExo4D takes directly from S3.
# Downloads ~1-5 MB per video (moov atom + GOP) instead of full files.
# Each take processes ~4 cameras in ~5-10 seconds.

MANIFEST=/capstor/scratch/cscs/tnanni/datasets/egoexo4d/job_manifest.json
OUT_DIR=/capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames

pixi run python utilities/extract_egoexo4d_frames.py \
    --manifest "$MANIFEST" \
    --out_dir  "$OUT_DIR"
