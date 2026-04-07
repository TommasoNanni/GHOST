#!/usr/bin/env bash
# Push all *_segmentation_reid.mp4 from a cluster output folder to your local machine.
#
# Usage (run on the CLUSTER):
#   bash scripts/download_reid_videos.sh <cluster_folder> <local_dest>
#
# Arguments:
#   cluster_folder  Name of the output folder, e.g. rich11_segmentation_test
#   local_dest      Destination on your local machine:
#                     user@mylaptop.local:/home/user/reid_videos
#                     tommaso@192.168.1.10:/Users/tommaso/Desktop/reid_videos
#
# Example:
#   bash scripts/download_reid_videos.sh rich11_segmentation_test tommaso@mylaptop:/tmp/reid

set -euo pipefail

CLUSTER_BASE="/cluster/project/cvg/students/tnanni/ghost/preprocessing_outputs"

FOLDER="${1:?Usage: $0 <cluster_folder> <local_dest>}"
LOCAL_DEST="${2:?Provide local destination, e.g. user@mylaptop:/path/to/dest}"

SOURCE_DIR="$CLUSTER_BASE/$FOLDER"

echo "Searching for *_segmentation_reid.mp4 under $SOURCE_DIR ..."

mapfile -t FILES < <(find "$SOURCE_DIR" -name "*_segmentation_reid.mp4" | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No *_segmentation_reid.mp4 files found. Exiting."
    exit 1
fi

echo "Found ${#FILES[@]} file(s):"
printf '  %s\n' "${FILES[@]}"
echo

for FILE in "${FILES[@]}"; do
    # Preserve scene/cam subfolder structure: strip SOURCE_DIR prefix
    REL="${FILE#$SOURCE_DIR/}"          # e.g. scene_01/cam_02/cam_02_segmentation_reid.mp4
    DEST_FILE="$LOCAL_DEST/$REL"
    DEST_DIR="$(dirname "$DEST_FILE")"

    echo "Uploading: $REL"
    # Create the remote subdirectory first, then copy the file
    ssh "${LOCAL_DEST%%:*}" "mkdir -p '${DEST_DIR#*:}'"
    scp "$FILE" "$DEST_FILE"
done

echo
echo "Done. Files pushed to: $LOCAL_DEST"
