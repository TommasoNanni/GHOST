#!/bin/bash
# Gate before deleting any EgoHumans .sqsh: is the downloaded iopsstor tree complete?
#
# For every scene under $DEST it checks the three things the production pipeline needs:
#   exo/<cam>/images/   raw frames  (prep_undistort_egohumans.py reads these)
#   colmap/workplace/   cameras.txt + images.txt (fisheye K/D for undistort + GT mapping)
#   processed_data/poses2d/  per-camera 2D GT with aria ids (the production ReID)
#
# Prints one line per scene and a per-activity tally. A scene is OK only if all three
# are present and non-empty. 010_basketball is expected to have no processed_data —
# it ships unannotated upstream (see scripts/download_egohumans_gt.py:437).
#
#   bash bash_jobs/verify_egohumans_download.sh            # all activities
#   bash bash_jobs/verify_egohumans_download.sh 03_fencing # one
set -uo pipefail

DEST="${DEST:-/iopsstor/scratch/cscs/tnanni/egohumans_raw}"
INNER="media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
ALL_ACTS=(01_tagging 02_lego 03_fencing 04_basketball 05_volleyball 06_badminton 07_tennis)
ACTS=("${1:-}"); [ -z "${ACTS[0]}" ] && ACTS=("${ALL_ACTS[@]}")

# Scenes known to ship without GT annotations upstream — not a download failure.
UNANNOTATED=" 010_basketball "

total_ok=0; total_bad=0
for ACT in "${ACTS[@]}"; do
    ROOT="$DEST/$INNER/$ACT"
    if [ ! -d "$ROOT" ]; then
        echo "### $ACT: MISSING ($ROOT)"; continue
    fi
    echo "### $ACT"
    ok=0; bad=0
    for S in "$ROOT"/*/; do
        [ -d "$S" ] || continue
        SN=$(basename "$S")

        cams=0; frames=0
        for C in "$S/exo"/*/; do
            [ -d "$C/images" ] || continue
            n=$(ls "$C/images" 2>/dev/null | wc -l)
            (( n > 0 )) && { cams=$((cams+1)); frames=$((frames+n)); }
        done

        [ -f "$S/colmap/workplace/cameras.txt" ] && [ -f "$S/colmap/workplace/images.txt" ] \
            && col=yes || col=NO
        gt=$(find "$S/processed_data/poses2d" -name '*.npy' 2>/dev/null | head -1)
        [ -n "$gt" ] && gtv=yes || gtv=NO
        [ -f "$S/.full_done" ] && sent=yes || sent=NO

        verdict=OK
        [ "$cams" -eq 0 ] && verdict=BAD
        [ "$col" = NO ] && verdict=BAD
        if [ "$gtv" = NO ]; then
            case "$UNANNOTATED" in *" $SN "*) : ;; *) verdict=BAD ;; esac
        fi

        printf "  %-22s cams=%-3s frames=%-6s colmap=%-4s gt=%-4s done=%-4s %s\n" \
               "$SN" "$cams" "$frames" "$col" "$gtv" "$sent" "$verdict"
        [ "$verdict" = OK ] && ok=$((ok+1)) || bad=$((bad+1))
    done
    echo "  -> $ACT: $ok OK, $bad BAD"
    total_ok=$((total_ok+ok)); total_bad=$((total_bad+bad))
done

echo
echo "TOTAL: $total_ok OK, $total_bad BAD"
if (( total_bad == 0 )); then
    echo "Safe to delete the corresponding .sqsh archives."
else
    echo "DO NOT delete any .sqsh — $total_bad scene(s) incomplete."
fi
