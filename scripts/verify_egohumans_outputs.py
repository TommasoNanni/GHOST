"""
Verify EgoHumans pipeline outputs across all activities/scenes/cameras.
Checks:
  1. Frame count consistency across cameras per scene
  2. Segmentation person count vs body_data person count per camera
  3. NPZ file corruption (mask_data.npz + person_*.npz)
"""
import os
import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans")

ISSUES = []

def warn(activity, scene, cam, msg):
    tag = f"{activity}/{scene}" + (f"/{cam}" if cam else "")
    print(f"  [WARN] {tag}: {msg}")
    ISSUES.append((tag, msg))

def check_npz(path, allow_pickle=False):
    try:
        with np.load(path, allow_pickle=allow_pickle) as f:
            _ = list(f.keys())
        return True
    except Exception as e:
        return str(e)

activities = sorted(os.listdir(OUTPUT_ROOT))
for activity in activities:
    act_dir = OUTPUT_ROOT / activity
    scenes = sorted(os.listdir(act_dir))
    print(f"\n{'='*60}")
    print(f"Activity: {activity}  ({len(scenes)} scenes)")
    print(f"{'='*60}")

    for scene in scenes:
        scene_dir = act_dir / scene
        cams = sorted([d for d in os.listdir(scene_dir) if d.startswith("cam")])

        if not cams:
            warn(activity, scene, None, "no camera dirs found")
            continue

        # ── Check 1: frame count consistency across cameras ──────────────
        frame_counts = {}
        for cam in cams:
            json_dir = scene_dir / cam / "json_data"
            if not json_dir.exists():
                frame_counts[cam] = None
                warn(activity, scene, cam, "json_data dir missing")
                continue
            n = len(list(json_dir.glob("mask_*.json")))
            frame_counts[cam] = n

        valid_counts = [v for v in frame_counts.values() if v is not None]
        if valid_counts:
            min_fc, max_fc = min(valid_counts), max(valid_counts)
            if max_fc - min_fc > 5:  # allow small boundary diff
                detail = " | ".join(f"{c}={frame_counts[c]}" for c in cams)
                warn(activity, scene, None, f"frame count mismatch: {detail}")
            else:
                print(f"  {scene}: frames OK ({min_fc}-{max_fc}) across {len(cams)} cams")
        else:
            warn(activity, scene, None, "no valid frame counts")

        # ── Check 2: seg person count vs body_data count ─────────────────
        for cam in cams:
            json_dir = scene_dir / cam / "json_data"
            body_dir = scene_dir / cam / "body_data"

            if not json_dir.exists():
                continue

            # unique person IDs seen in json_data
            seg_ids = set()
            bad_jsons = []
            for jf in json_dir.glob("mask_*.json"):
                try:
                    data = json.loads(jf.read_text())
                    # format: {"labels": [...], "boxes": [...]} or {"persons": {id: ...}}
                    if "labels" in data:
                        for lbl in data["labels"]:
                            seg_ids.add(int(lbl))
                    elif "persons" in data:
                        for pid in data["persons"]:
                            seg_ids.add(int(pid))
                    else:
                        # try flat dict with int keys
                        for k in data.keys():
                            try:
                                seg_ids.add(int(k))
                            except ValueError:
                                pass
                except Exception as e:
                    bad_jsons.append(jf.name)

            if bad_jsons:
                warn(activity, scene, cam, f"corrupt json files: {bad_jsons[:3]}")

            if not body_dir.exists():
                warn(activity, scene, cam, "body_data dir missing")
                continue

            body_npzs = list(body_dir.glob("person_*.npz"))
            n_body = len(body_npzs)
            n_seg = len(seg_ids)

            if n_body == 0:
                warn(activity, scene, cam, "body_data has 0 person files")
            elif n_seg > 0 and n_body != n_seg:
                warn(activity, scene, cam,
                     f"seg={n_seg} unique IDs but body={n_body} npz files")

            # ── Check 3: corruption on body npz files ───────────────────
            for npz_path in body_npzs:
                result = check_npz(npz_path, allow_pickle=True)
                if result is not True:
                    warn(activity, scene, cam, f"corrupt {npz_path.name}: {result}")

        # ── Check 3b: mask_data.npz corruption ───────────────────────────
        for cam in cams:
            mask_path = scene_dir / cam / "mask_data.npz"
            if not mask_path.exists():
                warn(activity, scene, cam, "mask_data.npz missing")
                continue
            result = check_npz(mask_path, allow_pickle=False)
            if result is not True:
                warn(activity, scene, cam, f"corrupt mask_data.npz: {result}")

print(f"\n{'='*60}")
print(f"SUMMARY: {len(ISSUES)} issues found")
if ISSUES:
    print("\nAll issues:")
    for tag, msg in ISSUES:
        print(f"  {tag}: {msg}")
else:
    print("All clean.")
