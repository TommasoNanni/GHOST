"""Test: verify GT person matching for a RICH scene (or all scenes in a root dir).

Loads a RICHFusionDatapoint for a given scene directory and reports:
  - which ghost person ID was matched to which RICH GT person ID
  - the mean 3D distance of the match (metres)
  - which ghost persons were discarded (no GT match)

Usage:
    pixi run python -m test.test_gt_person_matching
    pixi run python -m test.test_gt_person_matching --root /path/to/rich_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SCENE_DIR = Path(
    "test_outputs/rich10_segmentation_test/LectureHall_020_wipingtable1"
)
# ─────────────────────────────────────────────────────────────────────────────


def run(scene_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Scene: {scene_dir.name}")
    print(f"{'='*60}")

    # ── Patch _match_persons_to_gt to capture the matching before it remaps ──
    # We re-run the matching logic here manually so we can report it clearly,
    # without duplicating the actual implementation.

    # Load the datapoint — this runs the real matching internally
    dp = RICHFusionDatapoint(
        scene_dir=scene_dir,
        rich_data_root=CONFIG.data.rich_data_root,
        body_split=args.body_split,
    )

    # ── Ghost persons ──────────────────────────────────────────────────────
    all_ghost_pids: set[int] = set()
    for cam_persons in dp._raw:
        all_ghost_pids.update(cam_persons.keys())
    ghost_pids = sorted(all_ghost_pids)
    per_cam = {k: sorted(dp._raw[k].keys()) for k in range(len(dp._raw))}
    print(f"\nGhost persons detected (all cameras): {ghost_pids}")
    for k, pids in per_cam.items():
        print(f"  cam {k}: {pids}")

    # ── RICH GT persons (after remapping, keys are ghost pids) ────────────
    if not dp._gt or not dp._gt[0]:
        print("No GT loaded — inference mode or GT directory missing.")
        return

    matched_ghost_pids = sorted(dp._gt[0].keys())
    unmatched = sorted(set(ghost_pids) - set(matched_ghost_pids))

    print(f"\nGT person matching (ghost pid → RICH GT, world-frame 3D distance):")
    # Recompute distances for reporting using the already-remapped _gt
    # Aggregate translations across all cameras (same logic as _match_persons_to_gt)
    accum: dict[int, dict[int, list]] = {gpid: {} for gpid in ghost_pids}
    for cam_persons in dp._raw:
        for gpid, pdata in cam_persons.items():
            tr = pdata.get("smplx_transl")
            fi = pdata.get("frame_indices")
            if tr is None or fi is None:
                continue
            for i, f in enumerate(fi):
                accum[gpid].setdefault(int(f), []).append(tr[i])
    ghost_transl = {
        gpid: {f: np.mean(views, axis=0) for f, views in accum[gpid].items()}
        for gpid in ghost_pids
    }
    for gpid in matched_ghost_pids:
        gt_data = dp._gt[0][gpid]
        gt_tr   = gt_data.get("transl")
        gt_fi   = gt_data.get("frame_indices")
        if gt_tr is None or gt_fi is None:
            print(f"  ghost {gpid} → GT (no transl available)")
            continue
        gt_lut = {int(f): i for i, f in enumerate(gt_fi)}
        common = set(gt_lut.keys()) & set(ghost_transl.get(gpid, {}).keys())
        if common:
            dists = [
                np.linalg.norm(gt_tr[gt_lut[f]] - ghost_transl[gpid][f])
                for f in common
            ]
            print(
                f"  ghost {gpid} → matched GT "
                f"| mean dist = {np.mean(dists):.3f} m "
                f"| min = {np.min(dists):.3f} m "
                f"| max = {np.max(dists):.3f} m "
                f"| over {len(common)} common frames"
            )
        else:
            print(f"  ghost {gpid} → matched GT (no common frames to compute distance)")

    print(f"\nDiscarded ghost persons (no GT match): {unmatched if unmatched else 'none'}")

    # ── Sanity check: gt_valid coverage ───────────────────────────────────
    from data.fusion_dataset import RICHFusionDataset
    _, targets = RICHFusionDataset([dp])[0]
    gt_valid = targets["gt_valid"]   # (T, P)
    T, P = gt_valid.shape
    for p in range(P):
        n_valid = int(gt_valid[:, p].sum())
        print(f"  person slot {p}: {n_valid}/{T} frames have GT annotation")

    rich_to_ghost = getattr(dp, "_rich_to_ghost", {})
    print(f"\nRICH GT pid → ghost pid mapping: {rich_to_ghost}")
    print()
    return {
        "scene": scene_dir.name,
        "ghost_pids": ghost_pids,
        "matched": matched_ghost_pids,
        "unmatched": unmatched,
        "rich_to_ghost": rich_to_ghost,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory containing one sub-folder per scene. "
             "If omitted, runs on the single SCENE_DIR defined at the top.",
    )
    parser.add_argument(
        "--rich-root",
        type=Path,
        default=None,
        help="Override rich_data_root (must contain scan_calibration/ and body split dir).",
    )
    parser.add_argument(
        "--body-split",
        default="train_body",
        help="Which body split to use for GT (default: train_body; use test_body for test set).",
    )
    args = parser.parse_args()
    if args.rich_root is not None:
        CONFIG.data.rich_data_root = str(args.rich_root)

    if args.root is not None:
        scene_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
        print(f"Found {len(scene_dirs)} scenes under {args.root}\n")
        summaries = []
        for sd in scene_dirs:
            try:
                summary = run(sd)
                summaries.append(summary)
            except Exception as e:
                print(f"  ERROR in {sd.name}: {e}\n")
                summaries.append({"scene": sd.name, "error": str(e)})

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for s in summaries:
            if "error" in s:
                print(f"  {s['scene']:<50}  ERROR: {s['error']}")
            else:
                n_ghost   = len(s["ghost_pids"])
                n_matched = len(s["matched"])
                n_drop    = len(s["unmatched"])
                mapping   = ", ".join(
                    f"GT{rpid}→g{gpid}"
                    for rpid, gpid in sorted(s.get("rich_to_ghost", {}).items())
                )
                flag = "  <-- FP ghosts" if n_drop > 0 else ""
                print(
                    f"  {s['scene']:<50}  "
                    f"ghost={n_ghost}  matched={n_matched}  dropped={n_drop}"
                    f"  [{mapping}]{flag}"
                )
    else:
        run(SCENE_DIR)
