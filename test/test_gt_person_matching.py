"""Test: verify GT person matching for a RICH scene.

Loads a RICHFusionDatapoint for a given scene directory and reports:
  - which ghost person ID was matched to which RICH GT person ID
  - the mean 3D distance of the match (metres)
  - which ghost persons were discarded (no GT match)

Usage:
    pixi run python -m test.test_gt_person_matching
"""

from __future__ import annotations

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


def run(scene_dir: Path) -> None:
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

    print()


if __name__ == "__main__":
    run(SCENE_DIR)
