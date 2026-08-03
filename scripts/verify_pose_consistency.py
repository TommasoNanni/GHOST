"""Verify cross-view person ids using pose agreement — no GT, no calibration.

SMPL-X `smplx_body_pose` is a set of joint rotations relative to the parent joint, so it
is VIEW-INVARIANT: the same human must have (nearly) the same body pose in every camera at
the same frame. Two different humans do not.

So for a person id P, comparing P's pose in camera C against P's pose in a reference camera
should give a smaller angle than comparing it against a *different* id in the reference.
When it doesn't, the ids are swapped.

This catches errors that a projection-based check cannot: when two people stand close
together their projected positions are nearly identical (margins of 0.01 body-widths were
observed in ParkingLot1_004_005_greetingchattingeating1), but their poses still differ.

The reference camera is the one sharing the most frames with the others, so a single bad
reference cannot poison the whole scene. Scenes with one person are reported as trivially
consistent — there is nothing to swap.

    pixi run python scripts/verify_pose_consistency.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR


def load_pose(cam_dir: Path, pid: int) -> dict[int, np.ndarray] | None:
    p = cam_dir / "body_data" / f"person_{pid}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    if "smplx_body_pose" not in d.files:
        return None
    bp = d["smplx_body_pose"]
    fi = d["frame_indices"].astype(int)
    return {int(f): bp[i].reshape(-1, 3) for i, f in enumerate(fi)}


def pose_distance(a: dict, b: dict, n: int = 60) -> float | None:
    """Median over frames of the mean geodesic angle between joint rotations (degrees)."""
    common = sorted(set(a) & set(b))
    if len(common) < 20:
        return None
    vals = []
    for f in common[::max(1, len(common) // n)]:
        Ra = SciR.from_rotvec(a[f]).as_matrix()
        Rb = SciR.from_rotvec(b[f]).as_matrix()
        tr = np.einsum("nij,nij->n", Ra, Rb)
        vals.append(float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).mean()))
    return float(np.median(vals)) if vals else None


def check_scene(scene_dir: Path, pids: list[int]) -> dict:
    cams = sorted(d for d in scene_dir.iterdir()
                  if d.is_dir() and (d / "body_data").is_dir())
    poses = {c.name: {p: load_pose(c, p) for p in pids} for c in cams}
    # reference = camera holding every id with the most frames
    ref, best = None, -1
    for c, pp in poses.items():
        if all(pp[p] is not None for p in pids):
            n = min(len(pp[p]) for p in pids)
            if n > best:
                ref, best = c, n
    if ref is None:
        return {"status": "no reference camera holds all ids"}

    rows = []
    for c, pp in poses.items():
        if c == ref:
            continue
        for p in pids:
            if pp[p] is None:
                continue
            same = pose_distance(pp[p], poses[ref][p])
            others = [(q, pose_distance(pp[p], poses[ref][q]))
                      for q in pids if q != p and poses[ref][q] is not None]
            others = [(q, d) for q, d in others if d is not None]
            if same is None or not others:
                continue
            best_other_q, best_other = min(others, key=lambda qd: qd[1])
            rows.append({"cam": c, "pid": p, "same": same,
                         "other": best_other, "other_pid": best_other_q,
                         "swapped": best_other < same})
    return {"status": "ok", "ref": ref, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--only_remapped", action="store_true",
                    help="only scenes carrying a gt_reid_map.json")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = [d for d in sorted(args.ghost_root.iterdir())
              if d.is_dir() and (d / "cross_view_reid.json").exists()]
    if args.only_remapped:
        scenes = [d for d in scenes if (d / "gt_reid_map.json").exists()]
    if wanted:
        scenes = [d for d in scenes if d.name in wanted]

    n_single = n_ok = n_bad = 0
    bad_detail = []
    for sd in scenes:
        # candidate ids = those present in >=2 cameras (the ones an id error could affect)
        counts: dict[int, int] = {}
        for c in sd.iterdir():
            if not (c.is_dir() and (c / "body_data").is_dir()):
                continue
            for p in (c / "body_data").glob("person_*.npz"):
                pid = int(p.stem.split("_")[1])
                counts[pid] = counts.get(pid, 0) + 1
        pids = sorted(p for p, n in counts.items() if n >= 2)
        if len(pids) < 2:
            n_single += 1
            continue
        r = check_scene(sd, pids)
        if r["status"] != "ok":
            continue
        swapped = [x for x in r["rows"] if x["swapped"]]
        if swapped:
            n_bad += 1
            print(f"\n  {sd.name}  (ref {r['ref']}, ids {pids})")
            for x in swapped:
                print(f"     {x['cam']} person_{x['pid']}: same-id {x['same']:.1f} deg "
                      f"but closer to person_{x['other_pid']} at {x['other']:.1f} deg  <-- SWAPPED")
            bad_detail.append(sd.name)
        else:
            n_ok += 1

    print(f"\n  multi-id scenes consistent : {n_ok}")
    print(f"  multi-id scenes SWAPPED    : {n_bad}  {bad_detail if bad_detail else ''}")
    print(f"  single-id scenes (trivial) : {n_single}")


if __name__ == "__main__":
    main()
