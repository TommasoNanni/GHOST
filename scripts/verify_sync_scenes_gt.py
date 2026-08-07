"""Verify the sync-experiment scenes against RICH ground truth.

`evaluation/alignment_experiments_multi.py` picks the people it synchronises with
`common_persons()`: the intersection of the ghost person ids over all cameras. That is
an appearance-ReID product, so it answers "which id survived in every view", not "which
person is the annotated subject". Two failure modes hide behind it:

  * the intersection id is a bystander or a merged track, not the RICH subject;
  * cross-view ReID gave the subject a different id in one camera, so the subject drops
    out of the intersection entirely (ParkingLot1_004_005_greetingchattingeating1 loses
    protagonist 1 because of cam_03 alone).

RICH ships GT cameras, so identity can be settled geometrically. For every GT subject and
every camera this projects the GT root translation into the centered-crop image and takes
the track whose 2-D keypoints sit closest, by majority vote over sampled frames — the same
procedure as `scripts/gt_reid_rich.py`, but read-only and reporting every camera instead
of only the ones needing a relabel.

Two checks are reported per scene:

  A. presence — is there a ghost pid present in *every* camera (what the sync experiment
     currently relies on), and which;
  B. reprojection — does the GT subject actually land on that pid in each camera, and at
     what distance, measured in body-sizes (keypoint bbox diagonal).

A camera is VERIFIED when the GT-matched pid equals the intersection pid and the distance
is within --max_dist. MISMATCH means GT lands on a different track (ReID misfired there;
the view is recoverable by relabelling). FAIL means no track is close enough (the subject
is genuinely not detected there).

    pixi run python scripts/verify_sync_scenes_gt.py \\
        --ghost_root    /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root     /capstor/scratch/cscs/tnanni/datasets/rich \\
        --centered_root /tmp/centered_train_verify

Nothing is written unless --out is given.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# The 18 scenes the synchronisation experiment reports on.
SYNC_SCENES: list[str] = [
    "BBQ_001_guitar",
    "BBQ_001_juggle",
    "ParkingLot1_002_burpee3",
    "ParkingLot1_002_overfence1",
    "ParkingLot1_002_overfence2",
    "ParkingLot1_002_pushup1",
    "ParkingLot1_002_stretching1",
    "ParkingLot1_004_burpeejump1",
    "ParkingLot1_005_overfence1",
    "ParkingLot2_008_pushup1",
    "ParkingLot2_008_overfence1",
    "ParkingLot2_014_pushup2",
    "ParkingLot1_005_burpeejump2",
    "ParkingLot2_008_burpeejump1",
    "ParkingLot2_014_burpeejump1",
    "ParkingLot2_015_overfence1",
    "ParkingLot2_016_stretching1",
    "Pavallion_003_plankjack",
]


# ---------------------------------------------------------------------------
# GT / calibration (copied, not imported: a site-packages `scripts` package
# shadows this repo's scripts/ directory in the pixi env)
# ---------------------------------------------------------------------------

def load_calibration(rich_root: Path, scene: str) -> dict[int, dict]:
    calib_dir = rich_root / "scan_calibration" / scene.split("_")[0] / "calibration"
    out: dict[int, dict] = {}
    for xml_path in sorted(calib_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
            mats = {}
            for node in root:
                data = node.find("data")
                if data is None or data.text is None:
                    continue
                rows, cols = int(node.find("rows").text), int(node.find("cols").text)
                mats[node.tag] = np.array([float(v) for v in data.text.split()],
                                          dtype=np.float64).reshape(rows, cols)
            if "Intrinsics" in mats and "CameraMatrix" in mats:
                out[int(xml_path.stem)] = {"K": mats["Intrinsics"], "RT": mats["CameraMatrix"]}
        except Exception:
            continue
    return out


def load_crop_meta(centered_root: Path, scene: str) -> dict[str, dict]:
    p = centered_root / scene / "crop_meta.json"
    return json.load(open(p)).get("cameras", {}) if p.exists() else {}


def gt_subjects(rich_root: Path, body_split: str, scene: str) -> dict[int, dict[int, np.ndarray]]:
    """{rich_pid: {frame: root_translation (3,)}}."""
    out: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    gt_dir = rich_root / body_split / scene
    if not gt_dir.is_dir():
        return {}
    for fdir in sorted(p for p in gt_dir.iterdir() if p.name.isdigit()):
        frame = int(fdir.name)
        for pkl in sorted(fdir.glob("*.pkl")):
            try:
                d = pickle.load(open(pkl, "rb"))
            except Exception:
                continue
            t = d.get("transl")
            if t is not None:
                out[int(pkl.stem)][frame] = np.asarray(t, dtype=np.float64).reshape(-1)[:3]
    return dict(out)


def project(X: np.ndarray, K: np.ndarray, RT: np.ndarray) -> tuple[np.ndarray, float]:
    Xc = RT[:, :3] @ X + RT[:, 3]
    z = float(Xc[2])
    if z <= 1e-6:
        return np.array([np.nan, np.nan]), z
    return (K @ (Xc / z))[:2], z


def camera_tracks(cam_dir: Path) -> dict[int, dict]:
    """{pid: {'fi': frame_indices, 'kp': pred_keypoints_2d}}."""
    out = {}
    for p in sorted((cam_dir / "body_data").glob("person_*.npz")):
        d = np.load(p, allow_pickle=False)
        if "pred_keypoints_2d" not in d.files:
            continue
        out[int(p.stem.split("_")[1])] = {"fi": d["frame_indices"].astype(int),
                                          "kp": d["pred_keypoints_2d"]}
    return out


# ---------------------------------------------------------------------------
# Matching — as gt_reid_rich.match_camera, but keeps every candidate's distance
# so the intersection pid can be reported even when it is not the winner.
# ---------------------------------------------------------------------------

def match_camera_full(tracks: dict[int, dict], gt_frames: dict[int, np.ndarray],
                      cal: dict, cm: dict, orig_width: int,
                      n_frames: int) -> tuple[int | None, dict[int, float], int]:
    """-> (winning pid, {pid: median rel distance}, n frames actually voted on)."""
    frames = sorted(gt_frames)
    step = max(1, len(frames) // max(n_frames, 1))
    scale = cm["src_w"] / float(orig_width)

    votes: Counter = Counter()
    # Distance of every candidate on every voted frame, not only the winner's:
    # a MISMATCH is only informative if we can say how far the runner-up was.
    all_dists: dict[int, list[float]] = defaultdict(list)
    n_voted = 0
    for f in frames[::step][:n_frames]:
        uv, z = project(gt_frames[f], cal["K"], cal["RT"])
        if not np.isfinite(uv).all() or z <= 0:
            continue
        gx, gy = uv[0] * scale - cm["off_x"], uv[1] * scale - cm["off_y"]
        best = None
        for pid, tr in tracks.items():
            i = np.where(tr["fi"] == f)[0]
            if not i.size:
                continue
            k = tr["kp"][int(i[0])]
            k = k[np.isfinite(k).all(-1)]
            if not k.size:
                continue
            c = k.mean(0)
            size = float(np.hypot(*(k.max(0) - k.min(0)))) or 1.0
            r = float(np.hypot(c[0] - gx, c[1] - gy) / size)
            all_dists[pid].append(r)
            if best is None or r < best[1]:
                best = (pid, r)
        if best is not None:
            votes[best[0]] += 1
            n_voted += 1
    med = {pid: float(np.median(v)) for pid, v in all_dists.items()}
    if not votes:
        return None, med, n_voted
    return votes.most_common(1)[0][0], med, n_voted


# ---------------------------------------------------------------------------
# Per-scene verification
# ---------------------------------------------------------------------------

def verify_scene(scene_dir: Path, rich_root: Path, centered_root: Path, body_split: str,
                 orig_width: int, n_frames: int, max_dist: float) -> dict:
    scene = scene_dir.name
    report: dict = {"scene": scene, "cameras": {}, "errors": []}

    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())
    if len(cam_dirs) < 2:
        report["errors"].append(f"only {len(cam_dirs)} camera(s) with body_data")
        return report

    # ---- check A: presence ------------------------------------------------
    per_cam_pids = {d.name: {int(p.stem.split("_")[1])
                             for p in (d / "body_data").glob("person_*.npz")}
                    for d in cam_dirs}
    per_cam_pids = {c: s for c, s in per_cam_pids.items() if s}
    intersection = sorted(set.intersection(*per_cam_pids.values())) if per_cam_pids else []
    report["n_cameras"] = len(per_cam_pids)
    report["intersection"] = intersection
    report["union_size"] = len(set.union(*per_cam_pids.values())) if per_cam_pids else 0

    # ---- check B: GT reprojection ----------------------------------------
    gt = gt_subjects(rich_root, body_split, scene)
    calib = load_calibration(rich_root, scene)
    crop = load_crop_meta(centered_root, scene)
    if not gt:
        report["errors"].append(f"no GT under {rich_root / body_split / scene}")
    if not calib:
        report["errors"].append("no calibration XMLs")
    if not crop:
        report["errors"].append("no crop_meta.json")
    report["gt_subjects"] = sorted(gt)
    if report["errors"]:
        return report

    for cam_dir in cam_dirs:
        cam = cam_dir.name
        cal = calib.get(int(re.sub(r"\D", "", cam)))
        cm = crop.get(cam)
        if cal is None or cm is None:
            report["cameras"][cam] = {"status": "NO_CALIB"}
            continue
        tracks = camera_tracks(cam_dir)
        if not tracks:
            report["cameras"][cam] = {"status": "NO_TRACKS"}
            continue

        entry: dict = {"subjects": {}}
        for rpid, frames in gt.items():
            pid, med, n_voted = match_camera_full(tracks, frames, cal, cm,
                                                  orig_width, n_frames)
            dist = med.get(pid, float("inf")) if pid is not None else float("inf")
            sub = {"match": pid, "dist": None if pid is None else round(dist, 3),
                   "n_voted": n_voted}
            # How far the intersection pid sits, even when GT chose another track.
            for ip in intersection:
                if ip in med:
                    sub.setdefault("intersection_dist", {})[ip] = round(med[ip], 3)
            entry["subjects"][rpid] = sub
        # A camera is verified when every GT subject matched within threshold and,
        # for single-subject scenes, landed on the intersection pid.
        matched = [s for s in entry["subjects"].values()
                   if s["match"] is not None and s["dist"] is not None
                   and s["dist"] <= max_dist]
        if not matched:
            entry["status"] = "FAIL"
        elif len(intersection) == 1 and any(s["match"] != intersection[0] for s in matched):
            entry["status"] = "MISMATCH"
        else:
            entry["status"] = "OK"
        entry["src"] = [cm["src_w"], cm["src_h"]]
        entry["off"] = [cm["off_x"], cm["off_y"]]
        report["cameras"][cam] = entry

    n_ok = sum(1 for c in report["cameras"].values() if c.get("status") == "OK")
    report["n_ok"] = n_ok
    report["verdict"] = ("CLEAN" if n_ok == len(report["cameras"])
                         else "PARTIAL" if n_ok >= 2 else "BROKEN")
    return report


def print_report(rep: dict, max_dist: float) -> None:
    print(f"\n{rep['scene']}")
    if rep["errors"]:
        for e in rep["errors"]:
            print(f"    ERROR  {e}")
        return
    print(f"    cameras={rep['n_cameras']}  union={rep['union_size']}  "
          f"intersection={rep['intersection']}  gt_subjects={rep['gt_subjects']}")
    for cam, c in sorted(rep["cameras"].items()):
        status = c.get("status", "?")
        if "subjects" not in c:
            print(f"    {cam:<8} {status}")
            continue
        parts = []
        for rpid, s in sorted(c["subjects"].items()):
            d = "inf" if s["dist"] is None else f"{s['dist']:.2f}"
            parts.append(f"gt{rpid}->pid{s['match']} d={d} (n={s['n_voted']})")
            extra = s.get("intersection_dist", {})
            for ip, dv in sorted(extra.items()):
                if ip != s["match"]:
                    parts.append(f"[pid{ip} d={dv:.2f}]")
        print(f"    {cam:<8} {status:<9} {'  '.join(parts)}")
    print(f"    verdict: {rep['verdict']}  ({rep['n_ok']}/{len(rep['cameras'])} "
          f"cameras verified at max_dist={max_dist})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--centered_root", required=True, type=Path)
    ap.add_argument("--body_split", default="train_body")
    ap.add_argument("--orig_width", type=int, default=4112)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--max_dist", type=float, default=0.8,
                    help="reject a match beyond this many body-sizes")
    ap.add_argument("--scenes", default="",
                    help="comma-separated; defaults to the 18 sync scenes")
    ap.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.scenes.split(",") if s.strip()] or SYNC_SCENES

    reports = []
    for name in wanted:
        sd = args.ghost_root / name
        if not sd.is_dir():
            print(f"\n{name}\n    ERROR  missing under {args.ghost_root}")
            continue
        rep = verify_scene(sd, args.rich_root, args.centered_root, args.body_split,
                           args.orig_width, args.frames, args.max_dist)
        reports.append(rep)
        print_report(rep, args.max_dist)

    print("\n" + "=" * 78)
    print(f"{'scene':<34}{'cams':>5}{'inter':>7}{'ok':>4}  verdict")
    for r in reports:
        if r["errors"]:
            print(f"{r['scene']:<34}{'':>5}{'':>7}{'':>4}  ERROR")
            continue
        print(f"{r['scene']:<34}{r['n_cameras']:>5}{str(r['intersection']):>7}"
              f"{r['n_ok']:>4}  {r['verdict']}")
    clean = sum(1 for r in reports if r.get("verdict") == "CLEAN")
    print(f"\n{clean}/{len(reports)} scenes CLEAN")

    if args.out:
        json.dump(reports, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
