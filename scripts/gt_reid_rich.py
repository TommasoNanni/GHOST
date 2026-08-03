"""Assign RICH global person ids geometrically, from GT cameras.

Cross-view appearance ReID mislabels the annotated subject in some views: in
ParkingLot2_016_pushup2 the subject is pid 1 in cam_00/01/02/04/05 but pid 9 in cam_03,
even though it is detected there for all 471 frames and sits on the GT projection. Any
scene where the subject is split across ids loses views (or gets another person's pose
under the subject's id).

RICH has GT cameras, so identity does not need appearance matching at all. For each GT
subject and each camera we project the GT root and take the track whose 2-D keypoints are
closest, by majority vote over sampled frames. That track IS the subject in that camera.

Output is a per-scene `gt_reid_map.json`::

    {"canonical": {"<rich_pid>": <global_id>},
     "cameras": {"cam_03": {"9": 1}},          # old ghost pid -> new global id
     "quality": {"cam_03": 0.18}}              # median distance, in body-sizes

With --apply the person_*.npz files are renamed accordingly (a permutation, so a
conflicting id is swapped rather than overwritten). The map records the original names,
so --revert undoes it.

Cameras whose best match is worse than --max_dist are left untouched and reported: that
means the subject genuinely is not visible/detected there, and inventing a mapping would
be worse than leaving the view out.

    pixi run python scripts/gt_reid_rich.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \\
        --centered_root /tmp/ct_ver --body_split train_body        # dry run
        [--apply]
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


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

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


def match_camera(tracks: dict[int, dict], gt_frames: dict[int, np.ndarray],
                 cal: dict, cm: dict, orig_width: int,
                 n_frames: int) -> tuple[int | None, float]:
    """Which pid is this GT subject in this camera? -> (pid, median rel distance)."""
    frames = sorted(gt_frames)
    step = max(1, len(frames) // max(n_frames, 1))
    scale = cm["src_w"] / float(orig_width)

    votes: Counter = Counter()
    dists: dict[int, list[float]] = defaultdict(list)
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
            if best is None or r < best[1]:
                best = (pid, r)
        if best is not None:
            votes[best[0]] += 1
            dists[best[0]].append(best[1])
    if not votes:
        return None, float("inf")
    pid = votes.most_common(1)[0][0]
    return pid, float(np.median(dists[pid]))


def process_scene(scene_dir: Path, rich_root: Path, centered_root: Path, body_split: str,
                  orig_width: int, n_frames: int, max_dist: float) -> dict | None:
    scene = scene_dir.name
    gt = gt_subjects(rich_root, body_split, scene)
    calib = load_calibration(rich_root, scene)
    crop = load_crop_meta(centered_root, scene)
    if not gt or not calib or not crop:
        return None

    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())

    # canonical global id per GT subject: whatever id it already holds most often
    per_cam_match: dict[str, dict[int, tuple[int, float]]] = {}
    for cam_dir in cam_dirs:
        cal, cm = calib.get(int(re.sub(r"\D", "", cam_dir.name))), crop.get(cam_dir.name)
        if cal is None or cm is None:
            continue
        tracks = camera_tracks(cam_dir)
        if not tracks:
            continue
        per_cam_match[cam_dir.name] = {}
        for rpid, frames in gt.items():
            pid, dist = match_camera(tracks, frames, cal, cm, orig_width, n_frames)
            if pid is not None:
                per_cam_match[cam_dir.name][rpid] = (pid, dist)

    canonical: dict[int, int] = {}
    for rpid in gt:
        c = Counter(m[rpid][0] for m in per_cam_match.values()
                    if rpid in m and m[rpid][1] <= max_dist)
        canonical[rpid] = c.most_common(1)[0][0] if c else min(gt)

    remap: dict[str, dict[int, int]] = {}
    quality: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for cam, m in per_cam_match.items():
        pairs = {}
        worst = 0.0
        for rpid, (pid, dist) in m.items():
            if dist > max_dist:
                skipped[cam] = f"best match {dist:.2f} > {max_dist}"
                continue
            pairs[pid] = canonical[rpid]
            worst = max(worst, dist)
        pairs = {o: n for o, n in pairs.items() if o != n}
        if pairs:
            remap[cam] = pairs
            quality[cam] = round(worst, 3)
    return {"scene": scene, "canonical": {str(k): v for k, v in canonical.items()},
            "cameras": {c: {str(o): n for o, n in p.items()} for c, p in remap.items()},
            "quality": quality, "skipped": skipped}


def apply_remap(scene_dir: Path, plan: dict) -> list[str]:
    """Rename person_*.npz as a permutation (conflicts swap, never overwrite)."""
    actions = []
    for cam, pairs in plan["cameras"].items():
        bd = scene_dir / cam / "body_data"
        pairs = {int(o): int(n) for o, n in pairs.items()}
        # move every involved file aside first so a swap cannot clobber
        tmp = {}
        for old in list(pairs) + [v for v in pairs.values()
                                  if (bd / f"person_{v}.npz").exists()]:
            src = bd / f"person_{old}.npz"
            if src.exists():
                t = bd / f".tmp_person_{old}.npz"
                src.rename(t)
                tmp[old] = t
        inverse = {v: k for k, v in pairs.items()}
        for old, t in tmp.items():
            new = pairs.get(old, inverse.get(old, old) if old in inverse else old)
            t.rename(bd / f"person_{new}.npz")
            actions.append(f"{cam}: person_{old} -> person_{new}")
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--centered_root", required=True, type=Path)
    ap.add_argument("--body_split", default="train_body")
    ap.add_argument("--orig_width", type=int, default=4112)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--max_dist", type=float, default=0.8,
                    help="reject a match beyond this many body-sizes")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--apply", action="store_true", help="rename the npz files")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = [d for d in sorted(args.ghost_root.iterdir())
              if d.is_dir() and (d / "cross_view_reid.json").exists()]
    if wanted:
        scenes = [d for d in scenes if d.name in wanted]

    n_fix = n_scene = 0
    for sd in scenes:
        plan = process_scene(sd, args.rich_root, args.centered_root, args.body_split,
                             args.orig_width, args.frames, args.max_dist)
        if plan is None:
            continue
        if not plan["cameras"] and not plan["skipped"]:
            continue
        n_scene += 1
        fixes = sum(len(v) for v in plan["cameras"].values())
        n_fix += fixes
        print(f"\n{sd.name}  (canonical {plan['canonical']})")
        for cam, pairs in plan["cameras"].items():
            m = ", ".join(f"{o}->{n}" for o, n in pairs.items())
            print(f"    {cam:<8} {m:<20} dist={plan['quality'][cam]}")
        for cam, why in plan["skipped"].items():
            print(f"    {cam:<8} SKIPPED  {why}")
        if args.apply:
            for a in apply_remap(sd, plan):
                print(f"      applied {a}")
            json.dump(plan, open(sd / "gt_reid_map.json", "w"), indent=2)

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: "
          f"{n_fix} camera relabels across {n_scene} scenes")
    if not args.apply and n_fix:
        print("re-run with --apply to rename the files")


if __name__ == "__main__":
    main()
