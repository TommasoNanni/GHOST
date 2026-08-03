"""Verify that the person used for training is the SAME human in every camera.

The training datapoint picks ONE global ghost id for the GT subject and then reads
`person_<id>.npz` from every camera. That guarantees a single *id* across views — it
does NOT guarantee the id refers to the same *human* in each view. If cross-view ReID
mislabelled one camera, that camera silently contributes a different person's pose.

This checks the link directly, per camera:

    GT root --GT calibration--> pixel --scale/crop--> centered-crop pixels
    vs. the mean of that camera's stored `pred_keypoints_2d` for the matched id

A small distance means the stored track really is the annotated subject. A large one
means that camera holds somebody else under the same id.

Distances are reported in units of the subject's own 2D size (bbox diagonal), so they
are comparable across cameras and scenes: <0.5 is solidly the same person, >1.5 is
almost certainly a different one.

    pixi run python scripts/verify_rich_subject_identity.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \\
        --centered_root /tmp/ct_ver --body_split train_body
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.disable(logging.WARNING)

# Copied from scripts/verify_rich_train_preprocessing.py rather than imported: a
# dependency installs a top-level `scripts` package into the pixi env that shadows this
# repo's scripts/ directory, so `from scripts.x import y` resolves to the wrong module.

import pickle                                              # noqa: E402
import xml.etree.ElementTree as ET                         # noqa: E402


def load_calibration(rich_root: Path, scene: str) -> dict[int, dict]:
    """{cam_index: {'K': (3,3), 'RT': (3,4)}} from scan_calibration/<LOCATION>/calibration."""
    location = scene.split("_")[0]
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    out: dict[int, dict] = {}
    for xml_path in sorted(calib_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
            mats = {}
            for node in root:
                data = node.find("data")
                if data is None or data.text is None:
                    continue
                vals = [float(v) for v in data.text.split()]
                rows, cols = int(node.find("rows").text), int(node.find("cols").text)
                mats[node.tag] = np.array(vals, dtype=np.float64).reshape(rows, cols)
            if "Intrinsics" in mats and "CameraMatrix" in mats:
                out[int(xml_path.stem)] = {"K": mats["Intrinsics"], "RT": mats["CameraMatrix"]}
        except Exception:
            continue
    return out


def load_crop_meta(centered_root: Path | None, scene: str) -> dict[str, dict]:
    if centered_root is None:
        return {}
    p = centered_root / scene / "crop_meta.json"
    return json.load(open(p)).get("cameras", {}) if p.exists() else {}


def gt_root_translation(rich_root: Path, body_split: str, scene: str,
                        frame: int) -> np.ndarray | None:
    fdir = rich_root / body_split / scene / f"{frame:05d}"
    if not fdir.is_dir():
        return None
    pkls = sorted(fdir.glob("*.pkl"))
    if not pkls:
        return None
    try:
        d = pickle.load(open(pkls[0], "rb"))
    except Exception:
        return None
    t = d.get("transl")
    return np.asarray(t, dtype=np.float64).reshape(-1)[:3] if t is not None else None


def project(X: np.ndarray, K: np.ndarray, RT: np.ndarray) -> tuple[np.ndarray, float]:
    Xc = RT[:, :3] @ X + RT[:, 3]
    z = float(Xc[2])
    if z <= 1e-6:
        return np.array([np.nan, np.nan]), z
    uv = K @ (Xc / z)
    return uv[:2], z


def subject_global_id(scene_dir: Path, rich_root: Path, body_split: str) -> int | None:
    """The ghost global id the training datapoint would use for the GT subject."""
    from data.fusion_dataset import RICHFusionDatapoint
    try:
        dp = RICHFusionDatapoint(
            scene_dir=scene_dir, rich_data_root=rich_root,
            rich_gt_dir=rich_root, body_split=body_split,
        )
    except Exception:
        return None
    if dp.num_frames == 0 or not dp.has_gt:
        return None
    gt = getattr(dp, "_gt", None)
    if not gt or not gt[0]:
        return None
    return sorted(gt[0].keys())[0]      # remapped to ghost ids already


def check_scene(scene_dir: Path, rich_root: Path, centered_root: Path,
                body_split: str, orig_width: int, n_frames: int) -> dict | None:
    scene = scene_dir.name
    gid = subject_global_id(scene_dir, rich_root, body_split)
    if gid is None:
        return None

    calib = load_calibration(rich_root, scene)
    crop = load_crop_meta(centered_root, scene)
    gt_dir = rich_root / body_split / scene
    if not gt_dir.is_dir() or not calib or not crop:
        return None
    frames = sorted(int(p.name) for p in gt_dir.iterdir() if p.name.isdigit())
    step = max(1, len(frames) // max(n_frames, 1))
    frames = frames[::step][:n_frames]

    per_cam: dict[str, list[float]] = {}
    for cam_dir in sorted(d for d in scene_dir.iterdir()
                          if d.is_dir() and (d / "body_data").is_dir()):
        npz = cam_dir / "body_data" / f"person_{gid}.npz"
        cm = crop.get(cam_dir.name)
        cam_idx = int(re.sub(r"\D", "", cam_dir.name))
        cal = calib.get(cam_idx)
        if not npz.exists() or cm is None or cal is None:
            continue
        d = np.load(npz, allow_pickle=False)
        if "pred_keypoints_2d" not in d.files:
            continue
        fi = d["frame_indices"].astype(int)
        kp = d["pred_keypoints_2d"]                      # (n, J, 2) centered-crop px
        scale = cm["src_w"] / float(orig_width)

        rel = []
        for f in frames:
            idx = np.where(fi == f)[0]
            X = gt_root_translation(rich_root, body_split, scene, f)
            if not idx.size or X is None:
                continue
            uv, z = project(X, cal["K"], cal["RT"])
            if not np.isfinite(uv).all() or z <= 0:
                continue
            gx, gy = uv[0] * scale - cm["off_x"], uv[1] * scale - cm["off_y"]
            k = kp[int(idx[0])]
            k = k[np.isfinite(k).all(-1)]
            if k.size == 0:
                continue
            centre = k.mean(0)
            size = float(np.hypot(*(k.max(0) - k.min(0)))) or 1.0
            rel.append(float(np.hypot(centre[0] - gx, centre[1] - gy) / size))
        if rel:
            per_cam[cam_dir.name] = rel
    if not per_cam:
        return None
    return {"scene": scene, "gid": gid,
            "per_cam": {c: float(np.median(v)) for c, v in per_cam.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--centered_root", required=True, type=Path)
    ap.add_argument("--body_split", default="train_body")
    ap.add_argument("--orig_width", type=int, default=4112)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--bad", type=float, default=1.5,
                    help="relative distance above which a camera is a different person")
    ap.add_argument("--scenes", default="")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = [d for d in sorted(args.ghost_root.iterdir())
              if d.is_dir() and (d / "cross_view_reid.json").exists()]
    if wanted:
        scenes = [d for d in scenes if d.name in wanted]

    print(f"{'scene':<40}{'id':>4}{'cams':>6}{'median':>9}{'worst':>8}  suspect cameras")
    n_cam = n_bad = 0
    bad_scenes = []
    for sd in scenes:
        r = check_scene(sd, args.rich_root, args.centered_root,
                        args.body_split, args.orig_width, args.frames)
        if r is None:
            continue
        vals = r["per_cam"]
        n_cam += len(vals)
        sus = {c: v for c, v in vals.items() if v > args.bad}
        n_bad += len(sus)
        if sus:
            bad_scenes.append(r["scene"])
        med = float(np.median(list(vals.values())))
        worst = max(vals.values())
        s = "  " + ",".join(f"{c}:{v:.1f}" for c, v in sorted(sus.items(), key=lambda kv: -kv[1]))
        print(f"  {r['scene']:<38}{r['gid']:>4}{len(vals):>6}{med:>9.2f}{worst:>8.2f}{s if sus else ''}")

    print(f"\n  cameras checked: {n_cam}   suspect (>{args.bad} body-sizes off): {n_bad} "
          f"({100*n_bad/max(n_cam,1):.1f}%)")
    if bad_scenes:
        print(f"  scenes with >=1 suspect camera: {len(bad_scenes)}")


if __name__ == "__main__":
    main()
