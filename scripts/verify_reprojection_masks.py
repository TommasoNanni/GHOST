"""Verify person identity by reprojecting GT into each camera and reading the mask.

The existing check (scripts/verify_rich_subject_identity.py) compares a projected GT
root against track bbox *centres*, in body-width units. That is indirect: a camera at
0.7 body-widths is ambiguous, and the threshold (0.8/1.5) is a guess.

This asks the question directly: **project the GT body into the image and read which
person id the segmentation actually assigned to those pixels.** ``mask_data.npz`` stores
one uint16 array per frame (key ``mask_<frame:05d>_<cam>``) whose pixel value IS the
person id, so the answer is a lookup, not a distance.

Per camera we report the fraction of sampled GT joints that land on the id the pipeline
believes is the subject. Interpretation:

  hit >= 0.7   the id is that human
  0.3 - 0.7    partially overlapping / occluded / borderline -- inspect
  hit <  0.3   the id is a DIFFERENT human, or GT is not visible in this view

``bg`` is the fraction landing on background (value 0), which separates "wrong person"
(low hit, low bg) from "not visible / bad projection" (low hit, high bg).

    pixi run python scripts/verify_reprojection_masks.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \\
        --centered_root /tmp/ct_ver --body_split train_body [--scenes A,B]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ is not a package

from verify_rich_subject_identity import (               # noqa: E402
    load_calibration, load_crop_meta, project, subject_global_id,
)

_SMPLX_GT_KEYS = ("body_pose", "global_orient", "transl", "betas")


def gt_joints_for_frame(rich_root: Path, body_split: str, scene: str,
                        frame: int, smplx_model) -> np.ndarray | None:
    """GT SMPL-X joints (55, 3) in the multi-cam world frame for one frame."""
    import pickle
    import torch
    scene_short = scene.split("_")[0]
    seq = "_".join(scene.split("_")[:3])
    for pdir in sorted((rich_root / body_split).glob(f"{seq}*/{frame:05d}")):
        for pkl in sorted(pdir.glob("*.pkl")):
            with open(pkl, "rb") as fh:
                d = pickle.load(fh)
            try:
                out = smplx_model(
                    betas=torch.as_tensor(d["betas"], dtype=torch.float32).reshape(1, -1)[:, :10],
                    body_pose=torch.as_tensor(d["body_pose"], dtype=torch.float32).reshape(1, -1)[:, :63],
                    global_orient=torch.as_tensor(d["global_orient"], dtype=torch.float32).reshape(1, 3),
                    transl=torch.as_tensor(d["transl"], dtype=torch.float32).reshape(1, 3),
                )
                return out.joints[0, :55].detach().cpu().numpy()
            except Exception:
                return None
    return None


def mask_frame(cam_dir: Path, cam_tag: str, frame: int, cache: dict):
    """Read one frame's mask array (uint16, pixel value = person id)."""
    if "z" not in cache:
        p = cam_dir / "mask_data.npz"
        if not p.exists():
            cache["z"] = None
        else:
            cache["z"] = np.load(p)
    z = cache["z"]
    if z is None:
        return None
    for key in (f"mask_{frame:05d}_{cam_tag}", f"mask_{frame}_{cam_tag}", f"mask_{frame:05d}"):
        if key in z.files:
            return z[key]
    return None


def check_camera(cam_dir: Path, cal: dict, offs: tuple[int, int],
                 joints_by_frame: dict[int, np.ndarray], want_id: int) -> dict | None:
    """Fraction of projected GT joints landing on `want_id` in the mask."""
    cam_tag = cam_dir.name.split("_")[-1]
    cache: dict = {}
    hit = other = bg = total = 0
    other_ids: dict[int, int] = {}
    for frame, J in joints_by_frame.items():
        m = mask_frame(cam_dir, cam_tag, frame, cache)
        if m is None:
            continue
        uv, _ = project(J, cal["K"], cal["RT"])
        uv = uv - np.asarray(offs, dtype=uv.dtype)          # centred-image crop offset
        H, W = m.shape
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        vals = m[v[ok], u[ok]]
        total += vals.size
        hit += int((vals == want_id).sum())
        bg += int((vals == 0).sum())
        for x in np.unique(vals):
            if x not in (0, want_id):
                other_ids[int(x)] = other_ids.get(int(x), 0) + int((vals == x).sum())
        other += int(((vals != 0) & (vals != want_id)).sum())
    if total == 0:
        return None
    return {"hit": hit / total, "bg": bg / total, "other": other / total,
            "n": total, "top_other": sorted(other_ids.items(), key=lambda kv: -kv[1])[:2]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--centered_root", required=True, type=Path)
    ap.add_argument("--body_split", default="train_body")
    ap.add_argument("--smplx_model", default="body_models/SMPLX_NEUTRAL.pkl")
    ap.add_argument("--frames", type=int, default=6, help="frames sampled per scene")
    ap.add_argument("--warn", type=float, default=0.3,
                    help="hit fraction below this = suspect camera")
    ap.add_argument("--scenes", default="")
    args = ap.parse_args()

    import smplx
    smplx_model = smplx.create(
        str(Path(args.smplx_model).parent), model_type="smplx",
        gender="neutral", use_pca=False, batch_size=1,
    )

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = [d for d in sorted(args.ghost_root.iterdir())
              if d.is_dir() and (d / "cross_view_reid.json").exists()]
    if wanted:
        scenes = [d for d in scenes if d.name in wanted]

    print(f"{'scene':<42}{'cam':>8}{'hit':>7}{'bg':>7}{'other':>7}  verdict")
    n_cam = n_bad = 0
    bad: list[str] = []
    for sd in scenes:
        want = subject_global_id(sd, args.rich_root, args.body_split)
        if want is None:
            continue
        cals = load_calibration(args.rich_root, sd.name)
        crop = load_crop_meta(args.centered_root, sd.name)

        # sample frames that exist in GT
        seq = "_".join(sd.name.split("_")[:3])
        gt_dirs = sorted((args.rich_root / args.body_split).glob(f"{seq}*/*"))
        frames = [int(p.name) for p in gt_dirs if p.name.isdigit()]
        if not frames:
            continue
        step = max(1, len(frames) // args.frames)
        frames = frames[::step][:args.frames]
        joints = {}
        for f in frames:
            J = gt_joints_for_frame(args.rich_root, args.body_split, sd.name, f, smplx_model)
            if J is not None:
                joints[f] = J
        if not joints:
            continue

        for cam_dir in sorted(p for p in sd.iterdir()
                              if p.is_dir() and (p / "mask_data.npz").exists()):
            idx = int(cam_dir.name.split("_")[-1])
            if idx not in cals:
                continue
            off = crop.get(cam_dir.name, {})
            offs = (off.get("x", 0), off.get("y", 0))
            r = check_camera(cam_dir, cals[idx], offs, joints, want)
            if r is None:
                continue
            n_cam += 1
            verdict = "ok" if r["hit"] >= 0.7 else ("SUSPECT" if r["hit"] < args.warn else "borderline")
            if r["hit"] < args.warn:
                n_bad += 1
                bad.append(f"{sd.name}/{cam_dir.name}")
            print(f"{sd.name:<42}{cam_dir.name:>8}{r['hit']:7.2f}{r['bg']:7.2f}"
                  f"{r['other']:7.2f}  {verdict}"
                  + (f"  top_other={r['top_other']}" if r["hit"] < 0.7 else ""))

    print()
    print(f"  cameras checked: {n_cam}   suspect (hit < {args.warn}): {n_bad}")
    for b in bad:
        print(f"    {b}")


if __name__ == "__main__":
    main()
