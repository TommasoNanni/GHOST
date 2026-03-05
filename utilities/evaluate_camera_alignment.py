"""Evaluate and report camera alignment quality.

This utility uses `ghost.data.camera_alignment.CameraAlignment` to estimate
pairwise relative poses between cameras from `body_data/person_<id>.npz`
outputs, then computes simple diagnostics (RMSE, counts) and saves a CSV
report plus a couple of plots (`residuals_hist.png`, `stats.csv`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ghost.data.camera_alignment import CameraAlignment


def _load_person_npz(body_dir: Path, person_id: int):
    path = body_dir / f"person_{person_id}.npz"
    if not path.exists():
        return None
    with np.load(str(path)) as f:
        return dict(f)


def collect_correspondences(video_dirs: Dict[str, Path]):
    """Collect stacked point correspondences for every video pair.

    Returns dict[(vid_a, vid_b)] -> (pts_a, pts_b) where pts are (N,3)
    """
    video_persons = {}
    for vid_id, vid_dir in video_dirs.items():
        body_dir = Path(vid_dir) / "body_data"
        summary = body_dir / "body_params_summary.json"
        if not summary.exists():
            continue
        import json
        with open(summary) as f:
            s = json.load(f)
        persons = {}
        for k in s.get("persons", {}):
            pid = int(k)
            data = _load_person_npz(body_dir, pid)
            if data is not None and "frame_indices" in data:
                persons[pid] = data
        if persons:
            video_persons[vid_id] = persons

    vids = list(video_dirs.keys())
    results = {}
    for i, a in enumerate(vids):
        if a not in video_persons:
            continue
        for b in vids[i + 1 :]:
            if b not in video_persons:
                continue
            persons_a = video_persons[a]
            persons_b = video_persons[b]
            shared = sorted(set(persons_a) & set(persons_b))
            if not shared:
                continue
            pts_a_parts = []
            pts_b_parts = []
            for pid in shared:
                da = persons_a[pid]
                db = persons_b[pid]
                fa = {int(fi): idx for idx, fi in enumerate(da["frame_indices"])}
                fb = {int(fi): idx for idx, fi in enumerate(db["frame_indices"])}
                common = sorted(set(fa) & set(fb))
                for fi in common:
                    ra = fa[fi]
                    rb = fb[fi]
                    ja = None
                    jb = None
                    if "pred_keypoints_3d" in da:
                        k = da["pred_keypoints_3d"]
                        ja = k[ra] + da.get("pred_cam_t", np.zeros((1, 3)))[ra]
                    elif "vertices" in da:
                        ja = da["vertices"][ra]
                    if "pred_keypoints_3d" in db:
                        k = db["pred_keypoints_3d"]
                        jb = k[rb] + db.get("pred_cam_t", np.zeros((1, 3)))[rb]
                    elif "vertices" in db:
                        jb = db["vertices"][rb]
                    if ja is None or jb is None:
                        continue
                    n = min(len(ja), len(jb))
                    if n < 1:
                        continue
                    pts_a_parts.append(ja[:n])
                    pts_b_parts.append(jb[:n])
            if not pts_a_parts:
                continue
            pts_a = np.concatenate(pts_a_parts, axis=0).astype(np.float64)
            pts_b = np.concatenate(pts_b_parts, axis=0).astype(np.float64)
            results[(a, b)] = (pts_a, pts_b)
    return results


def evaluate(video_dirs: Dict[str, Path], out_dir: Path, min_correspondences: int = 30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CA = CameraAlignment()
    correspondences = collect_correspondences(video_dirs)
    rows = []
    all_residuals = []
    for (a, b), (pts_a, pts_b) in correspondences.items():
        n = len(pts_a)
        if n < min_correspondences:
            rows.append((a, b, n, float('nan'), float('nan')))
            continue
        try:
            R, t = CA._kabsch(pts_a, pts_b)
        except Exception:
            rows.append((a, b, n, float('nan'), float('nan')))
            continue
        residuals = pts_b - (pts_a @ R.T + t[None, :])
        errs = np.sqrt((residuals ** 2).sum(axis=-1))
        rmse = float(np.sqrt((errs ** 2).mean()))
        rows.append((a, b, n, rmse, float(np.linalg.norm(t))))
        all_residuals.extend(errs.tolist())

    # write CSV
    csv_path = out_dir / "camera_alignment_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vid_a", "vid_b", "n_correspondences", "rmse_m", "|t|_m"])
        for r in rows:
            writer.writerow(r)

    # residuals histogram
    if all_residuals:
        plt.figure()
        plt.hist(all_residuals, bins=50)
        plt.xlabel("per-point error (m)")
        plt.ylabel("count")
        plt.title("Alignment residuals")
        plt.tight_layout()
        plt.savefig(str(out_dir / "residuals_hist.png"))
        plt.close()

    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dirs", nargs="+", required=True,
                        help="Pairs of video_id=path entries, e.g. cam1=out/dir cam2=out/dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min", type=int, default=30)
    args = parser.parse_args()
    video_dirs = {}
    for token in args.video_dirs:
        if "=" not in token:
            continue
        vid, p = token.split("=", 1)
        video_dirs[vid] = Path(p)
    csv_path = evaluate(video_dirs, args.out_dir, args.min)
    print(f"Wrote report to {csv_path}")


if __name__ == "__main__":
    main()
