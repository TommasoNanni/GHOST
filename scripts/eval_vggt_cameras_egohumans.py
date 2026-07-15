"""
Evaluate VGGT-Omega camera predictions vs COLMAP GT on EgoHumans exo cameras.

For each scene with a `vggt_cameras_centered.npz`, finds the matching COLMAP
reconstruction (`colmap/workplace/images.txt`) in the raw data tree, and for the
exo cameras VGGT actually ran on computes (after re-rooting both GT and VGGT to
the first camera, so the arbitrary COLMAP world frame cancels):

  - rotation error (degrees)            ← the main "camera angle error"
  - translation direction error (deg)   ← angle between camera centers
  - scale ratio  ||C_gt|| / ||C_vggt||

Rotation/direction errors are invariant to the undistortion + resize VGGT saw
(those change intrinsics + pixels, not relative orientation), so this is a clean
check of whether the black-border undistortion artefacts hurt VGGT geometry.

COLMAP images.txt convention: per image line
  IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
gives world->cam rotation R(qvec) and translation t -> camera center C = -R^T t.
Exo cameras are static, so the first registered frame per camera is used.
"""

import argparse
import numpy as np
from pathlib import Path

# output tree (vggt npz) and raw data tree (colmap)
OUT_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans")
# inner path from a raw activity root (or squashfuse mount) to the camera_ready tree
INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
# raw data roots to search for colmap GT (badminton backup + any --raw-root mounts).
# Activity sqsh archives mount to /tmp and expose the same `media/...` tree.
RAW_ROOTS = [
    Path("/iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def quat_to_R(qw, qx, qy, qz):
    """COLMAP quaternion (w,x,y,z) -> 3x3 world->cam rotation."""
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def parse_colmap_images(images_txt):
    """Return {cam_name: (R, t)} for static exo cams (first registered frame)."""
    out = {}
    with open(images_txt) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or not ln.strip():
            i += 1
            continue
        p = ln.split()
        # pose line has >= 10 fields ending in NAME; skip the POINTS2D line after
        name = p[9]
        cam = name.split("/")[0]
        if cam.startswith("cam") and cam not in out:
            qw, qx, qy, qz = map(float, p[1:5])
            tx, ty, tz = map(float, p[5:8])
            out[cam] = (quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz], np.float64))
        i += 2   # skip the 2D-points line
    return out


def find_colmap(scene_rel):
    """scene_rel = '06_badminton/055_badminton' -> images.txt path or None.

    Deterministic path under each raw root (no glob — fast over fuse mounts):
      <root>/<INNER>/<activity>/<seq>/colmap/workplace/images.txt
    The badminton backup nests one extra <activity>/ level, so also try that.
    """
    activity, seq = scene_rel.split("/")
    rel = f"{INNER}/{activity}/{seq}/colmap/workplace/images.txt"
    for root in RAW_ROOTS:
        for cand in (root / rel, root / activity / rel):
            if cand.exists():
                return cand
    return None


def reroot(pose_dict, ref):
    R0, t0 = pose_dict[ref]
    out = {}
    for name, (Rk, tk) in pose_dict.items():
        R_rel = Rk @ R0.T
        t_rel = tk - R_rel @ t0
        out[name] = (R_rel, t_rel)
    return out


def cam_center(R, t):
    return -R.T @ t


def rot_err_deg(R1, R2):
    cos = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ── per-scene ─────────────────────────────────────────────────────────────────

def evaluate_scene(scene_rel):
    npz_path = OUT_ROOT / scene_rel / "vggt_cameras_centered.npz"
    if not npz_path.exists():
        return None, "no vggt npz"

    images_txt = find_colmap(scene_rel)
    if images_txt is None:
        return None, "no colmap"

    d          = np.load(npz_path)
    extrinsics = d["extrinsics"].astype(np.float64)   # (T,K,3,4)
    valid      = d["valid"]                            # (T,K)
    cam_names  = [n.decode() if isinstance(n, bytes) else n for n in d["camera_names"]]

    gt_all = parse_colmap_images(images_txt)
    # keep only cams VGGT used and that COLMAP registered
    gt = {c: gt_all[c] for c in cam_names if c in gt_all}

    ref = cam_names[0]
    if ref not in gt:
        return None, f"ref {ref} not in colmap"
    gt_rel = reroot(gt, ref)

    # average VGGT extrinsics over valid frames, re-orthonormalise R
    vggt = {}
    for k, name in enumerate(cam_names):
        fr = np.where(valid[:, k])[0]
        if len(fr) == 0:
            continue
        E = extrinsics[fr, k].mean(axis=0)
        U, _, Vt = np.linalg.svd(E[:, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        vggt[name] = (R, E[:, 3])
    if ref not in vggt:
        return None, f"ref {ref} not valid in vggt"
    vggt_rel = reroot(vggt, ref)

    rows = []
    for name in cam_names[1:]:
        if name not in gt_rel or name not in vggt_rel:
            continue
        R_gt, t_gt = gt_rel[name]
        R_vg, t_vg = vggt_rel[name]
        C_gt, C_vg = cam_center(R_gt, t_gt), cam_center(R_vg, t_vg)
        d_gt, d_vg = np.linalg.norm(C_gt), np.linalg.norm(C_vg)
        rows.append({
            "scene": scene_rel, "cam": name,
            "rot_err": rot_err_deg(R_vg, R_gt),
            "dir_err": angle_between(C_gt, C_vg),
            "scale_ratio": d_gt / d_vg if d_vg > 1e-9 else np.nan,
        })
    if not rows:
        return None, "no common cams"
    return rows, None


# ── main ──────────────────────────────────────────────────────────────────────

_ALL_ACTS = ["01_tagging", "02_lego", "03_fencing", "04_basketball",
             "05_volleyball", "06_badminton", "07_tennis"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", action="append", default=None,
                    help="activity folder(s) under output root; repeatable (default: all)")
    ap.add_argument("--raw-root", action="append", default=None,
                    help="extra raw/colmap root(s) to search (e.g. squashfuse mounts); repeatable")
    ap.add_argument("--out-root", default=None,
                    help="override the vggt-output root (e.g. a probe's temp_egohumans dir)")
    args = ap.parse_args()

    if args.raw_root:
        RAW_ROOTS.extend(Path(r) for r in args.raw_root)
    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)

    activities = args.activity or [a for a in _ALL_ACTS if (OUT_ROOT / a).is_dir()]
    scenes = []
    for act in activities:
        act_dir = OUT_ROOT / act
        if act_dir.is_dir():
            scenes.extend(sorted(f"{act}/{p.name}" for p in act_dir.iterdir() if p.is_dir()))
    print(f"Evaluating {len(scenes)} scenes across {len(activities)} activities...\n")

    all_rows, skipped = [], []
    for scene_rel in scenes:
        rows, why = evaluate_scene(scene_rel)
        if rows is None:
            skipped.append((scene_rel, why))
            continue
        all_rows.extend(rows)
        rot = [r["rot_err"] for r in rows]
        dr  = [r["dir_err"] for r in rows]
        print(f"  {scene_rel:<32}  cams={len(rows)}  "
              f"rot_mean={np.mean(rot):5.2f}°  rot_max={np.max(rot):5.2f}°  "
              f"dir_mean={np.nanmean(dr):5.2f}°")

    if not all_rows:
        print("\nNo data collected.")
        for s, w in skipped:
            print(f"  skip {s}: {w}")
        return

    # per-activity rotation summary (the headline: camera angle error per activity)
    print(f"\n{'='*64}")
    print(f"PER-ACTIVITY rotation error (°)")
    print(f"{'='*64}")
    by_act: dict[str, list] = {}
    for r in all_rows:
        by_act.setdefault(r["scene"].split("/")[0], []).append(r["rot_err"])
    for act in sorted(by_act):
        v = np.array(by_act[act])
        nsc = len({r["scene"] for r in all_rows if r["scene"].startswith(act)})
        print(f"  {act:<16} scenes={nsc:>3} pairs={len(v):>3}  "
              f"mean={v.mean():5.2f}  median={np.median(v):5.2f}  "
              f"p90={np.percentile(v,90):5.2f}  max={v.max():5.2f}")

    rot = np.array([r["rot_err"] for r in all_rows])
    dr  = np.array([r["dir_err"] for r in all_rows if not np.isnan(r["dir_err"])])
    sc  = np.array([r["scale_ratio"] for r in all_rows if not np.isnan(r["scale_ratio"])])

    print(f"\n{'='*64}")
    print(f"AGGREGATE  ({len(all_rows)} cam-scene pairs, {len(scenes)-len(skipped)} scenes)")
    print(f"{'='*64}")
    print(f"  Rotation error (°)          mean={rot.mean():.2f}  median={np.median(rot):.2f}  "
          f"std={rot.std():.2f}  p90={np.percentile(rot,90):.2f}  max={rot.max():.2f}")
    print(f"  Transl direction error (°)  mean={dr.mean():.2f}  median={np.median(dr):.2f}  "
          f"std={dr.std():.2f}  p90={np.percentile(dr,90):.2f}  max={dr.max():.2f}")
    print(f"  Scale ratio (GT/VGGT)       mean={sc.mean():.3f}  median={np.median(sc):.3f}  "
          f"p5={np.percentile(sc,5):.3f}  p95={np.percentile(sc,95):.3f}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s, w in skipped:
            print(f"  {s}: {w}")


if __name__ == "__main__":
    main()
