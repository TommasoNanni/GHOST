"""
Stage A data-prep for the EgoHumans balance=0 re-run.

Reads RAW full-size distorted frames from a (read-only) squashfuse mount, undistorts
them the CORRECT way (fisheye balance=0 → no black borders) AND resizes to a target
long side in a single remap, then writes the resized-undistorted frames + the adjusted
calibration into a writable temp data_root laid out exactly how
`scripts/egohumans_pipeline.py` expects:

    <out_root>/<activity>/<INNER>/<activity>/<seq>/exo/<cam>/
        images_undistorted/frames/<stem>.jpg     resized-undistorted (balance=0)
        calibration.json                          K scaled to the resized resolution

Writing into the `frames/` subdir means the pipeline uses them verbatim (no second
resize). Nothing is written to the sqsh; the source stays read-only.

Correctness of resize+undistort: undistortion needs the full-res K,D. We compute the
balance=0 new camera matrix at full res, scale it by the resize factor, and remap
straight to the target size — so a target pixel maps back through the scaled pinhole
K_new_r → fisheye distortion → full-res source. No intermediate full-res image, no
double resampling. The saved calibration is the scaled K_new_r.

Idempotent: a cam whose output frame count already matches the source is skipped.

Usage (one sequence, from a mounted activity sqsh):
    python scripts/prep_undistort_egohumans.py \
        --raw-root /tmp/mnt_<jobid> --activity 02_lego --seq 001_legoassemble \
        --out-root /iopsstor/scratch/cscs/tnanni/temp_egohumans --max-side 1440
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

KEEP_CAMS = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}


def parse_colmap_calibration(seq_dir: Path) -> dict[str, dict]:
    """Return {cam_name: {K, D, W, H}} from COLMAP OPENCV_FISHEYE cameras.txt + images.txt."""
    cameras_txt = seq_dir / "colmap" / "workplace" / "cameras.txt"
    images_txt  = seq_dir / "colmap" / "workplace" / "images.txt"

    cam_params: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if parts[1] != "OPENCV_FISHEYE":
                raise ValueError(f"{seq_dir.name}: cam {parts[0]} model {parts[1]} != OPENCV_FISHEYE")
            cid = int(parts[0])
            W, H = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = (float(x) for x in parts[4:8])
            k1, k2, k3, k4 = (float(x) for x in parts[8:12])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
            cam_params[cid] = {"K": K, "D": D, "W": W, "H": H}

    name_to_cid: dict[str, int] = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            cam = parts[9].split("/")[0]
            if cam.startswith("cam") and cam not in name_to_cid:
                name_to_cid[cam] = int(parts[8])

    return {n: cam_params[c] for n, c in name_to_cid.items() if c in cam_params}


def build_map(calib: dict, max_side: int):
    """balance=0 fisheye undistort folded with resize. Returns (K_new_r, W_r, H_r, map1, map2)."""
    K, D, W, H = calib["K"], calib["D"], calib["W"], calib["H"]
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0
    )
    s = max_side / max(W, H) if max_side else 1.0
    W_r, H_r = int(round(W * s)), int(round(H * s))
    S = np.diag([s, s, 1.0])
    K_new_r = S @ K_new
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new_r, (W_r, H_r), cv2.CV_16SC2
    )
    return K_new_r, W_r, H_r, map1, map2


def prep_camera(src_cam: Path, out_cam: Path, calib: dict, max_side: int) -> str:
    src_imgs = sorted((src_cam / "images").glob("*.jpg"))
    if not src_imgs:
        return f"  {src_cam.name}: no raw frames — skip"
    out_frames = out_cam / "images_undistorted" / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    if len(list(out_frames.glob("*.jpg"))) == len(src_imgs):
        return f"  {src_cam.name}: already done ({len(src_imgs)})"

    K_new_r, W_r, H_r, map1, map2 = build_map(calib, max_side)
    for f in src_imgs:
        img = cv2.imread(str(f))
        if img is None:
            continue
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_frames / f.name), und)

    with open(out_cam / "calibration.json", "w") as fh:
        json.dump({"K": K_new_r.tolist(), "width": W_r, "height": H_r}, fh, indent=2)
    return f"  {src_cam.name}: {len(src_imgs)} frames -> {W_r}x{H_r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="squashfuse mount (read-only source)")
    ap.add_argument("--activity", required=True)
    ap.add_argument("--seq", default=None, help="single sequence name; default all in activity")
    ap.add_argument("--out-root", required=True, help="writable temp data_root")
    ap.add_argument("--max-side", type=int, default=1440, help="resize long side (0 = no resize)")
    args = ap.parse_args()

    cam_ready = Path(args.raw_root) / INNER / args.activity
    if not cam_ready.is_dir():
        raise FileNotFoundError(f"not found: {cam_ready}")
    keep = KEEP_CAMS[args.activity]

    seq_dirs = sorted(p for p in cam_ready.iterdir() if p.is_dir())
    if args.seq:
        seq_dirs = [p for p in seq_dirs if p.name == args.seq]
    if not seq_dirs:
        raise SystemExit(f"no sequences matched (activity={args.activity} seq={args.seq})")

    for seq_dir in seq_dirs:
        print(f"[{args.activity}/{seq_dir.name}]")
        calib = parse_colmap_calibration(seq_dir)
        out_seq = Path(args.out_root) / args.activity / INNER / args.activity / seq_dir.name
        for cam in keep:
            if cam not in calib:
                print(f"  {cam}: no calibration — skip")
                continue
            print(prep_camera(seq_dir / "exo" / cam, out_seq / "exo" / cam,
                              calib[cam], args.max_side))
    print("Prep done.")


if __name__ == "__main__":
    main()
