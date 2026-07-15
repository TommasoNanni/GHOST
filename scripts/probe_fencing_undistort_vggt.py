"""
Probe: re-undistort an EgoHumans activity the CORRECT way (fisheye balance=0, no
black borders) into a temp dir, then run VGGT-Omega ONLY — to measure whether the
border fix improves VGGT camera angles, without touching source data / the sqsh and
without re-running segmentation / SAM3D / MapAnything.

Self-contained (copies the undistort + colmap-parse logic; does not import the old
pipeline). VGGT is loaded ONCE and reused across all scenes.

Outputs, per scene, under  <out>/<activity>/<seq>/ :
    exo/<cam>/images_undistorted/<stem>.jpg   undistorted (balance=0) frames
    exo/<cam>/calibration.json                K_new (pinhole, no distortion)
    vggt_cameras_centered.npz                 VGGT extrinsics/intrinsics (same name
                                              the pipeline uses; depth ignored here)

Nothing is written into the mounted raw data; the sqsh stays read-only.

Eval the result with:
    python scripts/eval_vggt_cameras_egohumans.py \
        --activity 03_fencing --raw-root <mount> --out-root <out>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.run_vggt import VGGTPreprocessor  # noqa: E402

# inner path from an activity root / squashfuse mount to the camera_ready tree
INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

# exo cameras VGGT runs on, per activity (mirrors utilities/undistort_egohumans.py)
KEEP_CAMS = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}


# ── COLMAP OPENCV_FISHEYE calibration ─────────────────────────────────────────

def parse_colmap_calibration(seq_dir: Path) -> dict[str, dict]:
    """Return {cam_name: {K, D, W, H}} from COLMAP cameras.txt + images.txt."""
    cameras_txt = seq_dir / "colmap" / "workplace" / "cameras.txt"
    images_txt  = seq_dir / "colmap" / "workplace" / "images.txt"

    cam_params: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            model = parts[1]
            if model != "OPENCV_FISHEYE":
                raise ValueError(
                    f"{seq_dir.name}: camera {parts[0]} model is {model}, "
                    f"expected OPENCV_FISHEYE (balance=0 undistort assumes fisheye)."
                )
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
            cam_name = parts[9].split("/")[0]
            if cam_name.startswith("cam") and cam_name not in name_to_cid:
                name_to_cid[cam_name] = int(parts[8])

    return {name: cam_params[cid]
            for name, cid in name_to_cid.items() if cid in cam_params}


def fisheye_balance0_map(calib: dict):
    """Precompute (K_new, map1, map2) for a balance=0 fisheye undistort (no borders)."""
    K, D, W, H = calib["K"], calib["D"], calib["W"], calib["H"]
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2
    )
    return K_new, map1, map2


# ── per-scene ─────────────────────────────────────────────────────────────────

def select_frame_stems(seq_dir: Path, cams: list[str], stride: int, max_frames: int):
    """Common frame stems present across all cams, subsampled by stride+cap."""
    per_cam = []
    for cam in cams:
        stems = {p.stem for p in (seq_dir / "exo" / cam / "images").glob("*.jpg")}
        per_cam.append(stems)
    common = sorted(set.intersection(*per_cam)) if per_cam else []
    common = common[::stride]
    if max_frames and len(common) > max_frames:
        idx = np.linspace(0, len(common) - 1, max_frames).round().astype(int)
        common = [common[i] for i in idx]
    return common


def process_scene(seq_dir: Path, out_dir: Path, cams: list[str],
                  calib: dict, stride: int, max_frames: int,
                  preprocessor: VGGTPreprocessor, device: str) -> str:
    stems = select_frame_stems(seq_dir, cams, stride, max_frames)
    if not stems:
        return f"SKIP {seq_dir.name}: no common frames across {cams}"

    # precompute undistort maps + write calibration.json per cam
    maps = {}
    for cam in cams:
        K_new, map1, map2 = fisheye_balance0_map(calib[cam])
        maps[cam] = (map1, map2)
        cam_out = out_dir / "exo" / cam / "images_undistorted"
        cam_out.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "exo" / cam / "calibration.json", "w") as f:
            json.dump({"K": K_new.tolist(),
                       "width": int(calib[cam]["W"]),
                       "height": int(calib[cam]["H"])}, f, indent=2)

    # undistort the selected frames
    for cam in cams:
        map1, map2 = maps[cam]
        src = seq_dir / "exo" / cam / "images"
        dst = out_dir / "exo" / cam / "images_undistorted"
        for stem in stems:
            img = cv2.imread(str(src / f"{stem}.jpg"))
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst / f"{stem}.jpg"), und)

    # build frame_paths[t][k]
    frame_paths: list[list[Path | None]] = []
    for stem in stems:
        row: list[Path | None] = []
        for cam in cams:
            p = out_dir / "exo" / cam / "images_undistorted" / f"{stem}.jpg"
            row.append(p if p.exists() else None)
        frame_paths.append(row)

    # VGGT (single device → reuses the already-loaded model, no reload)
    preprocessor.process_scene(
        frame_paths=frame_paths,
        camera_names=cams,
        output_dir=out_dir,
        devices=[device],
    )
    return f"OK {seq_dir.name}: {len(stems)} frames × {len(cams)} cams"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True,
                    help="activity root or squashfuse mount containing media/.../camera_ready")
    ap.add_argument("--activity", default="03_fencing")
    ap.add_argument("--out", required=True, help="output root (e.g. ghost/temp_egohumans)")
    ap.add_argument("--weights", required=True, help="VGGT-Omega checkpoint path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=10, help="frame subsample stride")
    ap.add_argument("--max-frames", type=int, default=60, help="cap frames per scene")
    ap.add_argument("--seq", default=None, help="limit to one sequence (debug)")
    args = ap.parse_args()

    cam_ready = Path(args.raw_root) / INNER / args.activity
    if not cam_ready.is_dir():
        raise FileNotFoundError(f"camera_ready not found: {cam_ready}")
    cams_keep = KEEP_CAMS[args.activity]

    seq_dirs = sorted(p for p in cam_ready.iterdir() if p.is_dir())
    if args.seq:
        seq_dirs = [p for p in seq_dirs if p.name == args.seq]
    print(f"Probe {args.activity}: {len(seq_dirs)} scenes, "
          f"stride={args.stride} max_frames={args.max_frames}, device={args.device}")

    preprocessor = VGGTPreprocessor(weights=args.weights, device=args.device)

    for seq_dir in seq_dirs:
        out_dir = Path(args.out) / args.activity / seq_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            calib = parse_colmap_calibration(seq_dir)
            cams = [c for c in cams_keep if c in calib]
            if len(cams) < 2:
                print(f"SKIP {seq_dir.name}: <2 keep-cams calibrated ({cams})")
                continue
            print(process_scene(seq_dir, out_dir, cams, calib,
                                args.stride, args.max_frames, preprocessor, args.device))
        except Exception as e:
            print(f"FAIL {seq_dir.name}: {e}")

    del preprocessor
    torch.cuda.empty_cache()
    print("Probe done.")


if __name__ == "__main__":
    main()
