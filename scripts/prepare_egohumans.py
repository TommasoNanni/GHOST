"""One-time preprocessing script for the EgoHumans dataset.

For each sequence archive found under `data_root`:
  1. Extracts the tar.gz into the same directory (skips if already done).
  2. For every exo camera (cam01–cam08): undistorts frames using the
     OPENCV_FISHEYE calibration from COLMAP's cameras.txt, and writes
     undistorted JPEGs to  exo/<cam>/images_undistorted/.
  3. For every ego (Aria) camera (aria01–ariaN): undistorts frames using
     the radtanthinprism model stored in ego/<aria>/calib/<frame>.txt,
     and writes undistorted JPEGs to ego/<aria>/images_undistorted/.
     Per-frame world-to-cam extrinsics are saved as
     aria_extrinsics/<aria>.npz  (shape N×3×4) because Aria moves.
  4. Converts SMPL body parameters (processed_data/smpl/*.npy) to SMPL-X
     by copying body-joint rotations and correcting the pelvis translation
     for the shape-dependent joint-regressor difference between the two
     models.  Results are written to processed_data/smplx/*.npy.
  5. Saves a calibration.json alongside each extracted sequence with
     pinhole (zero-distortion) intrinsics for all cameras.  Exo cameras
     include a single R, t; Aria cameras reference their extrinsics file.

Usage
-----
    python scripts/prepare_egohumans.py \\
        --data_root /capstor/scratch/cscs/tnanni/datasets/egohumans \\
        --smpl_model  /users/tnanni/ghost/body_models/SMPL_NEUTRAL.npz \\
        --smplx_model /users/tnanni/ghost/body_models/SMPLX_NEUTRAL.npz \\
        [--jobs 8]              # parallel workers for frame undistortion
        [--balance 0.0]         # fisheye crop factor: 0=crop black, 1=keep all
        [--skip_existing]       # skip sequences already fully processed
        [--skip_smpl]           # skip SMPL->SMPL-X conversion
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# COLMAP parsing helpers  (exo cameras)
# ---------------------------------------------------------------------------

def parse_cameras_txt(cameras_txt: Path) -> dict[int, dict]:
    """Parse COLMAP cameras.txt → {camera_id: {K, width, height, dist_coeffs}}.

    Only OPENCV_FISHEYE cameras are handled (all EgoHumans exo cameras).
    The distortion vector is [k1, k2, k3, k4].
    """
    cameras: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]

            if model == "OPENCV_FISHEYE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, k3, k4 = params[4], params[5], params[6], params[7]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                D = np.array([k1, k2, k3, k4], dtype=np.float64)
            else:
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                D = np.zeros(4, dtype=np.float64)

            cameras[cam_id] = {
                "K": K, "D": D, "width": width, "height": height, "model": model
            }
    return cameras


def parse_images_txt(images_txt: Path) -> dict[str, dict]:
    """Parse COLMAP images.txt → {name_prefix: {R, t, camera_id}}.

    Returns one representative extrinsic per camera prefix (e.g. 'cam01').
    For static exo cameras the extrinsic is constant; we average rotations
    (via SVD) and translations across all frames.

    COLMAP convention: x_cam = R @ x_world + t.
    """
    from scipy.spatial.transform import Rotation as SciR

    accum: dict[str, dict] = defaultdict(lambda: {"Rs": [], "ts": [], "cam_id": None})

    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 9:
            i += 1
            continue
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz      = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id          = int(parts[8])
        name            = parts[9]
        prefix          = name.split("/")[0]

        R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        accum[prefix]["Rs"].append(R)
        accum[prefix]["ts"].append(t)
        accum[prefix]["cam_id"] = cam_id
        i += 2

    result: dict[str, dict] = {}
    for prefix, data in accum.items():
        Rs   = np.stack(data["Rs"])
        ts   = np.stack(data["ts"])
        t_mean = ts.mean(axis=0)
        R_mean = Rs.mean(axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        R_rep = U @ Vt
        if np.linalg.det(R_rep) < 0:
            U[:, -1] *= -1
            R_rep = U @ Vt
        result[prefix] = {
            "R": R_rep,
            "t": t_mean,
            "camera_id": data["cam_id"],
        }
    return result


# ---------------------------------------------------------------------------
# Aria (ego) camera parsing  — radtanthinprism model
# ---------------------------------------------------------------------------

# The per-frame calib .txt layout (after the 2-line header):
#   <intrinsics: 15 floats>   for camera 0 (RGB, 1408×1408)
#   <extrinsic:  12 floats>   3×4 world-to-cam, row-major
#   <intrinsics: 15 floats>   for camera 1 (SLAM left)
#   <extrinsic:  12 floats>
#   <intrinsics: 15 floats>   for camera 2 (SLAM right)
#   <extrinsic:  12 floats>
#
# Intrinsics params: [f, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3]
# The RGB camera (index 0) has the largest focal length / cx.

_RGB_CAM_IDX = 0  # camera 0 in the per-frame calib file is Aria RGB


def parse_aria_calib(calib_txt: Path) -> dict:
    """Parse one per-frame Aria calib .txt.

    Returns a dict for the RGB camera:
        {intrinsics: (15,), extrinsic: (3,4), width: int, height: int}
    """
    with open(calib_txt) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Lines 0-1: header + serial.  Then pairs (intrinsics, extrinsic) per cam.
    cam_lines = lines[2:]  # skip header rows

    idx = _RGB_CAM_IDX
    intr_vals = [float(v) for v in cam_lines[idx * 2].split()]
    ext_vals  = [float(v) for v in cam_lines[idx * 2 + 1].split()]

    intr = np.array(intr_vals, dtype=np.float64)   # (15,)
    ext  = np.array(ext_vals,  dtype=np.float64).reshape(3, 4)

    # Aria RGB resolution is 1408×1408
    f, cx, cy = intr[0], intr[1], intr[2]
    # Round up resolution from cx/cy (≈ half-width/half-height)
    w = round(cx * 2)
    h = round(cy * 2)

    return {"intrinsics": intr, "extrinsic": ext, "width": w, "height": h}


def build_undistort_maps_radtanthinprism(
    intr: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cv2.remap maps for the radtanthinprism distortion model.

    For each destination pixel (u, v) in the undistorted image we compute
    its normalised undistorted position, apply the forward distortion model
    to obtain the distorted normalised position, then convert back to pixel
    coords in the original image.  The same K is used for both images
    (no field-of-view change).

    Parameters
    ----------
    intr : (15,) array  [f, cx, cy, k0..k5, p0, p1, s0..s3]

    Returns
    -------
    map1, map2 : float32 remap arrays for cv2.remap
    K_new      : (3,3) pinhole intrinsics of the undistorted image
    """
    f, cx, cy = intr[0], intr[1], intr[2]
    k  = intr[3:9]    # k0..k5 (radial)
    p0, p1 = intr[9], intr[10]
    s  = intr[11:15]  # s0..s3 (thin prism)

    us, vs = np.meshgrid(
        np.arange(width,  dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )

    # Normalised coords (no distortion) for each destination pixel
    xn = (us - cx) / f
    yn = (vs - cy) / f

    # Forward radtanthinprism distortion
    r2  = xn**2 + yn**2
    r4  = r2**2
    r6  = r2**3
    r8  = r4**2
    r10 = r4 * r6
    r12 = r6**2

    radial = 1.0 + k[0]*r2 + k[1]*r4 + k[2]*r6 + k[3]*r8 + k[4]*r10 + k[5]*r12

    xt = 2*p0*xn*yn + p1*(r2 + 2*xn**2)
    yt = p0*(r2 + 2*yn**2) + 2*p1*xn*yn

    xs = s[0]*r2 + s[1]*r4
    ys = s[2]*r2 + s[3]*r4

    xd = radial*xn + xt + xs
    yd = radial*yn + yt + ys

    # Source pixel coords in the distorted image
    map1 = (f * xd + cx).astype(np.float32)
    map2 = (f * yd + cy).astype(np.float32)

    K_new = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    return map1, map2, K_new


# ---------------------------------------------------------------------------
# Generic frame undistortion
# ---------------------------------------------------------------------------

def build_undistort_maps_fisheye(
    K: np.ndarray,
    D: np.ndarray,
    width: int,
    height: int,
    balance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (width, height), cv2.CV_16SC2
    )
    return map1, map2, K_new


def undistort_frame(
    src_path: Path,
    dst_path: Path,
    map1: np.ndarray,
    map2: np.ndarray,
) -> None:
    img = cv2.imread(str(src_path))
    if img is None:
        return
    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), undist, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ---------------------------------------------------------------------------
# Exo camera processing
# ---------------------------------------------------------------------------

EXO_PREFIX = "cam"


def process_exo_cameras(
    seq_dir: Path,
    balance: float,
    skip_existing: bool,
    jobs: int,
) -> dict[str, dict]:
    """Undistort exo camera frames and return calibration entries."""
    colmap_dir  = seq_dir / "colmap" / "workplace"
    cameras_txt = colmap_dir / "cameras.txt"
    images_txt  = colmap_dir / "images.txt"

    if not cameras_txt.exists() or not images_txt.exists():
        print(f"  WARN: no COLMAP calibration found in {seq_dir.name}")
        return {}

    colmap_cams = parse_cameras_txt(cameras_txt)
    colmap_imgs = parse_images_txt(images_txt)

    calibration: dict[str, dict] = {}

    for prefix, ext_data in colmap_imgs.items():
        if not prefix.startswith(EXO_PREFIX):
            continue

        cam_id = ext_data["camera_id"]
        if cam_id not in colmap_cams:
            print(f"  WARN {prefix}: camera_id {cam_id} not in cameras.txt")
            continue

        cam_info = colmap_cams[cam_id]
        K, D = cam_info["K"], cam_info["D"]
        W, H = cam_info["width"], cam_info["height"]

        map1, map2, K_new = build_undistort_maps_fisheye(K, D, W, H, balance)

        src_dir = seq_dir / "exo" / prefix / "images"
        dst_dir = seq_dir / "exo" / prefix / "images_undistorted"

        if src_dir.is_dir():
            src_frames = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
            if src_frames:
                if skip_existing and dst_dir.is_dir() and len(list(dst_dir.glob("*.jpg"))) == len(src_frames):
                    print(f"  SKIP undistortion for {prefix} (already done)")
                else:
                    print(f"  Undistorting {prefix}: {len(src_frames)} frames …")
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    def _worker(src: Path) -> None:
                        undistort_frame(src, dst_dir / src.name, map1, map2)

                    with ThreadPoolExecutor(max_workers=jobs) as pool:
                        list(pool.map(_worker, src_frames))

        R = ext_data["R"]
        t = ext_data["t"]
        calibration[prefix] = {
            "K":         K_new.tolist(),
            "R":         R.tolist(),
            "t":         t.tolist(),
            "width":     W,
            "height":    H,
            "image_dir": f"exo/{prefix}/images_undistorted",
        }

    return calibration


# ---------------------------------------------------------------------------
# Aria (ego) camera processing
# ---------------------------------------------------------------------------

EGO_PREFIX = "aria"


def process_aria_cameras(
    seq_dir: Path,
    skip_existing: bool,
    jobs: int,
) -> dict[str, dict]:
    """Undistort Aria ego-camera frames and return calibration entries.

    Because Aria is a moving camera, per-frame world-to-cam extrinsics are
    stored separately in  aria_extrinsics/<aria_id>.npz  (array 'extrinsics'
    of shape N×3×4, float64) rather than in the calibration.json directly.
    calibration.json carries only the (constant) pinhole intrinsics.
    """
    ego_root = seq_dir / "ego"
    if not ego_root.is_dir():
        return {}

    calibration: dict[str, dict] = {}
    ext_out_dir = seq_dir / "aria_extrinsics"
    ext_out_dir.mkdir(exist_ok=True)

    for aria_dir in sorted(ego_root.iterdir()):
        if not aria_dir.is_dir() or not aria_dir.name.startswith(EGO_PREFIX):
            continue

        aria_id  = aria_dir.name        # e.g. "aria01"
        calib_dir = aria_dir / "calib"
        src_dir   = aria_dir / "images"
        dst_dir   = aria_dir / "images_undistorted"

        if not calib_dir.is_dir() or not src_dir.is_dir():
            print(f"  WARN {aria_id}: missing calib/ or images/ folder")
            continue

        src_frames = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        if not src_frames:
            print(f"  WARN {aria_id}: no source frames found")
            continue

        if skip_existing and dst_dir.is_dir() and len(list(dst_dir.glob("*.jpg"))) == len(src_frames):
            # Still load extrinsics to populate calibration entry if needed
            ext_path = ext_out_dir / f"{aria_id}.npz"
            if ext_path.exists():
                print(f"  SKIP {aria_id} undistortion (already done)")
                data = parse_aria_calib(calib_dir / src_frames[0].with_suffix(".txt").name)
                f, cx, cy = data["intrinsics"][0], data["intrinsics"][1], data["intrinsics"][2]
                K_new = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
                calibration[aria_id] = {
                    "K":               K_new.tolist(),
                    "width":           data["width"],
                    "height":          data["height"],
                    "image_dir":       f"ego/{aria_id}/images_undistorted",
                    "extrinsics_path": f"aria_extrinsics/{aria_id}.npz",
                }
                continue

        # ── Parse intrinsics from the first frame (constant across frames) ──
        first_calib_name = src_frames[0].stem + ".txt"
        first_calib_path = calib_dir / first_calib_name
        if not first_calib_path.exists():
            print(f"  WARN {aria_id}: calib file not found: {first_calib_path.name}")
            continue

        first_data = parse_aria_calib(first_calib_path)
        intr = first_data["intrinsics"]
        W, H = first_data["width"], first_data["height"]
        map1, map2, K_new = build_undistort_maps_radtanthinprism(intr, W, H)

        # ── Undistort frames ──────────────────────────────────────────────
        print(f"  Undistorting {aria_id}: {len(src_frames)} frames …")
        dst_dir.mkdir(parents=True, exist_ok=True)

        def _worker(src: Path) -> None:
            undistort_frame(src, dst_dir / src.name, map1, map2)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(_worker, src_frames))

        # ── Collect per-frame extrinsics ──────────────────────────────────
        # We read all per-frame calib files; the extrinsic changes each frame.
        extrinsics_list = []
        for src in src_frames:
            calib_path = calib_dir / (src.stem + ".txt")
            if calib_path.exists():
                frame_data = parse_aria_calib(calib_path)
                extrinsics_list.append(frame_data["extrinsic"])
            else:
                # Fallback: repeat previous extrinsic if calib is missing
                if extrinsics_list:
                    extrinsics_list.append(extrinsics_list[-1])
                else:
                    extrinsics_list.append(np.zeros((3, 4), dtype=np.float64))

        extrinsics_arr = np.stack(extrinsics_list, axis=0)  # (N, 3, 4)
        ext_path = ext_out_dir / f"{aria_id}.npz"
        np.savez_compressed(str(ext_path), extrinsics=extrinsics_arr)
        print(f"  Saved {ext_path.name}: {extrinsics_arr.shape}")

        calibration[aria_id] = {
            "K":               K_new.tolist(),
            "width":           W,
            "height":          H,
            "image_dir":       f"ego/{aria_id}/images_undistorted",
            "extrinsics_path": f"aria_extrinsics/{aria_id}.npz",
        }

    return calibration


# ---------------------------------------------------------------------------
# SMPL → SMPL-X conversion
# ---------------------------------------------------------------------------

def _compute_pelvis(model: dict, betas: np.ndarray) -> np.ndarray:
    """Return the pelvis joint (joint 0) position in the shaped T-pose.

    Uses only the first len(betas) shape components.
    """
    v_template = model["v_template"]      # (V, 3)
    shapedirs  = model["shapedirs"]       # (V, 3, num_betas_max)
    J_reg      = model["J_regressor"]     # (num_joints, V)

    nb = min(len(betas), shapedirs.shape[2])
    v_shaped = v_template + np.einsum("ijk,k->ij", shapedirs[:, :, :nb], betas[:nb])
    J = J_reg @ v_shaped                  # (num_joints, 3)
    return J[0]                           # pelvis


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector (3,) → rotation matrix (3,3)."""
    angle = np.linalg.norm(rvec)
    if angle < 1e-8:
        return np.eye(3, dtype=rvec.dtype)
    axis = rvec / angle
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def convert_smpl_to_smplx_params(
    smpl_data: dict,
    smpl_model: dict,
    smplx_model: dict,
) -> dict:
    """Convert a single person's SMPL parameters to SMPL-X parameters.

    Body joints 1–21 are identical in both skeletons, so body_pose is
    copied directly (first 63 of SMPL's 69 entries; the last 6 are SMPL's
    hand endpoint joints, which SMPL-X handles separately).

    The pelvis translation is corrected for the shape-dependent offset
    between the two joint regressors:
        transl_smplx = transl_smpl + R_global @ (pelvis_smpl - pelvis_smplx)

    Hands, jaw, and expression are initialised to zero (rest pose).
    """
    betas        = np.asarray(smpl_data["betas"],        dtype=np.float64)
    global_orient = np.asarray(smpl_data["global_orient"], dtype=np.float64)
    body_pose    = np.asarray(smpl_data["body_pose"],    dtype=np.float64)
    transl       = np.asarray(smpl_data["transl"],       dtype=np.float64)

    pelvis_smpl  = _compute_pelvis(smpl_model,  betas)
    pelvis_smplx = _compute_pelvis(smplx_model, betas)

    R_global   = _rodrigues(global_orient)
    transl_out = transl + R_global @ (pelvis_smpl - pelvis_smplx)

    return {
        "global_orient":    global_orient.astype(np.float32),
        "body_pose":        body_pose[:63].astype(np.float32),  # joints 1–21
        "betas":            betas[:10].astype(np.float32),
        "transl":           transl_out.astype(np.float32),
        "left_hand_pose":   np.zeros(45, dtype=np.float32),
        "right_hand_pose":  np.zeros(45, dtype=np.float32),
        "jaw_pose":         np.zeros(3,  dtype=np.float32),
        "leye_pose":        np.zeros(3,  dtype=np.float32),
        "reye_pose":        np.zeros(3,  dtype=np.float32),
        "expression":       np.zeros(10, dtype=np.float32),
    }


def process_smpl_conversion(
    seq_dir: Path,
    smpl_model: dict,
    smplx_model: dict,
    skip_existing: bool,
) -> str:
    """Convert all per-frame SMPL .npy files to SMPL-X and write to smplx/."""
    smpl_dir  = seq_dir / "processed_data" / "smpl"
    smplx_dir = seq_dir / "processed_data" / "smplx"

    if not smpl_dir.is_dir():
        return "SKIP smpl_conversion (no smpl/ dir)"

    smpl_files = sorted(smpl_dir.glob("*.npy"))
    if not smpl_files:
        return "SKIP smpl_conversion (no .npy files)"

    if skip_existing and smplx_dir.is_dir() and len(list(smplx_dir.glob("*.npy"))) == len(smpl_files):
        return f"SKIP smpl_conversion ({len(smpl_files)} files already done)"

    smplx_dir.mkdir(parents=True, exist_ok=True)

    for npy_path in smpl_files:
        frame_data = np.load(npy_path, allow_pickle=True).item()
        out = {}
        for person_id, smpl_params in frame_data.items():
            try:
                out[person_id] = convert_smpl_to_smplx_params(smpl_params, smpl_model, smplx_model)
            except Exception as e:
                print(f"  WARN {npy_path.name}/{person_id}: conversion failed: {e}")
                out[person_id] = smpl_params  # keep original on failure
        np.save(str(smplx_dir / npy_path.name), out)

    return f"OK smpl_conversion ({len(smpl_files)} frames)"


# ---------------------------------------------------------------------------
# Per-sequence top-level processing
# ---------------------------------------------------------------------------

def process_sequence(
    archive_path: Path,
    balance: float,
    skip_existing: bool,
    jobs: int,
    smpl_model: dict | None,
    smplx_model: dict | None,
    skip_smpl: bool,
) -> str:
    activity_dir = archive_path.parent
    seq_name     = archive_path.name.replace(".tar.gz", "")
    seq_dir      = activity_dir / seq_name

    calib_json = seq_dir / "calibration.json"
    if skip_existing and calib_json.exists():
        calib = json.load(open(calib_json))
        if any(k.startswith("aria") for k in calib):
            return f"SKIP  {archive_path.name}"
        # calibration.json exists but has no aria entries — re-process

    # ── 1. Extract archive ────────────────────────────────────────────────
    if not seq_dir.is_dir():
        print(f"  Extracting {archive_path.name} …")
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                parts = Path(member.name).parts
                try:
                    idx = parts.index(seq_name)
                except ValueError:
                    continue
                rel = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else None
                if rel is None:
                    continue
                member.name = str(rel)
                tf.extract(member, path=seq_dir)
    else:
        print(f"  Already extracted: {seq_name}")

    # ── 2. Exo cameras ────────────────────────────────────────────────────
    calibration = process_exo_cameras(seq_dir, balance, skip_existing, jobs)

    # ── 3. Aria ego cameras ───────────────────────────────────────────────
    aria_calib = process_aria_cameras(seq_dir, skip_existing, jobs)
    calibration.update(aria_calib)

    # ── 4. Save calibration.json ──────────────────────────────────────────
    with open(calib_json, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  Saved {calib_json.name} with {len(calibration)} cameras "
          f"({len([k for k in calibration if k.startswith('cam')])} exo, "
          f"{len([k for k in calibration if k.startswith('aria')])} aria)")

    # ── 5. SMPL → SMPL-X conversion ───────────────────────────────────────
    if not skip_smpl and smpl_model is not None and smplx_model is not None:
        smpl_status = process_smpl_conversion(seq_dir, smpl_model, smplx_model, skip_existing)
        print(f"  {smpl_status}")

    return f"OK    {seq_name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EgoHumans dataset")
    parser.add_argument("--data_root", required=True,
                        help="Root dir containing activity folders (01_tagging, 02_lego, …)")
    parser.add_argument("--smpl_model",  default=None,
                        help="Path to SMPL_NEUTRAL.npz (required for SMPL→SMPL-X conversion)")
    parser.add_argument("--smplx_model", default=None,
                        help="Path to SMPLX_NEUTRAL.npz (required for SMPL→SMPL-X conversion)")
    parser.add_argument("--balance", type=float, default=0.0,
                        help="Fisheye crop factor for exo cameras: 0=crop black, 1=keep full FoV")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Threads per sequence for frame undistortion (default: 8)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sequences whose calibration.json already exists")
    parser.add_argument("--skip_smpl", action="store_true",
                        help="Skip SMPL→SMPL-X conversion")
    parser.add_argument("--activity", default=None,
                        help="Only process a specific activity folder (e.g. 02_lego). Default: all.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # Load body models once (shared across all sequences)
    smpl_model  = None
    smplx_model = None
    if not args.skip_smpl:
        if args.smpl_model and args.smplx_model:
            print("Loading body models …")
            smpl_model  = dict(np.load(args.smpl_model,  allow_pickle=True))
            smplx_model = dict(np.load(args.smplx_model, allow_pickle=True))
            print("  SMPL  J_regressor:", smpl_model["J_regressor"].shape)
            print("  SMPL-X J_regressor:", smplx_model["J_regressor"].shape)
        else:
            print("  NOTE: --smpl_model / --smplx_model not provided; skipping SMPL conversion.")

    # Collect archives
    if args.activity:
        activity_dirs = [data_root / args.activity]
    else:
        activity_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    archives: list[Path] = []
    for act_dir in activity_dirs:
        archives += sorted(act_dir.glob("*.tar.gz"))

    if not archives:
        print("No .tar.gz archives found — nothing to do.")
        return

    print(f"Found {len(archives)} sequence archive(s) under {data_root}")

    for archive in archives:
        print(f"\n{'='*60}")
        print(f"Processing: {archive.relative_to(data_root)}")
        status = process_sequence(
            archive,
            balance=args.balance,
            skip_existing=args.skip_existing,
            jobs=args.jobs,
            smpl_model=smpl_model,
            smplx_model=smplx_model,
            skip_smpl=args.skip_smpl,
        )
        print(f"  → {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
