"""
Compare SAM3D per-camera predictions against GT body parameters in each camera's frame.

For each scene in rich10_segmentation_test, for each camera:
  - Translation : pred_cam_t  vs  R_rel @ transl_world + t_rel
  - Rotation    : smplx_global_orient (pred, cam-i frame)
                  vs  R_rel @ R_gt_world  (GT global_orient rotated to cam-i frame)
  - Focal length: focal_length (pred, px)  vs  GT focal from calibration XML (px)
                  Note: SAM3D focal is in crop-image pixels; GT focal is for the full image.
                  The ratio pred/gt is reported alongside absolute values.

Frame matching: body_data frame_indices map local SAM3D frame numbers to RICH global
frame numbers (matching the train_body/{frame:05d}/ directory names).

Usage:
    pixi run python scripts/compare_pred_cam_t.py
"""

from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

# ── Paths ──────────────────────────────────────────────────────────────────────
SCENE_ROOT = Path(
    "/cluster/project/cvg/students/tnanni/ghost/test_outputs/rich10_segmentation_test"
)
RICH_ROOT = Path(
    "/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _scene_stem(name: str) -> str:
    """'BBQ_001_guitar' -> 'BBQ', 'LectureHall_018_wipingchairs1' -> 'LectureHall'."""
    import re
    m = re.match(r'^(.+?)_\d{3}_', name)
    return m.group(1) if m else name


def _parse_calib_xml(xml_path: Path) -> dict:
    """Parse OpenCV-format calibration XML. Returns extrinsics (3,4) and intrinsics (3,3)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def _parse(tag):
        node = root.find(tag)
        if node is None:
            return None
        rows = int(node.findtext("rows", default="1"))
        cols = int(node.findtext("cols", default="1"))
        data = list(map(float, node.findtext("data", default="").split()))
        return np.array(data, dtype=np.float64).reshape(rows, cols)

    return {"extrinsics": _parse("CameraMatrix"), "intrinsics": _parse("Intrinsics")}


def _geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle error in degrees between two (3,3) rotation matrices."""
    R_rel = R_gt.T @ R_pred
    cos = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def load_gt_cameras(scene_dir: Path) -> list[dict | None]:
    """
    Load GT camera extrinsics from RICH calibration XMLs, expressed relative to cam-0.

    Returns one dict per cam_dir:
      'R_rel': (3,3)  rotation   cam-0 → cam-i  (i.e. p_i = R_rel @ p_0 + t_rel)
      't_rel': (3,)   translation cam-0 → cam-i
      'focal': float  GT focal length in full-image pixels
    Camera-0 has identity R_rel and zero t_rel.
    """
    stem = _scene_stem(scene_dir.name)
    calib_dir = RICH_ROOT / "scan_calibration" / stem / "calibration"
    cam_dirs = sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam_"))

    raw = []
    for i, _ in enumerate(cam_dirs):
        xml_path = calib_dir / f"{i:03d}.xml"
        if not xml_path.exists():
            raw.append(None)
            continue
        calib = _parse_calib_xml(xml_path)
        ext = calib["extrinsics"]
        intr = calib["intrinsics"]
        raw.append({
            "R": ext[:3, :3],
            "t": ext[:3, 3],
            "focal": float(intr[0, 0]) if intr is not None else 1000.0,
        })

    if not raw or raw[0] is None:
        return []

    R0, t0 = raw[0]["R"], raw[0]["t"]
    result = []
    for v in raw:
        if v is None:
            result.append(None)
            continue
        R_rel = v["R"] @ R0.T       # cam-0 → cam-i rotation
        t_rel = v["t"] - R_rel @ t0 # cam-0 → cam-i translation
        result.append({"R_rel": R_rel, "t_rel": t_rel, "focal": v["focal"]})
    return result


def load_gt_body(scene_dir: Path) -> dict[int, dict[int, dict]]:
    """
    Load GT body parameters from RICH train_body pkl files.

    Returns:
        gt[person_id][frame_idx] = {
            'transl':        (3,)  body root translation in world/cam-0 space (m)
            'global_orient': (3,)  body root orientation axis-angle in world/cam-0 space
        }
    """
    gt: dict[int, dict[int, dict]] = {}
    gt_root = RICH_ROOT / "train_body" / scene_dir.name
    if not gt_root.is_dir():
        return gt
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            person_id = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            entry = {
                "transl":        np.asarray(data["transl"],        dtype=np.float64).squeeze(),
                "global_orient": np.asarray(data["global_orient"], dtype=np.float64).squeeze(),
            }
            gt.setdefault(person_id, {})[frame_idx] = entry
    return gt


def load_pred(cam_dir: Path) -> dict[int, dict]:
    """
    Load SAM3D predictions from body_data/*.npz.

    Returns:
        {person_id: {
            'frame_indices':     (T,)   RICH global frame numbers
            'pred_cam_t':        (T, 3) body root translation in cam space (m)
            'global_orient':     (T, 3) body root orientation axis-angle in cam space
            'focal_length':      (T,)   predicted focal length (px, crop-image scale)
        }}
    """
    body_dir = cam_dir / "body_data"
    result = {}
    for npz_path in sorted(body_dir.glob("person_*.npz")):
        pid = int(npz_path.stem.split("_")[1])
        d = np.load(npz_path)
        result[pid] = {
            "frame_indices":  d["frame_indices"].astype(int),
            "pred_cam_t":     d["pred_cam_t"].astype(np.float64),
            "global_orient":  d["smplx_global_orient"].astype(np.float64),  # (T, 3) axis-angle
            "focal_length":   d["focal_length"].astype(np.float64),          # (T,)
        }
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    scenes = sorted(d for d in SCENE_ROOT.iterdir() if d.is_dir())

    hdr = (
        f"{'Scene':<38} {'Cam':>3}  {'N':>5}  "
        f"{'Med-TE(m)':>10}  {'Med-RE(°)':>10}  "
        f"{'Med-f(px)':>10}  {'GT-f(px)':>9}  {'f-ratio':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    all_te: list[float] = []
    all_re: list[float] = []
    all_f_ratio: list[float] = []

    for scene_dir in scenes:
        gt_cameras = load_gt_cameras(scene_dir)
        gt_body    = load_gt_body(scene_dir)
        cam_dirs   = sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam_"))

        if not gt_cameras or not gt_body:
            print(f"{scene_dir.name:<38}  (missing GT cameras or body annotations)")
            continue

        scene_te: list[float] = []
        scene_re: list[float] = []
        first_line = True

        for cam_idx, cam_dir in enumerate(cam_dirs):
            if cam_idx >= len(gt_cameras) or gt_cameras[cam_idx] is None:
                continue
            if not (cam_dir / "body_data").is_dir():
                continue

            cam_gt = gt_cameras[cam_idx]
            R_rel  = cam_gt["R_rel"]
            t_rel  = cam_gt["t_rel"]
            gt_focal = cam_gt["focal"]

            pred_data = load_pred(cam_dir)
            if not pred_data:
                continue

            cam_te: list[float] = []
            cam_re: list[float] = []
            cam_f:  list[float] = []

            gt_persons = list(gt_body.keys())

            for pid, pdata in pred_data.items():
                gt_pid   = min(gt_persons, key=lambda p: abs(p - pid))
                gt_frames = gt_body[gt_pid]

                for local_t, gf in enumerate(pdata["frame_indices"]):
                    if gf not in gt_frames:
                        continue
                    gt_f = gt_frames[gf]

                    # ── Translation ──────────────────────────────────────────
                    transl_world = gt_f["transl"]
                    gt_t_cam = R_rel @ transl_world + t_rel
                    te = float(np.linalg.norm(pdata["pred_cam_t"][local_t] - gt_t_cam))
                    cam_te.append(te)

                    # ── Rotation ─────────────────────────────────────────────
                    # GT global_orient is in world/cam-0 space; rotate to cam-i.
                    R_gt_world = SciR.from_rotvec(gt_f["global_orient"]).as_matrix()
                    R_gt_cam   = R_rel @ R_gt_world
                    R_pred_cam = SciR.from_rotvec(pdata["global_orient"][local_t]).as_matrix()
                    re = _geodesic_deg(R_pred_cam, R_gt_cam)
                    cam_re.append(re)

                    # ── Focal length ─────────────────────────────────────────
                    cam_f.append(float(pdata["focal_length"][local_t]))

            if not cam_te:
                continue

            scene_te.extend(cam_te)
            scene_re.extend(cam_re)
            all_te.extend(cam_te)
            all_re.extend(cam_re)
            all_f_ratio.extend(f / gt_focal for f in cam_f)

            med_f    = float(np.median(cam_f))
            f_ratio  = med_f / gt_focal

            sname = scene_dir.name if first_line else ""
            first_line = False
            print(
                f"{sname:<38} {cam_idx:>3}  {len(cam_te):>5}  "
                f"{np.median(cam_te):>10.4f}  {np.median(cam_re):>10.2f}  "
                f"{med_f:>10.1f}  {gt_focal:>9.1f}  {f_ratio:>8.3f}"
            )

        if scene_te:
            print(
                f"  → scene: N={len(scene_te)}  "
                f"median TE={np.median(scene_te):.4f}m  "
                f"median RE={np.median(scene_re):.2f}°"
            )
        print()

    if all_te:
        print("=" * len(hdr))
        print(f"OVERALL  N={len(all_te)} frame-camera pairs")
        print(f"  Median TE      : {np.median(all_te):.4f} m")
        print(f"  Mean   TE      : {np.mean(all_te):.4f} m")
        print(f"  Median RE      : {np.median(all_re):.2f}°")
        print(f"  Mean   RE      : {np.mean(all_re):.2f}°")
        print(f"  Median f ratio : {np.median(all_f_ratio):.3f}  (pred / GT full-image focal)")
        errs = np.array(all_te)
        for thr in (0.1, 0.2, 0.5, 1.0):
            print(f"  TE < {thr:.1f}m      : {(errs < thr).mean() * 100:.1f}%")


if __name__ == "__main__":
    run()
