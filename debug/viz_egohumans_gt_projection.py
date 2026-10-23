#!/usr/bin/env python
"""Debug the EgoHumans GT-driven ReID projections on ONE frame of ONE camera.

Writes a 3-panel image:

  1. RAW fisheye frame + GT SMPL mesh, projected the way
     ``utilities/render_egohumans_gt.py`` does it (COLMAP OPENCV_FISHEYE, aria01
     anchor). Checks the 3D GT + calibration on their own.
  2. UNDISTORTED 1440 frame (what SAM3/SAM3D ran on) + the same mesh carried into
     that space with the production ``gt_points_to_undistorted`` (K_new_r from
     ``undistorted_intrinsics``), plus the GT ``poses2d`` keypoints the production
     matcher actually uses, mapped the same way. Checks the undistort/resize mapping.
  3. UNDISTORTED frame + SAM3 masks (one colour + label per track id) and each
     track's SAM3D ``pred_keypoints_2d`` centroid — the point the matcher compares
     to the GT centroid.

GT colours: aria01 red, aria02 green, aria03 blue, aria04 yellow.

Helpers are copied (not imported) from scripts/egohumans_pipeline_production.py and
utilities/render_egohumans_gt.py so this reproduces their exact maths.

    pixi run python debug/viz_egohumans_gt_projection.py \\
        --raw-seq  .../egohumans_raw/<INNER>/01_tagging/001_tagging \\
        --und-seq  .../temp_egohumans/01_tagging/<INNER>/01_tagging/001_tagging \\
        --ghost-seq .../ghost_outputs/egohumans_prod/01_tagging/001_tagging \\
        --cam cam06 --out logs/debug_viz/001_tagging_cam06.jpg
"""

import argparse
import io
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np

ARIA_BGR = {"aria01": (0, 0, 255), "aria02": (0, 200, 0),
            "aria03": (255, 80, 0), "aria04": (0, 220, 255)}
TRACK_BGR = [(255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
             (0, 255, 128), (255, 128, 128), (128, 255, 255), (200, 200, 200)]


# --- copied from utilities/render_egohumans_gt.py ---------------------------
def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _parse_cameras(cameras_txt: Path) -> dict[int, dict]:
    cam: dict[int, dict] = {}
    for line in cameras_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        fx, fy, cx, cy = (float(v) for v in p[4:8])
        cam[int(p[0])] = {
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
            "D": np.array([[float(v)] for v in p[8:12]], dtype=np.float64),
            "W": int(p[2]), "H": int(p[3]),
        }
    return cam


def _parse_images_for_cam(images_txt: Path, cam_name: str) -> tuple[int, dict]:
    camera_id, poses = -1, {}
    for line in images_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 10 or p[9].split("/")[0] != cam_name:
            continue
        R = _quat_wxyz_to_R(np.array([float(v) for v in p[1:5]]))
        poses[Path(p[9]).name] = (R, np.array([float(v) for v in p[5:8]]))
        camera_id = int(p[8])
    return camera_id, poses


# --- copied from scripts/egohumans_pipeline_production.py ---------------------
def undistorted_intrinsics(calib: dict, max_side: int):
    K, D, W, H = calib["K"], calib["D"], calib["W"], calib["H"]
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0)
    s = max_side / max(W, H) if max_side else 1.0
    return np.diag([s, s, 1.0]) @ K_new, int(round(W * s)), int(round(H * s))


def gt_points_to_undistorted(pts, calib, K_new_r):
    # Contiguous copy: a strided view (e.g. kps[:, :2]) is misread by OpenCV — only the
    # first point comes out right, the rest is garbage.
    p = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(p, calib["K"], calib["D"], R=np.eye(3), P=K_new_r).reshape(-1, 2)


# -----------------------------------------------------------------------------
def project_raw(Xw, R, t, calib):
    """World points -> raw fisheye pixels (all points in front of the camera)."""
    Xc = Xw @ R.T + t
    Xc = Xc[Xc[:, 2] > 1e-3]
    if not len(Xc):
        return np.empty((0, 2))
    uv, _ = cv2.fisheye.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                      calib["K"], calib["D"])
    return uv.reshape(-1, 2)


def draw_points(img, uv, color, step=1, r=1):
    H, W = img.shape[:2]
    for u, v in uv[::step]:
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(img, (int(u), int(v)), r, color, -1)


def label(img, text, uv, color):
    if len(uv):
        u, v = np.median(uv, 0)
        cv2.putText(img, text, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-seq", type=Path, required=True)
    ap.add_argument("--und-seq", type=Path, required=True)
    ap.add_argument("--ghost-seq", type=Path, required=True)
    ap.add_argument("--cam", default="cam06")
    ap.add_argument("--frame", type=int, default=None, help="image stem; default = middle masked frame")
    ap.add_argument("--max-side", type=int, default=1440)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cam_dir = a.ghost_seq / a.cam
    zf = zipfile.ZipFile(cam_dir / "mask_data.npz")
    stems = sorted(int(n.split("_")[1].split(".")[0]) for n in zf.namelist())
    frame = a.frame if a.frame is not None else stems[len(stems) // 2]
    fname = f"{frame:05d}"

    wk = a.raw_seq / "colmap" / "workplace"
    cams = _parse_cameras(wk / "cameras.txt")
    cam_id, poses = _parse_images_for_cam(wk / "images.txt", a.cam)
    calib = cams[cam_id]
    R, t = poses.get(f"{fname}.jpg") or next(iter(poses.values()))
    T = np.asarray(pickle.load(open(wk / "colmap_from_aria_transforms.pkl", "rb"))["aria01"], np.float64)
    K_new_r, W_r, H_r = undistorted_intrinsics(calib, a.max_side)

    raw = cv2.imread(str(a.raw_seq / "exo" / a.cam / "images" / f"{fname}.jpg"))
    und = cv2.imread(str(a.und_seq / "exo" / a.cam / "images_undistorted" / "frames" / f"{fname}.jpg"))
    print(f"frame {fname}  raw {raw.shape[1]}x{raw.shape[0]}  calib {calib['W']}x{calib['H']}  "
          f"undistorted {und.shape[1]}x{und.shape[0]}  K_new_r size {W_r}x{H_r}")

    # Panel 1 + 2: GT SMPL mesh
    p1, p2 = raw.copy(), und.copy()
    smpl = np.load(a.raw_seq / "processed_data" / "smpl" / f"{fname}.npy", allow_pickle=True).item()
    for name, prm in smpl.items():
        if not isinstance(prm, dict) or "vertices" not in prm:
            continue
        Xw = np.asarray(prm["vertices"], np.float64) @ T[:3, :3].T + T[:3, 3]
        uv_raw = project_raw(Xw, R, t, calib)
        col = ARIA_BGR.get(name, (255, 255, 255))
        draw_points(p1, uv_raw, col, step=3, r=2)
        label(p1, name, uv_raw, col)
        uv_und = gt_points_to_undistorted(uv_raw, calib, K_new_r) if len(uv_raw) else uv_raw
        draw_points(p2, uv_und, col, step=3, r=1)
        label(p2, f"{name} mesh", uv_und, col)

    # Panel 2: GT poses2d keypoints (what the matcher uses)
    kp_path = a.raw_seq / "processed_data" / "poses2d" / a.cam / "rgb" / f"{fname}.npy"
    if kp_path.exists():
        for person in np.load(kp_path, allow_pickle=True):
            if not isinstance(person, dict) or "keypoints" not in person:
                continue
            k = np.asarray(person["keypoints"], np.float64)
            k = k[k[:, 2] > 0.3][:, :2] if k.shape[1] > 2 else k[:, :2]
            if not len(k):
                continue
            col = ARIA_BGR.get(str(person["human_name"]), (255, 255, 255))
            ku = gt_points_to_undistorted(k, calib, K_new_r)
            for u, v in ku:
                cv2.drawMarker(p2, (int(u), int(v)), col, cv2.MARKER_TILTED_CROSS, 12, 2)
            c = ku.mean(0)
            cv2.circle(p2, (int(c[0]), int(c[1])), 9, col, 3)
            label(p2, f"{person['human_name']} kp2d", ku + [0, 25], col)
            draw_points(p1, k, col, r=6)
    else:
        print(f"no poses2d at {kp_path}")

    # Panel 3: SAM3 masks + SAM3D kp2d centroids
    p3 = und.copy()
    with zf.open(f"mask_{fname}.npy") as f:
        mask = np.load(io.BytesIO(f.read()))
    print(f"mask shape {mask.shape}  ids {sorted(int(i) for i in np.unique(mask) if i)}")
    overlay = p3.copy()
    for i, tid in enumerate(int(x) for x in np.unique(mask) if x):
        col = TRACK_BGR[i % len(TRACK_BGR)]
        m = mask == tid
        overlay[m] = col
        ys, xs = np.nonzero(m)
        label(p3, f"track {tid}", np.stack([xs, ys], 1), col)
        bd = cam_dir / "body_data" / f"person_{tid}.npz"
        if bd.exists():
            d = np.load(bd)
            j = np.where(d["frame_indices"].astype(int) == frame)[0]
            if j.size:
                k = np.asarray(d["pred_keypoints_2d"][j[0]], np.float64)[..., :2]
                k = k[np.isfinite(k).all(-1)]
                c = k.mean(0)
                cv2.drawMarker(p3, (int(c[0]), int(c[1])), (255, 255, 255), cv2.MARKER_CROSS, 30, 4)
                cv2.drawMarker(p3, (int(c[0]), int(c[1])), col, cv2.MARKER_CROSS, 30, 2)
    p3 = cv2.addWeighted(overlay, 0.45, p3, 0.55, 0)

    # Assemble: raw panel resized to the undistorted height
    h = p2.shape[0]
    p1 = cv2.resize(p1, (int(p1.shape[1] * h / p1.shape[0]), h))
    for p, txt in ((p1, "1 RAW fisheye + GT mesh (+ poses2d dots)"),
                   (p2, "2 UNDISTORTED 1440 + GT mesh + GT poses2d (matcher input)"),
                   (p3, "3 UNDISTORTED 1440 + SAM3 masks + SAM3D kp2d centroids")):
        cv2.putText(p, f"{txt}  frame {fname} {a.cam}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), np.concatenate([p1, p2, p3], 1), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
