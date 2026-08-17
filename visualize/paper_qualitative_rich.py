"""
Paper qualitative figure for RICH — GT-mesh reprojection + a 3-D scene render.

The RICH counterpart of visualize/paper_qualitative_egohumans.py, and deliberately
a separate file: the two datasets share not one input convention, and folding them
together would mean a change made for one silently reshaping the other.

Three PNGs for one (scene, frame):

  A, B : two cameras, far apart, with the GT SMPL-X mesh reprojected onto the
         image through RICH's own calibration.  No ghost output appears here.
  C    : cam A's viewpoint on a white page — the VGGT depth of that frame as a
         coloured point cloud, the GHOST predicted body as a shaded mesh, and the
         GT body as a bare outline.

RICH is easier than EgoHumans in one decisive way: its cameras are genuinely
calibrated and its GT lives in that same multi-camera frame, so there is no
aria->COLMAP anchor and none of the anchor error that makes some EgoHumans scenes
unusable for an overlay.  Camera 000 is the reference (R = I, t = 0), which is
also the frame ``fusion_dataset`` puts GT into, so the npz GT and the XML cameras
already agree.

It is harder in another way: the images the pipeline consumed are **centered
crops of a resized frame**, so projecting through the raw calibration lands in
the wrong place unless both steps are undone:

    full 4112 x 3008  --aspect-preserving resize to an 840 long side-->  840 x 614
                      --crop by (off_x, off_y) from crop_meta.json-->    crop_w x crop_h

which is the ``rich_crop_meta_offsets_bug`` this file is careful about:

    fx_c = fx * s,  cx_c = cx * s - off_x,   s = 840 / 4112

Everything in panel C is drawn in the RICH multi-camera frame:

    GT bodies       npz gt_* (already that frame)          no transform
    GT cameras      scan_calibration XML                   no transform
    ghost + depth   ghost world  --x ma_scale, then SE(3)--> multi-cam frame

The SE(3) is fitted from the ghost camera centres onto the GT camera centres, no
scaling, matching how the RICH evaluation aligns the two.

Usage
-----
    pixi run python -m visualize.paper_qualitative_rich --rank
    pixi run python -m visualize.paper_qualitative_rich \
        --scene ParkingLot2_009_impro1 --frame 456 --cams cam_00 cam_03
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── data roots ─────────────────────────────────────────────────────────────
GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test")
RICH_ROOT  = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
IMG_ROOT   = Path("/users/tnanni/rich_centered_mnt")     # centered_test.sqsh mount

# Native RICH capture resolution; the centered frames are an aspect-preserving
# resize of this to an 840-pixel long side, then cropped (see crop_meta.json).
FULL_W, FULL_H = 4112, 3008

PERSON_RGB = {0: (220, 50, 47), 1: (60, 170, 70), 2: (40, 110, 220), 3: (230, 170, 30)}

# Triangles rasterise in fixed point: rounding each triangle's corners to whole
# pixels independently leaves the seam between two of them unfilled, and whatever
# is behind shows through as speckle.
SUBPIX = 4
_S = 1 << SUBPIX


def _person_rgb(i: int) -> tuple[int, int, int]:
    return PERSON_RGB.get(i % 4, (120, 120, 120))


def _person_rgb_dark(i: int, f: float = 0.45) -> tuple[int, int, int]:
    return tuple(int(round(c * f)) for c in _person_rgb(i))


def poly_fx(uv: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(uv) * _S).astype(np.int32)


# ── geometry (copied — this file is standalone) ────────────────────────────
def se3_align(src: np.ndarray, dst: np.ndarray):
    """Kabsch SE(3), no scale: R,t minimising ||R@src+t-dst||."""
    sc, dc = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, dc - R @ sc


def _quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def project(points_w, R, t, K):
    Xc = np.asarray(points_w, dtype=np.float64) @ R.T + t
    z = Xc[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    return np.stack([u, v], -1), z


# ── RICH calibration ───────────────────────────────────────────────────────
def _location(scene: str) -> str:
    """"ParkingLot2_009_impro1" -> "ParkingLot2"."""
    return scene.split("_")[0]


def parse_calib(scene: str, cam: str):
    """(R, t, K_full) for one camera, from scan_calibration.

    R, t map the multi-camera world into the camera; K_full is at native
    4112x3008 resolution.
    """
    num = int(cam.split("_")[1])
    xml = RICH_ROOT / "scan_calibration" / _location(scene) / "calibration" / f"{num:03d}.xml"
    root = ET.parse(str(xml)).getroot()

    def _mat(name):
        node = root.find(name)
        rows, cols = int(node.find("rows").text), int(node.find("cols").text)
        return np.fromstring(node.find("data").text.replace("\n", " "),
                             sep=" ", dtype=np.float64).reshape(rows, cols)

    ext = _mat("CameraMatrix")            # (3, 4) = [R | t]
    return ext[:, :3], ext[:, 3], _mat("Intrinsics")


def crop_meta(scene: str) -> dict:
    with open(IMG_ROOT / scene / "crop_meta.json") as f:
        return json.load(f)["cameras"]


def centered_K(K_full: np.ndarray, meta: dict) -> tuple[np.ndarray, int, int]:
    """Undo resize + crop: calibration at native resolution -> the image on disk.

    The resize preserves aspect and targets a fixed long side, so one scalar
    covers both axes; the crop then only moves the principal point.
    """
    s = max(meta["src_w"], meta["src_h"]) / max(FULL_W, FULL_H)
    K = K_full.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = K_full[0, 2] * s - meta["off_x"]
    K[1, 2] = K_full[1, 2] * s - meta["off_y"]
    return K, int(meta["crop_w"]), int(meta["crop_h"])


def frame_path(scene: str, cam: str, frame: int) -> Path:
    return IMG_ROOT / scene / cam / f"{frame:05d}_{int(cam.split('_')[1]):02d}.jpg"


# ── bodies ─────────────────────────────────────────────────────────────────
def smplx_vertices(pose, shape, trans, smplx_model_dir):
    """SMPL-X forward -> (vertices (T,P,V,3), faces (F,3)).  Copied from the viewer."""
    import torch
    import smplx as smplx_lib
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

    def _6d_to_aa(p6):
        m = rotation_6d_to_matrix(torch.from_numpy(p6.astype(np.float32)))
        return matrix_to_axis_angle(m).numpy()

    T, P, J, _ = pose.shape
    if shape.ndim == 2:
        shape = np.broadcast_to(shape[None], (T, P, 10)).copy()
    p = Path(smplx_model_dir)
    kw = {"model_type": "smplx", "model_path": str(p)}
    if p.is_file():
        kw["ext"] = p.suffix.lstrip(".")
    model = smplx_lib.create(**kw, gender="neutral", use_pca=False, num_betas=10,
                             flat_hand_mean=True, batch_size=T * P)
    model.eval()

    def _t(x):
        return torch.from_numpy(x.reshape(T * P, -1).astype(np.float32))

    with torch.no_grad():
        out = model(global_orient=_t(_6d_to_aa(pose[:, :, 0, :])),
                    body_pose=_t(_6d_to_aa(pose[:, :, 1:22, :])),
                    betas=_t(shape), transl=_t(trans), return_verts=True)
    return out.vertices.numpy().reshape(T, P, -1, 3), model.faces.copy()


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    for c in range(3):
        np.add.at(n, faces[:, c], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def face_shading(verts, faces, R, base_rgb):
    fn = vertex_normals(verts, faces)[faces].mean(1)
    lam = np.clip(-(fn @ R.T)[:, 2], 0.0, 1.0)
    return np.clip(np.asarray(base_rgb, np.float64)[None] * (0.35 + 0.65 * lam)[:, None],
                   0, 255).astype(np.uint8)


def mesh_mask(verts, faces, R, t, K, H, W, close: bool = True) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    uv, z = project(verts, R, t, K)
    ok = np.isfinite(uv).all(1) & (z > 1e-6)
    keep = ok[faces].all(1)
    if keep.any():
        cv2.fillPoly(m, poly_fx(uv[faces[keep]]), 1, shift=SUBPIX)
    if not close:
        return m
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def draw_outline(img, mask, rgb, thickness):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, tuple(int(c) for c in rgb[::-1]), thickness,
                     lineType=cv2.LINE_AA)


def paint(img, prims):
    """Painter's algorithm over balls and triangles together, far to near."""
    order = np.argsort([-p[1] for p in prims], kind="stable")
    for i in order:
        kind, _, geom, rgb = prims[i]
        col = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        if kind == "p":
            cv2.circle(img, geom[0], geom[1], col, -1, lineType=cv2.LINE_AA)
        else:
            cv2.fillConvexPoly(img, geom, col, lineType=cv2.LINE_8, shift=SUBPIX)


# ── depth ──────────────────────────────────────────────────────────────────
def depth_context(scene: str):
    d = GHOST_ROOT / scene
    cam_npz = np.load(d / "vggt_cameras_centered.npz", allow_pickle=False)
    dep_npz = np.load(d / "vggt_depth_centered.npz", allow_pickle=False)
    sc = np.load(d / "mapanything_scale_baseline.npy")
    sc = float(np.median(np.asarray(sc)[np.asarray(sc) > 0]))
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in cam_npz["camera_names"]]
    return {"names": names, "extr": cam_npz["extrinsics"], "intr": cam_npz["intrinsics"],
            "oc": cam_npz["original_coords"], "cam_valid": cam_npz["valid"],
            "depth_mm": dep_npz["depth"], "depth_conf": dep_npz["depth_conf"],
            "depth_valid": dep_npz["depth_valid"], "scale": sc}


def depth_cloud(ctx, t, k, bgr, stride, conf_thr):
    """Unproject camera k's depth into metric ghost world; colours from the frame."""
    if (t >= ctx["depth_mm"].shape[0] or not ctx["depth_valid"][t, k]
            or not ctx["cam_valid"][t, k]):
        return None
    d = ctx["depth_mm"][t, k][::stride, ::stride].astype(np.float32) / 1000.0 * ctx["scale"]
    conf = ctx["depth_conf"][t, k][::stride, ::stride].astype(np.float32)
    h_d, w_d = d.shape
    vv, uu = np.mgrid[0:h_d, 0:w_d].astype(np.float32) * stride
    x1, y1, x2, y2 = ctx["oc"][t, k]
    mask = ((d > 1e-4) & (conf >= conf_thr)
            & (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2))
    if not mask.any():
        return None
    u, v, z = uu[mask], vv[mask], d[mask]
    intr = ctx["intr"][t, k]
    pts_cam = np.stack([(u - intr[0, 2]) / intr[0, 0] * z,
                        (v - intr[1, 2]) / intr[1, 1] * z, z], -1)
    R_d = ctx["extr"][t, k, :3, :3]
    t_v = ctx["extr"][t, k, :3, 3] * ctx["scale"]
    pts = ((pts_cam - t_v) @ R_d).astype(np.float64)
    if bgr is not None:
        fh, fw = bgr.shape[:2]
        iu = np.clip(((u - x1) * fw / max(1e-6, x2 - x1)).astype(np.int32), 0, fw - 1)
        iv = np.clip(((v - y1) * fh / max(1e-6, y2 - y1)).astype(np.int32), 0, fh - 1)
        cols = bgr[iv, iu][:, ::-1]
    else:
        cols = np.full((len(z), 3), 128, np.uint8)
    return pts, cols.astype(np.uint8)


# ── prediction ─────────────────────────────────────────────────────────────
def load_pred(pred_path: Path, scene: str):
    d = np.load(pred_path, allow_pickle=False)
    fmin = None
    for f in sorted((GHOST_ROOT / scene).glob("cam_*/body_data/person_*.npz")):
        fi = np.load(str(f), allow_pickle=False)["frame_indices"].astype(int)
        if fi.size:
            fmin = int(fi.min()) if fmin is None else min(fmin, int(fi.min()))
    if fmin is None:
        raise RuntimeError(f"no body_data under {GHOST_ROOT / scene}")
    return d, fmin


def ghost_cam_centres(camera_t: np.ndarray) -> np.ndarray:
    """(K,3) metric ghost-world camera centres from a (K,8) row.

    Encoding is [quat_wxyz(4), t_w2c(3), focal(1)] with the translation already
    multiplied by the MapAnything scale, so no further scaling is needed.
    """
    out = np.full((camera_t.shape[0], 3), np.nan)
    for k in range(camera_t.shape[0]):
        R = _quat_wxyz_to_R(camera_t[k, :4])
        out[k] = -R.T @ camera_t[k, 4:7].astype(np.float64)
    return out


def ghost_to_rich(camera_t, cam_names, scene):
    """SE(3) mapping the ghost world onto RICH's multi-camera frame."""
    gc, pc = [], []
    centres = ghost_cam_centres(camera_t)
    for k, cam in enumerate(cam_names):
        if not np.isfinite(centres[k]).all():
            continue
        R, t, _ = parse_calib(scene, cam)
        gc.append(-R.T @ t)
        pc.append(centres[k])
    if len(pc) < 2:
        raise RuntimeError("fewer than 2 usable cameras for the ghost->RICH SE(3)")
    R_a, t_a = se3_align(np.stack(pc), np.stack(gc))
    resid = np.linalg.norm(np.stack(pc) @ R_a.T + t_a - np.stack(gc), axis=1)
    print(f"  ghost->RICH SE(3) from {len(pc)} cams, centre residual "
          f"{resid.mean() * 100:.1f} cm mean / {resid.max() * 100:.1f} cm max")
    return R_a, t_a


# ── panels ─────────────────────────────────────────────────────────────────
def render_gt_overlay(scene, cam, frame, d, fmin, faces, args):
    """Panel A/B — the image with the GT SMPL-X mesh reprojected on it."""
    p = frame_path(scene, cam, frame)
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(p)
    H, W = bgr.shape[:2]
    R, t, K_full = parse_calib(scene, cam)
    K, cw, ch = centered_K(K_full, crop_meta(scene)[cam])
    if (cw, ch) != (W, H):
        print(f"  note: {cam} crop_meta says {cw}x{ch}, image is {W}x{H}")

    ti = frame - fmin
    verts, _ = smplx_vertices(d["gt_body_pose"][ti:ti + 1], d["gt_body_shape"],
                              np.nan_to_num(d["gt_body_transl_world"][ti:ti + 1]),
                              args.smplx_model)
    overlay = bgr.copy()
    drawn = 0
    for i in range(verts.shape[1]):
        if not d["gt_valid"][ti, i]:
            continue
        v = verts[0, i]
        uv, z = project(v, R, t, K)
        ok = np.isfinite(uv).all(1) & (z > 1e-6)
        keep = ok[faces].all(1)
        if not keep.any():
            continue
        fc = face_shading(v, faces[keep], R, _person_rgb(i))
        zc = z[faces[keep]].max(1)
        poly = poly_fx(uv[faces[keep]])
        paint(overlay, [("t", float(zc[j]), poly[j], fc[j]) for j in range(len(zc))])
        drawn += 1
    out = cv2.addWeighted(overlay, args.overlay_alpha, bgr, 1 - args.overlay_alpha, 0)
    for i in range(verts.shape[1]):
        if not d["gt_valid"][ti, i]:
            continue
        m = mesh_mask(verts[0, i], faces, R, t, K, H, W)
        if m.any():
            draw_outline(out, m, _person_rgb_dark(i), args.outline_thickness)
    return out, drawn


def render_scene_panel(scene, cam, frame, d, fmin, faces, args):
    """Panel C — white page: depth cloud + predicted body + GT outline."""
    ti = frame - fmin
    cam_names = [str(n) for n in d["camera_names"]]
    R_a, t_a = ghost_to_rich(d["camera"][ti], cam_names, scene)

    pv, pfaces = smplx_vertices(d["pose"][ti:ti + 1], d["shape"],
                                np.nan_to_num(d["body_transl_world"][ti:ti + 1]),
                                args.smplx_model)
    finite_p = (np.isfinite(d["body_transl_world"][ti]).all(1)
                & np.isfinite(d["pose"][ti]).all((1, 2)))
    pv_rich = pv[0] @ R_a.T + t_a

    gt_verts, _ = smplx_vertices(d["gt_body_pose"][ti:ti + 1], d["gt_body_shape"],
                                 np.nan_to_num(d["gt_body_transl_world"][ti:ti + 1]),
                                 args.smplx_model)
    gt_verts = gt_verts[0]

    R_cam, t_cam, K_full = parse_calib(scene, cam)
    K, W, H = centered_K(K_full, crop_meta(scene)[cam])
    img = np.full((H, W, 3), 255, np.uint8)
    prims: list = []

    # Silhouette of the predicted body in the RENDER camera.  Carving against a
    # mask built in the source camera instead would not land under the body drawn
    # over it, and the mismatch reads as a white rim around the person.
    pred_union = np.zeros((H, W), np.uint8)
    if args.carve:
        for i in range(pv_rich.shape[0]):
            if finite_p[i]:
                pred_union |= mesh_mask(pv_rich[i], pfaces, R_cam, t_cam, K, H, W,
                                        close=False)

    ctx = depth_context(scene)
    if ctx["depth_mm"].shape[0] != d["pose"].shape[0]:
        raise RuntimeError(f"depth T={ctx['depth_mm'].shape[0]} != prediction "
                           f"T={d['pose'].shape[0]}: not the same frame axis")
    src = ctx["names"] if args.depth_cams == "all" else [cam]
    pts_all, cols_all = [], []
    for cname in src:
        if cname not in ctx["names"]:
            continue
        k = ctx["names"].index(cname)
        bgr = cv2.imread(str(frame_path(scene, cname, frame)))
        got = depth_cloud(ctx, ti, k, bgr, args.depth_stride, args.conf_thr)
        if got is None:
            continue
        pts_all.append(got[0] @ R_a.T + t_a)
        cols_all.append(got[1])

    n_pts = 0
    if pts_all:
        pts = np.concatenate(pts_all)
        cols = np.concatenate(cols_all)
        uv, z = project(pts, R_cam, t_cam, K)
        vis = (np.isfinite(uv).all(1) & (z > 1e-6)
               & (uv[:, 0] > -10) & (uv[:, 0] < W + 10)
               & (uv[:, 1] > -10) & (uv[:, 1] < H + 10))
        if args.carve:
            iu = np.clip(np.round(np.nan_to_num(uv[:, 0])).astype(np.int32), 0, W - 1)
            iv = np.clip(np.round(np.nan_to_num(uv[:, 1])).astype(np.int32), 0, H - 1)
            vis &= pred_union[iv, iu] == 0
        uv, z, cols = uv[vis], z[vis], cols[vis]
        if len(z) > args.max_points:
            sel = np.random.default_rng(0).choice(len(z), args.max_points, replace=False)
            uv, z, cols = uv[sel], z[sel], cols[sel]
        n_pts = len(z)
        uvi = np.round(uv).astype(np.int32)
        for i in range(len(z)):
            prims.append(("p", float(z[i]), ((int(uvi[i, 0]), int(uvi[i, 1])),
                                             args.ball_radius), cols[i]))

    n_bodies = 0
    for i in range(pv_rich.shape[0]):
        if not finite_p[i]:
            continue
        v = pv_rich[i]
        uv, z = project(v, R_cam, t_cam, K)
        ok = np.isfinite(uv).all(1) & (z > 1e-6)
        keep = ok[pfaces].all(1)
        if not keep.any():
            continue
        fc = face_shading(v, pfaces[keep], R_cam, _person_rgb(i))
        zc = z[pfaces[keep]].max(1)
        poly = poly_fx(uv[pfaces[keep]])
        for j in range(len(zc)):
            prims.append(("t", float(zc[j]), poly[j], fc[j]))
        n_bodies += 1

    print(f"  panel C: {n_pts} balls (r={args.ball_radius}), {n_bodies} predicted, "
          f"{int(d['gt_valid'][ti].sum())} GT")
    paint(img, prims)

    for i in range(gt_verts.shape[0]):
        if not d["gt_valid"][ti, i]:
            continue
        m = mesh_mask(gt_verts[i], faces, R_cam, t_cam, K, H, W)
        if m.any():
            rgb = {"black": (0, 0, 0), "white": (255, 255, 255)}.get(
                args.gt_outline, _person_rgb_dark(i))
            draw_outline(img, m, rgb, args.outline_thickness)
    return img


# ── frame ranking ──────────────────────────────────────────────────────────
def rank(scene: str, pred: Path, top: int):
    """Per-frame root placement error, prediction vs GT, in the RICH frame.

    RICH has no per-frame eval dump to read, so this measures the quantity panel C
    is actually showing: the predicted pelvis carried into the RICH frame by the
    camera-fitted SE(3), against the GT pelvis.  No FK is needed, so it is cheap
    enough to sweep the whole sequence.
    """
    d, fmin = load_pred(pred, scene)
    cam_names = [str(n) for n in d["camera_names"]]
    rows = []
    for ti in range(d["pose"].shape[0]):
        if not d["gt_valid"][ti].all():
            continue
        pr = d["body_transl_world"][ti]
        gt = d["gt_body_transl_world"][ti]
        if not (np.isfinite(pr).all() and np.isfinite(gt).all()):
            continue
        try:
            R_a, t_a = ghost_to_rich_quiet(d["camera"][ti], cam_names, scene)
        except RuntimeError:
            continue
        err = np.linalg.norm(pr @ R_a.T + t_a - gt, axis=-1).mean()
        rows.append((err, fmin + ti))
    rows.sort()
    print(f"\n{scene}: {len(rows)} frames with valid GT")
    print(f"{'rank':>4s} {'frame':>7s} {'root err (mm)':>14s}")
    for i, (e, f) in enumerate(rows[:top]):
        print(f"{i + 1:4d} {f:7d} {e * 1000:14.1f}")
    if rows:
        med = np.median([r[0] for r in rows]) * 1000
        print(f"  median over the sequence: {med:.1f} mm")
    return rows


def ghost_to_rich_quiet(camera_t, cam_names, scene):
    gc, pc = [], []
    centres = ghost_cam_centres(camera_t)
    for k, cam in enumerate(cam_names):
        if not np.isfinite(centres[k]).all():
            continue
        R, t, _ = parse_calib(scene, cam)
        gc.append(-R.T @ t)
        pc.append(centres[k])
    if len(pc) < 2:
        raise RuntimeError("too few cameras")
    return se3_align(np.stack(pc), np.stack(gc))


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="ParkingLot2_009_impro1")
    ap.add_argument("--rank", action="store_true", help="rank frames by error and exit")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--frame", type=int)
    ap.add_argument("--cams", nargs=2, metavar=("CAM_A", "CAM_B"),
                    help="panel C is rendered from CAM_A")
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "figures" / "qualitative_rich")
    ap.add_argument("--panels", choices=["abc", "ab", "c"], default="abc")
    ap.add_argument("--smplx-model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--depth-cams", choices=["self", "all"], default="all")
    ap.add_argument("--depth-stride", type=int, default=1)
    ap.add_argument("--conf-thr", type=float, default=0.0)
    ap.add_argument("--ball-radius", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--carve", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overlay-alpha", type=float, default=0.6)
    ap.add_argument("--outline-thickness", type=int, default=2)
    ap.add_argument("--gt-outline", choices=["black", "white", "person"], default="white")
    args = ap.parse_args()

    if args.pred is None:
        args.pred = _REPO_ROOT / "fusion_outputs" / f"{args.scene}.npz"
    if not Path(args.pred).exists():
        sys.exit(f"prediction not found: {args.pred}")
    if not (IMG_ROOT / args.scene).is_dir():
        sys.exit(f"images not found: {IMG_ROOT / args.scene}\n"
                 f"mount centered_test.sqsh at {IMG_ROOT} first")

    if args.rank or args.frame is None:
        rank(args.scene, args.pred, args.top)
        return
    if args.cams is None:
        sys.exit("--frame needs --cams")

    args.out.mkdir(parents=True, exist_ok=True)
    d, fmin = load_pred(args.pred, args.scene)
    if not (0 <= args.frame - fmin < d["pose"].shape[0]):
        sys.exit(f"frame {args.frame} outside [{fmin}, {fmin + d['pose'].shape[0] - 1}]")

    import smplx as _smplx_lib
    faces = _smplx_lib.create(model_type="smplx", model_path=str(args.smplx_model),
                              ext=Path(args.smplx_model).suffix.lstrip("."),
                              gender="neutral", use_pca=False, num_betas=10).faces.copy()

    stem = f"{args.scene}_f{args.frame:05d}"
    if args.panels in ("abc", "ab"):
        for cam in args.cams:
            img, drawn = render_gt_overlay(args.scene, cam, args.frame, d, fmin, faces, args)
            p = args.out / f"{stem}_{cam}_gt.png"
            cv2.imwrite(str(p), img)
            print(f"  {p.name}: {drawn} GT bodies projected")
    if args.panels == "ab":
        return
    img = render_scene_panel(args.scene, args.cams[0], args.frame, d, fmin, faces, args)
    p = args.out / f"{stem}_{args.cams[0]}_scene.png"
    cv2.imwrite(str(p), img)
    print(f"  {p.name}")


if __name__ == "__main__":
    main()
