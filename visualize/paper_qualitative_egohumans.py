"""
Paper qualitative figure for EgoHumans — GT-mesh reprojection + a 3-D scene render.

Produces three PNGs for one (scene, frame):

  A, B : two exo views, far apart, showing the raw undistorted RGB with the GT
         SMPL mesh reprojected on top through the GT (COLMAP) camera.  If the
         GT calibration and the GT bodies are right, the mesh sits exactly on
         the person.  These panels contain no ghost output at all — they are the
         reference that the third panel is judged against.

  C    : the same viewpoint as panel A, but nothing photographic: a white page
         carrying (i) the VGGT depth of that frame as a coloured ball cloud,
         (ii) the GHOST predicted bodies as shaded meshes, and (iii) the GT
         bodies as a bare outline.  Where the outline hugs the shaded mesh,
         ghost placed the person correctly.

Everything in panel C is drawn in the **aria01 world**, which is the frame the
EgoHumans GT lives in:

    GT SMPL vertices      already aria01 world             (no transform)
    GT exo cameras        COLMAP  --inv(colmap_from_aria["aria01"])-->  aria01
    ghost preds + depth   ghost world  --x ma_scale, then SE(3)-->      aria01

The SE(3) is fitted per frame from the ghost camera centres onto the GT camera
centres, exactly as ``evaluation/evaluate_egohumans_median.py`` does for W-MPJPE†
(no scaling — the metric scale comes from ``mapanything_scale_baseline.npy``).
The figure therefore shows the same alignment the paper's table reports.

Standalone by design: the COLMAP parsing, the Kabsch/Sim(3) solvers, the SMPL-X
forward pass and the depth unprojection are copied here rather than imported, so
editing this figure can never move a number in the eval or the viewer.

Usage
-----
Rank the candidate frames (no GPU, reads the eval dumps only)::

    pixi run python -m visualize.paper_qualitative_egohumans --rank

Render one frame (needs the prediction .npz from scripts/infer_scene.py)::

    pixi run python -m visualize.paper_qualitative_egohumans \
        --scene 03_fencing/005_fencing --frame 300 --cams cam04 cam10 \
        --pred fusion_outputs/005_fencing.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── data roots ─────────────────────────────────────────────────────────────
GHOST_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
GT_ROOT     = Path("/iopsstor/scratch/cscs/tnanni/egohumans_gt_full")
COLMAP_ROOT = Path("/iopsstor/scratch/cscs/tnanni/sync_egohumans")
UNDIST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted")
DUMP_ROOT   = _REPO_ROOT / "eval_egohumans" / "dumps_smpl24_median"
INNER       = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

# The 14 sequences that still have undistorted exo frames on iopsstor.  Every
# other EgoHumans sequence has outputs but no imagery, so it cannot be figured.
SUBSET = [
    "01_tagging/007_tagging",    "01_tagging/011_tagging",
    "02_lego/001_legoassemble",  "02_lego/003_legoassemble",
    "03_fencing/005_fencing",    "03_fencing/006_fencing",
    "04_basketball/001_basketball", "04_basketball/011_basketball",
    "05_volleyball/004_volleyball", "05_volleyball/011_volleyball",
    "06_badminton/022_badminton", "06_badminton/031_badminton",
    "07_tennis/007_tennis",      "07_tennis/012_tennis",
]

# aria id -> RGB, matching utilities/render_egohumans_gt.py so figures agree.
PERSON_RGB = {
    1: (220,  50,  47),   # aria01 red
    2: ( 60, 170,  70),   # aria02 green
    3: ( 40, 110, 220),   # aria03 blue
    4: (230, 170,  30),   # aria04 yellow
}


def _person_rgb(pid: int) -> tuple[int, int, int]:
    return PERSON_RGB.get(pid, (120, 120, 120))


def _person_rgb_dark(pid: int, f: float = 0.45) -> tuple[int, int, int]:
    """The person's colour, darkened — used for that person's GT outline so the
    outline reads as the same subject as the shaded body it is being compared to."""
    return tuple(int(round(c * f)) for c in _person_rgb(pid))


def visible_masks(items):
    """[(key, mask, depth)] -> {key: mask with nearer items punched out}.

    Outlines are drawn after the bodies, so without this a person standing
    behind another still gets a full outline painted over the body in front —
    which is exactly what makes an occluded subject look wrong.  Depth is the
    mean of the mesh's vertices in camera space; that is crude for two meshes
    that interpenetrate, but people in these scenes are metres apart.
    """
    kern = np.ones((5, 5), np.uint8)
    out = {}
    for key, m, d in items:
        occ = np.zeros(m.shape, np.uint8)
        for key2, m2, d2 in items:
            # key is (source, person).  A subject's own predicted body must not
            # occlude its own GT outline — they are the same human, and letting
            # them cancel leaves only the sliver where they disagree instead of
            # the closed contour the figure is asking the reader to compare.
            if key2 == key or key2[1] == key[1]:
                continue
            if d2 < d:
                occ |= m2.astype(np.uint8)
        # Close the occluder and open the remainder.  Rasterised masks carry
        # single-pixel pinholes, and subtracting one from another turns every
        # pinhole into an isolated island that drawContours then rings — which
        # reads as dark speckle sprayed over the body, not as an outline.
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kern)
        vis = (m.astype(np.uint8) & (1 - occ))
        out[key] = cv2.morphologyEx(vis, cv2.MORPH_OPEN, kern).astype(bool)
    return out


# ── path helpers ───────────────────────────────────────────────────────────
def _split(scene: str) -> tuple[str, str]:
    act, seq = scene.split("/")
    return act, seq


def _undist_seq_dir(scene: str) -> Path:
    act, seq = _split(scene)
    return UNDIST_ROOT / act / INNER / act / seq


def _frames_dir(scene: str, cam: str) -> Path:
    return _undist_seq_dir(scene) / "exo" / cam / "images_undistorted" / "frames"


# ── geometry (copied) ──────────────────────────────────────────────────────
def se3_align(src: np.ndarray, dst: np.ndarray):
    """Kabsch SE(3) (no scale): R,t minimising ||R@src+t-dst||. src,dst (N,3)."""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, dc - R @ sc


def sim3_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return pred aligned to gt by Sim(3). pred,gt (N,3)."""
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    s = np.sqrt((g0 ** 2).sum() / ((p0 ** 2).sum() + 1e-12))
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return s * p0 @ R.T + gc


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ── COLMAP / GT calibration ────────────────────────────────────────────────
def load_colmap_poses(scene: str) -> dict[str, dict[str, tuple]]:
    """{cam: {image_basename: (R, t)}}, COLMAP world -> camera."""
    act, seq = _split(scene)
    images_txt = COLMAP_ROOT / act / seq / "colmap" / "workplace" / "images.txt"
    out: dict[str, dict[str, tuple]] = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 10:
                continue
            name = p[9]
            cam = name.split("/")[0]
            if not cam.startswith("cam"):
                continue
            try:
                q = np.array([float(v) for v in p[1:5]], dtype=np.float64)
                t = np.array([float(v) for v in p[5:8]], dtype=np.float64)
            except ValueError:
                continue
            out.setdefault(cam, {})[Path(name).name] = (_quat_wxyz_to_R(q), t)
    return out


def colmap_to_aria(scene: str) -> np.ndarray:
    """4x4 COLMAP world -> aria01 world."""
    act, seq = _split(scene)
    p = GT_ROOT / act / seq / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    with open(p, "rb") as f:
        d = pickle.load(f)
    return np.linalg.inv(np.asarray(d["aria01"], dtype=np.float64))


def gt_camera_aria(poses: dict, T_c2a: np.ndarray, cam: str, frame: int):
    """(R, t) mapping aria01 world -> camera, for one exo cam.

    The exo rig is static, so a missing entry for this exact frame falls back to
    the first pose COLMAP registered for that camera.
    """
    R_c, t_c = _colmap_pose(poses, cam, frame)
    T_a2c = np.linalg.inv(T_c2a)                       # aria -> colmap
    R = R_c @ T_a2c[:3, :3]
    t = R_c @ T_a2c[:3, 3] + t_c
    return R, t


def _colmap_pose(poses: dict, cam: str, frame: int):
    per_img = poses[cam]
    return per_img.get(f"{frame:05d}.jpg", per_img[sorted(per_img)[0]])


def gt_camera_centre_aria(poses: dict, T_c2a: np.ndarray, cam: str, frame: int):
    """Camera centre in aria01 world.

    Taken in the COLMAP frame first, where the extrinsic 3x3 really is a
    rotation, then mapped over.  Inverting the aria-folded extrinsic instead
    would be wrong: colmap_from_aria carries a scale, so its 3x3 is s*R and
    ``-R.T @ t`` inflates the centre by s^2.  Same order as
    evaluate_egohumans_median.py::_gt_exo_cameras_aria.
    """
    R_c, t_c = _colmap_pose(poses, cam, frame)
    C_colmap = -R_c.T @ t_c
    return T_c2a[:3, :3] @ C_colmap + T_c2a[:3, 3]


def undist_K(scene: str, cam: str) -> tuple[np.ndarray, int, int]:
    """Pinhole K of the undistorted frames, plus their size.

    prep_undistort_egohumans.py rectifies with R = I, so this K shares the
    fisheye camera's rotation — the COLMAP extrinsics apply unchanged.
    """
    import json
    with open(_undist_seq_dir(scene) / "exo" / cam / "calibration.json") as f:
        c = json.load(f)
    return np.asarray(c["K"], dtype=np.float64), int(c["width"]), int(c["height"])


# ── GT bodies ──────────────────────────────────────────────────────────────
def smpl_faces() -> np.ndarray:
    d = pickle.load(open(_REPO_ROOT / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl", "rb"),
                    encoding="latin1")
    return np.asarray(d["f"], dtype=np.int64)


def gt_bodies(scene: str, frame: int) -> dict[int, np.ndarray]:
    """{pid: (6890,3) vertices in aria01 world} for one frame."""
    act, seq = _split(scene)
    p = GT_ROOT / act / seq / "processed_data" / "smpl" / f"{frame:05d}.npy"
    if not p.exists():
        return {}
    data = np.load(str(p), allow_pickle=True).item()
    out: dict[int, np.ndarray] = {}
    for name, v in data.items():
        if not isinstance(v, dict) or not name.startswith("aria"):
            continue
        verts = v.get("vertices", v.get("verts"))
        if verts is None:
            continue
        out[int(name.replace("aria", ""))] = np.asarray(verts, dtype=np.float64)
    return out


# ── ghost predictions ──────────────────────────────────────────────────────
def ghost_frame_axis(scene: str) -> tuple[int, list[int]]:
    """(frame_start, pids) — the T-axis origin and person order of the prediction.

    infer_scene.py does not save either, and both are a pure function of what is
    on disk: the pids are the person_<id>.npz stems under body_data_clean and the
    origin is the smallest frame index any camera saw.  Recomputed here rather
    than re-derived by hand so the figure cannot silently drift off by a frame.
    """
    scene_dir = GHOST_ROOT / scene
    fmin, pids = None, set()
    for cam in sorted(d.name for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data_clean").is_dir()):
        for f in sorted((scene_dir / cam / "body_data_clean").glob("person_*.npz")):
            pids.add(int(f.stem.split("_")[1]))
            fi = np.load(str(f), allow_pickle=False)["frame_indices"].astype(int)
            if fi.size:
                fmin = int(fi.min()) if fmin is None else min(fmin, int(fi.min()))
    if fmin is None:
        raise RuntimeError(f"no body_data_clean tracks under {scene_dir}")
    return fmin, sorted(pids)


def smplx_vertices(pose, shape, trans, smplx_model_dir):
    """SMPL-X forward -> (vertices (T,P,V,3) float32, faces (F,3)).  Copied from
    visualize/visualize_rerun.py so the figure has no import into the viewer."""
    import torch
    import smplx as smplx_lib
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

    def _6d_to_aa(p6: np.ndarray) -> np.ndarray:
        m = rotation_6d_to_matrix(torch.from_numpy(p6.astype(np.float32)))
        return matrix_to_axis_angle(m).numpy()

    T, P, J, _ = pose.shape
    if shape.ndim == 2:
        shape = np.broadcast_to(shape[None], (T, P, 10)).copy()

    p = Path(smplx_model_dir)
    kw: dict = {"model_type": "smplx", "model_path": str(p)}
    if p.is_file():
        kw["ext"] = p.suffix.lstrip(".")
    model = smplx_lib.create(**kw, gender="neutral", use_pca=False, num_betas=10,
                             flat_hand_mean=True, batch_size=T * P)
    model.eval()

    go = _6d_to_aa(pose[:, :, 0, :])
    bp = _6d_to_aa(pose[:, :, 1:22, :])

    def _t(x):
        return torch.from_numpy(x.reshape(T * P, -1).astype(np.float32))

    with torch.no_grad():
        out = model(global_orient=_t(go), body_pose=_t(bp), betas=_t(shape),
                    transl=_t(trans), return_verts=True)
    V = out.vertices.shape[1]
    return out.vertices.numpy().reshape(T, P, V, 3), model.faces.copy()


def ghost_cam_centres(camera_t: np.ndarray) -> np.ndarray:
    """(K,3) camera centres in metric ghost world from a (K,8) camera row.

    Encoding is [quat_wxyz(4), t_w2c(3), focal(1)] with t_w2c already multiplied
    by the MapAnything scale (see _build_vggt_cameras in scripts/inference.py),
    so these centres are metres and need no further scaling.
    """
    out = np.full((camera_t.shape[0], 3), np.nan)
    for k in range(camera_t.shape[0]):
        R = _quat_wxyz_to_R(camera_t[k, :4])
        out[k] = -R.T @ camera_t[k, 4:7].astype(np.float64)
    return out


# ── depth ──────────────────────────────────────────────────────────────────
def depth_context(scene: str):
    scene_dir = GHOST_ROOT / scene
    cam_npz = np.load(scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    dep_npz = np.load(scene_dir / "vggt_depth_centered.npz", allow_pickle=False)
    scale = np.load(scene_dir / "mapanything_scale_baseline.npy")
    s = float(np.median(np.asarray(scale)[np.asarray(scale) > 0]))
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in cam_npz["camera_names"]]
    return {
        "names": names,
        "extr": cam_npz["extrinsics"],          # (T,K,3,4) cam-from-world, VGGT units
        "intr": cam_npz["intrinsics"],          # (T,K,3,3)
        "oc":   cam_npz["original_coords"],     # (T,K,4)
        "cam_valid": cam_npz["valid"],          # (T,K)
        "depth_mm": dep_npz["depth"],           # (T,K,h,w) uint16
        "depth_conf": dep_npz["depth_conf"],    # (T,K,h,w)
        "depth_valid": dep_npz["depth_valid"],  # (T,K)
        "scale": s,
    }


def depth_cloud(ctx, t, k, bgr, stride, conf_thr):
    """Unproject camera k's depth at index t into metric ghost world.

    Returns (points (N,3), colours (N,3) uint8) or None.  Copied from the
    viewer's _depth_cloud, minus the mask-based person removal (EgoHumans ships
    no mask_data.npz) — panel C carves the people with the projected meshes.
    """
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
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    pts_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)

    R_d = ctx["extr"][t, k, :3, :3]
    t_v = ctx["extr"][t, k, :3, 3] * ctx["scale"]
    pts = ((pts_cam - t_v) @ R_d).astype(np.float64)

    if bgr is not None:
        fh, fw = bgr.shape[:2]
        iu = np.clip(((u - x1) * fw / (x2 - x1)).astype(np.int32), 0, fw - 1)
        iv = np.clip(((v - y1) * fh / (y2 - y1)).astype(np.int32), 0, fh - 1)
        cols = bgr[iv, iu][:, ::-1]
    else:
        cols = np.full((len(z), 3), 128, np.uint8)
    return pts, cols.astype(np.uint8)


def depth_grid(ctx, t, k, stride, conf_thr, carve=None):
    """Metric depth grid for camera k with the invalid cells set to NaN.

    Returns ``(d, uu, vv)`` on the strided grid, in the depth map's own pixel
    coordinates (so ``ctx["intr"]`` applies directly), or None.
    """
    if (t >= ctx["depth_mm"].shape[0] or not ctx["depth_valid"][t, k]
            or not ctx["cam_valid"][t, k]):
        return None
    d = ctx["depth_mm"][t, k][::stride, ::stride].astype(np.float32) / 1000.0 * ctx["scale"]
    conf = ctx["depth_conf"][t, k][::stride, ::stride].astype(np.float32)
    h_d, w_d = d.shape
    vv, uu = np.mgrid[0:h_d, 0:w_d].astype(np.float32) * stride
    x1, y1, x2, y2 = ctx["oc"][t, k]
    bad = ~((d > 1e-4) & (conf >= conf_thr)
            & (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2))
    if carve is not None:
        bad |= carve[::stride, ::stride][:h_d, :w_d].astype(bool)
    d = d.copy()
    d[bad] = np.nan
    if not np.isfinite(d).any():
        return None
    return d, uu, vv


def depth_mesh(ctx, t, k, bgr, stride, conf_thr, carve=None, disc_rel: float = 0.06):
    """Triangulate camera k's depth into a surface in metric ghost world.

    A quad becomes two triangles only when all four corners are finite and free
    of a depth jump, so foreground and background are never rubber-sheeted into
    one another. Copied from the viewer's _depth_grid_mesh / _grid_vertex_colors.
    Returns (verts (M,3), tris (F,3), colours (M,3) uint8) or None.
    """
    got = depth_grid(ctx, t, k, stride, conf_thr, carve)
    if got is None:
        return None
    d, uu, vv = got
    h_d, w_d = d.shape
    intr = ctx["intr"][t, k]
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    pts_cam = np.stack([(uu - cx) / fx * d, (vv - cy) / fy * d, d], axis=-1)
    R_d = ctx["extr"][t, k, :3, :3]
    t_v = ctx["extr"][t, k, :3, 3] * ctx["scale"]
    verts_all = (pts_cam.reshape(-1, 3) - t_v) @ R_d

    quad = np.stack([d[:-1, :-1], d[1:, :-1], d[:-1, 1:], d[1:, 1:]], 0)
    ok = np.all(np.isfinite(quad), 0)
    with np.errstate(all="ignore"):
        ok &= (np.nanmax(quad, 0) - np.nanmin(quad, 0)) < disc_rel * np.nanmean(quad, 0)
    if not ok.any():
        return None
    I, J = np.mgrid[0:h_d - 1, 0:w_d - 1]
    v00 = (I * w_d + J)[ok]; v10 = ((I + 1) * w_d + J)[ok]
    v01 = (I * w_d + J + 1)[ok]; v11 = ((I + 1) * w_d + J + 1)[ok]
    tris = np.concatenate([np.stack([v00, v10, v11], -1),
                           np.stack([v00, v11, v01], -1)], 0)
    used, tris = np.unique(tris, return_inverse=True)
    tris = tris.reshape(-1, 3)
    verts = np.nan_to_num(verts_all[used])

    x1, y1, x2, y2 = ctx["oc"][t, k]
    if bgr is not None:
        u = uu.reshape(-1)[used]; v = vv.reshape(-1)[used]
        fh, fw = bgr.shape[:2]
        iu = np.clip(((u - x1) * fw / max(1e-6, float(x2 - x1))).astype(np.int32), 0, fw - 1)
        iv = np.clip(((v - y1) * fh / max(1e-6, float(y2 - y1))).astype(np.int32), 0, fh - 1)
        cols = bgr[iv, iu][:, ::-1]
    else:
        cols = np.full((len(used), 3), 128, np.uint8)
    return verts, tris, cols.astype(np.uint8)


# ── rasterisation ──────────────────────────────────────────────────────────
def border_box(bgr, thr: int = 12, frac: float = 0.6) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the frame's real content.

    The fisheye undistort leaves black bands on some cameras (cam06 of the
    tagging scenes is the worst), and they read as a heavy border in the figure.
    A row/column counts as content when more than ``frac`` of it is above
    ``thr``, so a genuinely dark scene edge is kept and a black band is not.
    Projection still happens in full-frame coordinates — the crop is applied to
    the finished canvas, so it cannot disturb the camera model.
    """
    g = np.asarray(bgr).max(axis=2)
    h, w = g.shape
    rows = np.where((g > thr).sum(1) > frac * w)[0]
    cols = np.where((g > thr).sum(0) > frac * h)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def cross_camera_mask(pts, cam_idx, radius: float = 0.10, min_cams: int = 1) -> np.ndarray:
    """Keep a point only if >= min_cams OTHER cameras reconstructed something
    within `radius` metres of it.

    Statistical outlier removal cannot touch this failure: a camera that invents
    a surface where the others see nothing produces a locally DENSE patch, so
    every one of its points has close neighbours and passes.  What marks it out
    is that the neighbours all come from the same camera.  Requiring corroboration
    from a different viewpoint is what actually tests the reconstruction.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    cams = np.unique(cam_idx)
    if len(cams) < 2:
        return np.ones(len(pts), bool)
    trees = {int(c): cKDTree(pts[cam_idx == c]) for c in cams}
    support = np.zeros(len(pts), np.int32)
    for c in cams:
        m = cam_idx == c
        for c2 in cams:
            if c2 == c:
                continue
            d, _ = trees[int(c2)].query(pts[m], k=1,
                                        distance_upper_bound=radius, workers=-1)
            support[m] += np.isfinite(d)
    return support >= min_cams


def statistical_outlier_mask(pts, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Open3D's remove_statistical_outlier, on scipy: keep a point when its mean
    distance to its k nearest neighbours is within std_ratio sigma of the global
    mean of that statistic.  Isolated points — one camera's reconstruction noise
    sitting where no other camera put anything — fail it; dense surface does not.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= k:
        return np.ones(len(pts), bool)
    dist, _ = cKDTree(pts).query(pts, k=k + 1, workers=-1)
    md = dist[:, 1:].mean(1)                      # column 0 is the point itself
    return md <= md.mean() + std_ratio * md.std()


def project(points_w, R, t, K):
    """(N,3) world -> (uv (N,2) float, z (N,)) through a pinhole camera."""
    Xc = np.asarray(points_w, dtype=np.float64) @ R.T + t
    z = Xc[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    return np.stack([u, v], -1), z


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    for c in range(3):
        np.add.at(n, faces[:, c], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-9)


def _face_lambert(verts, faces, R):
    """(F,) Lambert term, lit from the camera. Negative = the face points away."""
    fn = vertex_normals(verts, faces)[faces].mean(1)
    return -(fn @ R.T)[:, 2]                            # camera looks down +z


def front_facing(verts, faces, R) -> np.ndarray:
    """(F,) bool — faces whose outward normal turns toward the camera.

    A body is a closed surface, so its far side projects onto the same pixels as
    its near side.  Painter's algorithm sorts by mean depth, and across a thin
    limb the two sides tie often enough that a dark back face lands on top of the
    lit front one — which is the speckle over the bodies.  Culling the back
    faces removes the ambiguity outright; nothing behind them is visible anyway.
    """
    return _face_lambert(verts, faces, R) > 0.0


def face_shading(verts, faces, R, base_rgb):
    """(F,3) uint8 Lambert-shaded face colours, lit from the camera."""
    lam = np.clip(_face_lambert(verts, faces, R), 0.0, 1.0)
    shade = (0.35 + 0.65 * lam)[:, None]
    return np.clip(np.asarray(base_rgb, np.float64)[None] * shade, 0, 255).astype(np.uint8)


def mesh_mask(verts, faces, R, t, K, H, W, close: bool = True) -> np.ndarray:
    """Filled silhouette of a mesh, uint8 {0,1}, (H,W)."""
    m = np.zeros((H, W), np.uint8)
    uv, z = project(verts, R, t, K)
    ok = np.isfinite(uv).all(1) & (z > 1e-6)
    keep = ok[faces].all(1)
    if keep.any():
        cv2.fillPoly(m, poly_fx(uv[faces[keep]]), 1, shift=SUBPIX)
    # Filling thousands of triangles leaves pinholes between them; left alone
    # they make the silhouette's contour ragged and punch false holes in every
    # occlusion test that uses this mask.  Closing grows the mask by a couple of
    # pixels though, so the carve asks for close=False: a carve wider than the
    # body drawn over it is precisely the white rim around each person.
    if not close:
        return m
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def draw_outline(img, mask, rgb, thickness):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, tuple(int(c) for c in rgb[::-1]), thickness,
                     lineType=cv2.LINE_AA)


# Triangles are rasterised in fixed point with this many fractional bits.
# Rounding each triangle's corners to whole pixels independently means two
# triangles that share an edge do not agree on it, so the seam between them goes
# unfilled and whatever lies behind shows through as speckle.  1/16 px is enough
# for adjacent triangles to land on the same edge.
SUBPIX = 4
_S = 1 << SUBPIX


def poly_fx(uv: np.ndarray) -> np.ndarray:
    """(...,2) float pixel coords -> fixed-point ints for fillConvexPoly(shift=)."""
    return np.round(np.asarray(uv) * _S).astype(np.int32)


def paint(img, prims):
    """Painter's algorithm over a merged primitive list, far to near.

    prims entries are ("p", z, (u,v), rgb) for a ball and ("t", z, poly, rgb)
    for a triangle.  Sorting balls and triangles together is what lets the depth
    cloud occlude a body and the body occlude the cloud in the same image.
    """
    order = np.argsort([-p[1] for p in prims], kind="stable")
    for i in order:
        kind, _, geom, rgb = prims[i]
        col = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        if kind == "p":
            cv2.circle(img, geom[0], geom[1], col, -1, lineType=cv2.LINE_AA)
        else:
            # LINE_8, not LINE_AA.  A body is thousands of sub-pixel triangles;
            # antialiasing every one blends its edge with whatever is underneath,
            # so the background bleeds through the mesh as speckle and the body
            # stops reading as a surface.  Antialiasing is for the outline.
            cv2.fillConvexPoly(img, geom, col, lineType=cv2.LINE_8, shift=SUBPIX)


# ── panels ─────────────────────────────────────────────────────────────────
def render_gt_overlay(scene, cam, frame, poses, T_c2a, faces, alpha, thickness,
                      crop: bool = True):
    """Panel A/B — undistorted RGB with the GT SMPL mesh reprojected on it."""
    img_p = _frames_dir(scene, cam) / f"{frame:05d}.jpg"
    bgr = cv2.imread(str(img_p))
    if bgr is None:
        raise FileNotFoundError(img_p)
    H, W = bgr.shape[:2]
    K, Wc, Hc = undist_K(scene, cam)
    if (Wc, Hc) != (W, H):
        K = K * np.array([[W / Wc], [H / Hc], [1.0]])
    R, t = gt_camera_aria(poses, T_c2a, cam, frame)

    people = gt_bodies(scene, frame)
    overlay = bgr.copy()
    drawn = 0
    for pid, verts in sorted(people.items()):
        uv, z = project(verts, R, t, K)
        ok = np.isfinite(uv).all(1) & (z > 1e-6)
        keep = ok[faces].all(1)
        if not keep.any():
            continue
        rgb = _person_rgb(pid)
        fc = face_shading(verts, faces[keep], R, rgb)
        zc = z[faces[keep]].max(1)
        poly = poly_fx(uv[faces[keep]])
        prims = [("t", float(zc[i]), poly[i], fc[i]) for i in range(len(zc))]
        paint(overlay, prims)
        drawn += 1

    out = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0.0)
    items = []
    for pid, verts in sorted(people.items()):
        m = mesh_mask(verts, faces, R, t, K, H, W)
        if not m.any():
            continue
        _, z = project(verts, R, t, K)
        items.append((("gt", pid), m, float(np.nanmean(z))))
    for (_, pid), m in visible_masks(items).items():
        if m.any():
            draw_outline(out, m, _person_rgb_dark(pid), thickness)
    if crop:
        x0, y0, x1, y1 = border_box(bgr)
        out = out[y0:y1, x0:x1]
    return out, drawn, len(people)


def render_scene_panel(scene, cam, frame, poses, T_c2a, gt_faces, args):
    """Panel C — white page: depth balls + predicted meshes + GT outlines."""
    pred = np.load(args.pred, allow_pickle=False)
    frame_start, pids = ghost_frame_axis(scene)
    t_idx = frame - frame_start
    pose, shape = pred["pose"], pred["shape"]
    transl, camera = pred["body_transl_world"], pred["camera"]
    cam_names = [str(n) for n in pred["camera_names"]]
    if not (0 <= t_idx < pose.shape[0]):
        raise ValueError(f"frame {frame} outside prediction range "
                         f"[{frame_start}, {frame_start + pose.shape[0] - 1}]")
    if pose.shape[1] != len(pids):
        raise ValueError(f"prediction has {pose.shape[1]} persons, disk has {len(pids)}")

    # ghost world -> aria01 world, from this frame's camera centres.
    T_a2c = np.linalg.inv(T_c2a)
    gc, pc = [], []
    ghost_c = ghost_cam_centres(camera[t_idx])
    for k, cname in enumerate(cam_names):
        if cname not in poses or not np.isfinite(ghost_c[k]).all():
            continue
        gc.append(gt_camera_centre_aria(poses, T_c2a, cname, frame))
        pc.append(ghost_c[k])
    if len(pc) < 2:
        raise RuntimeError("fewer than 2 usable cameras for the ghost->aria SE(3)")
    R_a, t_a = se3_align(np.stack(pc), np.stack(gc))
    resid = np.linalg.norm(np.stack(pc) @ R_a.T + t_a - np.stack(gc), axis=1)
    print(f"  ghost->aria SE(3) from {len(pc)} cams, centre residual "
          f"{resid.mean() * 100:.1f} cm mean / {resid.max() * 100:.1f} cm max")

    # predicted bodies (ghost world) -> aria world
    pv, pfaces = smplx_vertices(pose[t_idx:t_idx + 1], shape,
                                np.nan_to_num(transl[t_idx:t_idx + 1]),
                                args.smplx_model)
    pv = pv[0]                                            # (P, V, 3)
    finite_p = np.isfinite(transl[t_idx]).all(1) & np.isfinite(pose[t_idx]).all((1, 2))
    pv_aria = pv @ R_a.T + t_a

    # render camera = GT pose of `cam`, undistorted pinhole intrinsics
    K, W, H = undist_K(scene, cam)
    R_cam, t_cam = gt_camera_aria(poses, T_c2a, cam, frame)

    gt_people = gt_bodies(scene, frame)
    img = np.full((H, W, 3), 255, np.uint8)
    prims: list = []

    # ---- depth balls -------------------------------------------------------
    # Silhouette of the predicted bodies in the RENDER camera.  The depth is
    # carved against this rather than against a mask built in the source
    # camera: the two cameras differ by the ghost->aria SE(3) residual and the
    # source mask is quantised to the depth grid, so a source-side hole does not
    # land under the body that is drawn over it, and the mismatch shows up as a
    # white rim around every person.  Carving here makes the removed region and
    # the drawn body the same shape by construction.
    pred_union = np.zeros((H, W), np.uint8)
    if args.carve:
        for p in range(pv_aria.shape[0]):
            if finite_p[p]:
                pred_union |= mesh_mask(pv_aria[p], pfaces, R_cam, t_cam, K, H, W,
                                        close=False)

    ctx = depth_context(scene)
    if ctx["depth_mm"].shape[0] != pose.shape[0]:
        raise RuntimeError(
            f"depth T={ctx['depth_mm'].shape[0]} != prediction T={pose.shape[0]}; "
            "the two arrays do not share a frame axis")
    src_cams = ctx["names"] if args.depth_cams == "all" else [cam]
    n_pts = 0
    cloud_pts: list = []
    cloud_cols: list = []
    cloud_cam: list = []
    for cname in src_cams:
        if cname not in ctx["names"]:
            continue
        k = ctx["names"].index(cname)
        fp = _frames_dir(scene, cname) / f"{frame:05d}.jpg"
        bgr = cv2.imread(str(fp))
        # Carve the people out of the depth with the projected meshes: it
        # otherwise contains the real person and would bury the prediction.
        Rv = ctx["extr"][t_idx, k, :3, :3]
        tv = ctx["extr"][t_idx, k, :3, 3] * ctx["scale"]
        Kv = ctx["intr"][t_idx, k]
        # Carve with the PREDICTED bodies only.  The GT bodies sit slightly off
        # the prediction, so carving them too removes cloud that no drawn mesh
        # covers afterwards — that leftover is the white aura around each person.
        # Undilated, every carved pixel ends up behind the body drawn over it.
        bodies_ghost = [pv[p] for p in range(pv.shape[0]) if finite_p[p]]
        if args.carve_gt:
            bodies_ghost += [(v - t_a) @ R_a for v in gt_people.values()]
        h_c, w_c = ctx["depth_mm"][t_idx, k].shape
        carve = np.zeros((h_c, w_c), np.uint8)
        for v in bodies_ghost:
            uvv = (v @ Rv.T + tv)
            zc = uvv[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                cu = Kv[0, 0] * uvv[:, 0] / zc + Kv[0, 2]
                cv_ = Kv[1, 1] * uvv[:, 1] / zc + Kv[1, 2]
            okv = np.isfinite(cu) & np.isfinite(cv_) & (zc > 1e-6)
            kf = okv[pfaces].all(1) if v.shape[0] == pv.shape[1] else okv[gt_faces].all(1)
            ff = pfaces if v.shape[0] == pv.shape[1] else gt_faces
            if kf.any():
                poly = np.round(np.stack([cu[ff[kf]], cv_[ff[kf]]], -1)).astype(np.int32)
                cv2.fillPoly(carve, poly, 1)
        if not args.carve:
            carve[:] = 0
        if args.carve_dilate > 0:
            r = args.carve_dilate
            carve = cv2.dilate(carve, np.ones((2 * r + 1, 2 * r + 1), np.uint8))

        if args.bg == "mesh":
            got = depth_mesh(ctx, t_idx, k, bgr, args.depth_stride, args.conf_thr,
                             None, args.disc_rel)
            if got is None:
                continue
            verts_g, tris, cols = got
            uv, z = project(verts_g @ R_a.T + t_a, R_cam, t_cam, K)
            ok = np.isfinite(uv).all(1) & (z > 1e-6)
            keep = ok[tris].all(1)
            if keep.any():
                # Drop the quads that land on a predicted body — that surface is
                # the real person, and the body drawn over it replaces it exactly.
                cen = uv[tris].mean(1)
                cu = np.clip(np.round(cen[:, 0]).astype(np.int32), 0, W - 1)
                cvv = np.clip(np.round(cen[:, 1]).astype(np.int32), 0, H - 1)
                keep &= pred_union[cvv, cu] == 0
            if not keep.any():
                continue
            tk = tris[keep]
            poly = poly_fx(uv[tk])
            zc = z[tk].max(1)
            fcol = cols[tk].mean(1).astype(np.uint8)
            n_pts += len(tk)
            for i in range(len(tk)):
                prims.append(("t", float(zc[i]), poly[i], fcol[i]))
            continue

        got = depth_cloud(ctx, t_idx, k, bgr, args.depth_stride, args.conf_thr)
        if got is None:
            continue
        pts_g, cols = got
        cloud_pts.append(pts_g @ R_a.T + t_a)     # ghost world -> aria world
        cloud_cols.append(cols)
        cloud_cam.append(np.full(len(pts_g), k, np.int32))

    if args.bg == "balls" and cloud_pts:
        pts_a = np.concatenate(cloud_pts)
        cols = np.concatenate(cloud_cols)
        cam_idx = np.concatenate(cloud_cam)
        if args.xcam_radius > 0 and len(np.unique(cam_idx)) > 1:
            keep_x = cross_camera_mask(pts_a, cam_idx, args.xcam_radius,
                                       args.xcam_min_cams)
            print(f"  cross-camera (r={args.xcam_radius} m, >={args.xcam_min_cams} "
                  f"other cams): {len(pts_a)} -> {int(keep_x.sum())} points "
                  f"({100 * (1 - keep_x.mean()):.1f}% dropped)")
            pts_a, cols, cam_idx = pts_a[keep_x], cols[keep_x], cam_idx[keep_x]
        # Statistical outlier removal on the MERGED cloud, so a point is judged
        # against every camera's geometry rather than only its own — the noise
        # this is here to kill is exactly the points one camera puts where the
        # others put nothing.
        if args.sor_neighbors > 0:
            keep_sor = statistical_outlier_mask(pts_a, args.sor_neighbors, args.sor_std)
            print(f"  SOR(k={args.sor_neighbors}, std={args.sor_std}): "
                  f"{len(pts_a)} -> {int(keep_sor.sum())} points "
                  f"({100 * (1 - keep_sor.mean()):.1f}% dropped)")
            pts_a, cols = pts_a[keep_sor], cols[keep_sor]

        # Carve in the RENDER camera, same as the mesh branch: a hole punched in
        # the source camera does not land under the body drawn over it.
        uv, z = project(pts_a, R_cam, t_cam, K)
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
            print(f"  max-points: subsampled to {args.max_points}")
        n_pts += len(z)
        uvi = np.round(uv).astype(np.int32)
        for i in range(len(z)):
            prims.append(("p", float(z[i]), ((int(uvi[i, 0]), int(uvi[i, 1])),
                                             args.ball_radius), cols[i]))

    # ---- predicted bodies --------------------------------------------------
    n_bodies = 0
    occluders: list = []
    for p in range(pv_aria.shape[0]):
        if not finite_p[p]:
            continue
        verts = pv_aria[p]
        uv, z = project(verts, R_cam, t_cam, K)
        ok = np.isfinite(uv).all(1) & (z > 1e-6)
        keep = ok[pfaces].all(1)
        if not keep.any():
            continue
        pm = mesh_mask(verts, pfaces, R_cam, t_cam, K, H, W)
        occluders.append((("pred", pids[p]), pm.astype(bool), float(np.nanmean(z))))
        rgb = _person_rgb(pids[p])
        fc = face_shading(verts, pfaces[keep], R_cam, rgb)
        # Bias the body toward the camera.  With the carve off, the depth
        # surface still holds the real person, sitting a few centimetres from
        # the predicted body; without a bias the painter's sort lets the
        # photographed person win in patches and paint over the prediction.
        zc = z[pfaces[keep]].max(1) - args.body_bias
        poly = poly_fx(uv[pfaces[keep]])
        for i in range(len(zc)):
            prims.append(("t", float(zc[i]), poly[i], fc[i]))
        n_bodies += 1

    print(f"  panel C: {n_pts} balls (r={args.ball_radius}), {n_bodies} predicted bodies, "
          f"{len(gt_people)} GT bodies")
    paint(img, prims)

    # ---- GT outlines on top ------------------------------------------------
    gt_items = []
    for pid, verts in sorted(gt_people.items()):
        m = mesh_mask(verts, gt_faces, R_cam, t_cam, K, H, W)
        if not m.any():
            continue
        _, z = project(verts, R_cam, t_cam, K)
        gt_items.append((("gt", pid), m.astype(bool), float(np.nanmean(z))))
    # A GT body hidden behind a nearer body — its own or a predicted one — must
    # not get an outline painted over the thing in front of it.
    vis = visible_masks(gt_items + occluders)
    for (kind, pid), m in vis.items():
        if kind != "gt" or not m.any():
            continue
        rgb = {"black": (0, 0, 0), "white": (255, 255, 255)}.get(
            args.gt_outline, _person_rgb_dark(pid))
        draw_outline(img, m, rgb, args.outline_thickness)
    if args.border_crop:
        ref = cv2.imread(str(_frames_dir(scene, cam) / f"{frame:05d}.jpg"))
        if ref is not None:
            x0, y0, x1, y1 = border_box(ref)
            img = img[y0:y1, x0:x1]
    return img


# ── frame ranking ──────────────────────────────────────────────────────────
def rank(top: int):
    """Per-frame error over the 14 figurable scenes, from the eval dumps.

    ``pred`` in a dump is already SE(3)-aligned into aria world, so its distance
    to ``gt`` is the W-MPJPE† of that frame; GA re-aligns the frame by a Sim(3)
    over all persons at once.  Only frames where every person is valid qualify —
    a figure with a missing body is not a figure.
    """
    rows = []
    for scene in SUBSET:
        act, seq = _split(scene)
        d_path = DUMP_ROOT / act / f"{seq}.npz"
        if not d_path.exists():
            print(f"{scene}: no dump")
            continue
        d = np.load(d_path, allow_pickle=False)
        pred, gt, frames = d["pred"], d["gt"], d["frames"]
        best = []
        for i in range(pred.shape[0]):
            pr, gtt = pred[i], gt[i]
            ok = np.isfinite(pr).all((1, 2)) & np.isfinite(gtt).all((1, 2))
            if not ok.all() or ok.sum() == 0:
                continue
            w = np.linalg.norm(pr[ok] - gtt[ok], axis=-1).mean()
            a = sim3_align(pr[ok].reshape(-1, 3), gtt[ok].reshape(-1, 3)).reshape(pr[ok].shape)
            ga = np.linalg.norm(a - gtt[ok], axis=-1).mean()
            best.append((w, ga, int(frames[i]), int(ok.sum())))
        if not best:
            print(f"{scene}: no frame with all persons valid")
            continue
        best.sort()
        rows.append((best[0][0], scene, best))
    rows.sort()
    print(f"\n{'scene':32s} {'best W†':>9s} {'GA':>8s} {'frame':>7s}  P   next frames")
    for w, scene, best in rows:
        nxt = " ".join(str(b[2]) for b in best[1:top])
        print(f"{scene:32s} {w * 1000:9.1f} {best[0][1] * 1000:8.1f} "
              f"{best[0][2]:7d}  {best[0][3]}   {nxt}")
    return rows


def camera_spread(scene: str):
    """Pairwise angle (deg) between exo cameras seen from the rig centroid."""
    poses = load_colmap_poses(scene)
    T_c2a = colmap_to_aria(scene)
    cams = sorted(c for c in poses if (_undist_seq_dir(scene) / "exo" / c).is_dir())
    C = {c: gt_camera_centre_aria(poses, T_c2a, c, 1) for c in cams}
    mid = np.mean(list(C.values()), 0)
    out = []
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            va, vb = C[a] - mid, C[b] - mid
            cosang = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
            out.append((np.degrees(np.arccos(np.clip(cosang, -1, 1))), a, b))
    out.sort(reverse=True)
    return out


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", action="store_true", help="rank frames and exit")
    ap.add_argument("--top", type=int, default=6, help="frames listed per scene by --rank")
    ap.add_argument("--spread", metavar="SCENE", help="print exo camera pair angles and exit")
    ap.add_argument("--scene", help="<activity>/<sequence>, e.g. 03_fencing/005_fencing")
    ap.add_argument("--frame", type=int, help="global 1-based frame index")
    ap.add_argument("--cams", nargs=2, metavar=("CAM_A", "CAM_B"),
                    help="the two exo cameras; panel C is rendered from CAM_A")
    ap.add_argument("--pred", type=Path, help="prediction .npz from scripts/infer_scene.py")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "figures" / "qualitative_egohumans")
    ap.add_argument("--panels", choices=["abc", "ab", "c"], default="abc",
                    help="which panels to write; 'c' leaves the GT overlays alone")
    ap.add_argument("--smplx-model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--bg", choices=["mesh", "balls"], default="mesh",
                    help="'mesh' triangulates the depth into a surface; 'balls' "
                         "draws one sphere per depth pixel")
    ap.add_argument("--disc-rel", type=float, default=float("inf"),
                    help="mesh only: relative depth jump that breaks a quad, so "
                         "foreground and background are not rubber-sheeted")
    ap.add_argument("--depth-cams", choices=["self", "all"], default="self",
                    help="'self' clouds only CAM_A's depth, 'all' merges every exo cam")
    ap.add_argument("--depth-stride", type=int, default=1,
                    help="mesh quad size in depth pixels; smaller looks "
                         "photographic, larger reads as a surface")
    ap.add_argument("--conf-thr", type=float, default=0.0,
                    help="drop depth below this VGGT confidence; 0 keeps "
                         "everything VGGT produced")
    ap.add_argument("--ball-radius", type=int, default=3)
    ap.add_argument("--max-points", type=int, default=250_000)
    ap.add_argument("--xcam-radius", type=float, default=0.10,
                    help="metres: keep a point only if another camera has geometry "
                         "within this distance; 0 disables")
    ap.add_argument("--xcam-min-cams", type=int, default=1,
                    help="how many OTHER cameras must corroborate a point")
    ap.add_argument("--sor-neighbors", type=int, default=20,
                    help="statistical outlier removal neighbour count; 0 disables")
    ap.add_argument("--sor-std", type=float, default=2.0,
                    help="statistical outlier removal std-dev ratio")
    ap.add_argument("--carve-dilate", type=int, default=0,
                    help="grow the person carve by this many depth-grid cells; "
                         "any growth shows up as a white halo around each subject")
    ap.add_argument("--body-bias", type=float, default=0.5,
                    help="metres to pull predicted bodies toward the camera in "
                         "the depth sort, so the real person in the depth "
                         "surface cannot paint over them")
    ap.add_argument("--carve", action=argparse.BooleanOptionalAction, default=True,
                    help="cut the people out of the depth surface; off by default "
                         "because any hole that the drawn body does not cover "
                         "exactly shows up as white around the person")
    ap.add_argument("--carve-gt", action="store_true",
                    help="also carve the cloud with the GT bodies (leaves an aura "
                         "wherever GT and prediction disagree)")
    ap.add_argument("--overlay-alpha", type=float, default=0.6,
                    help="GT mesh opacity in panels A and B")
    ap.add_argument("--outline-thickness", type=int, default=2)
    ap.add_argument("--gt-outline", choices=["black", "white", "person"],
                    default="person",
                    help="'person' draws each GT outline in a darkened shade of "
                         "that subject's own colour instead of one flat black")
    ap.add_argument("--border-crop", action=argparse.BooleanOptionalAction, default=True,
                    help="trim the undistort's black bands off the finished panels")
    args = ap.parse_args()

    if args.spread:
        for ang, a, b in camera_spread(args.spread):
            print(f"{a} {b} {ang:7.1f} deg")
        return
    if args.rank or not args.scene:
        rank(args.top)
        return

    if args.frame is None or args.cams is None:
        sys.exit("--scene needs --frame and --cams")
    if args.pred is None:
        args.pred = _REPO_ROOT / "fusion_outputs" / f"{_split(args.scene)[1]}.npz"
    if not Path(args.pred).exists():
        sys.exit(f"prediction not found: {args.pred}\n"
                 f"run scripts/infer_scene.py --dataset egohumans first")

    args.out.mkdir(parents=True, exist_ok=True)
    poses = load_colmap_poses(args.scene)
    T_c2a = colmap_to_aria(args.scene)
    gt_faces = smpl_faces()
    seq = _split(args.scene)[1]
    stem = f"{seq}_f{args.frame:05d}"

    for cam in (args.cams if args.panels in ("abc", "ab") else []):
        img, drawn, total = render_gt_overlay(args.scene, cam, args.frame, poses, T_c2a,
                                              gt_faces, args.overlay_alpha,
                                              args.outline_thickness, args.border_crop)
        p = args.out / f"{stem}_{cam}_gt.png"
        cv2.imwrite(str(p), img)
        print(f"  {p.name}: {drawn}/{total} GT bodies projected")

    if args.panels == "ab":
        return
    img = render_scene_panel(args.scene, args.cams[0], args.frame, poses, T_c2a,
                             gt_faces, args)
    p = args.out / f"{stem}_{args.cams[0]}_scene.png"
    cv2.imwrite(str(p), img)
    print(f"  {p.name}")


if __name__ == "__main__":
    main()
