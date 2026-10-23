"""Offscreen flyover render of a GHOST scene: per-frame VGGT pointmap + fused
SMPL-X body meshes, orbiting camera, MP4 out.

WHAT IS RENDERED IS EXACTLY WHAT THE PAPER PIPELINE PRODUCES. Nothing here
invents geometry or re-estimates scale:

* Bodies      — ``<cam>/body_data_clean`` -> geodesic ``median_fuse`` ->
                ``BodyPlacer.estimate_procrustes_dlt_mhr(scale=MapAnything
                baseline, smooth_window=0)`` -> SMPL-X FK vertices. This is a
                copy of ``predict_scene`` in
                ``evaluation/evaluate_egohumans_median.py``, the shipped
                evaluation path, with the SMPL-24 regression replaced by the FK
                vertices it already computes.
* Scene       — the PER-FRAME VGGT pointmap from ``vggt_depth_centered.npz``,
                unprojected with the per-frame VGGT intrinsics/extrinsics and
                multiplied by the SAME metric scale the placer placed with
                (``median(mapanything_scale_baseline.npy)``). There is no
                temporal median depth, no pose stabilisation, no synthetic
                surface mesh and no scale inferred from camera-translation
                ratios. VGGT is an independent per-frame reconstruction, so the
                cloud *will* shimmer under a moving camera — that is the data.
* World frame — the VGGT world (camera-0 gauge, RDF). No flips are applied.

The only display-side choices are: pixels covered by a detected person are
carved out of the cloud so the depth surface cannot occlude the body meshes
(``--keep-person-pixels`` disables), an optional voxel thinning of the cloud,
and the flyover camera path itself.

Rendering is a small torch rasteriser in this file (PyTorch3D in this env ships
without its compiled `_C` extension, so its rasterisers cannot run). Points and
triangles land in ONE shared z-buffer, so cloud/mesh occlusion is exact and no
OpenGL/EGL context is needed — it runs headless on a compute or login node.

Example (EgoHumans fencing 001):

    pixi run python -m visualize.render_flyover \
        --scene /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/03_fencing/001_fencing \
        --frames-root /iopsstor/scratch/cscs/tnanni/temp_egohumans/03_fencing/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/03_fencing/001_fencing/exo \
        --out /capstor/scratch/cscs/tnanni/ghost_renders/fencing_001_flyover.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("render_flyover")

_NUM_JOINTS = 55                       # SMPL-X joints fed to fusion (root + 54)

# Distinct, reasonably light body colours (RGB 0-1); cycled over person ids.
_PERSON_COLORS = np.array([
    [0.90, 0.36, 0.31],
    [0.32, 0.58, 0.89],
    [0.42, 0.78, 0.45],
    [0.93, 0.72, 0.28],
    [0.68, 0.45, 0.83],
    [0.35, 0.80, 0.80],
], dtype=np.float32)


# ── copied from evaluation/evaluate_egohumans_median.py ─────────────────────
# Kept byte-identical (modulo docstrings) so the rendered bodies are the same
# estimator as the reported numbers.

def _remap_conf_to_packed(conf: np.ndarray, num_joints: int = 55) -> np.ndarray:
    """Canonical-SMPL-X confidence -> `_pack_pose` packed layout."""
    if conf.shape[-1] < num_joints:
        return conf
    out = np.ones(conf.shape[:-1] + (num_joints,), dtype=np.float32)
    out[..., 0:22]  = conf[..., 0:22]
    out[..., 22:37] = conf[..., 25:40]
    out[..., 37:52] = conf[..., 40:55]
    return out


def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """(J,3) axis-angle -> (J,6) = first two ROWS of R (training convention)."""
    from scipy.spatial.transform import Rotation as SciR
    Rm = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()      # (J,3,3)
    return Rm[:, :2, :].reshape(-1, 6)                        # first two rows


def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    """(J,6) -> (J,3) axis-angle; 6 = first two ROWS of R (Gram-Schmidt)."""
    from scipy.spatial.transform import Rotation as SciR
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    Rm = np.stack([b1, b2, b3], axis=1)                      # rows are b1,b2,b3
    return SciR.from_matrix(Rm).as_rotvec()


def _sixd_to_matrix(sixd: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt on the first two rows."""
    a1, a2 = sixd[..., :3], sixd[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def median_fuse(pose_t: torch.Tensor, mask_t: torch.Tensor,
                iters: int = 5, eps: float = 1e-3) -> torch.Tensor:
    """Fuse per-camera poses by the GEODESIC MEDIAN over the visible cameras.

    Identical estimator to `median_fuse` in evaluation/evaluate_rich_median.py
    and evaluation/evaluate_egohumans_median.py — the shipped fusion method.
    """
    R = _sixd_to_matrix(pose_t)                             # (B,T,K,P,J,3,3)
    J = R.shape[4]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)

    empty = (mask_t.sum(dim=2) == 0)[..., None]             # (B,T,P,1) -> over J

    def _chordal(w: torch.Tensor) -> torch.Tensor:
        ww = w[..., None, None]
        M = (R * ww).sum(dim=2) / ww.sum(dim=2).clamp_min(1e-8)
        if bool(empty.any()):
            M = torch.where(empty[..., None, None], eye.expand_as(M), M)
        U, _, Vh = torch.linalg.svd(M)
        d = torch.linalg.det(U @ Vh)                        # (B,T,P,J)  ±1
        D = eye.expand(*d.shape, 3, 3).clone()
        D[..., 2, 2] = d
        return U @ D @ Vh

    w0 = mask_t[..., None].expand(*mask_t.shape, J)         # (B,T,K,P,J)
    R_bar = _chordal(w0)                                    # chordal-mean seed
    for _ in range(iters):
        rel = R @ R_bar[:, :, None].transpose(-1, -2)       # (B,T,K,P,J,3,3)
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        theta = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7))  # (B,T,K,P,J) rad
        R_bar = _chordal(w0 / (theta + eps))

    return torch.cat([R_bar[..., 0, :], R_bar[..., 1, :]], dim=-1)


def _load_clean_tracks(ghost_scene: Path, cam_names, valid_mask, pids,
                       body_dir: str = "body_data_clean", offsets=None):
    """{cam_idx: {pid: {frame: entry}}} from the per-camera track directory.

    EgoHumans ships ``body_data_clean`` (manual track ops applied); EgoExo4D and
    RICH only have ``body_data``, already globally consistent because cross-view
    ReID remaps person ids in place.
    """
    per_cam = {}
    for k, cam in enumerate(cam_names):
        if not valid_mask[k]:
            per_cam[k] = {}
            continue
        bd = ghost_scene / cam / body_dir
        cam_map = {}
        for pid in pids:
            f = bd / f"person_{pid}.npz"
            if not f.exists():
                continue
            d = np.load(str(f), allow_pickle=False)
            fi = d["frame_indices"].astype(int)
            # A desynchronised camera is one whose clock is off: at output
            # frame t it hands over what it actually recorded at t + offset.
            # Re-keying the track here is the whole simulation — nothing is
            # re-estimated, so the reconstruction downstream breaks exactly the
            # way it would have if the streams had never been aligned.
            off = 0 if offsets is None else int(offsets.get(k, 0))
            frames_map = {}
            for t, gfr in enumerate(fi):
                gfr = int(gfr) - off
                e = {"go": d["smplx_global_orient"][t].reshape(3),
                     "bp": d["smplx_body_pose"][t].reshape(63)}
                if "smplx_left_hand_pose" in d.files:
                    e["lh"] = d["smplx_left_hand_pose"][t].reshape(-1, 3)
                if "smplx_right_hand_pose" in d.files:
                    e["rh"] = d["smplx_right_hand_pose"][t].reshape(-1, 3)
                if "smplx_betas" in d.files:
                    e["betas"] = d["smplx_betas"][t][:10]
                if "pred_joint_confidence" in d.files:
                    e["jc"] = _remap_conf_to_packed(
                        d["pred_joint_confidence"][t].reshape(-1)
                    )                                                     # (55,)
                frames_map[int(gfr)] = e
            if frames_map:
                cam_map[pid] = frames_map
        per_cam[k] = cam_map
    return per_cam


def _pack_pose(entry) -> np.ndarray:
    parts = [entry["go"].reshape(1, 3), entry["bp"].reshape(21, 3)]
    parts.append(entry.get("lh", np.zeros((15, 3))).reshape(15, 3))
    parts.append(entry.get("rh", np.zeros((15, 3))).reshape(15, 3))
    aa = np.concatenate(parts, 0).astype(np.float64)
    if aa.shape[0] < _NUM_JOINTS:
        aa = np.concatenate([aa, np.zeros((_NUM_JOINTS - aa.shape[0], 3))], 0)
    return _aa_to_6d(aa)[1:]            # (54,6) root excluded


def _clean_scene_view(ghost_scene: Path, cam_names) -> Path:
    """Temp scene dir: root vggt/scale files symlinked; each cam/body_data ->
    the real cam/body_data_clean. Lets BodyPlacer read the clean tracks."""
    view = Path(tempfile.mkdtemp(prefix="cleanview_"))
    for f in ghost_scene.iterdir():
        if f.is_file():
            os.symlink(f, view / f.name)
    for cam in cam_names:
        clean = ghost_scene / cam / "body_data_clean"
        if clean.is_dir():
            (view / cam).mkdir(parents=True, exist_ok=True)
            os.symlink(clean, view / cam / "body_data")
    return view


# ── bodies: fused pose -> placed SMPL-X vertices ───────────────────────────
def discover_pids(ghost_scene: Path, cam_names,
                  body_dir: str = "body_data_clean") -> list[int]:
    """Global person ids present in any camera's track directory."""
    pids: set[int] = set()
    for cam in cam_names:
        bd = ghost_scene / cam / body_dir
        if not bd.is_dir():
            continue
        for f in bd.glob("person_*.npz"):
            try:
                pids.add(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(pids)


def predict_vertices(ghost_scene: Path, frames, pids, device, smplx_arg,
                     smooth_window: int = 0, min_views: int = 2,
                     body_dir: str = "body_data_clean",
                     crop_meta_path=None, offsets=None):
    """Placed SMPL-X vertices per person per frame, via the shipped median path.

    Copy of ``predict_scene`` in evaluation/evaluate_egohumans_median.py; the
    only change is that the SMPL-24 regression at the end is replaced by the FK
    vertices (which that function already computes and throws away), batched
    over frames instead of one FK call per frame — same arithmetic.

    ``smooth_window`` is the placer's Savitzky-Golay root filter. The EgoHumans
    table is computed with it OFF (0) because those metrics follow CHROMM's
    single-frame protocol; any non-zero value here is the placer's own shipped
    default behaviour but NOT the reported configuration.

    Returns:
        verts_by_pid: {pid: {frame: (V,3) float32 world vertices}}
        faces:        (F,3) int64 SMPL-X faces
        ma_scale:     float, metres per VGGT unit (the placer's scale)
    """
    vggt = np.load(ghost_scene / "vggt_cameras_centered.npz", allow_pickle=False)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    valid_mask = vggt["valid"][0]

    per_cam = _load_clean_tracks(ghost_scene, cam_names, valid_mask, pids, body_dir,
                                 offsets=offsets)
    K, P = len(cam_names), len(pids)
    pid_slot = {p: i for i, p in enumerate(pids)}
    fmin, fmax = frames[0], frames[-1]
    T = fmax - fmin + 1

    # dense per-frame tensors (T,1,K,P,54,6) — temporal length 1 => no temporal attn
    pose  = np.zeros((T, 1, K, P, 54, 6), dtype=np.float32)
    mask  = np.zeros((T, 1, K, P), dtype=np.float32)
    jconf = np.zeros((T, 1, K, P, 54), dtype=np.float32)
    betas_acc = {p: [] for p in pids}
    for k, cam_map in per_cam.items():
        for pid, fm in cam_map.items():
            s = pid_slot[pid]
            for gfr, e in fm.items():
                if fmin <= gfr <= fmax:
                    t = gfr - fmin
                    pose[t, 0, k, s] = _pack_pose(e)
                    mask[t, 0, k, s] = 1.0
                    jc = e.get("jc")
                    jconf[t, 0, k, s] = (
                        jc[1:_NUM_JOINTS] if jc is not None and jc.size >= _NUM_JOINTS else 1.0
                    )
                    if "betas" in e:
                        betas_acc[pid].append(e["betas"])
    betas_by_pid = {p: (np.mean(v, 0).astype(np.float32) if v else np.zeros(10, np.float32))
                    for p, v in betas_acc.items()}
    ncam = mask[:, 0].sum(axis=1)                             # (T,P) views per person

    # Per-person visibility: which cameras actually detected them, and how the
    # per-frame view count is distributed. A root triangulated from few or
    # inconsistent views is the first thing to suspect when a body drifts.
    for pid in pids:
        s = pid_slot[pid]
        per_k = mask[:, 0, :, s].sum(axis=0).astype(int)      # (K,) frames per cam
        hist = np.bincount(ncam[:, s].astype(int), minlength=K + 1)
        logger.info("pid %d: seen in %s | views/frame: %s",
                    pid,
                    ", ".join(f"{c}={int(n)}/{T}" for c, n in zip(cam_names, per_k)),
                    ", ".join(f"{v}cam:{int(h)}" for v, h in enumerate(hist) if h))

    with torch.no_grad():
        chunks = []
        for t0 in range(0, T, 256):
            pt = torch.from_numpy(pose[t0:t0 + 256]).to(device)   # (b,1,K,P,54,6)
            mt = torch.from_numpy(mask[t0:t0 + 256]).to(device)   # (b,1,K,P)
            chunks.append(median_fuse(pt, mt)[:, 0].cpu().numpy())
        fused = np.concatenate(chunks, 0)                         # (T,P,54,6)

    # BodyPlacer hard-codes cam/body_data. EgoHumans keeps its tracks in
    # body_data_clean, so it needs a symlinked view; the other datasets already
    # store them where the placer looks.
    view = _clean_scene_view(ghost_scene, cam_names) if body_dir != "body_data" else ghost_scene
    try:
        placer = BodyPlacer(view, smplx_arg, crop_meta_path=crop_meta_path)
        fused_pose_by_pid = {p: fused[:, pid_slot[p]] for p in pids}

        scale = placer.load_mapanything_scale(filename="mapanything_scale_baseline.npy")
        if scale is None:
            raise RuntimeError("mapanything_scale_baseline.npy missing/mismatched")
        ma_scale = float(np.median(np.asarray(scale)))

        trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
            scale=scale, all_pids=set(pids), pred_betas_by_pid=betas_by_pid,
            fused_pose_by_pid=fused_pose_by_pid, frame_start=fmin,
            smooth_window=smooth_window)

        faces = np.asarray(placer._smplx_model.faces, dtype=np.int64)

        verts_by_pid: dict[int, dict[int, np.ndarray]] = {}
        for pid in pids:
            s = pid_slot[pid]
            placed = [(gfr, trans_dict.get(pid, {}).get(gfr),
                       orient_dict.get(pid, {}).get(gfr)) for gfr in frames]
            placed = [(g, pw, R_m) for g, pw, R_m in placed
                      if pw is not None and R_m is not None]
            if min_views > 1:
                # The root comes from a DLT triangulation across the cameras that
                # saw this person. With a single view the depth along that
                # camera's ray is unconstrained and the root slides along it —
                # a body gliding through the scene. The pipeline's own
                # reconstruction gate is >= 2 views, so hold the render to it.
                n_before = len(placed)
                placed = [(g, pw, R_m) for g, pw, R_m in placed
                          if ncam[g - fmin, s] >= min_views]
                if n_before - len(placed):
                    logger.info("pid %d: dropped %d/%d frames seen by < %d cameras",
                                pid, n_before - len(placed), n_before, min_views)
            if not placed:
                verts_by_pid[pid] = {}
                continue

            idx = np.array([g - fmin for g, _, _ in placed])
            bp_aa = np.stack([_6d_to_aa(fused[t, s, :21]).reshape(63) for t in idx])
            betas_p = np.repeat(betas_by_pid[pid][np.newaxis], len(idx), axis=0)
            J55, verts = placer._smplx_fk(
                betas_p, bp_aa.astype(np.float32),
                np.zeros((len(idx), 3), np.float32), return_verts=True)
            pelvis = np.asarray(J55)[:, 0].astype(np.float64)          # (n,3)
            verts = np.asarray(verts).astype(np.float64)               # (n,V,3)

            out = {}
            for i, (gfr, pw, R_m) in enumerate(placed):
                out[gfr] = ((R_m @ (verts[i] - pelvis[i]).T).T + pw).astype(np.float32)
            verts_by_pid[pid] = out
    finally:
        if view != ghost_scene:              # only the temp symlink view is ours
            shutil.rmtree(view, ignore_errors=True)

    return verts_by_pid, faces, ma_scale


# ── scene: per-frame VGGT pointmap ─────────────────────────────────────────
def monocular_bodies(scene_dir: Path, cam: str, k: int, pids, frames,
                     smplx_arg: str, ma_scale: float, body_dir: str):
    """Bodies from ONE camera, at that camera's own predicted depth.

    Multi-view placement triangulates each joint and solves for the root; with a
    single view there is nothing to intersect, so the only depth available is the
    one SAM3D itself predicted for the crop (``pred_cam_t``).  That is what a
    monocular method has to trust, and its error is exactly what this render is
    meant to show: the body lands wherever that estimate says along the camera
    ray.

    No fusion either — one camera cannot have a median over cameras — so the pose
    is that camera's raw per-frame estimate.
    """
    import smplx as smplx_lib

    f = scene_dir / cam / body_dir
    cam_npz = np.load(scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    extr = cam_npz["extrinsics"]
    out: dict[int, dict[int, np.ndarray]] = {}
    pm = Path(smplx_arg)
    kw = {"model_path": str(pm), "ext": pm.suffix.lstrip(".")} if pm.is_file() \
        else {"model_path": str(pm), "ext": "pkl"}

    for pid in pids:
        npz = f / f"person_{pid}.npz"
        if not npz.exists():
            out[pid] = {}
            continue
        with np.load(str(npz), allow_pickle=False) as d:
            need = {"smplx_body_pose", "smplx_global_orient", "smplx_betas",
                    "pred_cam_t", "frame_indices"}
            if not need.issubset(d.files):
                out[pid] = {}
                continue
            fi = d["frame_indices"].astype(int)
            keep = [i for i, g in enumerate(fi) if g in set(frames)]
            if not keep:
                out[pid] = {}
                continue
            model = smplx_lib.create(**kw, model_type="smplx", gender="neutral",
                                     use_pca=False, num_betas=10,
                                     batch_size=len(keep))
            with torch.no_grad():
                res = model(betas=torch.from_numpy(d["smplx_betas"][keep][:, :10].astype(np.float32)),
                            body_pose=torch.from_numpy(d["smplx_body_pose"][keep].astype(np.float32)),
                            global_orient=torch.from_numpy(d["smplx_global_orient"][keep].astype(np.float32)),
                            return_verts=True)
            verts = res.vertices.numpy().astype(np.float64)      # camera frame, root at 0
            cam_t = d["pred_cam_t"][keep].astype(np.float64)     # SAM3D's own depth
        per_frame = {}
        for i, idx in enumerate(keep):
            gfr = int(fi[idx])
            t_idx = min(max(gfr - 1, 0), extr.shape[0] - 1)
            R = extr[t_idx, k, :3, :3].astype(np.float64)
            tv = extr[t_idx, k, :3, 3].astype(np.float64) * ma_scale
            v_cam = verts[i] + cam_t[i][None, :]
            per_frame[gfr] = ((v_cam - tv) @ R).astype(np.float32)
        out[pid] = per_frame
        logger.info("mono %s: person %d placed in %d frames", cam, pid, len(per_frame))
    return out


def load_depth_context(scene_dir: Path, ma_scale: float):
    """VGGT depth + camera arrays, with the placer's metric scale attached.

    The scale is the scalar the placer placed with — NOT re-derived from camera
    translation ratios. ``load_mapanything_scale`` already broadcasts one robust
    median over the scene, so a single float is the whole story.
    """
    depth_npz = np.load(scene_dir / "vggt_depth_centered.npz", mmap_mode="r")
    cam_npz   = np.load(scene_dir / "vggt_cameras_centered.npz")
    extr = cam_npz["extrinsics"]
    cam_valid = cam_npz["valid"]
    if not cam_valid.any():
        cam_valid = ~np.isnan(extr[:, :, 0, 0])
    names = [n.decode() if isinstance(n, bytes) else n for n in cam_npz["camera_names"]]
    return {
        "depth_mm":    depth_npz["depth"],        # (T,K,h,w) uint16, VGGT units x1000
        "depth_conf":  depth_npz["depth_conf"],   # (T,K,h,w) float16
        "depth_valid": depth_npz["depth_valid"],  # (T,K) bool
        "extr": extr, "intr": cam_npz["intrinsics"],
        "oc": cam_npz["original_coords"],         # (T,K,4) valid region in image px
        "cam_valid": cam_valid, "names": names, "scale": float(ma_scale),
    }


def load_person_boxes(scene_dir: Path, cam_name: str) -> dict[int, list]:
    """{abs_frame: [(x1,y1,x2,y2), …]} person boxes in original image pixels.

    Most EgoHumans scenes ship no ``mask_data.npz`` (fencing has none) but every
    per-person npz carries a per-frame ``bbox`` from the same segmentation.
    Union of both track sets: ANY detection occludes the meshes, not only the
    ReID-kept people.
    """
    boxes: dict[int, list] = {}
    for sub in ("body_data_clean", "body_data"):
        bd = Path(scene_dir) / cam_name / sub
        if not bd.is_dir():
            continue
        for f in sorted(bd.glob("person_*.npz")):
            try:
                d = np.load(str(f), allow_pickle=False)
            except Exception:
                continue
            if "bbox" not in d.files or "frame_indices" not in d.files:
                continue
            for fr, bb in zip(d["frame_indices"].astype(int), d["bbox"]):
                boxes.setdefault(int(fr), []).append(np.asarray(bb, dtype=np.float32))
    return boxes


def _read_frame(frames_root: Path | None, cam_name: str, abs_frame: int):
    """BGR image for (camera, absolute frame), or None."""
    if frames_root is None:
        return None
    for pat in ("images_undistorted/frames", "images_undistorted", "frames",
                "images", "."):
        d = frames_root / cam_name / pat
        if not d.is_dir():
            continue
        # RICH names frames <frame>_<camera suffix>.jpg, e.g. cam_01/00001_01.jpg
        suffix = cam_name.rsplit("_", 1)[-1]
        for name in (f"{abs_frame:05d}.jpg", f"{abs_frame:06d}.jpg",
                     f"frame_{abs_frame:06d}.jpg",          # EgoExo4D
                     f"{abs_frame:05d}_{suffix}.jpg",       # RICH
                     f"{abs_frame:05d}.jpeg", f"{abs_frame:06d}.jpeg",
                     f"{abs_frame:05d}.png", f"{abs_frame:08d}.jpg"):
            p = d / name
            if p.exists():
                return cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None


def _voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float,
                      snap: bool = True):
    """Keep one point per ``voxel``-sized cell, optionally snapped to its centre.

    Keeping the arbitrary first hit makes a STATIC surface boil: the winning
    point is re-picked independently every frame, so the rendered point set
    churns even where the geometry never moved — an artefact of this thinning,
    not of the reconstruction. Snapping to the cell centre makes the same cell
    render at the same place every frame. It quantises positions to the voxel
    grid (<= voxel/2 displacement); colours still come from the surviving point.
    """
    if len(pts) == 0 or voxel <= 0.0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    if snap:
        return ((keys[idx] + 0.5) * voxel).astype(np.float32), cols[idx]
    return pts[idx], cols[idx]


def body_silhouette(verts_list, ctx, t, k, stride, h_d, w_d, dilate: int):
    """(h_d, w_d) bool mask of the depth grid covered by the placed bodies.

    Projects the SMPL-X vertices with the same extrinsics/intrinsics and the same
    scaled translation that ``depth_cloud`` unprojects with, so the mask lands
    exactly on the depth pixels those bodies produced. Tighter than a bounding
    box, and it still carves a person a given camera failed to detect.
    """
    if not verts_list:
        return None
    v = np.concatenate(verts_list, 0).astype(np.float64)
    R_d = ctx["extr"][t, k, :3, :3].astype(np.float64)
    t_vec = ctx["extr"][t, k, :3, 3].astype(np.float64) * ctx["scale"]
    pc = v @ R_d.T + t_vec
    z = pc[:, 2]
    ok = z > 1e-4
    if not ok.any():
        return None
    intr = ctx["intr"][t, k]
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    iu = np.round((fx * pc[ok, 0] / z[ok] + cx) / stride).astype(np.int32)
    iv = np.round((fy * pc[ok, 1] / z[ok] + cy) / stride).astype(np.int32)
    inb = (iu >= 0) & (iu < w_d) & (iv >= 0) & (iv < h_d)
    if not inb.any():
        return None
    m = np.zeros((h_d, w_d), np.uint8)
    m[iv[inb], iu[inb]] = 1
    if dilate > 0:
        # This marks where VERTICES land, so its density depends on how many
        # grid pixels the body spans: at --depth-stride 1 a person covers ~4x
        # the pixels of stride 2 and the same 10475 vertices leave holes. Close
        # first to fill those gaps, then dilate to cover the silhouette rim.
        ker = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
        m = cv2.dilate(m, ker)
    return m.astype(bool)


_CAM_COLORS = np.array([
    [0.95, 0.25, 0.25], [0.20, 0.55, 0.95],
    [0.20, 0.75, 0.35], [0.95, 0.65, 0.10],
    [0.70, 0.35, 0.85], [0.15, 0.75, 0.75],
], dtype=np.float32)


def camera_frustums(ctx, t, depth_m: float, spacing: float = 0.01):
    """Wireframe frustums for the ESTIMATED cameras at frame t, as points.

    Drawn from the same per-frame VGGT extrinsics/intrinsics the cloud is
    unprojected with, and scaled by the same MapAnything scalar, so the frustums
    sit where the pipeline actually thinks the cameras are — they are a view of
    the estimate, not an annotation drawn over it.

    Edges are sampled into points rather than rasterised as lines: the shared
    z-buffer then occludes them against the cloud for free.
    """
    h, w = ctx["depth_mm"].shape[2], ctx["depth_mm"].shape[3]
    s = ctx["scale"]
    pts, cols = [], []
    for k in range(ctx["extr"].shape[1]):
        if not ctx["cam_valid"][t, k]:
            continue
        intr = ctx["intr"][t, k]
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        R = ctx["extr"][t, k, :3, :3].astype(np.float64)
        tv = ctx["extr"][t, k, :3, 3].astype(np.float64) * s

        centre = -R.T @ tv                                   # camera centre, world
        corners = []
        for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
            pc = np.array([(u - cx) / fx * depth_m, (v - cy) / fy * depth_m, depth_m])
            corners.append(R.T @ (pc - tv))

        edges = [(centre, c) for c in corners]               # centre -> image plane
        edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]   # image rect
        col = _CAM_COLORS[k % len(_CAM_COLORS)]
        for a, b in edges:
            n = max(2, int(np.linalg.norm(b - a) / spacing))
            seg = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
            pts.append(seg.astype(np.float32))
            cols.append(np.repeat(col[None], n, axis=0))
    if not pts:
        return None
    return np.concatenate(pts, 0), np.concatenate(cols, 0)


def radius_outlier_filter(pts: torch.Tensor, cols: torch.Tensor, voxel: float,
                          min_neighbors: int, device):
    """Drop points whose voxel neighbourhood is nearly empty.

    A point on a real surface has many neighbours within a few centimetres; a
    point VGGT hallucinated in mid-air has almost none. Counting occupied cells
    in the 26-neighbourhood separates them without any smoothing — this only
    ever removes points, it never moves or invents one.

    Costs one searchsorted per offset because the cloud is already snapped to
    the voxel grid, so no KD-tree is needed.

    Thin real structures (curtain edges, leaves, fine lego parts) are sparse
    too, so a high threshold eats them — keep it low.
    """
    if pts.numel() == 0 or min_neighbors <= 0 or voxel <= 0.0:
        return pts, cols
    k = torch.floor(pts / voxel).to(torch.int64)
    k = k - k.min(0).values + 1                       # shift so -1 offsets stay >= 0
    n = k.max(0).values + 3
    code = (k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2]
    occupied = torch.unique(code)                     # sorted
    cnt = torch.zeros(pts.shape[0], dtype=torch.int16, device=device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nb = ((k[:, 0] + dx) * n[1] + (k[:, 1] + dy)) * n[2] + (k[:, 2] + dz)
                pos = torch.searchsorted(occupied, nb).clamp(max=occupied.numel() - 1)
                cnt += (occupied[pos] == nb).to(torch.int16)
    keep = cnt >= min_neighbors
    return pts[keep], cols[keep]


def cull_points_near_bodies(pts: torch.Tensor, cols: torch.Tensor,
                            verts_list, margin: float, device):
    """Drop cloud points lying within ``margin`` of any placed body.

    The per-camera silhouette carve is a 2D approximation: it only removes what
    a body projects onto in that view, so any placement or depth error leaves a
    rim of the photographed person behind, which then renders as squares on top
    of their own mesh. This cull is view-independent — the people are drawn as
    meshes, so their points do not belong in the cloud from ANY camera.

    AABB prefilter, then exact nearest-vertex distance on the survivors.
    """
    if not verts_list or pts.numel() == 0 or margin <= 0.0:
        return pts, cols
    keep = torch.ones(pts.shape[0], dtype=torch.bool, device=device)
    for v in verts_list:
        vt = torch.as_tensor(v, dtype=torch.float32, device=device)
        lo, hi = vt.min(0).values - margin, vt.max(0).values + margin
        inbox = ((pts >= lo) & (pts <= hi)).all(1) & keep
        idx = torch.nonzero(inbox, as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        for c0 in range(0, idx.numel(), 4096):        # chunk: |chunk| x V distances
            ck = idx[c0:c0 + 4096]
            d = torch.cdist(pts[ck], vt).min(dim=1).values
            keep[ck] &= d > margin
    return pts[keep], cols[keep]


def depth_cloud(ctx, t, k, abs_frame, bgr, stride, conf_thr, boxes,
                box_pad: float, voxel: float, silhouette=None,
                disc_rel: float = 0.0, voxel_snap: bool = True):
    """Unproject camera k's depth at frame t into VGGT world space (RDF).

    Metric conversion is ``depth/1000 * ma_scale`` with the placer's scale, and
    the extrinsic translation gets the same factor — identical to how the bodies
    were placed.

    ``disc_rel`` drops pixels whose depth disagrees with a 4-neighbour by more
    than that relative amount: those are the flying pixels VGGT interpolates
    across occlusion boundaries, which otherwise hang in mid-air in front of the
    bodies. ``silhouette`` carves the pixels the placed bodies cover. Both only
    ever REMOVE points.
    """
    depth_mm = ctx["depth_mm"]
    off = int(ctx.get("offsets", {}).get(k, 0))
    if off:
        t = t + off
        if t < 0 or t >= depth_mm.shape[0]:
            return None
    if (t >= depth_mm.shape[0] or k >= depth_mm.shape[1]
            or not ctx["depth_valid"][t, k] or not ctx["cam_valid"][t, k]):
        return None

    s = ctx["scale"]
    d = depth_mm[t, k][::stride, ::stride].astype(np.float32) / 1000.0 * s
    conf = ctx["depth_conf"][t, k][::stride, ::stride].astype(np.float32)
    h_d, w_d = d.shape
    vv, uu = np.mgrid[0:h_d, 0:w_d].astype(np.float32) * stride
    x1, y1, x2, y2 = ctx["oc"][t, k]

    keep = (d > 1e-4) & (conf >= conf_thr) \
        & (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2)

    if disc_rel > 0.0:
        # Flying pixels: a valid surface point agrees with its neighbours; a
        # point interpolated across an occlusion edge does not.
        jump = np.zeros_like(d, bool)
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            nb = np.roll(d, sh, axis=ax)
            rel = np.abs(d - nb) / np.maximum(np.minimum(d, nb), 1e-6)
            edge = np.zeros_like(d, bool)
            if ax == 0:
                edge[0 if sh > 0 else -1, :] = True
            else:
                edge[:, 0 if sh > 0 else -1] = True
            jump |= (rel > disc_rel) & ~edge
        keep &= ~jump

    if silhouette is not None:
        keep &= ~silhouette

    # Carve out detected people so the depth surface cannot occlude the meshes.
    if boxes is not None and bgr is not None:
        fh, fw = bgr.shape[:2]
        for bb in boxes.get(abs_frame, []):
            bx1, by1, bx2, by2 = [float(v) for v in bb[:4]]
            pw = box_pad * max(1.0, bx2 - bx1)
            ph = box_pad * max(1.0, by2 - by1)
            # image px -> depth-grid px (inverse of the colour lookup below)
            gu1 = x1 + (bx1 - pw) / fw * (x2 - x1)
            gu2 = x1 + (bx2 + pw) / fw * (x2 - x1)
            gv1 = y1 + (by1 - ph) / fh * (y2 - y1)
            gv2 = y1 + (by2 + ph) / fh * (y2 - y1)
            keep &= ~((uu >= gu1) & (uu <= gu2) & (vv >= gv1) & (vv <= gv2))

    if not keep.any():
        return None
    u, v, z = uu[keep], vv[keep], d[keep]

    intr = ctx["intr"][t, k]
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    pts_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)

    R_d   = ctx["extr"][t, k, :3, :3]
    t_vec = ctx["extr"][t, k, :3, 3] * s
    pts_world = ((pts_cam - t_vec) @ R_d).astype(np.float32)   # world == RDF, no flip

    if bgr is not None:
        fh, fw = bgr.shape[:2]
        iu = np.clip(((u - x1) / (x2 - x1) * fw).astype(np.int32), 0, fw - 1)
        iv = np.clip(((v - y1) / (y2 - y1) * fh).astype(np.int32), 0, fh - 1)
        cols = bgr[iv, iu][:, ::-1].astype(np.float32) / 255.0     # BGR -> RGB
    else:
        cols = np.full((len(z), 3), 0.5, np.float32)

    return _voxel_downsample(pts_world, cols.astype(np.float32), voxel, voxel_snap)


# ── flyover camera path ────────────────────────────────────────────────────
def world_up(ctx) -> np.ndarray:
    """Up direction from the estimated cameras: -mean(camera down axis).

    The VGGT world gauge is camera-0's frame, so its Y axis is only roughly
    gravity. Averaging every camera's own down axis (row 1 of R_w2c, expressed
    in world) is a better rig-derived up and uses nothing but the pipeline's own
    camera estimates.
    """
    down = []
    for k in range(ctx["extr"].shape[1]):
        val = ctx["cam_valid"][:, k]
        if not val.any():
            continue
        down.append(ctx["extr"][val, k, 1, :3].mean(0))
    if not down:
        return np.array([0.0, -1.0, 0.0], np.float32)
    d = np.mean(down, 0)
    n = np.linalg.norm(d)
    return (-d / n).astype(np.float32) if n > 1e-6 else np.array([0.0, -1.0, 0.0], np.float32)


def smooth_track(x: np.ndarray, win: int) -> np.ndarray:
    """Centred box filter along axis 0 with edge padding (camera path only)."""
    if win <= 1 or len(x) < 3:
        return x
    win = min(win | 1, len(x) if len(x) % 2 else len(x) - 1)
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(win, np.float32) / win
    return np.stack([np.convolve(xp[:, i], ker, mode="valid") for i in range(x.shape[1])], 1)


def orbit_eyes(targets: np.ndarray, up: np.ndarray, radius: float,
               turns: float, elev_deg: float, swing_deg: float,
               az0_deg: float) -> np.ndarray:
    """Camera positions orbiting each frame's target on the plane normal to up."""
    n = len(targets)
    e1 = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(np.dot(e1, up))) > 0.9:
        e1 = np.array([0.0, 0.0, 1.0], np.float32)
    e1 = e1 - np.dot(e1, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    ph = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
    az = np.deg2rad(az0_deg + 360.0 * turns * ph)
    el = np.deg2rad(elev_deg + swing_deg * 0.5 * (1.0 - np.cos(2 * np.pi * ph)))

    horiz = (np.cos(az)[:, None] * e1[None] + np.sin(az)[:, None] * e2[None])
    dirs = np.cos(el)[:, None] * horiz + np.sin(el)[:, None] * up[None]
    return targets + radius * dirs


# ── rendering: minimal torch rasteriser ────────────────────────────────────
# PyTorch3D in this environment ships WITHOUT its compiled `_C` extension, so
# none of its rasterisers can run. Points and triangles are rasterised here
# instead, straight into ONE shared z-buffer — which also makes cloud/mesh
# compositing exact and free instead of a second blending pass.
#
# Camera convention is the CV one the rest of the pipeline uses: x right,
# y down, z forward, u = fx·x/z + cx.

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->camera rotation; rows are the camera x (right), y (down), z (fwd)."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    down = -up - np.dot(-up, f) * f                 # orthogonalise against fwd
    n = np.linalg.norm(down)
    if n < 1e-6:                                    # degenerate: looking along up
        alt = np.array([0.0, 0.0, 1.0], np.float32)
        down = alt - np.dot(alt, f) * f
        n = np.linalg.norm(down)
    down = down / n
    right = np.cross(down, f)                       # y × z = x
    return np.stack([right, down, f], 0).astype(np.float32)


def intrinsics(W: int, H: int, fov_deg: float):
    """Square-pixel pinhole from a VERTICAL field of view."""
    fy = 0.5 * H / np.tan(0.5 * np.deg2rad(fov_deg))
    return float(fy), float(fy), W * 0.5, H * 0.5


def _project(pts: torch.Tensor, R: torch.Tensor, eye: torch.Tensor, intr):
    fx, fy, cx, cy = intr
    pc = (pts - eye) @ R.T                          # (N,3) camera coords
    z = pc[:, 2]
    zc = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    return fx * pc[:, 0] / zc + cx, fy * pc[:, 1] / zc + cy, z


def vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals."""
    tri = v[f]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = torch.zeros_like(v)
    for c in range(3):
        n.index_add_(0, f[:, c], fn)
    return n / (n.norm(dim=1, keepdim=True) + 1e-12)


def point_fragments(pts, cols, R, eye, intr, W, H, znear, size_m, r_max):
    """Screen fragments for a point cloud; splat radius follows perspective."""
    fx, fy, cx, cy = intr
    u, v, z = _project(pts, R, eye, intr)
    ok = (z > znear) & (u > -r_max) & (u < W + r_max) & (v > -r_max) & (v < H + r_max)
    if not bool(ok.any()):
        return None
    u, v, z, cols = u[ok], v[ok], z[ok], cols[ok]

    r = torch.clamp(torch.round(fy * size_m / z), 0, r_max).long()
    ui, vi = torch.round(u).long(), torch.round(v).long()

    pix_l, z_l, col_l = [], [], []
    for rv in torch.unique(r):
        sel = r == rv
        rr = int(rv)
        d = torch.arange(-rr, rr + 1, device=pts.device)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        off = (dy.reshape(-1), dx.reshape(-1))
        m = off[0].numel()
        px = (ui[sel][:, None] + off[1][None]).reshape(-1)
        py = (vi[sel][:, None] + off[0][None]).reshape(-1)
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not bool(inb.any()):
            continue
        pix_l.append((py[inb] * W + px[inb]))
        z_l.append(z[sel].repeat_interleave(m)[inb])
        col_l.append(cols[sel].repeat_interleave(m, dim=0)[inb])
    if not pix_l:
        return None
    return torch.cat(pix_l), torch.cat(z_l), torch.cat(col_l)


def mesh_fragments(verts, faces, vcol, R, eye, intr, W, H, znear,
                   ambient, diffuse, max_area=8192):
    """Screen fragments for a triangle mesh: barycentric fill + Lambert shading.

    Every face is expanded over its screen bounding box, so cost tracks covered
    area, not image size. Depth is perspective-correct; normals are interpolated
    from area-weighted vertex normals and shaded two-sided by a head light.
    """
    fx, fy, cx, cy = intr
    u, v, z = _project(verts, R, eye, intr)
    nrm = vertex_normals(verts, faces)

    uf = u[faces]                                            # (F,3)
    vf = v[faces]
    zf = z[faces]
    live = ((zf > znear).all(1)
            & (uf.max(1).values >= 0) & (uf.min(1).values <= W - 1)
            & (vf.max(1).values >= 0) & (vf.min(1).values <= H - 1))
    if not bool(live.any()):
        return None
    faces = faces[live]; uf, vf, zf = uf[live], vf[live], zf[live]

    umin = uf.min(1).values.floor().clamp(0, W - 1).long()
    umax = uf.max(1).values.ceil().clamp(0, W - 1).long()
    vmin = vf.min(1).values.floor().clamp(0, H - 1).long()
    vmax = vf.max(1).values.ceil().clamp(0, H - 1).long()
    bw, bh = umax - umin + 1, vmax - vmin + 1
    n = bw * bh
    keep = (n > 0) & (n <= max_area)
    if not bool(keep.any()):
        return None
    faces = faces[keep]
    uf, vf, zf = uf[keep], vf[keep], zf[keep]
    umin, vmin, bw, n = umin[keep], vmin[keep], bw[keep], n[keep]

    starts = torch.cumsum(n, 0) - n
    fidx = torch.repeat_interleave(torch.arange(n.numel(), device=verts.device), n)
    local = torch.arange(int(n.sum()), device=verts.device) - starts[fidx]
    wrep = bw[fidx]
    px = umin[fidx] + local % wrep
    py = vmin[fidx] + local // wrep

    xs, ys = px.float() + 0.5, py.float() + 0.5
    x0, x1, x2 = uf[fidx, 0], uf[fidx, 1], uf[fidx, 2]
    y0, y1, y2 = vf[fidx, 0], vf[fidx, 1], vf[fidx, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    safe = torch.where(area.abs() < 1e-9, torch.full_like(area, 1e-9), area)
    l0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / safe
    l1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / safe
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (area.abs() >= 1e-9)
    if not bool(inside.any()):
        return None

    fidx, px, py = fidx[inside], px[inside], py[inside]
    l0, l1, l2 = l0[inside], l1[inside], l2[inside]
    z0, z1, z2 = zf[fidx, 0], zf[fidx, 1], zf[fidx, 2]
    iz = l0 / z0 + l1 / z1 + l2 / z2
    zpix = 1.0 / torch.where(iz.abs() < 1e-9, torch.full_like(iz, 1e-9), iz)
    b = torch.stack([l0 / z0, l1 / z1, l2 / z2], 1) * zpix[:, None]  # persp-correct

    fv = faces[fidx]                                                # (M,3)
    nrm_p = (nrm[fv] * b[..., None]).sum(1)
    nrm_p = nrm_p / (nrm_p.norm(dim=1, keepdim=True) + 1e-12)

    # Head light: shade against the viewing ray through each pixel.
    dir_cam = torch.stack([(px.float() + 0.5 - cx) / fx,
                           (py.float() + 0.5 - cy) / fy,
                           torch.ones_like(zpix)], 1)
    dir_world = dir_cam @ R                                          # R^T applied
    dir_world = dir_world / (dir_world.norm(dim=1, keepdim=True) + 1e-12)
    lam = (nrm_p * dir_world).sum(1).abs()                           # two-sided
    shade = (ambient + diffuse * lam).clamp(0.0, 1.5)[:, None]

    col = (vcol[fv] * b[..., None]).sum(1) * shade
    return py * W + px, zpix, col.clamp(0.0, 1.0)


def render_frame(frag_list, W, H, bg_rgb, device):
    """Resolve fragments through one z-buffer -> (H,W,3) uint8 BGR for cv2.

    ``frag_list`` holds ``(fragments, bias)`` pairs. The bias is subtracted from
    that group's depth for the comparison only: giving the body meshes a small
    bias makes a cloud point win a pixel only if it is in front by more than
    that margin, which kills the residual speckle that sits within centimetres
    of a body surface without hiding anything genuinely in front of it.
    """
    img = torch.tensor(bg_rgb, dtype=torch.float32, device=device).repeat(H * W, 1)
    frags = [(f, b) for f, b in frag_list if f is not None]
    if frags:
        pix = torch.cat([f[0] for f, _ in frags])
        z   = torch.cat([f[1] - b for f, b in frags])
        col = torch.cat([f[2] for f, _ in frags])
        zbuf = torch.full((H * W,), float("inf"), device=device)
        zbuf.scatter_reduce_(0, pix, z, "amin", include_self=True)
        win = z <= zbuf[pix] + 1e-6
        img[pix[win]] = col[win]
    out = (img.view(H, W, 3).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(out[:, :, ::-1])                     # RGB -> BGR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("egohumans", "egoexo", "rich"),
                    default="egohumans",
                    help="selects the track directory, frame-id convention and "
                         "placer arguments; the fusion/placement estimator is "
                         "the same geodesic median in all three.")
    ap.add_argument("--orbit-frames", type=int, default=0,
                    help="render this many output frames from a SINGLE scene "
                         "frame, orbiting a frozen reconstruction (0 = off). "
                         "Required for EgoExo4D, whose source videos are gone.")
    ap.add_argument("--crop-meta", type=Path, default=None,
                    help="RICH crop_meta.json for the SAM3D kp2d offset "
                         "correction; without it the placer runs offsets=0.")
    ap.add_argument("--scene", required=True, type=Path,
                    help="ghost output scene dir (has vggt_*_centered.npz).")
    ap.add_argument("--frames-root", type=Path, default=None,
                    help="dir holding <cam>/images_undistorted/frames/%%05d.jpg; "
                         "without it the cloud renders grey.")
    ap.add_argument("--out", required=True, type=Path, help="output .mp4")
    ap.add_argument("--smplx-model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame.")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))

    ap.add_argument("--depth-stride", type=int, default=2,
                    help="subsample the depth grid by this factor.")
    ap.add_argument("--conf-thr", type=float, default=5.0,
                    help="VGGT depth confidence floor. NOT 0-1: measured range on "
                         "EgoHumans is 1.0-30.5 (median ~7.7), so any value below "
                         "1.0 is a no-op. 5.0 keeps ~73%% of pixels and removes "
                         "most mid-air floaters — the points VGGT itself doubts.")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="voxel thinning of the cloud, metres (0 = off).")
    ap.add_argument("--voxel-snap", action=argparse.BooleanOptionalAction, default=True,
                    help="snap thinned points to their voxel centre so static "
                         "surfaces stop boiling between frames.")
    ap.add_argument("--point-size", type=float, default=0.03,
                    help="splat radius in METRES; projected per point.")
    ap.add_argument("--point-px-max", type=int, default=4,
                    help="cap on the projected splat radius, pixels.")
    ap.add_argument("--clip-front", type=float, default=-1.0,
                    help="drop cloud points nearer the camera than "
                         "dist(eye,target) - this, in metres: removes the wall / "
                         "ceiling the orbit passes behind. -1 = auto (scene span), "
                         "0 = off. Bodies are never clipped.")
    ap.add_argument("--mono", default=None, metavar="CAM",
                    help="single-view mode: only this camera contributes depth, and "
                         "bodies are placed at SAM3D's own monocular depth "
                         "(pred_cam_t) through this camera's pose instead of by "
                         "multi-view triangulation. Everything the camera cannot "
                         "see is simply absent — that is the point.")
    ap.add_argument("--fake-drift", default=None, metavar="DRIFT_M,JITTER_M[,SEED]",
                    help="displace each placed body by a drift that grows over the "
                         "clip plus per-frame jitter. FABRICATED: the placer's own "
                         "output is overwritten after the fact, so nothing here is "
                         "a property of the reconstruction.")
    ap.add_argument("--cam-noise", default=None, metavar="DEG,CM[,SEED]",
                    help="perturb every camera pose by a random rotation of DEG "
                         "degrees and a translation of CM centimetres. SYNTHETIC: "
                         "this is a hand-made degradation for illustration, not a "
                         "run of the pipeline under any condition.")
    ap.add_argument("--desync", default=None, metavar="SPEC",
                    help="render WITHOUT synchronisation: 'cam03:-62,cam04:41' "
                         "shifts each camera's stream by that many frames, both "
                         "its tracks and its point cloud. Nothing is re-estimated "
                         "— this is what the scene looks like unaligned.")
    ap.add_argument("--carve", choices=("mesh", "bbox", "both"), default="mesh",
                    help="how to remove people from the cloud: projected body "
                         "silhouette (tight), per-camera detection boxes "
                         "(rectangular, leaves big holes), or both.")
    ap.add_argument("--carve-dilate", type=int, default=3,
                    help="dilation of the body silhouette, depth-grid pixels.")
    ap.add_argument("--min-neighbors", type=int, default=0,
                    help="drop cloud points with fewer than this many occupied "
                         "cells in their 26-voxel neighbourhood (0 = off). "
                         "Removes points floating in mid-air; too high also eats "
                         "genuinely thin structures.")
    ap.add_argument("--cull-3d", type=float, default=0.08,
                    help="drop cloud points within this many metres of any "
                         "placed body (view-independent; 0 = off). This is what "
                         "removes the photographed people from the cloud.")
    ap.add_argument("--mesh-bias", type=float, default=0.15,
                    help="a cloud point must be in front of a body by this many "
                         "metres to win the pixel; kills residual speckle.")
    ap.add_argument("--disc-rel", type=float, default=0.0,
                    help="drop depth pixels disagreeing with a neighbour by more "
                         "than this (flying pixels at occlusion edges); 0 = off. "
                         "OFF by default: it cannot tell a flying pixel from a "
                         "real object edge, so it deletes foreground structure "
                         "(it erased the barrier in RICH ParkingLot2 overfence). "
                         "--conf-thr already removes the floaters it targeted.")
    ap.add_argument("--pids", type=int, nargs="+", default=None,
                    help="person ids to render (default: every track on disk). "
                         "evaluate_egohumans_median.py derives its person list "
                         "from the GT aria ids, so extra tracks found here are "
                         "never scored and their placement is unmeasured.")
    ap.add_argument("--min-views", type=int, default=2,
                    help="skip a person on frames seen by fewer cameras than "
                         "this; 1 = draw single-view bodies, whose root is "
                         "unconstrained along the viewing ray.")
    ap.add_argument("--root-smooth", type=int, default=15,
                    help="placer Savitzky-Golay root smoothing window. 0 = the "
                         "CHROMM single-frame protocol the EgoHumans table uses.")
    ap.add_argument("--keep-person-pixels", action="store_true",
                    help="do NOT carve detected people out of the cloud.")
    ap.add_argument("--no-points", action="store_true", help="bodies only.")
    ap.add_argument("--no-bodies", action="store_true", help="pointmap only.")
    ap.add_argument("--draw-cameras", action="store_true",
                    help="draw the estimated camera frustums, one colour each.")
    ap.add_argument("--frustum-depth", type=float, default=0.35,
                    help="frustum length in metres.")

    ap.add_argument("--orbit-turns", type=float, default=0.35,
                    help="full revolutions over the whole clip.")
    ap.add_argument("--azimuth0", type=float, default=0.0)
    ap.add_argument("--elev-deg", type=float, default=18.0)
    ap.add_argument("--elev-swing", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=None, help="orbit radius, metres.")
    ap.add_argument("--radius-scale", type=float, default=1.0)
    ap.add_argument("--span-pct", type=float, default=90.0,
                    help="percentile of body-centroid spread used to size the "
                         "orbit; 100 = worst case, and one stray body sets the shot.")
    ap.add_argument("--target-smooth", type=int, default=61,
                    help="box-filter width on the look-at target, frames.")
    ap.add_argument("--ssaa", type=int, default=2,
                    help="supersampling factor: rasterise at this multiple of "
                         "--size and box-filter down. 2 costs ~4x per frame.")
    ap.add_argument("--fov", type=float, default=55.0, help="VERTICAL fov, degrees.")
    ap.add_argument("--ambient", type=float, default=0.35)
    ap.add_argument("--diffuse", type=float, default=0.75)
    ap.add_argument("--bg", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                    help="background RGB 0-1; default white for figures.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene = args.scene

    vggt = np.load(scene / "vggt_cameras_centered.npz", allow_pickle=False)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    T_total = vggt["extrinsics"].shape[0]

    # Absolute frame ids are 1-based in EgoHumans; depth index t = frame - 1.
    body_dir = "body_data_clean" if args.dataset == "egohumans" else "body_data"

    # Frame ids come from the tracks, never assumed. Absolute ids are NOT always
    # 1-based and the depth/camera arrays hold exactly the covered span:
    #   EgoHumans lego 001  1..601   T=601
    #   EgoExo   covid_02_4 67..67   T=1     (videos gone, single GT frame)
    #   RICH     Gym_010     1..779  T=779
    #   RICH     ParkingLot2 150..559 T=410  <- 1-based indexing renders nothing
    # so the depth index is f - first_track_frame in every case.
    track_frames = sorted({int(f) for cam in cam_names
                           for p in (scene / cam / body_dir).glob("person_*.npz")
                           for f in np.load(str(p), allow_pickle=False)["frame_indices"]})
    if not track_frames:
        raise SystemExit(f"no frame_indices under {scene}/*/{body_dir}")
    origin = track_frames[0]
    if len(track_frames) != T_total:
        logger.warning("tracks cover %d frames but depth/cameras hold %d — "
                       "indices may be misaligned", len(track_frames), T_total)

    f0 = max(origin, args.frame_start) if args.frame_start is not None else origin
    f1 = min(track_frames[-1], args.frame_end) if args.frame_end is not None \
        else track_frames[-1]
    frames_all = list(range(f0, f1 + 1))
    frames = frames_all[::max(1, args.stride)]

    if args.orbit_frames > 0:
        frames = [frames[0]] * args.orbit_frames
        logger.info("freeze-frame orbit: %d output frames from scene frame %d",
                    args.orbit_frames, frames[0])

    # Absolute frame id -> index into the per-frame depth/camera arrays.
    depth_index = {f: f - origin for f in frames_all}
    logger.info("scene=%s dataset=%s cams=%s scene-frames=%d..%d (%d rendered)",
                scene.name, args.dataset, cam_names,
                frames_all[0], frames_all[-1], len(frames))

    pids = discover_pids(scene, cam_names, body_dir)
    if not pids:
        raise SystemExit(f"no body_data_clean/person_*.npz under {scene}")
    if args.pids:
        missing = sorted(set(args.pids) - set(pids))
        if missing:
            raise SystemExit(f"--pids {missing} not found on disk (have {pids})")
        logger.info("person ids: %s (of %s on disk)", sorted(args.pids), pids)
        pids = sorted(args.pids)
    else:
        logger.info("person ids: %s", pids)

    offsets = {}
    if args.desync:
        by_name = {c: i for i, c in enumerate(cam_names)}
        for part in args.desync.split(","):
            name, _, val = part.partition(":")
            name, val = name.strip(), val.strip()
            if name not in by_name:
                sys.exit(f"--desync: unknown camera {name!r}; have {cam_names}")
            offsets[by_name[name]] = int(val)
        logger.warning("DESYNCHRONISED render: %s — tracks and clouds are read at "
                       "t+offset per camera, so cross-camera triangulation mixes "
                       "instants. This is the unaligned scene, not a GHOST result.",
                       ", ".join(f"{c}{offsets.get(i, 0):+d}"
                                 for i, c in enumerate(cam_names)))

    # Bodies use the full contiguous range (the placer needs it); render subset after.
    verts_by_pid, faces, ma_scale = predict_vertices(
        scene, frames_all, pids, device, str(args.smplx_model),
        smooth_window=args.root_smooth, min_views=args.min_views,
        body_dir=body_dir, crop_meta_path=args.crop_meta, offsets=offsets)
    logger.info("MapAnything baseline scale = %.4f m/unit", ma_scale)

    mono_k = None
    if args.mono:
        if args.mono not in cam_names:
            sys.exit(f"--mono: unknown camera {args.mono!r}; have {cam_names}")
        mono_k = cam_names.index(args.mono)
        verts_by_pid = monocular_bodies(scene, cam_names[mono_k], mono_k, pids,
                                        frames_all, str(args.smplx_model), ma_scale,
                                        body_dir)
        logger.warning("MONOCULAR mode on %s: bodies come from that camera's own "
                       "SAM3D depth, not from triangulation, and only its depth map "
                       "feeds the cloud.", args.mono)

    if args.fake_drift:
        parts = [x.strip() for x in args.fake_drift.split(",")]
        drift_m, jitter_m = float(parts[0]), float(parts[1])
        seed = int(parts[2]) if len(parts) > 2 else 0
        rng = np.random.default_rng(seed)
        for pid, per_frame in verts_by_pid.items():
            if not per_frame:
                continue
            fr = sorted(per_frame)
            span = max(fr[-1] - fr[0], 1)
            direction = rng.normal(size=3)
            direction /= max(np.linalg.norm(direction), 1e-12)
            for f in fr:
                ramp = (f - fr[0]) / span                  # 0 at the start, 1 at the end
                shift = direction * drift_m * ramp + rng.normal(size=3) * jitter_m
                per_frame[f] = (per_frame[f] + shift.astype(np.float32))
        logger.warning("FABRICATED body drift applied: %.2f m over the clip + %.3f m "
                       "jitter, seed %d. The placer's output was overwritten; these "
                       "bodies are not where the pipeline put them.",
                       drift_m, jitter_m, seed)
    if args.root_smooth:
        logger.warning(
            "root smoothing ON (savgol w=%d) — steadier bodies, but this is NOT "
            "the CHROMM single-frame protocol the EgoHumans numbers use. Pass "
            "--root-smooth 0 to match the reported configuration.",
            args.root_smooth)

    ctx = load_depth_context(scene, ma_scale)
    ctx["offsets"] = offsets

    if args.cam_noise:
        parts = [x.strip() for x in args.cam_noise.split(",")]
        deg, cm = float(parts[0]), float(parts[1])
        seed = int(parts[2]) if len(parts) > 2 else 0
        rng = np.random.default_rng(seed)
        extr = np.array(ctx["extr"], dtype=np.float64, copy=True)
        for k in range(extr.shape[1]):
            axis = rng.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-12)
            ang = np.radians(deg) * rng.normal()
            Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
            dR = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)
            dt = rng.normal(size=3) * (cm / 100.0) / max(ma_scale, 1e-9)
            extr[:, k, :3, :3] = dR @ extr[:, k, :3, :3]
            extr[:, k, :3, 3] = (dR @ extr[:, k, :3, 3].T).T + dt
        ctx["extr"] = extr
        logger.warning("SYNTHETIC camera perturbation applied: %.2f deg, %.1f cm, "
                       "seed %d. The cloud below is fabricated degradation, not a "
                       "measurement of anything.", deg, cm, seed)
    use_bbox = (not args.keep_person_pixels) and args.carve in ("bbox", "both")
    use_mesh = (not args.keep_person_pixels) and args.carve in ("mesh", "both")
    boxes_by_cam = ({k: load_person_boxes(scene, c) for k, c in enumerate(cam_names)}
                    if use_bbox else None)

    # ── camera path from the placed bodies ────────────────────────────────
    targets = []
    for gfr in frames:
        pts = [v[gfr] for v in verts_by_pid.values() if gfr in v]
        targets.append(np.concatenate(pts, 0).mean(0) if pts else np.zeros(3, np.float32))
    targets = np.nan_to_num(np.asarray(targets, np.float32))
    filled = np.array([any(gfr in v for v in verts_by_pid.values()) for gfr in frames])
    if filled.any():
        targets[~filled] = targets[filled][0]
    targets = smooth_track(targets, args.target_smooth)

    # Framing span: a PERCENTILE of the per-person, per-frame body centroid
    # distance from the mean target, not the max. A single body placed far from
    # the action — one person genuinely walking off, or one bad placement —
    # would otherwise set the orbit radius for the entire clip. Camera framing
    # only; the bodies and the cloud are rendered unchanged either way.
    cen = np.asarray([v[g].mean(0) for v in verts_by_pid.values()
                      for g in list(v)[::5]], np.float32) if verts_by_pid else targets
    d = np.linalg.norm(cen - targets.mean(0), axis=1)
    span = float(np.percentile(d, args.span_pct)) + 1.0 if len(d) else 1.0
    radius = args.radius if args.radius is not None else args.radius_scale * max(2.5 * span, 4.0)
    up = world_up(ctx)
    eyes = orbit_eyes(targets, up, radius, args.orbit_turns,
                      args.elev_deg, args.elev_swing, args.azimuth0)
    clip_front = span if args.clip_front < 0.0 else args.clip_front
    logger.info("orbit radius = %.2f m, up = [%.2f %.2f %.2f], span = %.2f m, "
                "clip_front = %.2f m, turns = %.2f",
                radius, *up, span, clip_front, args.orbit_turns)

    W, H = int(args.size[0]), int(args.size[1])
    # Supersampling: rasterise at ssaa x resolution and box-filter down. Splats
    # and triangle edges are hard-edged (one z-buffer winner per pixel, no
    # analytic coverage), so this is the only thing that antialiases them. The
    # splat radius follows the render height automatically via fy; its pixel
    # clamp has to be scaled by hand.
    ss = max(1, int(args.ssaa))
    Wr, Hr = W * ss, H * ss
    intr = intrinsics(Wr, Hr, args.fov)
    px_max = args.point_px_max * ss
    znear = 0.05
    if ss > 1:
        logger.info("supersampling %dx: rasterising %dx%d -> %dx%d", ss, Wr, Hr, W, H)

    faces_t = torch.from_numpy(faces).to(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"cv2.VideoWriter failed to open {args.out}")

    try:
        for i, gfr in enumerate(frames):
            t = depth_index[gfr]                          # depth/camera array index
            R = torch.from_numpy(look_at(eyes[i], targets[i], up)).to(device)
            eye = torch.from_numpy(eyes[i].astype(np.float32)).to(device)
            frag_list = []
            n_pts = 0

            if not args.no_points:
                frame_verts = [v[gfr] for v in verts_by_pid.values() if gfr in v]
                h_d = -(-ctx["depth_mm"].shape[2] // args.depth_stride)
                w_d = -(-ctx["depth_mm"].shape[3] // args.depth_stride)
                pw, pc, n_img = [], [], 0
                for k, cam in enumerate(cam_names):
                    if mono_k is not None and k != mono_k:
                        continue          # single-view: no other camera fills in
                    bgr = _read_frame(args.frames_root, cam, gfr)
                    if bgr is not None:
                        n_img += 1
                    sil = (body_silhouette(frame_verts, ctx, t, k, args.depth_stride,
                                           h_d, w_d, args.carve_dilate)
                           if use_mesh else None)
                    got = depth_cloud(ctx, t, k, gfr, bgr, args.depth_stride,
                                      args.conf_thr,
                                      None if boxes_by_cam is None else boxes_by_cam[k],
                                      0.05, args.voxel, silhouette=sil,
                                      disc_rel=args.disc_rel,
                                      voxel_snap=args.voxel_snap)
                    if got is not None:
                        pw.append(got[0]); pc.append(got[1])
                if pw:
                    p = torch.from_numpy(np.concatenate(pw, 0)).to(device)
                    c = torch.from_numpy(np.concatenate(pc, 0)).to(device)
                    if clip_front > 0.0:
                        # Camera-space depth; keep only what sits at or beyond the
                        # subject, so whatever wall/ceiling the orbit passes behind
                        # stops occluding. Purely a view crop — the bodies and the
                        # far side of the room are untouched.
                        zc = (p - eye) @ R[2]
                        d_t = float(np.linalg.norm(eyes[i] - targets[i]))
                        vis = zc >= (d_t - clip_front)
                        p, c = p[vis], c[vis]
                    # A frames-root that resolves to nothing renders a grey cloud
                    # and looks like a bad reconstruction rather than a bad path,
                    # so refuse instead of producing a plausible-looking lie.
                    if i == 0 and args.frames_root is not None and n_img == 0:
                        raise SystemExit(
                            f"--frames-root {args.frames_root} matched no image for "
                            f"any of {cam_names} at frame {gfr}. The cloud would "
                            f"render grey. Check the filename pattern in _read_frame.")
                    if args.cull_3d > 0.0:
                        p, c = cull_points_near_bodies(
                            p, c, frame_verts, args.cull_3d, device)
                    if args.min_neighbors > 0:
                        p, c = radius_outlier_filter(
                            p, c, args.voxel, args.min_neighbors, device)
                    n_pts = int(p.shape[0])
                    frag_list.append((point_fragments(
                        p, c, R, eye, intr, Wr, Hr, znear,
                        args.point_size, px_max), 0.0))

            if args.draw_cameras:
                # Bypasses the cloud filters on purpose: these are drawn
                # geometry, not reconstructed points.
                fr = camera_frustums(ctx, t, args.frustum_depth)
                if fr is not None:
                    frag_list.append((point_fragments(
                        torch.from_numpy(fr[0]).to(device),
                        torch.from_numpy(fr[1]).to(device),
                        R, eye, intr, Wr, Hr, znear,
                        args.point_size * 0.6, px_max), 0.0))

            if not args.no_bodies:
                vlist, flist, clist, off = [], [], [], 0
                for j, pid in enumerate(pids):
                    v = verts_by_pid.get(pid, {}).get(gfr)
                    if v is None:
                        continue
                    vt = torch.from_numpy(np.ascontiguousarray(v)).to(device)
                    vlist.append(vt)
                    flist.append(faces_t + off)
                    clist.append(torch.from_numpy(
                        _PERSON_COLORS[j % len(_PERSON_COLORS)]).to(device)
                        .expand(vt.shape[0], 3))
                    off += vt.shape[0]
                if vlist:
                    frag_list.append((mesh_fragments(
                        torch.cat(vlist), torch.cat(flist), torch.cat(clist),
                        R, eye, intr, Wr, Hr, znear, args.ambient, args.diffuse),
                        args.mesh_bias))

            img = render_frame(frag_list, Wr, Hr, tuple(args.bg), device)
            if ss > 1:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            writer.write(img)
            if i % 25 == 0:
                logger.info("frame %d/%d (abs %d) points=%d", i + 1, len(frames), gfr, n_pts)
    finally:
        writer.release()

    logger.info("wrote %s (%d frames, %.1f fps)", args.out, len(frames), args.fps)


if __name__ == "__main__":
    main()
