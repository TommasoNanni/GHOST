"""Run fusion inference on a single RICH scene and (optionally) launch the visualizer.

Usage
-----
    pixi run python scripts/infer_scene.py --scene BBQ_001_guitar
    pixi run python scripts/infer_scene.py --scene BBQ_001_guitar --no_visualize
    pixi run python scripts/infer_scene.py --scene BBQ_001_guitar \\
        --scenes_root test_outputs/rich10_segmentation_test

The script:
  1. Loads the RICHFusionDatapoint for the requested scene.
  2. Fuses the per-camera SAM3D poses by the geodesic median (no checkpoint).
  3. Places each body with the Procrustes-DLT BodyPlacer.
  4. Saves predictions to fusion_outputs/<scene_name>.npz  (visualizer-ready format).
  5. Launches the viewer unless --no_visualize is set.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import tyro

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.placer import BodyPlacer

# Import PnP placement helpers from inference.py (same scripts/ dir)
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from inference import _run_placer, _build_vggt_cameras

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _R_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to 6D (first two rows)."""
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


# ── Fusion: geodesic median over cameras ─────────────────────────────────────
# Copied verbatim from evaluation/evaluate_rich_median.py so this script stands
# alone.  No learned module, no checkpoint.

def _sixd_to_matrix(sixd: torch.Tensor) -> torch.Tensor:
    """(..., 6) 6D rotation → (..., 3, 3) rotation matrix (Gram-Schmidt)."""
    r0, r1 = sixd[..., :3], sixd[..., 3:]
    b1 = r0 / (r0.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def median_fuse(pose_t: torch.Tensor, mask_t: torch.Tensor,
                iters: int = 5, eps: float = 1e-3) -> torch.Tensor:
    """Fuse per-camera poses by the GEODESIC MEDIAN over the visible cameras.

    The chordal mean minimises sum_k ||R - R_k||_F^2 (an L2 criterion), so a
    single badly-wrong camera drags the estimate. The geodesic median minimises
    the L1 criterion on SO(3) instead:

        R_bar = argmin_R  sum_k  w_k * d_geo(R, R_k),    d_geo = angle of R^T R_k

    solved by Weiszfeld/IRLS: seed with the chordal mean, then repeatedly re-take
    a chordal mean with weights w_k / (theta_k + eps). Each step is the same
    closed-form SVD projection, so this costs `iters` extra 3x3 batched SVDs and
    nothing else — no training, no parameters.

    Args:
        pose_t: (B, T, K, P, J, 6) per-camera SAM3D poses in 6D rotation form.
        mask_t: (B, T, K, P) 1.0 where camera k has a detection for person p at t.
        iters:  IRLS steps. 3-5 is converged; 5 used for every reported number.
        eps:    radians, guards 1/theta when a camera sits on the current estimate.

    Returns:
        (B, T, P, J, 6) — rows 0 and 1 of R_bar. R_bar is already orthonormal, so
        a downstream Gram-Schmidt is a no-op on it.
    """
    R = _sixd_to_matrix(pose_t)                             # (B,T,K,P,J,3,3)
    J = R.shape[4]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)

    # No camera saw this (t, p): the mean matrix is all-zero and its SVD is
    # arbitrary. Seed the identity so the decomposition stays well-conditioned;
    # these slots are dropped by the visibility mask before any metric sees them.
    empty = (mask_t.sum(dim=2) == 0)[..., None]             # (B,T,P,1) -> over J

    def _chordal(w: torch.Tensor) -> torch.Tensor:
        """Weighted chordal mean. w: (B,T,K,P,J) -> (B,T,P,J,3,3)."""
        ww = w[..., None, None]
        M = (R * ww).sum(dim=2) / ww.sum(dim=2).clamp_min(1e-8)
        if bool(empty.any()):
            M = torch.where(empty[..., None, None], eye.expand_as(M), M)
        # Nearest rotation to M in Frobenius norm; the det correction blocks
        # reflections when M is near-degenerate.
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


def _clean_scene_view(ghost_scene: Path, cam_names: list[str]) -> Path:
    """Temp scene dir: root vggt/scale files symlinked; each cam/body_data ->
    the real cam/body_data_clean.  Lets the datapoint + BodyPlacer read the
    curated (global-id, ops-applied) tracks without touching hard-coded
    ``body_data`` paths.  Copied from evaluation/evaluate_egohumans.py.
    """
    import tempfile
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


# ── EgoHumans forward (Route B: bypass the datapoint) ────────────────────────
# The EgoHumans datapoint couples camera loading (COLMAP cameras.txt, absent in
# the extracted seq dirs) with body loading and drops every camera without it —
# fatal, and the fusion/placement path needs no GT cameras.  Instead we pack the
# model input straight from body_data_clean, copying evaluate_egohumans.py's
# proven packing (54-joint convention).  GT overlay is disabled in this path.
_NUM_JOINTS = 55   # SMPL-X joints packed before the root is dropped


def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """(J,3) axis-angle → (J,6) = first two ROWS of R (training convention).
    Copied from evaluation/evaluate_egohumans.py."""
    from scipy.spatial.transform import Rotation as SciR
    Rm = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()
    return Rm[:, :2, :].reshape(-1, 6)


def _egohumans_pack_pose(entry: dict) -> np.ndarray:
    """SMPL-X aa params → (54, 6) root-excluded 6-D pose.  Copied from
    evaluate_egohumans._pack_pose."""
    parts = [entry["go"].reshape(1, 3), entry["bp"].reshape(21, 3)]
    parts.append(entry.get("lh", np.zeros((15, 3))).reshape(15, 3))
    parts.append(entry.get("rh", np.zeros((15, 3))).reshape(15, 3))
    aa = np.concatenate(parts, 0).astype(np.float64)
    if aa.shape[0] < _NUM_JOINTS:
        aa = np.concatenate([aa, np.zeros((_NUM_JOINTS - aa.shape[0], 3))], 0)
    return _aa_to_6d(aa)[1:]            # (54, 6)


def _egohumans_load_tracks(scene_dir: Path, cam_names: list[str]):
    """{cam_idx: {pid: {frame: entry}}} + sorted pid list, from body_data_clean.
    Adapted from evaluate_egohumans._load_clean_tracks: all cams treated valid,
    pids discovered from disk instead of a GT list."""
    per_cam: dict[int, dict] = {}
    pid_set: set[int] = set()
    for k, cam in enumerate(cam_names):
        bd = scene_dir / cam / "body_data_clean"
        cam_map: dict[int, dict] = {}
        for f in sorted(bd.glob("person_*.npz")):
            pid = int(f.stem.split("_")[1])
            d = np.load(str(f), allow_pickle=False)
            fi = d["frame_indices"].astype(int)
            frames_map = {}
            for t, gfr in enumerate(fi):
                e = {"go": d["smplx_global_orient"][t].reshape(3),
                     "bp": d["smplx_body_pose"][t].reshape(63)}
                if "smplx_left_hand_pose" in d.files:
                    e["lh"] = d["smplx_left_hand_pose"][t].reshape(-1, 3)
                if "smplx_right_hand_pose" in d.files:
                    e["rh"] = d["smplx_right_hand_pose"][t].reshape(-1, 3)
                if "smplx_betas" in d.files:
                    e["betas"] = d["smplx_betas"][t][:10]
                frames_map[int(gfr)] = e
            if frames_map:
                cam_map[pid] = frames_map
                pid_set.add(pid)
        per_cam[k] = cam_map
    return per_cam, sorted(pid_set)


def _egohumans_forward(
    scene_dir: Path,
    cam_names: list[str],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    """Pack pose/mask from body_data_clean, fuse over cameras, return the same
    raw_arrays contract as _run_forward with GT set to zeros (no GT overlay)."""
    per_cam, pids = _egohumans_load_tracks(scene_dir, cam_names)
    if not pids:
        raise RuntimeError(f"No persons found in body_data_clean under {scene_dir}")
    K, P = len(cam_names), len(pids)
    pid_slot = {p: i for i, p in enumerate(pids)}

    all_frames = [gfr for cm in per_cam.values() for fm in cm.values() for gfr in fm]
    fmin, fmax = min(all_frames), max(all_frames)
    T = fmax - fmin + 1

    pose = np.zeros((T, K, P, 54, 6), dtype=np.float32)
    mask = np.zeros((T, K, P), dtype=np.float32)
    betas_acc: dict[int, list] = {p: [] for p in pids}
    for k, cam_map in per_cam.items():
        for pid, fm in cam_map.items():
            s = pid_slot[pid]
            for gfr, e in fm.items():
                t = gfr - fmin
                pose[t, k, s] = _egohumans_pack_pose(e)
                mask[t, k, s] = 1.0
                if "betas" in e:
                    betas_acc[pid].append(e["betas"])
    pred_shape = np.stack([
        (np.mean(betas_acc[p], 0) if betas_acc[p] else np.zeros(10, np.float32))
        for p in pids
    ]).astype(np.float32)                                   # (P, 10)

    # Geodesic median over the visible cameras.  Pose is already root-excluded
    # (54 joints); no autocast — the IRLS runs batched 3x3 SVDs in fp32.
    with torch.no_grad():
        pt = torch.from_numpy(pose[None]).to(device)        # (1, T, K, P, 54, 6)
        mt = torch.from_numpy(mask[None]).to(device)        # (1, T, K, P)
        fused = median_fuse(pt, mt)                          # (1, T, P, 54, 6)
    pred_pose_54 = fused[0].float().cpu().numpy().astype(np.float32)   # (T, P, 54, 6)

    raw_arrays = {
        "pred_pose_54":         pred_pose_54,
        "pred_shape":           pred_shape,
        "gt_body_pose":         np.zeros((T, P, 55, 6), dtype=np.float32),
        "gt_body_shape":        np.zeros((P, 10), dtype=np.float32),
        "gt_camera":            np.zeros((T, K, 8), dtype=np.float32),
        "gt_body_transl_world": np.zeros((T, P, 3), dtype=np.float32),
        "gt_valid":             np.zeros((T, P), dtype=bool),
    }
    return raw_arrays, {"pids": pids, "frame_start": fmin, "T": T}


def _run_forward(
    dp:      RICHFusionDatapoint,
    device:  torch.device,
) -> dict[str, np.ndarray]:
    """Fuse the full sequence over cameras and return predictions + GT arrays.

    Returns pred_pose_54 (root excluded) so that main() can prepend the
    PnP-estimated global_orient to build the full 55-joint pose.
    """
    ds     = RICHFusionDataset([dp])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    def _s(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).float().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch

            inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in inputs.items()}

            # Geodesic median over the visible cameras.  Joint 0 of inp["pose"] is
            # each camera's own root orient — dropped here, the placer re-estimates
            # it by Procrustes DLT.  Per-joint SAM3D confidence is deliberately
            # unused: the IRLS weights are the 0/1 visibility mask only.
            pose_aggr = median_fuse(
                inp["pose"][..., 1:, :],           # (B, T, K, P, 54, 6)
                inp["person_mask"],                # (B, T, K, P)
            )                                       # (B, T, P, 54, 6)

            pred_pose_54 = _s(pose_aggr)   # (T, P, 54, 6)
            # Mean SAM3D betas: visibility-weighted mean over T and K.
            mask = inp["person_mask"].float()                               # (B, T, K, P)
            shape_sum = (inp["shape"] * mask.unsqueeze(-1)).sum(dim=[1,2]) # (B, P, 10)
            denom     = mask.sum(dim=[1,2]).clamp(min=1).unsqueeze(-1)     # (B, P, 1)
            pred_shape = _s(shape_sum / denom)                             # (P, 10)

            # GT arrays (cam-0 = world frame)
            gt_pose  = _s(targets["pose"])   # (T, P, 55, 6)
            gt_shape = _s(targets["shape"])  # (T, P, 10)
            gt_trans = _s(targets["trans"])  # (T, P, 3)
            gt_valid = _s(targets["gt_valid"])  # (T, P)

            # GT camera is static (K, 8); tile to (T, K, 8) for the visualizer.
            gt_cam_static = _s(targets["camera"])               # (K, 8)
            T_local = pred_pose_54.shape[0]
            gt_cam_tiled = np.broadcast_to(
                gt_cam_static[None], (T_local,) + gt_cam_static.shape
            ).copy()  # (T, K, 8)

            break  # single scene, single batch

    return {
        "pred_pose_54":      pred_pose_54,    # (T, P, 54, 6) — root excluded
        "pred_shape":        pred_shape,      # (P, 10)
        "gt_body_pose":      gt_pose,         # (T, P, 55, 6)
        "gt_body_shape":     gt_shape[0],     # (P, 10) — first frame
        "gt_camera":         gt_cam_tiled,    # (T, K, 8)
        "gt_body_transl_world": gt_trans,     # (T, P, 3)
        "gt_valid":          gt_valid,        # (T, P)
    }


def main(
    scene:        str,
    scenes_root:  Path = Path(CONFIG.data.output_directory),
    out_dir:      Path = Path(CONFIG.data.fusion_output_dir),
    port:         int  = 9090,
    no_visualize: bool = False,
    frame_start:  int  = 0,
    show_gt:      bool = True,
    show_depth:   bool = False,
    all_people:         bool = False,
    body_split:         str  = "train_body",
    device:             str  = "cuda" if torch.cuda.is_available() else "cpu",
    centered_data_root: Path | None = None,
    images_root:        Path | None = None,
    dataset:            str  = "rich",
    viewer:             str  = "rerun",
    frames_dir:         Path | None = None,
) -> None:
    """
    Parameters
    ----------
    scene          : scene directory name, e.g. "BBQ_001_guitar"
    scenes_root    : root that contains the scene directory (default: config output_directory)
    out_dir        : directory where the predictions .npz is written
    port           : viser WebSocket port
    no_visualize   : save predictions only — do not launch the viewer
    frame_start    : first RICH frame index (for image loading in the viewer)
    show_gt        : show GT body in the viewer (if available)
    show_depth     : show VGGT depth maps as point clouds in the viewer
    all_people     : run inference on ALL foreground persons, not only the
                     GT-matched subject (background people get no GT overlay)
    device         : "cuda" or "cpu"
    images_root    : override for the RICH images root (cam_XX subdirs); defaults to config value
    dataset        : "rich" | "egohumans".  egohumans packs pose from
                     body_data_clean/, bypasses the datapoint (no colmap needed),
                     drops crop_meta, and has no GT overlay.  Cameras come from
                     the placer's VGGT cameras.
    viewer         : "rerun" (visualize_rerun) | "viser" (visualize_fusion)
    """
    scene_dir = Path(scenes_root) / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # ── load scene ────────────────────────────────────────────────────────────
    logger.info(f"Loading scene: {scene_dir}  (dataset={dataset})")
    _rich_data_root = str(images_root) if images_root is not None else CONFIG.data.rich_data_root

    dev = torch.device(device)

    # work_dir = scene root the placer reads body_data from.  egohumans keeps
    # curated tracks in body_data_clean/, so mount a temp clean-view whose
    # cam/body_data symlinks point at cam/body_data_clean (root vggt/scale files
    # symlinked too).  RICH reads body_data directly.
    cleanup_view: Path | None = None
    if dataset == "egohumans":
        cam_names = sorted(
            d.name for d in scene_dir.iterdir()
            if d.is_dir() and (d / "body_data_clean").is_dir()
        )
        if not cam_names:
            raise FileNotFoundError(
                f"dataset=egohumans but no cam*/body_data_clean found in {scene_dir}"
            )
        show_gt = False   # Route B has no calibrated GT overlay
        work_dir = _clean_scene_view(scene_dir, cam_names)   # for the placer
        cleanup_view = work_dir
        logger.info(f"Fusing (geodesic median) on {dev} … (egohumans, {len(cam_names)} cams)")
        raw_arrays, ego_meta = _egohumans_forward(scene_dir, cam_names, dev)
        frame_start_val  = ego_meta["frame_start"]
        all_pids_ordered = ego_meta["pids"]
        logger.info(
            f"  {ego_meta['T']} frames, {len(cam_names)} cameras, "
            f"{len(all_pids_ordered)} persons"
        )
    else:
        work_dir = scene_dir
        dp = RICHFusionDatapoint(
            scene_dir      = scene_dir,
            rich_data_root = _rich_data_root,
            rich_gt_dir    = CONFIG.data.rich_gt_dir,
            body_split     = body_split,
            restrict_to_gt_persons = not all_people,
            min_foreground_cams    = 1 if all_people else None,
        )
        T = dp._frame_end - dp._frame_start
        logger.info(f"  {T} frames, {dp.num_cameras} cameras, {dp.max_persons} persons")
        logger.info(f"Fusing (geodesic median) on {dev} …")
        raw_arrays = _run_forward(dp, dev)
        frame_start_val  = dp._frame_start
        all_pids_ordered = sorted(set(pid for pids_k in dp._pid_order for pid in pids_k))

    pred_pose_54 = raw_arrays["pred_pose_54"]   # (T, P, 54, 6)
    pred_shape   = raw_arrays["pred_shape"]      # (P, 10)
    T_scene = pred_pose_54.shape[0]
    P       = pred_pose_54.shape[1]

    # ── BodyPlacer: PnP translation + orientation + VGGT cameras ─────────────
    # cam_dirs come from work_dir so egohumans resolves body_data → body_data_clean.
    cam_dirs = sorted(
        d for d in work_dir.iterdir()
        if d.is_dir() and (d / "body_data").is_dir()
    )
    # raw dict for _run_placer (it needs frame_indices from body data)
    raw_body: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_persons: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid_num = int(npz_path.stem.split("_")[1])
            if pid_num not in all_pids_ordered:
                continue
            d = np.load(npz_path, allow_pickle=False)
            cam_persons[pid_num] = {k: d[k] for k in d.files}
        raw_body.append(cam_persons)

    _smplx_arg = Path(CONFIG.data.smplx_model_path)

    # SAM3D kp2d are in uncropped source pixels; VGGT cameras live in centered-crop
    # space. crop_meta.json (next to the centered images) reconciles them in the placer.
    # egohumans already stores kp2d in centered-crop space → no crop_meta.
    if dataset == "egohumans":
        crop_meta_path = None
    else:
        _centered_root = Path(centered_data_root) if centered_data_root is not None else Path(CONFIG.data.rich_data_root)
        crop_meta_path = _centered_root / scene / "crop_meta.json"

    logger.info("Running BodyPlacer (Procrustes DLT) …")
    root_translation, orient_R, vggt_cameras = _run_placer(
        scene_dir      = work_dir,
        cam_dirs       = cam_dirs,
        raw            = raw_body,
        all_pids       = all_pids_ordered,
        frame_start    = frame_start_val,
        T              = T_scene,
        smplx_model_path = _smplx_arg,
        fused_pose     = pred_pose_54,   # (T, P, 54, 6) fused pose for Procrustes FK
        crop_meta_path = crop_meta_path,
    )
    # orient_R: (T, P, 3, 3) — NaN where Procrustes DLT failed

    # ── Build full 55-joint pose: Procrustes root + fusion non-root ──────────
    # Convert Procrustes rotation matrices to 6D; fall back to GT root where NaN.
    gt_root_6d = raw_arrays["gt_body_pose"][:, :, 0, :]   # (T, P, 6)
    proc_root_6d = np.where(
        np.isnan(orient_R[:, :, 0, 0])[..., np.newaxis],   # (T, P, 1) mask
        gt_root_6d,
        _R_to_6d(orient_R),                                 # (T, P, 6)
    )  # (T, P, 6)
    pred_pose_55 = np.concatenate(
        [proc_root_6d[:, :, np.newaxis, :], pred_pose_54], axis=2
    )  # (T, P, 55, 6)

    # Real camera names (slot order) so the viewer locates each cam's frames.
    # Same order as the vggt_cameras K axis (both from vggt_cameras_centered.npz).
    _vggt_cam_npz = np.load(work_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    camera_names = np.array([
        n.decode() if isinstance(n, bytes) else str(n)
        for n in _vggt_cam_npz["camera_names"]
    ])

    # ── Assemble visualizer-ready dict ────────────────────────────────────────
    arrays: dict[str, np.ndarray] = {
        # Predicted
        "pose":              pred_pose_55,             # (T, P, 55, 6)
        "shape":             pred_shape,               # (P, 10)
        "camera":            vggt_cameras,             # (T, K, 8) — VGGT cameras
        "camera_names":      camera_names,             # (K,) real cam names, slot order
        "body_transl_world": root_translation,         # (T, P, 3) — PnP translation
        # GT reference (for visualizer comparison)
        "gt_body_pose":          raw_arrays["gt_body_pose"],
        "gt_body_shape":         raw_arrays["gt_body_shape"],
        "gt_camera":             raw_arrays["gt_camera"],
        "gt_body_transl_world":  raw_arrays["gt_body_transl_world"],
        "gt_valid":              raw_arrays["gt_valid"],
    }

    # ── save predictions ──────────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{scene}.npz"
    np.savez_compressed(str(out_file), **arrays)
    logger.info(f"Predictions saved → {out_file}  (keys: {list(arrays)})")

    # clean-view only fed the datapoint + placer (both already ran); the viewer
    # reads the real scene_dir + predictions npz, so drop the temp symlinks now.
    if cleanup_view is not None:
        import shutil
        shutil.rmtree(cleanup_view, ignore_errors=True)

    # ── launch visualizer ─────────────────────────────────────────────────────
    repo_root = Path(__file__).parent.parent
    show_gt_flag    = "--show-gt" if show_gt else "--no-show-gt"
    show_depth_flag = "--show-depth" if show_depth else "--no-show-depth"
    vis_script = "visualize/visualize_rerun.py" if viewer == "rerun" else "visualize/visualize_fusion.py"
    # egohumans frames are 1-indexed and start at fmin → align the viewer's frame
    # numbering with the predictions (frame t ↔ image {frame_start_val+t}).
    vis_frame_start = frame_start_val if dataset == "egohumans" else frame_start
    vis_cmd = (
        f"pixi run python {vis_script}"
        f" --predictions {out_file}"
        f" --scene-dir {scene_dir}"
        f" --smplx-model-dir {CONFIG.data.smplx_model_path}"
        f" --rich-data-root {_rich_data_root}"
        f" --frame-start {vis_frame_start}"
        f" {show_gt_flag}"
        f" {show_depth_flag}"
        f" --port {port}"
    )
    if frames_dir is not None and viewer == "rerun":
        vis_cmd += f" --frames-dir {frames_dir}"

    if no_visualize:
        logger.info("\nTo visualize, run:")
        logger.info(f"  {vis_cmd}")
    else:
        logger.info("Launching visualizer …")
        subprocess.run(vis_cmd, shell=True, cwd=str(repo_root))


if __name__ == "__main__":
    tyro.cli(main)
