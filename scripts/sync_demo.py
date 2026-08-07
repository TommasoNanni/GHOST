#!/usr/bin/env python
"""scripts/sync_demo.py — before/after temporal-synchronisation figure.

Renders two stills of the *same* physical instant of one RICH scene:

  ``<out>/<scene>_desync.png``
      Per-camera delays δ_k ∈ [0, max_shift] are injected, so every stage that
      follows sees frames from different instants: VGGT estimates cameras and
      depth from mismatched images, the fusion model averages poses that are
      out of phase, and the Procrustes-DLT placer triangulates rays that never
      met.  Bad on purpose.

      One VGGT forward pass runs per condition — on the rendered frame only.
      The fusion model takes no camera input and the placer reads cameras one
      frame at a time, so no other row can reach the figure.

  ``<out>/<scene>_sync.png``
      The Synchronizer estimates δ̂_k from the *desynced* 3D joint tracks; the
      residual δ − δ̂ is re-applied and the identical pipeline is re-run.

Each still shows the VGGT depth maps of every camera back-projected into the
common world frame (points coloured by their source image) plus the fused
SMPL-X mesh.  Rendering is a plain numpy/cv2 z-buffer — no OpenGL, no EGL, no
viewer — so it runs headless on a compute node and writes a PNG directly.

Only the frame pairing changes between the two conditions.  The MapAnything
metric scale is held fixed so the figure isolates synchronisation.

Usage
-----
    pixi run python scripts/sync_demo.py --scene LectureHall_021_sidebalancerun1

Notes
-----
* Unlike ``scripts/infer_scene.py``, a failed Procrustes DLT is **not** patched
  with the ground-truth root orientation — that fallback would silently rescue
  the desynced panel and make it look better than the method actually is.
  Failures stay NaN and the frame is reported as unplaced.
* The centered RICH frames must be mounted (see bash_jobs/sync_demo.sh).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import tyro

from configuration import CONFIG
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from fusion.fusion_module_v2 import PoseFusionModule
from preprocessing.run_vggt import VGGT_RESOLUTION, VGGTPreprocessor
from synchronize_videos.synchronizer import Synchronizer
from visualize.visualize_rerun import _build_smplx_vertices

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from inference import _run_placer          # noqa: E402  (same scripts/ dir)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ══════════════════════════════════════════════════════════════════════════════
#  Scene I/O
# ══════════════════════════════════════════════════════════════════════════════

def _camera_names(scene_dir: Path) -> list[str]:
    """Camera names in the K-axis order of vggt_cameras_centered.npz."""
    npz = np.load(scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    return [n.decode() if isinstance(n, bytes) else str(n) for n in npz["camera_names"]]


def _cam_frame_files(centered_root: Path, scene: str, cam: str) -> list[Path]:
    """Sorted image files of one camera — index i is VGGT row i / body frame i.

    The VGGT preprocessing indexed frames positionally in this same sorted
    order, so positional indexing keeps images, cameras and body_data aligned
    regardless of the filename numbering base.
    """
    base = Path(centered_root) / scene / cam
    for sub in (base, base / "frames", base / "images"):
        if sub.is_dir():
            files = sorted(p for p in sub.iterdir() if p.suffix.lower() in _IMG_EXTS)
            if files:
                return files
    raise FileNotFoundError(
        f"No frames for {scene}/{cam} under {base} — is the centered squashfs mounted?"
    )


def _load_tracks(scene_dir: Path, cams: list[str]) -> list[dict[int, dict]]:
    """tracks[k][pid] = {array_name: array} straight from body_data/person_*.npz."""
    tracks: list[dict[int, dict]] = []
    for cam in cams:
        per_pid: dict[int, dict] = {}
        for npz_path in sorted((scene_dir / cam / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            with np.load(npz_path, allow_pickle=False) as d:
                per_pid[pid] = {k: d[k] for k in d.files}
        tracks.append(per_pid)
    return tracks


# ══════════════════════════════════════════════════════════════════════════════
#  Frame + offset selection
# ══════════════════════════════════════════════════════════════════════════════

def _pick_frame(
    tracks: list[dict[int, dict]],
    T_scene: int,
    window: int,
    max_shift: int,
    delays: np.ndarray,
) -> int:
    """Frame where the injected delays do the most damage.

    Scores each candidate by how far each camera's own view of the body moves
    over *its own* delay — ``mean_k mean_j ‖kp_k(t + δ_k) − kp_k(t)‖`` — which
    is what makes the views disagree.  A plain 1-frame speed is a poor proxy:
    on a slow motion the fastest frame still shifts the body only a couple of
    centimetres over 19 frames, and both panels come out identical.
    """
    lo = window // 2
    hi = T_scene - window // 2 - max_shift - 1
    if hi <= lo:
        raise ValueError(f"Scene too short: T={T_scene}, window={window}, max_shift={max_shift}")

    score = np.zeros(T_scene, dtype=np.float64)
    count = np.zeros(T_scene, dtype=np.float64)
    for k, per_pid in enumerate(tracks):
        d = int(delays[k])
        if d == 0:
            continue                                    # anchor camera: no damage
        for pdata in per_pid.values():
            fi = pdata["frame_indices"].astype(int)
            kp = pdata["pred_keypoints_3d"].astype(np.float32)      # (T_i, J, 3)
            dense = np.full((T_scene, kp.shape[1], 3), np.nan, np.float32)
            keep = (fi >= 0) & (fi < T_scene)
            dense[fi[keep]] = kp[keep]
            shifted = np.full_like(dense, np.nan)
            shifted[:T_scene - d] = dense[d:]
            disp = np.linalg.norm(shifted - dense, axis=-1).mean(axis=-1)   # (T,)
            ok = np.isfinite(disp)
            score[ok] += disp[ok]
            count[ok] += 1.0

    mean_score = np.where(count > 0, score / np.maximum(count, 1e-6), 0.0)
    mean_score = np.convolve(mean_score, np.ones(5) / 5.0, mode="same")
    mean_score[:lo] = -np.inf
    mean_score[hi:] = -np.inf
    t_star = int(np.argmax(mean_score))
    logger.info(f"  auto frame t*={t_star}  (body moves {mean_score[t_star] * 100:.1f} cm "
                f"over the injected delays — the damage the figure shows)")
    return t_star


def _sample_offsets(K: int, max_shift: int, seed: int) -> np.ndarray:
    """Per-camera delay δ_k in frames; camera 0 is the anchor (δ_0 = 0)."""
    rng = np.random.default_rng(seed)
    delays = rng.integers(0, max_shift + 1, size=K).astype(int)
    delays[0] = 0
    return delays


# ══════════════════════════════════════════════════════════════════════════════
#  Synchronizer
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_offsets(
    tracks: list[dict[int, dict]],
    delays: np.ndarray,
    T_scene: int,
    device: str,
    max_shift: int,
) -> np.ndarray:
    """Run the Synchronizer on the *desynced* tracks; return δ̂ anchored at cam 0.

    Tracks are densified onto one global timeline after the injected shift, so
    the synchronizer sees exactly what a genuinely async capture would give it.
    The estimate uses the whole sequence (not just the rendered window) — it is
    cheap and a longer overlap gives a better cross-correlation.
    """
    shared_pids = sorted(set.intersection(*[set(t.keys()) for t in tracks])) if tracks else []
    if not shared_pids:
        raise RuntimeError("No person id is present in every camera — cannot synchronize.")
    logger.info(f"  sync uses person ids {shared_pids} on {len(tracks)} cameras")

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []
    for k, per_pid in enumerate(tracks):
        per_person_j, per_person_c = [], []
        for pid in shared_pids:
            pdata = per_pid[pid]
            # Synchronizer._compute_cost_matrix converts its input with
            # _axis_angle_to_rot_mat and scores an SO(3) geodesic, so it must be
            # fed axis-angle ROTATIONS, not 3D positions.  Feeding
            # pred_keypoints_3d (MHR70 positions) made the cost surface
            # meaningless: it still correlated on large motions but went flat on
            # slow ones, where estimate_couple_offset fell back to offset 0.
            # Same packing as utilities.body_data.load_person_smplx_pose, which
            # evaluation/alignment_experiments_multi.py (the correct caller) uses.
            rot = np.concatenate([
                pdata["smplx_body_pose"],        # (T_i, 63)
                pdata["smplx_left_hand_pose"],   # (T_i, 45)
                pdata["smplx_right_hand_pose"],  # (T_i, 45)
            ], axis=1).astype(np.float32).reshape(-1, 51, 3)
            J = rot.shape[1]                                          # 51
            conf = pdata["pred_joint_confidence"][:, 1:52].astype(np.float32)
            fi = pdata["frame_indices"].astype(int) - int(delays[k])  # desynced timeline
            dense_j = np.zeros((T_scene, J, 3), dtype=np.float32)
            dense_c = np.zeros((T_scene, J),    dtype=np.float32)
            keep = (fi >= 0) & (fi < T_scene)
            dense_j[fi[keep]] = rot[keep]
            dense_c[fi[keep]] = conf[keep]
            per_person_j.append(torch.from_numpy(dense_j).to(device))
            per_person_c.append(torch.from_numpy(dense_c).to(device))
        joints_list.append(per_person_j)
        confs_list.append(per_person_c)

    sync = Synchronizer(device=device, max_shift=max(2 * max_shift, 60), verbose=False)
    offset_matrix = sync.estimate_offset_matrix(joints_list, confs_list)
    weights       = sync.cycle_consistency_weights(offset_matrix)
    times         = sync.estimate_initial_times(offset_matrix, weights).cpu().numpy()
    times         = times - times[0]                      # anchor on camera 0, like δ
    return np.rint(times).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
#  Condition build-up:  view directory, VGGT, forward pass
# ══════════════════════════════════════════════════════════════════════════════

def _build_view(
    scene_dir: Path,
    view_dir:  Path,
    cams:      list[str],
    offsets:   np.ndarray,
    w0: int,
    w1: int,
) -> None:
    """Scene view where camera k's frame ``t`` holds physical frame ``t + o_k``.

    Everything except body_data is symlinked; body_data is rewritten with
    ``frame_indices -= o_k`` and cropped to the render window so the datapoint,
    the model and the placer all work on the window only.
    """
    view_dir.mkdir(parents=True, exist_ok=True)
    skip_root = {
        "vggt_cameras_centered.npz",     # rewritten per condition
        "mapanything_scale_baseline.npy",  # windowed copy written per condition
        "vggt_depth_centered.npz",       # unused by the placer, 1.9 GB
    }
    for entry in scene_dir.iterdir():
        if entry.is_file() and entry.name not in skip_root:
            link = view_dir / entry.name
            if not link.exists():
                os.symlink(entry, link)

    for k, cam in enumerate(cams):
        src_cam = scene_dir / cam
        dst_cam = view_dir / cam
        dst_cam.mkdir(parents=True, exist_ok=True)
        for entry in src_cam.iterdir():
            if entry.name == "body_data":
                continue
            link = dst_cam / entry.name
            if not link.exists():
                os.symlink(entry, link)

        body_out = dst_cam / "body_data"
        body_out.mkdir(exist_ok=True)
        for npz_path in sorted((src_cam / "body_data").glob("person_*.npz")):
            with np.load(npz_path, allow_pickle=False) as d:
                arrays = {key: d[key] for key in d.files}
            fi_old = arrays["frame_indices"].astype(int)
            fi_new = fi_old - int(offsets[k])
            keep   = (fi_new >= w0) & (fi_new < w1)
            if not keep.any():
                continue
            out = {
                key: (val[keep] if val.ndim > 0 and val.shape[0] == len(fi_old) else val)
                for key, val in arrays.items()
            }
            out["frame_indices"] = fi_new[keep].astype(np.int32)
            np.savez(body_out / npz_path.name, **out)


def _vggt_at_frame(
    runner:       VGGTPreprocessor,
    files_by_cam: list[list[Path]],
    offsets:      np.ndarray,
    w0: int,
    w1: int,
    t_star: int,
    original_npz: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One VGGT forward pass, on the frame pairing of ``t_star``.

    The fusion model consumes no camera parameters and the placer indexes
    cameras strictly per frame (``extrinsics[global_t - frame_start, k]``), so
    the rendered instant is the only row that can affect the figure.  Exactly
    one forward pass runs per condition; the remaining window rows are copied
    from the scene's own cameras and are never read for the render.

    Returns ``(cameras, depth_pack)``: window-length camera arrays in the layout
    of vggt_cameras_centered.npz, and the depth maps / intrinsics / extrinsics /
    source images of every camera at ``t_star``.
    """
    K = len(files_by_cam)
    paths = [files_by_cam[k][t_star + int(offsets[k])] for k in range(K)]
    logger.info(f"  VGGT forward on 1 frame ({K} cameras) — "
                f"pairing {[t_star + int(o) for o in offsets]}")
    res = runner.run_frame(paths)

    cameras = {
        "extrinsics":      original_npz["extrinsics"][w0:w1].copy(),
        "intrinsics":      original_npz["intrinsics"][w0:w1].copy(),
        "original_coords": original_npz["original_coords"][w0:w1].copy(),
        "original_size":   original_npz["original_size"][w0:w1].copy(),
        "valid":           original_npz["valid"][w0:w1].copy(),
        "camera_names":    original_npz["camera_names"].copy(),
    }
    row = t_star - w0
    ki = res["present_indices"]
    cameras["extrinsics"][row]      = np.nan
    cameras["intrinsics"][row]      = np.nan
    cameras["valid"][row]           = False
    cameras["extrinsics"][row, ki]      = res["extrinsics"]
    cameras["intrinsics"][row, ki]      = res["intrinsics"]
    cameras["original_coords"][row, ki] = res["original_coords"]
    cameras["original_size"][row, ki]   = res["original_size"]
    cameras["valid"][row, ki]           = True

    depth_pack = {
        "depth":      res["depth"],
        "depth_conf": res["depth_conf"],
        "intrinsics": res["intrinsics"],
        "extrinsics": res["extrinsics"],
        "present":    ki,
        "paths":      paths,
    }
    return cameras, depth_pack


def _build_model() -> PoseFusionModule:
    arch = CONFIG.fusion.architecture
    return PoseFusionModule(
        embedding_dim    = arch.embedding_dimension,
        num_heads        = arch.num_heads,
        num_layers       = arch.num_layers,
        max_temporal_len = arch.max_temporal_len,
        dropout          = arch.dropout,
        temporal_window  = arch.temporal_window,
    )


def _forward(model: PoseFusionModule, dp: RICHFusionDatapoint, device: torch.device) -> dict:
    """Fusion forward pass over the window (copied from scripts/infer_scene.py)."""
    loader = torch.utils.data.DataLoader(RICHFusionDataset([dp]), batch_size=1, shuffle=False)
    model.eval().to(device)

    def _s(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).float().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        for inputs, targets in loader:
            inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in inputs.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pose_aggr = model(
                    pose        = inp["pose"],
                    person_mask = inp["person_mask"],
                    joint_mask  = inp.get("joint_mask"),
                )
            mask      = inp["person_mask"].float()
            shape_sum = (inp["shape"] * mask.unsqueeze(-1)).sum(dim=[1, 2])
            denom     = mask.sum(dim=[1, 2]).clamp(min=1).unsqueeze(-1)
            return {
                "pred_pose_54": _s(pose_aggr),
                "pred_shape":   _s(shape_sum / denom),
                "gt_pose":      _s(targets["pose"]),
                "gt_trans":     _s(targets["trans"]),
                "gt_valid":     _s(targets["gt_valid"]),
            }
    raise RuntimeError("Empty dataloader — no data in the window.")


def _run_condition(
    tag:            str,
    scene:          str,
    scene_dir:      Path,
    cams:           list[str],
    offsets:        np.ndarray,
    files_by_cam:   list[list[Path]],
    original_npz:   dict[str, np.ndarray],
    scale_full:     np.ndarray,
    work_root:      Path,
    centered_root:  Path,
    runner:         VGGTPreprocessor,
    model:          PoseFusionModule,
    device:         torch.device,
    body_split:     str,
    w0: int, w1: int, t_star: int,
) -> dict:
    """One condition end-to-end: view → VGGT → fusion → placer → render inputs."""
    logger.info(f"[{tag}] pairing offsets = {offsets.tolist()}")
    # The datapoint derives the scene name from the directory name (GT
    # calibration and GT bodies are looked up by it), so the view must keep the
    # scene's own name — only its parent identifies the condition.
    view_dir = work_root / tag / scene
    _build_view(scene_dir, view_dir, cams, offsets, w0, w1)

    cameras, depth_pack = _vggt_at_frame(
        runner, files_by_cam, offsets, w0, w1, t_star, original_npz
    )

    dp = RICHFusionDatapoint(
        scene_dir              = view_dir,
        rich_data_root         = str(centered_root),
        rich_gt_dir            = CONFIG.data.rich_gt_dir,
        body_split             = body_split,
        restrict_to_gt_persons = True,
        min_foreground_cams    = None,
    )
    frame_start = dp._frame_start
    T_local     = dp._frame_end - dp._frame_start
    logger.info(f"[{tag}] datapoint: frames [{dp._frame_start}, {dp._frame_end}), "
                f"{dp.num_cameras} cams, {dp.max_persons} persons")

    # Camera rows and the metric scale must line up with the datapoint's frame
    # 0 — the placer indexes them as ``global_t - frame_start``.
    lo = frame_start - w0
    hi = lo + T_local
    if lo < 0 or hi > (w1 - w0):
        raise RuntimeError(
            f"[{tag}] datapoint range [{frame_start}, {frame_start + T_local}) "
            f"escapes the VGGT window [{w0}, {w1})"
        )
    np.savez_compressed(
        view_dir / "vggt_cameras_centered.npz",
        extrinsics      = cameras["extrinsics"][lo:hi],
        intrinsics      = cameras["intrinsics"][lo:hi],
        original_coords = cameras["original_coords"][lo:hi],
        original_size   = cameras["original_size"][lo:hi],
        valid           = cameras["valid"][lo:hi],
        camera_names    = cameras["camera_names"],
    )
    np.save(view_dir / "mapanything_scale_baseline.npy", scale_full[frame_start:frame_start + T_local])

    logger.info(f"[{tag}] fusion forward …")
    out = _forward(model, dp, device)
    pred_pose_54 = out["pred_pose_54"]                     # (T, P, 54, 6)
    all_pids = sorted({pid for pids in dp._pid_order for pid in pids})

    cam_dirs = sorted(d for d in view_dir.iterdir() if d.is_dir() and (d / "body_data").is_dir())
    raw_body: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        per_pid: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            if pid not in all_pids:
                continue
            with np.load(npz_path, allow_pickle=False) as d:
                per_pid[pid] = {k: d[k] for k in d.files}
        raw_body.append(per_pid)

    logger.info(f"[{tag}] BodyPlacer (Procrustes DLT) …")
    root_translation, orient_R, _ = _run_placer(
        scene_dir        = view_dir,
        cam_dirs         = cam_dirs,
        raw              = raw_body,
        all_pids         = all_pids,
        frame_start      = frame_start,
        T                = pred_pose_54.shape[0],
        smplx_model_path = Path(CONFIG.data.smplx_model_path),
        fused_pose       = pred_pose_54,
        crop_meta_path   = Path(centered_root) / scene / "crop_meta.json",
    )

    return {
        "tag": tag,
        "offsets": offsets,
        "view_dir": view_dir,
        "frame_start": frame_start,
        "pred_pose_54": pred_pose_54,
        "pred_shape": out["pred_shape"],
        "root_translation": root_translation,
        "orient_R": orient_R,
        "gt_pose": out["gt_pose"],
        "gt_trans": out["gt_trans"],
        "gt_valid": out["gt_valid"],
        "depth_pack": depth_pack,
        "scale": float(np.median(scale_full[frame_start:frame_start + pred_pose_54.shape[0]])),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry → mesh + point cloud at t*
# ══════════════════════════════════════════════════════════════════════════════

def _R_to_6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices → (..., 6) first-two-ROWS encoding.

    This used to take the first two COLUMNS, which is the transpose convention:
    ``pytorch3d.rotation_6d_to_matrix`` stacks its reconstructed basis along
    ``dim=-2`` (rows), so a column-encoded rotation decodes to Rᵀ — verified
    numerically (rows round-trip to 3e-8; columns match Rᵀ to 3e-8).  Every body
    rendered through _mesh_at was therefore inversely oriented: persons near the
    canonical orientation looked fine, while a substantially rotated person came
    out visibly wrong.  Matches infer_scene._R_to_6d and the training convention
    in evaluation/evaluate_egohumans._aa_to_6d.
    """
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def _mesh_at(result: dict, t_local: int, smplx_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """SMPL-X vertices of every placed person at one local frame.

    A person whose Procrustes DLT failed stays unplaced — no ground-truth
    orientation fallback, unlike scripts/infer_scene.py.
    """
    pose_54 = result["pred_pose_54"][t_local]                 # (P, 54, 6)
    orient  = result["orient_R"][t_local]                     # (P, 3, 3)
    transl  = result["root_translation"][t_local]             # (P, 3)
    placed  = [p for p in range(pose_54.shape[0])
               if np.isfinite(orient[p]).all() and np.isfinite(transl[p]).all()]
    if not placed:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), []

    root_6d = _R_to_6d(orient[placed])                        # (P', 6)
    pose_55 = np.concatenate(
        [root_6d[:, None, :], pose_54[placed]], axis=1
    )[None]                                                   # (1, P', 55, 6)
    verts, faces = _build_smplx_vertices(
        pose  = pose_55,
        shape = result["pred_shape"][placed],
        trans = transl[placed][None],
        smplx_model_dir = smplx_dir,
    )
    return verts[0].astype(np.float32), faces.astype(np.int32), placed


_CAM_TINT = np.array([                     # BGR, one per camera slot
    [ 60, 100, 240], [ 60, 200,  90], [230, 160,  50], [200,  90, 220],
    [ 70, 220, 230], [120, 120, 120], [120,  70, 220], [ 90, 200, 160],
], dtype=np.uint8)


def _vggt_view(path: Path, resolution: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Reproduce the exact pixels VGGT saw for one image.

    ``load_and_preprocess_images`` centre-crops extreme aspect ratios, resizes to
    a patch-aligned target, then **centre-pads** every image to the batch's
    common size.  ``run_vggt`` stores ``original_coords`` as the full canvas, so
    that padding is invisible downstream: sampling colour across the whole canvas
    is only correct for the one camera that needed no padding, and the depth in
    the padded border is meaningless.  Recomputing the content rect here fixes
    both the colours and the border junk.

    Returns ``(bgr (th, tw, 3), crop_box)`` where crop_box is the aspect crop
    applied to the source image, in source pixels.
    """
    from PIL import Image
    from vggt_omega.utils.load_fn import _balanced_target_shape

    with Image.open(path) as im:
        src_w, src_h = im.size
        ar = src_h / max(src_w, 1)
        if ar < 0.5:                                   # too wide → centre crop
            cw = min(src_w, max(1, int(round(src_h / 0.5))))
            box = ((src_w - cw) // 2, 0, (src_w - cw) // 2 + cw, src_h)
        elif ar > 2.0:                                 # too tall → centre crop
            ch = min(src_h, max(1, int(round(src_w * 2.0))))
            box = (0, (src_h - ch) // 2, src_w, (src_h - ch) // 2 + ch)
        else:
            box = (0, 0, src_w, src_h)
        cropped = im.convert("RGB").crop(box)
        th, tw = _balanced_target_shape(cropped.size[1] / max(cropped.size[0], 1),
                                        resolution, 16)
        rgb = np.asarray(cropped.resize((tw, th), Image.Resampling.BICUBIC))
    return rgb[:, :, ::-1].copy(), box                 # RGB → BGR


def _person_pixels(
    scene_dir: Path, cam: str, frame_idx: int, box: tuple[int, int, int, int],
    th: int, tw: int,
) -> np.ndarray | None:
    """(th, tw) bool mask, True where a person is — warped like the VGGT input."""
    mpath = scene_dir / cam / "mask_data.npz"
    if not mpath.exists():
        return None
    digits = "".join(ch for ch in cam if ch.isdigit())
    cam_idx = int(digits) if digits else 0
    with np.load(mpath, mmap_mode="r") as z:
        key = next((k for k in (f"mask_{frame_idx:05d}_{cam_idx:02d}",
                                f"mask_{frame_idx:05d}", f"mask_{frame_idx}")
                    if k in z), None)
        if key is None:
            return None
        m = np.asarray(z[key])
    m = m[box[1]:box[3], box[0]:box[2]]
    return cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0


def _point_cloud(
    depth_pack:  dict,
    scale:       float,
    cams:        list[str],
    frame_by_cam: dict[str, int],
    scene_dir:   Path,
    resolution:  int,
    stride:      int,
    conf_thr:    float,
    voxel:       float,
    mask_people: bool,
    colour_by_cam: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project every camera's VGGT depth into world metres.

    Mirrors ``visualize_rerun._depth_cloud``: confidence gate, voxel
    downsampling, and person pixels dropped so the bodies never appear twice
    (once as points, once as a mesh).  Adds the padded-canvas correction that
    the stored ``original_coords`` cannot express.
    """
    depth = depth_pack["depth"]            # (K', H, W) VGGT units
    conf  = depth_pack["depth_conf"]       # (K', H, W)
    K_int = depth_pack["intrinsics"]       # (K', 3, 3), padded-canvas pixels
    E_ext = depth_pack["extrinsics"]       # (K', 3, 4) cam-from-world, VGGT units
    paths = depth_pack["paths"]
    present = list(depth_pack["present"])
    H, W = depth.shape[1:]

    pts_all, col_all = [], []
    for i, k in enumerate(present):
        cam = cams[k]
        bgr, box = _vggt_view(paths[k], resolution)
        th, tw = bgr.shape[:2]
        top, left = (H - th) // 2, (W - tw) // 2        # centre padding
        if top < 0 or left < 0:
            logger.warning(f"  {cam}: content {th}x{tw} exceeds canvas {H}x{W} — skipped")
            continue

        d = depth[i][top:top + th, left:left + tw][::stride, ::stride].astype(np.float64)
        c = conf[i][top:top + th, left:left + tw][::stride, ::stride].astype(np.float64)
        vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        vv = vv * stride
        uu = uu * stride

        ok = np.isfinite(d) & (d > 1e-6) & (c >= conf_thr)
        if mask_people:
            people = _person_pixels(scene_dir, cam, frame_by_cam[cam], box, th, tw)
            if people is not None:
                ok &= ~people[::stride, ::stride]
        if not ok.any():
            logger.warning(f"  {cam}: no depth pixels survived the gates")
            continue

        u_pad = uu[ok] + left                            # back to canvas pixels
        v_pad = vv[ok] + top
        z     = d[ok]
        Ki = K_int[i].astype(np.float64)
        X_cam = np.stack([(u_pad - Ki[0, 2]) / Ki[0, 0] * z,
                          (v_pad - Ki[1, 2]) / Ki[1, 1] * z,
                          z], axis=-1)
        R = E_ext[i][:3, :3].astype(np.float64)
        t = E_ext[i][:3,  3].astype(np.float64)
        X_world = ((X_cam - t) @ R) * scale              # R^T(X − t), then metres

        if colour_by_cam:
            colours = np.broadcast_to(_CAM_TINT[k % len(_CAM_TINT)],
                                      (len(X_world), 3)).copy()
        else:
            colours = bgr[vv[ok], uu[ok]]

        pts = X_world.astype(np.float32)
        cols = colours.astype(np.uint8)
        if voxel > 0.0:
            keys = np.floor(pts / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts, cols = pts[idx], cols[idx]
        logger.info(f"  {cam}: {len(pts)} points")
        pts_all.append(pts)
        col_all.append(cols)

    if not pts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(pts_all), np.concatenate(col_all)


def _camera_frusta(
    depth_pack: dict,
    scale: float,
    cams: list[str],
    resolution: int,
    size: float = 0.6,
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Wireframe frustum per camera: ``(segments (S,2,3) world metres, BGR)``.

    Same idea as the pinhole frustums the Rerun viewer logs — they make the rig
    layout readable, as in "Reconstructing People, Places, and Cameras".
    """
    K_int = depth_pack["intrinsics"]
    E_ext = depth_pack["extrinsics"]
    paths = depth_pack["paths"]
    present = list(depth_pack["present"])
    H, W = depth_pack["depth"].shape[1:]

    out = []
    for i, k in enumerate(present):
        bgr, _ = _vggt_view(paths[k], resolution)
        th, tw = bgr.shape[:2]
        top, left = (H - th) // 2, (W - tw) // 2
        Ki = K_int[i].astype(np.float64)
        corners_px = np.array([[left, top], [left + tw, top],
                               [left + tw, top + th], [left, top + th]], np.float64)
        rays = np.stack([(corners_px[:, 0] - Ki[0, 2]) / Ki[0, 0],
                         (corners_px[:, 1] - Ki[1, 2]) / Ki[1, 1],
                         np.ones(4)], axis=-1)
        R = E_ext[i][:3, :3].astype(np.float64)
        t = E_ext[i][:3,  3].astype(np.float64)
        centre = (-t @ R) * scale                                   # −Rᵀt
        far    = ((rays * (size / scale) - t) @ R) * scale
        segs = [np.stack([centre, far[j]]) for j in range(4)]
        segs += [np.stack([far[j], far[(j + 1) % 4]]) for j in range(4)]
        out.append((np.stack(segs).astype(np.float32),
                    tuple(int(x) for x in _CAM_TINT[k % len(_CAM_TINT)])))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Headless renderer (numpy z-buffer + cv2 — no OpenGL)
# ══════════════════════════════════════════════════════════════════════════════

def _virtual_camera(
    target: np.ndarray,
    dist:   float,
    azim:   float,
    elev:   float,
    up:     np.ndarray,
    width:  int,
    height: int,
    fov_deg: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orbit camera looking at *target*. Returns (R_w2c, t_w2c, K)."""
    up = up / np.linalg.norm(up)
    # Build a horizontal basis orthogonal to `up`.
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(up, seed); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    a, e = np.radians(azim), np.radians(elev)
    direction = np.cos(e) * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(e) * up
    eye = target + dist * direction

    forward = target - eye; forward /= np.linalg.norm(forward)
    right   = np.cross(forward, up); right /= np.linalg.norm(right)
    down    = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward])                    # world → camera
    t_w2c = -R_w2c @ eye

    f = 0.5 * width / np.tan(0.5 * np.radians(fov_deg))
    K = np.array([[f, 0.0, width / 2.0],
                  [0.0, f, height / 2.0],
                  [0.0, 0.0, 1.0]])
    return R_w2c, t_w2c, K


def _project(P: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points → (uv (N,2) float, z (N,) camera depth)."""
    X = P @ R.T + t[None]
    z = X[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = np.stack([K[0, 0] * X[:, 0] / safe + K[0, 2],
                   K[1, 1] * X[:, 1] / safe + K[1, 2]], axis=-1)
    return uv, z


def _render(
    points:  np.ndarray,
    colours: np.ndarray,
    verts:   np.ndarray,
    faces:   np.ndarray,
    cam:     tuple[np.ndarray, np.ndarray, np.ndarray],
    width:   int,
    height:  int,
    point_radius: int = 1,
    mesh_bgr: tuple[int, int, int] = (70, 130, 240),
    bg: int = 255,
    frusta:  list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """Painter's-algorithm render of a coloured point cloud plus meshes.

    Points and meshes are rasterised into separate colour/depth layers and
    composited by depth, so the body correctly occludes (and is occluded by)
    the reconstructed scene.
    """
    R, t, K = cam
    img  = np.full((height, width, 3), bg, np.uint8)
    zbuf = np.full((height, width), np.inf, np.float32)

    # ── point cloud: far → near, nearer writes win ────────────────────────────
    if len(points):
        uv, z = _project(points, R, t, K)
        vis = (z > 1e-3) & np.isfinite(uv).all(axis=1)
        uv, z, cols = uv[vis], z[vis], colours[vis]
        u = np.rint(uv[:, 0]).astype(np.int64)
        v = np.rint(uv[:, 1]).astype(np.int64)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z, cols = u[inside], v[inside], z[inside], cols[inside]
        order = np.argsort(-z)
        u, v, z, cols = u[order], v[order], z[order], cols[order]
        for du in range(-point_radius, point_radius + 1):
            for dv in range(-point_radius, point_radius + 1):
                uu = np.clip(u + du, 0, width - 1)
                vv = np.clip(v + dv, 0, height - 1)
                img[vv, uu]  = cols
                zbuf[vv, uu] = z

    # ── meshes: flat-shaded, painter's order, composited against zbuf ─────────
    if len(verts) and len(faces):
        mesh_col = np.zeros((height, width, 3), np.uint8)
        mesh_z   = np.full((height, width), np.inf, np.float32)
        X = verts @ R.T + t[None]
        uv, z = _project(verts, R, t, K)
        tri = faces
        tri_z = z[tri].mean(axis=1)
        v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        shade = np.clip(0.35 + 0.65 * np.abs(n[:, 2]), 0.0, 1.0)
        base  = np.array(mesh_bgr, np.float32)
        valid = (z[tri] > 1e-3).all(axis=1) & np.isfinite(uv[tri]).all(axis=(1, 2))
        order = np.argsort(-tri_z)
        order = order[valid[order]]
        poly  = np.rint(uv[tri]).astype(np.int32)               # (F, 3, 2)
        for f_idx in order:
            colour = tuple(int(x) for x in np.clip(base * shade[f_idx], 0, 255))
            cv2.fillConvexPoly(mesh_col, poly[f_idx], colour)
            cv2.fillConvexPoly(mesh_z,   poly[f_idx], float(tri_z[f_idx]))
        take = mesh_z < zbuf
        img[take]  = mesh_col[take]
        zbuf[take] = mesh_z[take]

    # ── camera frusta: drawn on top so the rig stays readable ─────────────────
    for segs, colour in (frusta or []):
        flat = segs.reshape(-1, 3)
        uv, z = _project(flat, R, t, K)
        uv = uv.reshape(-1, 2, 2)
        z  = z.reshape(-1, 2)
        for (p0, p1), (z0, z1) in zip(uv, z):
            if z0 <= 1e-3 or z1 <= 1e-3 or not np.isfinite([p0, p1]).all():
                continue
            cv2.line(img, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)),
                     colour, 2, cv2.LINE_AA)

    return img


def _subject_box(
    verts_list: list[np.ndarray],
    cam: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    pad: float = 1.9,
) -> tuple[int, int, int, int] | None:
    """Screen box around every body, shared by both panels so the crop matches."""
    R, t, K = cam
    pts = np.concatenate([v.reshape(-1, 3) for v in verts_list if len(v)]) \
        if any(len(v) for v in verts_list) else None
    if pts is None:
        return None
    uv, z = _project(pts, R, t, K)
    uv = uv[(z > 1e-3) & np.isfinite(uv).all(axis=1)]
    if not len(uv):
        return None
    cx, cy = uv[:, 0].mean(), uv[:, 1].mean()
    half = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) * pad / 2.0
    half = max(half, 40.0)
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, width)), int(min(cy + half, height))
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None


def _add_inset(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    size: int = 520,
    colour: tuple[int, int, int] = (40, 40, 40),
) -> np.ndarray:
    """Magnified crop of *box* pasted bottom-right, with the source outlined.

    The wide shot carries the place and the rig; at that scale the body is a few
    dozen pixels, so the difference the figure is about needs magnifying.
    """
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return img
    h, w = crop.shape[:2]
    s = size / max(h, w)
    crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)
    ch, cw = crop.shape[:2]

    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
    m = 24
    py, px = out.shape[0] - ch - m, out.shape[1] - cw - m
    cv2.rectangle(out, (px - 3, py - 3), (px + cw + 3, py + ch + 3), colour, -1)
    out[py:py + ch, px:px + cw] = crop
    # connector between the source box and the inset
    cv2.line(out, (x1, y1), (px, py), colour, 1, cv2.LINE_AA)
    return out


def _annotate(img: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Dark header bar with the condition and its per-camera offsets."""
    bar_h = 70
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (35, 35, 35), -1)
    cv2.putText(out, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

_CACHE_KEYS = ("offsets", "pred_pose_54", "pred_shape", "root_translation",
               "orient_R", "gt_trans")


def _save_cache(path: Path, cond: dict) -> None:
    """Persist one condition so rendering tweaks skip VGGT + fusion + placer."""
    dp = cond["depth_pack"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_start = np.int64(cond["frame_start"]),
        scale       = np.float32(cond["scale"]),
        d_depth     = dp["depth"],      d_conf = dp["depth_conf"],
        d_intr      = dp["intrinsics"], d_extr = dp["extrinsics"],
        d_present   = np.asarray(dp["present"]),
        d_paths     = np.array([str(p) for p in dp["paths"]]),
        **{k: np.asarray(cond[k]) for k in _CACHE_KEYS},
    )


def _load_cache(path: Path, tag: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        cond = {k: z[k] for k in _CACHE_KEYS}
        cond["tag"]         = tag
        cond["frame_start"] = int(z["frame_start"])
        cond["scale"]       = float(z["scale"])
        cond["depth_pack"]  = {
            "depth": z["d_depth"], "depth_conf": z["d_conf"],
            "intrinsics": z["d_intr"], "extrinsics": z["d_extr"],
            "present": z["d_present"],
            "paths": [Path(str(p)) for p in z["d_paths"]],
        }
    return cond


def _autocrop(img: np.ndarray, bg: int = 255, pad: int = 24) -> np.ndarray:
    """Trim the empty background margin so the reconstruction fills the frame."""
    ink = np.any(img != bg, axis=2)
    if not ink.any():
        return img
    ys, xs = np.where(ink)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def _fit_distance(
    target: np.ndarray,
    keypts: np.ndarray,
    width:  int,
    height: int,
    fov_deg: float,
    margin: float = 1.15,
) -> float:
    """Orbit radius that keeps every key point inside the frame."""
    if len(keypts) == 0:
        return 5.0
    r = float(np.linalg.norm(keypts - target[None], axis=-1).max())
    half_h = np.radians(fov_deg) / 2.0
    half_v = np.arctan(np.tan(half_h) * height / width)
    return max(r / np.sin(min(half_h, half_v)) * margin, 1.0)


def main(
    scene:         str  = "LectureHall_010_sidebalancerun1",
    scenes_root:   Path = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"),
    centered_root: Path = Path("/tmp/centered_test"),
    body_split:    str  = "test_body",
    checkpoint:    Path | None = None,
    out_dir:       Path = Path("figures/sync_demo"),
    cache_root:    Path = Path("figures/.sync_demo_cache"),
    work_dir:      Path | None = None,
    frame:         int | None = None,
    window:        int  = 64,
    max_shift:     int  = 30,
    seed:          int  = 0,
    azim:          float = 45.0,
    elev:          float = 25.0,
    dist:          float = -1.0,
    fov:           float = 50.0,
    width:         int  = 1600,
    height:        int  = 1200,
    point_stride:  int  = 1,
    point_radius:  int  = 1,
    conf_thr:      float = 0.5,
    depth_voxel:   float = 0.02,
    zoom:          float = 1.0,
    inset:         bool = False,
    render_only:   bool = False,
    sweep:         str | None = None,
    mask_people:   bool = True,
    show_frusta:   bool = True,
    colour_by_cam: bool = False,
    device:        str  = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Render the desynchronised / synchronised pair for one scene.

    Parameters
    ----------
    scene         : scene directory name under *scenes_root*
    centered_root : root of the centered (principal-point-cropped) RICH frames
    frame         : global frame index to render; ``None`` picks the fastest-
                    moving frame, where a temporal offset does the most damage
    window        : frames of temporal context handed to the fusion model
                    around ``frame`` (VGGT itself runs on ``frame`` only)
    max_shift     : largest injected per-camera delay, in frames
    azim/elev     : orbit viewpoint, degrees
    dist          : orbit radius in metres; <= 0 fits the whole rig in frame
    depth_voxel   : voxel size for cloud thinning, as in the Rerun viewer
    mask_people   : drop person pixels from the cloud so bodies appear once,
                    as a mesh (matches visualize_rerun's background-only cloud)
    show_frusta   : draw a wireframe frustum per camera
    colour_by_cam : tint each camera's points instead of using image colour
    """
    scene_dir = Path(scenes_root) / scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_dir) if work_dir is not None else out_dir / f".work_{scene}"
    work_root.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        checkpoint = Path(CONFIG.fusion.checkpoint_dir) / "best.pt"
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cams = _camera_names(scene_dir)
    with np.load(scene_dir / "vggt_cameras_centered.npz", allow_pickle=False) as d:
        original_npz = {k: d[k] for k in d.files}
    T_scene = original_npz["extrinsics"].shape[0]
    scale_full = np.load(scene_dir / "mapanything_scale_baseline.npy").astype(np.float32)
    logger.info(f"Scene {scene}: {T_scene} frames, cameras {cams}, "
                f"scale median {float(np.median(scale_full)):.4f} m/VGGT-unit")

    # No silent fallback: without crop_meta the placer would run with
    # offsets = 0 and quietly mis-place both panels (see eval_rich_prod.sh).
    crop_meta = Path(centered_root) / scene / "crop_meta.json"
    if not crop_meta.exists():
        raise FileNotFoundError(
            f"crop_meta.json missing at {crop_meta} — the centered squashfs is "
            f"probably not mounted; refusing to run with kp2d offsets = 0"
        )

    files_by_cam = [_cam_frame_files(centered_root, scene, cam) for cam in cams]
    for cam, files in zip(cams, files_by_cam):
        if len(files) < T_scene:
            raise RuntimeError(
                f"{cam}: {len(files)} frames on disk but {T_scene} VGGT rows — "
                f"image and camera indexing would disagree"
            )

    tracks = _load_tracks(scene_dir, cams)
    delays = _sample_offsets(len(cams), max_shift, seed)
    logger.info(f"Injected delays δ = {dict(zip(cams, delays.tolist()))}")

    # Delays are drawn first: the frame is chosen for how much *these* delays hurt.
    t_star = (_pick_frame(tracks, T_scene, window, max_shift, delays)
              if frame is None else int(frame))
    w0 = t_star - window // 2
    w1 = w0 + window
    if w0 < 0 or w1 + max_shift > T_scene:
        raise ValueError(f"Window [{w0}, {w1}) + max_shift {max_shift} escapes the scene ({T_scene} frames)")

    logger.info("Estimating offsets with the Synchronizer on the desynced tracks …")
    est = _estimate_offsets(tracks, delays, T_scene, device, max_shift)
    # estimate_initial_times reports the shift that was *applied* to the stream
    # (a camera pulled δ frames earlier comes back as −δ), so undoing it adds
    # the estimate to the injected delay rather than subtracting it.
    residual = delays + est
    predicted = -est            # same sign convention as the injected delay

    logger.info("  synchronizer result — did it recover the injected delay?")
    logger.info(f"    {'camera':<10}{'injected':>10}{'predicted':>11}{'error':>8}   verdict")
    for k, cam in enumerate(cams):
        err = int(residual[k])
        verdict = "exact" if err == 0 else ("off by 1" if abs(err) == 1 else f"WRONG by {err:+d}")
        logger.info(f"    {cam:<10}{int(delays[k]):>10}{int(predicted[k]):>11}{err:>8}   {verdict}")
    n_exact = int((residual == 0).sum())
    logger.info(f"    {n_exact}/{len(cams)} cameras exact, "
                f"mean |error| {np.abs(residual).mean():.2f} frames, "
                f"max |error| {int(np.abs(residual).max())} frames")
    if np.abs(residual).sum() > np.abs(delays).sum():
        logger.warning(
            "Residual is larger than the injected delay — the synchronizer's sign "
            "convention may be inverted relative to this script."
        )

    # Everything above is cheap (numpy + the synchronizer).  Below is the 4-minute
    # part — VGGT, the fusion model, the placer — so its outputs are cached and
    # --render-only replays them, making viewpoint work a seconds-long loop.
    # Deliberately NOT under out_dir: the cache is a property of the scene +
    # frame, not of where a particular render is written, so switching --out-dir
    # to try a second framing must not orphan it.
    cache_dir = Path(cache_root) / scene
    conditions = []
    if render_only:
        for tag in ("desync", "sync"):
            cpath = cache_dir / f"{tag}.npz"
            if not cpath.exists():
                raise FileNotFoundError(
                    f"--render-only needs {cpath}; run once without it first"
                )
            cond = _load_cache(cpath, tag)
            # The squashfs mount point carries the PID, so cached absolute paths
            # die with the run that wrote them.  Rebase on the current mount.
            cond["depth_pack"]["paths"] = [
                Path(centered_root) / p.parent.parent.name / p.parent.name / p.name
                for p in cond["depth_pack"]["paths"]
            ]
            conditions.append(cond)
        logger.info(f"Loaded cached conditions from {cache_dir} (no VGGT / fusion / placer)")
    else:
        runner = VGGTPreprocessor(weights=CONFIG.data.vggt_omega_checkpoint, device=device)
        model  = _build_model()
        state  = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state["model"])
        logger.info(f"Loaded checkpoint {checkpoint} (epoch {state.get('epoch', '?')})")
        dev = torch.device(device)

        for tag, offsets in (("desync", delays), ("sync", residual)):
            cond = _run_condition(
                tag=tag, scene=scene, scene_dir=scene_dir, cams=cams, offsets=offsets,
                files_by_cam=files_by_cam, original_npz=original_npz, scale_full=scale_full,
                work_root=work_root, centered_root=centered_root, runner=runner, model=model,
                device=dev, body_split=body_split, w0=w0, w1=w1, t_star=t_star,
            )
            _save_cache(cache_dir / f"{tag}.npz", cond)
            conditions.append(cond)
        logger.info(f"Cached conditions → {cache_dir}  (re-render with --render-only)")

    # Shared viewpoint: centred on the synchronised subject so both panels are
    # rendered from exactly the same pose — only the reconstruction differs.
    smplx_dir = Path(CONFIG.data.smplx_model_path)
    ref = conditions[1]
    t_ref = t_star - ref["frame_start"]
    ref_trans = ref["root_translation"][t_ref]
    finite = np.isfinite(ref_trans).all(axis=-1)
    if finite.any():
        centre = ref_trans[finite].mean(axis=0)
    else:
        gt = ref["gt_trans"][t_ref]
        centre = gt[np.isfinite(gt).all(axis=-1)].mean(axis=0)
        logger.warning("Synchronised placement failed at t* — centring the view on the GT root")
    # World up from the placed body's own vertical axis (SMPL-X canonical +Y),
    # which beats assuming the VGGT world is gravity-aligned — cam_00 defines
    # that frame, so its tilt would tilt the whole figure.
    up = np.array([0.0, -1.0, 0.0])
    _orient = ref["orient_R"][t_ref]
    _ok = [p for p in range(_orient.shape[0]) if np.isfinite(_orient[p]).all()]
    if _ok:
        up = _orient[_ok[0]][:, 1] / max(np.linalg.norm(_orient[_ok[0]][:, 1]), 1e-9)
        logger.info(f"World up from body axis: {np.round(up, 3).tolist()}")

    summary = {
        "scene": scene, "frame": t_star, "window": [w0, w1], "seed": seed,
        "cameras": cams,
        "injected_delay": delays.tolist(),
        "predicted_delay": predicted.tolist(),
        "raw_initial_times": est.tolist(),
        "per_camera_error": residual.tolist(),
        "exact_cameras": n_exact,
        "mean_abs_error_frames": float(np.abs(residual).mean()),
        "max_abs_error_frames": int(np.abs(residual).max()),
    }

    # Build both scenes before choosing the viewpoint: the orbit radius is fitted
    # so the whole rig — every camera frustum and the reconstructed place — stays
    # in frame, the layout used in "Reconstructing People, Places, and Cameras".
    for cond in conditions:
        t_local = t_star - cond["frame_start"]
        cond["mesh"] = _mesh_at(cond, t_local, smplx_dir)
        frame_by_cam = {c: t_star + int(o) for c, o in zip(cams, cond["offsets"].tolist())}
        logger.info(f"[{cond['tag']}] depth cloud (frames {frame_by_cam}):")
        cond["cloud"] = _point_cloud(
            cond["depth_pack"], cond["scale"], cams, frame_by_cam, scene_dir,
            VGGT_RESOLUTION, point_stride, conf_thr, depth_voxel, mask_people,
            colour_by_cam,
        )
        cond["frusta"] = (
            _camera_frusta(cond["depth_pack"], cond["scale"], cams, VGGT_RESOLUTION)
            if show_frusta else []
        )

    # How much did the desync actually move the body?  Cheap, and it says
    # whether this frame is worth rendering without having to eyeball the PNGs.
    v_desync, v_sync = conditions[0]["mesh"][0], conditions[1]["mesh"][0]
    if len(v_desync) and v_desync.shape == v_sync.shape:
        dv = np.linalg.norm(v_desync - v_sync, axis=-1)
        t_d = conditions[0]["root_translation"][t_star - conditions[0]["frame_start"]]
        t_s = conditions[1]["root_translation"][t_star - conditions[1]["frame_start"]]
        root_shift = np.linalg.norm(t_d - t_s, axis=-1)
        root_shift = float(np.nanmean(root_shift)) if np.isfinite(root_shift).any() else float("nan")
        logger.info(f"desync vs sync at t*: mean vertex shift {dv.mean() * 100:.1f} cm, "
                    f"max {dv.max() * 100:.1f} cm, root shift {root_shift * 100:.1f} cm")
        summary["mesh_shift_cm"] = float(dv.mean() * 100)
        summary["mesh_shift_max_cm"] = float(dv.max() * 100)
        summary["root_shift_cm"] = root_shift * 100

    # Fit on the rig (camera centres + subject), NOT the cloud: VGGT depth runs
    # out to the far background, and fitting that pushes the eye tens of metres
    # back and shrinks the subject to a speck.  Distant scene points simply fall
    # outside the frame.
    cam_centres = np.array([segs[0, 0] for segs, _ in conditions[1]["frusta"]], np.float32) \
        if conditions[1]["frusta"] else np.zeros((0, 3), np.float32)
    keypts = np.concatenate([a for a in (cam_centres, centre[None]) if len(a)])
    view_centre = (keypts.min(axis=0) + keypts.max(axis=0)) / 2.0
    dist_used = dist if dist > 0 else _fit_distance(view_centre, keypts, width, height, fov)
    logger.info(f"View: centre {np.round(view_centre, 2).tolist()}, "
                f"dist {dist_used:.2f} m, azim {azim}, elev {elev}")
    cam = _virtual_camera(view_centre, dist_used * zoom, azim, elev, up, width, height, fov)
    inset_box = _subject_box([c["mesh"][0] for c in conditions], cam, width, height) \
        if inset else None
    summary["view"] = {"centre": view_centre.tolist(), "dist": float(dist_used),
                       "azim": azim, "elev": elev, "fov": fov}

    def _render_cond(cond: dict, cam_p) -> np.ndarray:
        verts, faces, _ = cond["mesh"]
        pts, cols = cond["cloud"]
        return _render(
            pts, cols, verts.reshape(-1, 3) if len(verts) else np.zeros((0, 3), np.float32),
            np.concatenate([faces + i * verts.shape[1] for i in range(len(verts))])
            if len(verts) else np.zeros((0, 3), np.int32),
            cam_p, width, height, point_radius, frusta=cond["frusta"],
        )

    # ── viewpoint sweep: one contact sheet, pick the framing, then render it ──
    if sweep:
        specs = []
        for item in sweep.split(","):
            parts = [float(x) for x in item.split(":")]
            specs.append((parts[0],                                   # azim
                          parts[1] if len(parts) > 1 else elev,       # elev
                          parts[2] if len(parts) > 2 else zoom))      # zoom
        tiles, tile_h = [], 460
        for a, e, z in specs:
            cam_s = _virtual_camera(view_centre, dist_used * z, a, e, up, width, height, fov)
            tile = _autocrop(_render_cond(conditions[1], cam_s))
            s = tile_h / tile.shape[0]
            tile = cv2.resize(tile, (int(tile.shape[1] * s), tile_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 34), (35, 35, 35), -1)
            cv2.putText(tile, f"azim {a:g}  elev {e:g}  zoom {z:g}", (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
            logger.info(f"  sweep azim={a:g} elev={e:g} zoom={z:g} → {tile.shape[1]}x{tile.shape[0]}")

        cols_n = min(3, len(tiles))
        rows_n = int(np.ceil(len(tiles) / cols_n))
        cell_w = max(t.shape[1] for t in tiles)
        sheet = np.full((rows_n * tile_h, cols_n * cell_w, 3), 255, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols_n)
            sheet[r * tile_h:(r + 1) * tile_h, c * cell_w:c * cell_w + t.shape[1]] = t
        sheet_path = out_dir / f"{scene}_sweep.png"
        cv2.imwrite(str(sheet_path), sheet)
        logger.info(f"Wrote {sheet_path}  ({sheet.shape[1]}x{sheet.shape[0]}) — "
                    f"pick one, then re-run with --azim/--elev/--zoom")
        return

    for cond in conditions:
        verts, faces, placed = cond["mesh"]
        pts, cols = cond["cloud"]
        logger.info(f"[{cond['tag']}] {len(pts)} depth points, "
                    f"{len(placed)} placed persons at t*={t_star}")

        cond["img"] = _render(
            pts, cols, verts.reshape(-1, 3) if len(verts) else np.zeros((0, 3), np.float32),
            np.concatenate([faces + i * verts.shape[1] for i in range(len(verts))])
            if len(verts) else np.zeros((0, 3), np.int32),
            cam, width, height, point_radius, frusta=cond["frusta"],
        )
        if inset_box is not None:
            cond["img"] = _add_inset(cond["img"], inset_box)
        summary[f"{cond['tag']}_placed_persons"] = len(placed)
        summary[f"{cond['tag']}_depth_points"]   = int(len(pts))

    # One crop box for both panels: the pair must stay pixel-comparable, and an
    # independent trim per image would shift the scene between them.
    ink = np.zeros(conditions[0]["img"].shape[:2], bool)
    for cond in conditions:
        ink |= np.any(cond["img"] != 255, axis=2)
    ys, xs = np.where(ink)
    if len(ys):
        pad = 24
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, height)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, width)
        logger.info(f"Auto-crop: {width}x{height} → {x1 - x0}x{y1 - y0} "
                    f"({100 * (1 - (x1 - x0) * (y1 - y0) / (width * height)):.0f}% empty margin removed)")
        for cond in conditions:
            cond["img"] = cond["img"][y0:y1, x0:x1]

    for cond in conditions:
        img = cond["img"]
        if cond["tag"] == "desync":
            title = "Without temporal synchronisation"
            sub = "injected per-camera delay  " + ", ".join(
                f"{c}:{d:+d}" for c, d in zip(cams, delays.tolist()))
        else:
            title = "After temporal synchronisation"
            sub = ("predicted delay  "
                   + ", ".join(f"{c}:{d:+d}" for c, d in zip(cams, predicted.tolist()))
                   + f"   |   error {residual.tolist()} frames"
                   + f"   |   {n_exact}/{len(cams)} exact")
        img = _annotate(img, f"{title}   -   {scene}, frame {t_star}", sub)

        out_path = out_dir / f"{scene}_{cond['tag']}.png"
        cv2.imwrite(str(out_path), img)
        logger.info(f"Wrote {out_path}  ({img.shape[1]}x{img.shape[0]})")

    with open(out_dir / f"{scene}_sync_demo.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {out_dir / f'{scene}_sync_demo.json'}")


if __name__ == "__main__":
    tyro.cli(main)
