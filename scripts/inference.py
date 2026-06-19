"""Inference script: fuse multi-view body estimates with BodyPlacer.

Usage
-----
    python scripts/inference.py \
        --scene_dir  /path/to/scene_output \
        --checkpoint /path/to/fusion_checkpoint.pt \
        --output     /path/to/inference_result.npz \
        [--device    cuda]

Input layout (``scene_dir``)
-----------------------------
    scene_dir/
        <cam_A>/body_data/person_<id>.npz   ← ghost pipeline outputs
        <cam_B>/body_data/person_<id>.npz
        ...
        vggt_cameras.npz                    ← VGGT extrinsics / intrinsics
        vggt_depth.npz                      ← VGGT depth maps

Each person_<id>.npz must contain at least:
    frame_indices       (T_local,)
    smplx_body_pose     (T_local, 63)
    smplx_global_orient (T_local, 3)
    smplx_betas         (T_local, 10)
    pred_keypoints_3d   (T_local, J, 3)
    pred_keypoints_2d   (T_local, J, 2+)

Output
------
    inference_result.npz containing:
        fused_pose          (T, P, 54, 6) — fused body pose in 6D (root excluded)
        fused_betas         (P, 10)       — refined per-person shape
        root_translation    (T, P, 3)     — root 3D position in world (cam0) space, NaN where invisible
        global_orient_R     (N_valid, P, 3, 3)  — body orientation matrices where estimated
        global_orient_frames (N_valid, P)         — global frame indices for the orient estimates
        person_ids          (P,)          — ghost person IDs in slot order
        camera_names        (K,)          — camera names in slot order
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the repo root is on sys.path when called from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.fusion_module_v2 import PoseFusionModule
from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Axis-angle → 6D rotation  (row-major, matches fusion_dataset.py training convention)
# ---------------------------------------------------------------------------

def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle ``(..., 3)`` to 6-D rotation ``(..., 6)``.

    Uses the first two rows of the rotation matrix — matches the convention in
    fusion_dataset.py used when training the fusion model.
    Falls back to identity on failure.
    """
    from scipy.spatial.transform import Rotation as SciR
    shape = aa.shape[:-1]
    try:
        mats = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()  # (N, 3, 3)
    except Exception:
        return np.zeros(shape + (6,), dtype=np.float32)
    sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)  # rows → (N, 6)
    return sixd.reshape(shape + (6,)).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_scene(scene_dir: Path) -> tuple[
    list[Path],             # cam_dirs in sorted order
    list[dict[int, dict]],  # raw[cam_idx][pid] = {field: array}
]:
    """Load body_data from all camera subdirectories."""
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())
    if not cam_dirs:
        raise FileNotFoundError(
            f"No camera directories with body_data/ found in {scene_dir}"
        )

    raw: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_persons: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            data = np.load(npz_path, allow_pickle=False)
            cam_persons[pid] = {k: data[k] for k in data.files}
        raw.append(cam_persons)
        pids = list(cam_persons.keys())
        logger.info(f"  {cam_dir.name}: {len(pids)} person(s) — PIDs {pids}")

    return cam_dirs, raw


# ---------------------------------------------------------------------------
# Tensor assembly
# ---------------------------------------------------------------------------

def _build_tensors(
    raw: list[dict[int, dict]],
    num_joints: int = 55,
) -> tuple[
    torch.Tensor,   # pose         (1, T, K, P, J-1, 6)  root excluded
    torch.Tensor,   # person_mask  (1, T, K, P)
    torch.Tensor,   # shape        (1, T, K, P, 10)
    list[int],      # person_ids   (P,)
    int,            # frame_start
]:
    """Assemble model input tensors from the loaded body data.

    The root joint (index 0) is excluded from ``pose`` because BodyPlacer
    handles global orientation separately.  PoseFusionModule also drops it
    when it receives all 55 joints, but excluding it here makes intent clear.
    """
    # Collect the union of all person IDs and frame indices.
    all_pids: list[int] = sorted(
        {pid for cam in raw for pid in cam}
    )
    all_frames: list[int] = sorted(
        {int(fi) for cam in raw for pdata in cam.values()
         for fi in pdata["frame_indices"]}
    )
    if not all_pids or not all_frames:
        raise RuntimeError("No person data found across any camera.")

    frame_start = all_frames[0]
    frame_end   = all_frames[-1] + 1
    T = frame_end - frame_start
    K = len(raw)
    P = len(all_pids)
    J = num_joints - 1  # root excluded

    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    pose_arr  = np.zeros((T, K, P, J, 6), dtype=np.float32)
    mask_arr  = np.zeros((T, K, P),       dtype=np.float32)
    shape_arr = np.zeros((T, K, P, 10),   dtype=np.float32)

    for k, cam in enumerate(raw):
        for pid, pdata in cam.items():
            p = pid_to_slot[pid]
            fi = pdata["frame_indices"].astype(int)

            go = pdata.get("smplx_global_orient")   # (T_local, 3)
            bp = pdata.get("smplx_body_pose")        # (T_local, 63)
            lh = pdata.get("smplx_left_hand_pose")   # (T_local, 45) or None
            rh = pdata.get("smplx_right_hand_pose")  # (T_local, 45) or None
            betas = pdata.get("smplx_betas")         # (T_local, 10)

            for local_t, global_t in enumerate(fi):
                t = global_t - frame_start
                if t < 0 or t >= T:
                    continue

                if go is None or bp is None:
                    continue

                # Stack global_orient + body_pose (+ hands if present)
                parts = [go[local_t].reshape(1, 3), bp[local_t].reshape(21, 3)]
                if lh is not None:
                    parts.append(lh[local_t].reshape(15, 3))
                if rh is not None:
                    parts.append(rh[local_t].reshape(15, 3))
                aa = np.concatenate(parts, axis=0)  # (22 or 52, 3)

                # Pad face joints (jaw, leye, reye) with zeros → identity
                if aa.shape[0] < num_joints:
                    aa = np.concatenate(
                        [aa, np.zeros((num_joints - aa.shape[0], 3), dtype=np.float32)],
                        axis=0,
                    )  # (55, 3)

                sixd = _aa_to_6d(aa)   # (55, 6)
                # slot 0 = global_orient, excluded from pose
                pose_arr[t, k, p] = sixd[1:]
                mask_arr[t, k, p] = 1.0

                if betas is not None:
                    shape_arr[t, k, p] = betas[local_t, :10]

    pose_t  = torch.from_numpy(pose_arr).unsqueeze(0)   # (1, T, K, P, J, 6)
    mask_t  = torch.from_numpy(mask_arr).unsqueeze(0)   # (1, T, K, P)
    shape_t = torch.from_numpy(shape_arr).unsqueeze(0)  # (1, T, K, P, 10)

    logger.info(
        f"Assembled tensors: T={T} K={K} P={P} J={J} | "
        f"visible frames: {int(mask_arr.sum())}/{T*K*P}"
    )
    return pose_t, mask_t, shape_t, all_pids, frame_start


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint: Path, device: torch.device) -> PoseFusionModule:
    """Load PoseFusionModule from a checkpoint.

    The checkpoint must be a dict with either:
    - ``"model_state_dict"`` (saved by TrainerV2), or
    - raw state dict at top level.

    Hyper-parameters are inferred from the checkpoint; defaults match training.
    """
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    embedding_dim    = state["joint_id_embedding.weight"].shape[1]
    num_joints       = state["joint_id_embedding.weight"].shape[0]
    num_layers       = sum(1 for k in state if k.startswith("layers.") and k.endswith(".ff.norm.weight"))
    max_temporal_len = state["temporal_pe.pe"].shape[0]

    model = PoseFusionModule(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_joints=num_joints,
        max_temporal_len=max_temporal_len,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    logger.info(
        f"Loaded checkpoint: embedding_dim={embedding_dim} "
        f"num_layers={num_layers} num_joints={num_joints}"
    )
    return model


# ---------------------------------------------------------------------------
# Placement via PnP
# ---------------------------------------------------------------------------

def _build_vggt_cameras(
    placer: BodyPlacer,
    scale_per_frame: np.ndarray,
) -> np.ndarray:
    """Convert VGGT per-frame extrinsics to visualizer camera format ``(T, K, 8)``.

    Format matches the GT camera encoding in fusion_dataset.py:
        [quat_wxyz (4), t_w2c (3), focal (1)]
    where ``quat_wxyz`` encodes R_w2c (world-to-camera) and ``t_w2c`` is the
    raw world-to-camera translation column, in metres after per-frame scaling.

    Uses raw per-frame VGGT extrinsics (not median-stabilized) so the camera
    array is temporally consistent with the per-frame scale used in PnP.

    Args:
        scale_per_frame: ``(T,)`` float32 — metres per VGGT unit, one per frame.
    """
    from scipy.spatial.transform import Rotation as SciR_local

    T, K = placer.extrinsics.shape[:2]
    out = np.zeros((T, K, 8), dtype=np.float32)
    out[:, :, 0] = 1.0  # default identity quaternion w=1

    for t in range(T):
        s = float(scale_per_frame[t])
        for k in range(K):
            if not placer.cam_valid[t, k]:
                continue
            R_w2c = placer.extrinsics[t, k, :3, :3].astype(np.float64)
            t_w2c = placer.extrinsics[t, k, :3, 3].astype(np.float32) * s
            focal  = float(placer.intrinsics[t, k, 0, 0])

            # Re-orthogonalise (VGGT R may drift slightly from SO(3))
            U, _, Vt = np.linalg.svd(R_w2c)
            R_w2c_ortho = (U @ Vt).astype(np.float32)

            quat_xyzw = SciR_local.from_matrix(R_w2c_ortho).as_quat()  # xyzw
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            out[t, k, :4]  = quat_wxyz
            out[t, k, 4:7] = t_w2c
            out[t, k, 7]   = focal

    return out


def _run_placer(
    scene_dir: Path,
    cam_dirs: list[Path],
    raw: list[dict[int, dict]],
    all_pids: list[int],
    frame_start: int,
    T: int,
    smplx_model_path: Path,
    fused_pose: np.ndarray,
    crop_meta_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate world-space root translation + orientation via Procrustes DLT.

    For each (person, frame): DLT-triangulates SMPL-X joints from 2D keypoints
    across all cameras, then Procrustes-aligns the canonical FK skeleton to
    recover global_orient and pelvis translation jointly.

    Args:
        fused_pose:  (T, P, 54, 6) fused body pose from the fusion model (6D).
                     Used for the Procrustes FK step; falls back to raw SAM3D
                     per-camera body_pose if None.

    Returns
    -------
    root_translation  : (T, P, 3) float32, NaN where invisible
    orient_R          : (T, P, 3, 3) float32, NaN where invisible
    vggt_cameras      : (T, K, 8) float32, visualizer camera format
    """
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    if isinstance(smplx_model_path, dict):
        _smplx_arg = smplx_model_path
    elif _gender_json.exists():
        _smplx_arg = resolve_smplx_models(scene_dir.name, Path(smplx_model_path).parent, _gender_json)
    else:
        _smplx_arg = smplx_model_path
    placer = BodyPlacer(scene_dir, _smplx_arg, crop_meta_path=crop_meta_path)
    P = len(all_pids)
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    # Betas: mean SAM3D betas across all cameras and frames.
    _betas_lists: dict[int, list[np.ndarray]] = {}
    for cam_dir in cam_dirs:
        for pid in all_pids:
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if bf.exists():
                d = np.load(bf, allow_pickle=False)
                if "smplx_betas" in d.files:
                    _betas_lists.setdefault(pid, []).append(d["smplx_betas"].mean(0))
    betas_by_pid: dict[int, np.ndarray] = {
        pid: np.mean(v, axis=0).astype(np.float32)
        for pid, v in _betas_lists.items()
    }
    for pid in all_pids:
        betas_by_pid.setdefault(pid, np.zeros(10, dtype=np.float32))

    # Build fused_betas_map for scale estimation
    fused_betas_map = {
        cam_dir / "body_data" / f"person_{pid}.npz": betas_by_pid[pid]
        for cam_dir in cam_dirs
        for pid in all_pids
        if (cam_dir / "body_data" / f"person_{pid}.npz").exists()
    }

    # Build per-pid fused pose arrays (used for FK in both scale estimation and Procrustes).
    fused_pose_by_pid: dict[int, np.ndarray] = {
        pid: fused_pose[:, pid_to_slot[pid]]   # (T, 54, 6)
        for pid in all_pids
    }

    scale_per_frame = placer.load_mapanything_scale()
    if scale_per_frame is not None:
        logger.info(f"  scale: using MapAnything  median={float(np.median(scale_per_frame)):.4f} m/VGGT-unit")
    else:
        logger.info("Estimating VGGT depth scale (triangulation) — MapAnything scale not found ...")
        scale_per_frame = placer.estimate_scale_triangulated(
            fused_betas_map=fused_betas_map,
            fused_pose_by_pid=fused_pose_by_pid,
            frame_start=frame_start,
        )
        logger.info(f"  scale: triangulated  median={float(np.median(scale_per_frame)):.4f} m/VGGT-unit")
    # Depth arrays (~3 GB) are only needed for depth-based methods; free them
    # before the DLT loop to avoid running out of memory on long sequences.
    del placer.depth_mm, placer.depth_conf
    placer.depth_mm = placer.depth_conf = None

    logger.info("Running Procrustes DLT translation + orientation (MHR70) ...")
    trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
        scale=scale_per_frame,
        all_pids=set(all_pids),
        pred_betas_by_pid=betas_by_pid,
        fused_pose_by_pid=fused_pose_by_pid,
        frame_start=frame_start,
    )

    root_translation = np.full((T, P, 3),    np.nan, dtype=np.float32)
    orient_R         = np.full((T, P, 3, 3), np.nan, dtype=np.float32)

    for pid, frames in trans_dict.items():
        p = pid_to_slot.get(pid)
        if p is None:
            continue
        for global_t, t_vec in frames.items():
            t_rel = int(global_t) - frame_start
            if 0 <= t_rel < T:
                root_translation[t_rel, p] = t_vec

    for pid, frames in orient_dict.items():
        p = pid_to_slot.get(pid)
        if p is None:
            continue
        for global_t, R_mat in frames.items():
            t_rel = int(global_t) - frame_start
            if 0 <= t_rel < T:
                orient_R[t_rel, p] = R_mat

    # Canonical pelvis offset J_can[0] per person (depends only on betas).
    _zero_pose   = np.zeros((1, 63), dtype=np.float32)
    _zero_orient = np.zeros((1,  3), dtype=np.float32)
    J0_by_pid: dict[int, np.ndarray] = {
        pid: placer._smplx_fk(
            betas_by_pid[pid][np.newaxis], _zero_pose, _zero_orient
        )[0, 0]  # (3,) canonical pelvis offset
        for pid in all_pids
    }

    # ── Single-view fallback ──────────────────────────────────────────────
    # Procrustes DLT needs ≥2 cameras, so persons seen in a single view stay
    # NaN above.  Place them from that camera's SAM3D estimate instead:
    # pelvis_cam = J_can[0] + smplx_transl, pushed through the (scaled) VGGT
    # extrinsic to world.  Less accurate (SAM3D depth) but fine for viewing.
    from scipy.spatial.transform import Rotation as _SciR
    n_fallback = 0
    for k, cam_persons in enumerate(raw):
        for pid, pdata in cam_persons.items():
            p = pid_to_slot.get(pid)
            if p is None:
                continue
            fi = pdata.get("frame_indices")
            tr = pdata.get("smplx_transl")
            go = pdata.get("smplx_global_orient")
            if fi is None or tr is None or go is None:
                continue
            for i, gf in enumerate(fi.astype(int)):
                t_rel = int(gf) - frame_start
                if not (0 <= t_rel < T):
                    continue
                if np.isfinite(root_translation[t_rel, p, 0]):
                    continue
                if not placer.cam_valid[t_rel, k]:
                    continue
                s   = float(scale_per_frame[t_rel])
                ext = placer.extrinsics[t_rel, k].astype(np.float64)
                R_k, t_k = ext[:3, :3], ext[:3, 3] * s
                pelvis_cam   = J0_by_pid[pid] + tr[i]
                pelvis_world = R_k.T @ (pelvis_cam - t_k)
                root_translation[t_rel, p] = pelvis_world.astype(np.float32)
                R_cam = _SciR.from_rotvec(go[i]).as_matrix()
                orient_R[t_rel, p] = (R_k.T @ R_cam).astype(np.float32)
                n_fallback += 1
    if n_fallback:
        logger.info(f"  single-view fallback placement: {n_fallback} (frame, person) entries")

    # placer returns pelvis_world = R @ J_can[0] + t, but downstream code
    # (visualizer, evaluator) expects SMPL-X transl = t = pelvis_world - J_can[0].
    for i, pid in enumerate(all_pids):
        valid = np.isfinite(root_translation[:, i, 0])
        root_translation[valid, i] -= J0_by_pid[pid]

    vggt_cameras = _build_vggt_cameras(placer, scale_per_frame)

    return root_translation, orient_R, vggt_cameras


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost inference: fuse poses and place bodies.")
    parser.add_argument("--scene_dir",       required=True, type=Path, help="Scene output directory.")
    parser.add_argument("--checkpoint",      required=True, type=Path, help="FusionWithBetas checkpoint (.pt).")
    parser.add_argument("--smplx_model",     required=True, type=Path, help="Path to SMPLX_NEUTRAL.pkl.")
    parser.add_argument("--output",          type=Path, default=None,  help="Output .npz path (default: scene_dir/inference_result.npz).")
    parser.add_argument("--crop_meta",       type=Path, default=None,  help="Path to crop_meta.json (centered-image crop offsets) for SAM3D kp2d→VGGT correction.")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    output    = args.output or (scene_dir / "inference_result.npz")
    device    = torch.device(args.device)

    logger.info(f"Scene dir : {scene_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device    : {device}")

    # ── 1. Load raw body data ────────────────────────────────────────────────
    logger.info("Loading body data ...")
    cam_dirs, raw = _load_scene(scene_dir)

    # ── 2. Build input tensors ───────────────────────────────────────────────
    logger.info("Building input tensors ...")
    pose_t, mask_t, shape_t, all_pids, frame_start = _build_tensors(raw)

    T = pose_t.shape[1]

    # ── 3. Load model and run forward pass ──────────────────────────────────
    logger.info("Loading model ...")
    model = _load_model(args.checkpoint, device)

    pose_t  = pose_t.to(device)
    mask_t  = mask_t.to(device)
    shape_t = shape_t.to(device)

    logger.info("Running fusion forward pass ...")
    with torch.no_grad():
        pose_aggr = model(pose_t, mask_t)
        # pose_aggr : (1, T, P, J, 6)

    fused_pose  = pose_aggr[0].cpu().numpy()              # (T, P, J, 6)

    # ── 4. BodyPlacer (Procrustes DLT) ──────────────────────────────────────
    logger.info("Running BodyPlacer (Procrustes DLT) ...")
    root_translation, orient_R, vggt_cameras = _run_placer(
        scene_dir, cam_dirs, raw, all_pids, frame_start, T,
        smplx_model_path=args.smplx_model,
        fused_pose=fused_pose,
        crop_meta_path=args.crop_meta,
    )
    # orient_R: (T, P, 3, 3) — NaN where not estimated

    # ── 5. Save ──────────────────────────────────────────────────────────────
    logger.info(f"Saving results to {output} ...")
    save_dict: dict[str, np.ndarray] = {
        "fused_pose":       fused_pose,       # (T, P, J, 6)
        "root_translation": root_translation, # (T, P, 3)
        "global_orient_R":  orient_R,         # (T, P, 3, 3) — NaN where missing
        "vggt_cameras":     vggt_cameras,     # (T, K, 8) — VGGT predicted cameras
        "person_ids":       np.array(all_pids, dtype=np.int32),
        "camera_names":     np.array([d.name for d in cam_dirs]),
        "frame_start":      np.array(frame_start, dtype=np.int32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **save_dict)
    logger.info("Done.")


if __name__ == "__main__":
    main()
