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

from fusion.fusion_module_v2 import FusionWithBetas, PoseFusionModule, BetasAggregator
from fusion.placer import BodyPlacer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Axis-angle → 6D rotation  (column-major, matches convert_pose in dataset)
# ---------------------------------------------------------------------------

def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle ``(..., 3)`` to 6-D rotation ``(..., 6)``.

    Uses the first two columns of the rotation matrix (continuous representation).
    Falls back to identity on failure.
    """
    from scipy.spatial.transform import Rotation as SciR
    shape = aa.shape[:-1]
    try:
        mats = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()  # (N, 3, 3)
    except Exception:
        return np.zeros(shape + (6,), dtype=np.float32)
    sixd = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1)  # (N, 6)
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

def _load_model(checkpoint: Path, device: torch.device) -> FusionWithBetas:
    """Load FusionWithBetas from a checkpoint.

    The checkpoint must be a dict with either:
    - ``"model_state_dict"`` (saved by TrainerV2), or
    - raw state dict at top level.

    Hyper-parameters are inferred from the checkpoint; defaults match training.
    """
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Infer embedding_dim from joint_id_embedding weight shape.
    embedding_dim = state["pose_module.joint_id_embedding.weight"].shape[1]
    num_joints    = state["pose_module.joint_id_embedding.weight"].shape[0]
    num_layers    = sum(1 for k in state if k.startswith("pose_module.layers.") and k.endswith(".ff.norm.weight"))

    pose_module = PoseFusionModule(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_joints=num_joints,
    )
    betas_agg = BetasAggregator()
    model = FusionWithBetas(pose_module, betas_agg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    logger.info(
        f"Loaded checkpoint: embedding_dim={embedding_dim} "
        f"num_layers={num_layers} num_joints={num_joints}"
    )
    return model


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _run_placer(
    scene_dir: Path,
    cam_dirs: list[Path],
    raw: list[dict[int, dict]],
    all_pids: list[int],
    frame_start: int,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BodyPlacer to get world-space root translations and orientations.

    Returns
    -------
    root_translation  : (T, P, 3) float32, NaN where not visible
    orient_R          : (N_frames, P, 3, 3) float32
    orient_frames     : (N_frames, P) int32
    """
    placer = BodyPlacer(scene_dir)

    P = len(all_pids)
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    logger.info("Estimating VGGT depth scale ...")
    scale = placer.estimate_scale()
    logger.info(f"  scale = {scale:.4f} m / VGGT unit")

    root_translation = np.full((T, P, 3), np.nan, dtype=np.float32)

    for k, (cam_dir, cam) in enumerate(zip(cam_dirs, raw)):
        for pid, pdata in cam.items():
            p = pid_to_slot[pid]
            body_file = cam_dir / "body_data" / f"person_{pid}.npz"
            logger.info(f"  root translation: {cam_dir.name} / person {pid} ...")
            t_cam = placer.estimate_root_translation(k, body_file, scale)
            # t_cam is (T_local, 3); map back to global timeline
            fi = pdata["frame_indices"].astype(int)
            for local_t, global_t in enumerate(fi):
                t = global_t - frame_start
                if 0 <= t < T:
                    root_translation[t, p] = t_cam[local_t]

    # Global orientation: triangulate across cameras per person.
    # collect union of frame indices covered by any camera for each person.
    all_orient_frames: list[np.ndarray] = []
    all_orient_R: list[np.ndarray] = []

    # We store per-slot results in lists, then align to a common frame set.
    slot_fi: list[np.ndarray | None] = [None] * P
    slot_R:  list[np.ndarray | None] = [None] * P

    for pid in all_pids:
        p = pid_to_slot[pid]
        body_files_per_cam: dict[int, Path] = {}
        for k, (cam_dir, cam) in enumerate(zip(cam_dirs, raw)):
            if pid in cam:
                body_files_per_cam[k] = cam_dir / "body_data" / f"person_{pid}.npz"

        if not body_files_per_cam:
            continue

        logger.info(f"  global orient: person {pid} ({len(body_files_per_cam)} cams) ...")
        fi_out, R_out = placer.estimate_global_orient(body_files_per_cam)
        slot_fi[p] = fi_out  # (N,) global frame indices
        slot_R[p]  = R_out   # (N, 3, 3)

    # Find the union of all frame indices across all slots.
    all_fi_sets = [set(fi.tolist()) for fi in slot_fi if fi is not None]
    if all_fi_sets:
        union_frames = sorted(set.union(*all_fi_sets))
    else:
        union_frames = []

    N_frames = len(union_frames)
    orient_R      = np.full((N_frames, P, 3, 3), np.nan, dtype=np.float32)
    orient_frames = np.array(union_frames, dtype=np.int32)  # (N_frames,)

    frame_to_row = {f: i for i, f in enumerate(union_frames)}
    for p in range(P):
        if slot_fi[p] is None:
            continue
        for local_i, global_f in enumerate(slot_fi[p]):
            row = frame_to_row.get(int(global_f))
            if row is not None:
                orient_R[row, p] = slot_R[p][local_i]

    return root_translation, orient_R, orient_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost inference: fuse poses and place bodies.")
    parser.add_argument("--scene_dir",  required=True, type=Path, help="Scene output directory.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="FusionWithBetas checkpoint (.pt).")
    parser.add_argument("--output",     type=Path, default=None,  help="Output .npz path (default: scene_dir/inference_result.npz).")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
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
        pose_aggr, betas_out = model(pose_t, mask_t, shape=shape_t)
        # pose_aggr : (1, T, P, J, 6)
        # betas_out : (1, P, 10) or None

    fused_pose  = pose_aggr[0].cpu().numpy()              # (T, P, J, 6)
    fused_betas = betas_out[0].cpu().numpy() if betas_out is not None else None  # (P, 10)

    # ── 4. BodyPlacer ────────────────────────────────────────────────────────
    logger.info("Running BodyPlacer ...")
    root_translation, orient_R, orient_frames = _run_placer(
        scene_dir, cam_dirs, raw, all_pids, frame_start, T
    )

    # ── 5. Save ──────────────────────────────────────────────────────────────
    logger.info(f"Saving results to {output} ...")
    save_dict: dict[str, np.ndarray] = {
        "fused_pose":           fused_pose,          # (T, P, J, 6)
        "root_translation":     root_translation,    # (T, P, 3)
        "global_orient_R":      orient_R,            # (N_frames, P, 3, 3)
        "global_orient_frames": orient_frames,       # (N_frames,)
        "person_ids":           np.array(all_pids, dtype=np.int32),
        "camera_names":         np.array([d.name for d in cam_dirs]),
        "frame_start":          np.array(frame_start, dtype=np.int32),
    }
    if fused_betas is not None:
        save_dict["fused_betas"] = fused_betas       # (P, 10)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **save_dict)
    logger.info("Done.")


if __name__ == "__main__":
    main()
