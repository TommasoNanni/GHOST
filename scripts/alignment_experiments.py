"""
Alignment experiments on real pipeline output.

Loads body keypoints produced by the ghost pipeline for a given scene,
applies random temporal shifts to simulate unsynchronised cameras, then
runs the Synchronizer to recover those shifts and measures the error.

We use smplx_body_pose (joint rotations in axis-angle) as the time series
for DTW. Joint rotations are camera-frame-independent — they encode pose
without any dependency on the global orientation or camera position.
smplx_global_orient is excluded because it IS camera-dependent.

Edit the CONFIG block below to change the scene directory and parameters.

Usage (GPU node):
    pixi run python scripts/alignment_experiments.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from synchronize_videos.synchronizer import Synchronizer

VERBOSE = False   # set True to see per-person DTW offsets and shift distributions

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE_DIR = Path("test_outputs/reid_logging_segmentation_test/BBQ_001_juggle")
N_TRIALS  = 10     # number of random-shift trials
MAX_SHIFT = 120     # maximum absolute shift (frames)
SEED      = 42



def load_scene(scene_dir: Path) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Load joint rotation sequences for all cameras in a scene.

    Loads smplx_body_pose (T, 63), smplx_left_hand_pose (T, 45) and
    smplx_right_hand_pose (T, 45), concatenates them and reshapes to
    (T, 51, 3): 21 body + 15 left hand + 15 right hand joints, each as
    a 3D axis-angle vector. Global orientation is excluded because it
    depends on the camera frame.

    Confidence comes from pred_joint_confidence[:, 1:52], which covers
    the same 51 joints in SMPL-X joint ordering (skipping root at index 0).

    Returns
    -------
    cam_data : {cam_id: {person_id: (rotations T×51×3, conf T×51)}}
    """
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    for cam_dir in sorted(scene_dir.iterdir()):
        body_dir = cam_dir / "body_data"
        if not cam_dir.is_dir() or not body_dir.exists():
            continue

        persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            with np.load(str(npz_path)) as d:
                required = {"smplx_body_pose", "smplx_left_hand_pose",
                            "smplx_right_hand_pose", "pred_joint_confidence"}
                if not required.issubset(d.files):
                    logger.warning(f"{npz_path}: missing pose params, skipping")
                    continue
                # Concatenate all pose params (excluding global orient) → (T, 153)
                # then reshape to (T, 51, 3): 21 body + 15 left hand + 15 right hand
                pose = np.concatenate([
                    d["smplx_body_pose"],       # (T, 63)  — 21 joints
                    d["smplx_left_hand_pose"],   # (T, 45)  — 15 joints
                    d["smplx_right_hand_pose"],  # (T, 45)  — 15 joints
                ], axis=1)  # (T, 153)
                rotations = torch.from_numpy(pose.astype(np.float32)).reshape(-1, 51, 3)
                # Confidence for the same 51 joints (indices 1..51, skipping root at 0)
                conf = torch.from_numpy(
                    d["pred_joint_confidence"][:, 1:52].astype(np.float32)
                )  # (T, 51)

            persons[pid] = (rotations, conf)

        if persons:
            cam_data[cam_dir.name] = persons
            logger.info(f"  {cam_dir.name}: {len(persons)} person(s), "
                        f"T={next(iter(persons.values()))[0].shape[0]}")

    return cam_data


def common_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Return person IDs present in every camera."""
    sets = [set(persons.keys()) for persons in cam_data.values()]
    return sorted(set.intersection(*sets))


def embed_sequence(seq: torch.Tensor, shift: int, total_len: int) -> torch.Tensor:
    """Place *seq* (T × ...) into a timeline of length *total_len* starting at *shift*.

    Positions before/after the sequence are edge-replicated.
    """
    T = seq.shape[0]
    out = torch.zeros(total_len, *seq.shape[1:])
    start     = max(shift, 0)
    end       = min(shift + T, total_len)
    src_start = start - shift
    src_end   = end   - shift
    out[start:end] = seq[src_start:src_end]
    if start > 0:
        out[:start] = seq[0]
    if end < total_len:
        out[end:] = seq[src_end - 1]
    return out


def apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    pids: list[int],
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Build shifted rotation sequences for the synchronizer."""
    cam_ids   = list(shifts.keys())
    T_base    = min(cam_data[c][p][0].shape[0] for c in cam_ids for p in pids)
    max_shift = max(abs(s) for s in shifts.values())
    total_len = T_base + max_shift + 10

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []

    for cam_id in cam_ids:
        shift = shifts[cam_id]
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            per_person_joints.append(embed_sequence(rotations, shift, total_len).to(DEVICE))
            per_person_confs.append(embed_sequence(conf,      shift, total_len).to(DEVICE))
        joints_list.append(per_person_joints)
        confs_list.append(per_person_confs)

    return joints_list, confs_list


def run_trial(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    pids: list[int],
    true_shifts: dict[str, int],
    sync: Synchronizer,
) -> dict:
    """Run one alignment experiment and return a result dict."""
    cam_ids = list(true_shifts.keys())

    joints_list, confs_list = apply_shifts(cam_data, true_shifts, pids)

    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    estimated  = sync.estimate_initial_times(offset_mat)  # K, normalised so min=0

    true_t = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
    true_t = true_t - true_t.min()

    # Log pairwise offset matrix vs. true pairwise offsets
    K = len(cam_ids)
    logger.info("  Pairwise offset matrix (estimated vs. true):")
    for i in range(K):
        for j in range(i + 1, K):
            est_pair  = offset_mat[i, j].item()
            true_pair = float(true_shifts[cam_ids[j]] - true_shifts[cam_ids[i]])
            err_pair  = abs(est_pair - true_pair)
            flag = " <-- WRONG" if err_pair > 1.5 else ""
            logger.info(
                f"    ({cam_ids[i]} → {cam_ids[j]}): "
                f"estimated={est_pair:+.1f}  true={true_pair:+.0f}  "
                f"pairwise_err={err_pair:.1f}{flag}"
            )

    errors = (estimated.cpu() - true_t).abs()
    mae    = errors.mean().item()

    return {
        "true_shifts": {c: true_shifts[c] for c in cam_ids},
        "true_times":  true_t.tolist(),
        "estimated":   estimated.cpu().tolist(),
        "errors":      errors.tolist(),
        "mae":         mae,
        "within_1":    (errors <= 1).float().mean().item(),
        "within_2":    (errors <= 2).float().mean().item(),
    }


if __name__ == "__main__":
    logger.info(f"Scene: {SCENE_DIR}")
    logger.info(f"Device: {DEVICE}  |  trials: {N_TRIALS}  |  max_shift: {MAX_SHIFT}")

    cam_data = load_scene(SCENE_DIR)
    if len(cam_data) < 2:
        raise RuntimeError(f"Need at least 2 cameras, found {len(cam_data)}")

    pids = common_persons(cam_data)
    if not pids:
        raise RuntimeError("No person ID common across all cameras — run cross-view ReID first.")

    cam_ids = list(cam_data.keys())
    logger.info(f"Cameras: {cam_ids}")
    logger.info(f"Common persons: {pids}")

    sync = Synchronizer(device=DEVICE)
    rng  = np.random.default_rng(SEED)

    results = []
    for trial in range(N_TRIALS):
        raw_shifts  = [0] + rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=len(cam_ids) - 1).tolist()
        true_shifts = {cam_id: int(s) for cam_id, s in zip(cam_ids, raw_shifts)}

        logger.info(f"\n── Trial {trial + 1}/{N_TRIALS}  true shifts: {true_shifts}")

        result = run_trial(cam_data, pids, true_shifts, sync)
        results.append(result)

        for cam_id, true_t, est, err in zip(cam_ids, result["true_times"], result["estimated"], result["errors"]):
            logger.info(f"   {cam_id}: true={true_t:+.0f}  "
                        f"estimated={est:+.1f}  error={err:.1f}")
        logger.info(f"   MAE={result['mae']:.2f}  "
                    f"within-1={result['within_1']*100:.0f}%  "
                    f"within-2={result['within_2']*100:.0f}%")

    all_mae      = [r["mae"]      for r in results]
    all_within_1 = [r["within_1"] for r in results]
    all_within_2 = [r["within_2"] for r in results]

    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY over {N_TRIALS} trials")
    logger.info(f"  MAE        mean={np.mean(all_mae):.2f}  median={np.median(all_mae):.2f}  max={np.max(all_mae):.2f}")
    logger.info(f"  Within 1fr {np.mean(all_within_1)*100:.1f}%")
    logger.info(f"  Within 2fr {np.mean(all_within_2)*100:.1f}%")
