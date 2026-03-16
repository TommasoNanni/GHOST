"""
Alignment experiments on multiple scenes from a common root folder.

Iterates over every subdirectory of SCENES_ROOT, runs the same random-shift
alignment experiment as alignment_experiments.py on each scene that has at
least 2 cameras and at least one common person ID across all cameras, then
prints per-scene and aggregate summaries.

Scenes listed in SKIP_SCENES are silently skipped — use this to exclude
scenes where you know cross-view re-ID is unreliable.

Usage (GPU node):
    pixi run python evaluation/alignment_experiments_multi.py
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

VERBOSE     = False   # set True to see per-person DTW offsets and shift distributions
SCENES_ROOT = Path("test_outputs/rich9_segmentation_test")
# Scene folder names (relative to SCENES_ROOT) to skip entirely.
# Add scenes here when you know cross-view re-ID is bad on them.
SKIP_SCENES: list[str] = [
    "BBQ_001_juggle",
    "Pavallion_003_018_tossball"
]
N_TRIALS  = 10    # random-shift trials per scene
MAX_SHIFT = 100   # maximum absolute shift (frames)
SEED      = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


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
                pose = np.concatenate([
                    d["smplx_body_pose"],
                    d["smplx_left_hand_pose"],
                    d["smplx_right_hand_pose"],
                ], axis=1)  # (T, 153)
                rotations = torch.from_numpy(pose.astype(np.float32)).reshape(-1, 51, 3)
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


def apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    pids: list[int],
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Give each camera all frames from its natural start position to end of recording."""
    cam_ids = list(shifts.keys())
    T_base  = min(cam_data[c][p][0].shape[0] for c in cam_ids for p in pids)
    max_s   = max(shifts.values())
    logger.info(f"  T_base={T_base}  shift_spread={max_s - min(shifts.values())}")

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []

    for cam_id in cam_ids:
        s = max_s - shifts[cam_id]
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            per_person_joints.append(rotations[s : T_base].to(DEVICE))
            per_person_confs .append(conf     [s : T_base].to(DEVICE))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)

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
    weights    = sync.cycle_consistency_weights(offset_mat)
    estimated  = sync.estimate_initial_times(offset_mat, weights)

    true_t = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
    true_t = true_t - true_t.min()

    K = len(cam_ids)
    logger.info("  Pairwise offset matrix (estimated vs. true):")
    for i in range(K):
        for j in range(i + 1, K):
            est_pair  = offset_mat[i, j].item()
            true_pair = float(true_shifts[cam_ids[j]] - true_shifts[cam_ids[i]])
            err_pair  = abs(est_pair - true_pair)
            w         = weights[i, j].item()
            flag = " <-- WRONG" if err_pair > 1.5 else ""
            logger.info(
                f"    ({cam_ids[i]} → {cam_ids[j]}): "
                f"estimated={est_pair:+.1f}  true={true_pair:+.0f}  "
                f"pairwise_err={err_pair:.1f}  weight={w:.3f}{flag}"
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


def run_scene(
    scene_dir: Path,
    sync: Synchronizer,
    rng: np.random.Generator,
) -> dict | None:
    """Run N_TRIALS experiments for one scene.

    Returns a summary dict, or None if the scene is not usable
    (fewer than 2 cameras, no common persons, or missing pose data).
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Scene: {scene_dir.name}")

    cam_data = load_scene(scene_dir)
    if len(cam_data) < 2:
        logger.warning(f"  Skipping: need ≥2 cameras, found {len(cam_data)}")
        return None

    pids = common_persons(cam_data)
    if not pids:
        logger.warning("  Skipping: no person ID common across all cameras — run cross-view ReID first.")
        return None

    cam_ids = list(cam_data.keys())
    logger.info(f"  Cameras: {cam_ids}")
    logger.info(f"  Common persons: {pids}")

    results = []
    for trial in range(N_TRIALS):
        raw_shifts  = [0] + rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=len(cam_ids) - 1).tolist()
        true_shifts = {cam_id: int(s) for cam_id, s in zip(cam_ids, raw_shifts)}

        logger.info(f"\n  ── Trial {trial + 1}/{N_TRIALS}  true shifts: {true_shifts}")

        result = run_trial(cam_data, pids, true_shifts, sync)
        results.append(result)

        for cam_id, true_t, est, err in zip(cam_ids, result["true_times"], result["estimated"], result["errors"]):
            logger.info(f"     {cam_id}: true={true_t:+.0f}  "
                        f"estimated={est:+.1f}  error={err:.1f}")
        logger.info(f"     MAE={result['mae']:.2f}  "
                    f"within-1={result['within_1']*100:.0f}%  "
                    f"within-2={result['within_2']*100:.0f}%")

    all_mae      = [r["mae"]      for r in results]
    all_within_1 = [r["within_1"] for r in results]
    all_within_2 = [r["within_2"] for r in results]
    all_spreads  = [
        max(r["true_shifts"].values()) - min(r["true_shifts"].values())
        for r in results
    ]

    logger.info(f"\n  SUMMARY — {scene_dir.name}  ({N_TRIALS} trials)")
    logger.info(f"    Shift spread  mean={np.mean(all_spreads):.1f}  median={np.median(all_spreads):.1f}  max={np.max(all_spreads):.1f}")
    logger.info(f"    MAE           mean={np.mean(all_mae):.2f}  median={np.median(all_mae):.2f}  max={np.max(all_mae):.2f}")
    logger.info(f"    Within 1fr    {np.mean(all_within_1)*100:.1f}%")
    logger.info(f"    Within 2fr    {np.mean(all_within_2)*100:.1f}%")

    return {
        "scene":      scene_dir.name,
        "n_cameras":  len(cam_ids),
        "n_persons":  len(pids),
        "mae_mean":   float(np.mean(all_mae)),
        "mae_median": float(np.median(all_mae)),
        "mae_max":    float(np.max(all_mae)),
        "within_1":   float(np.mean(all_within_1)),
        "within_2":   float(np.mean(all_within_2)),
        "trial_results": results,
    }

if __name__ == "__main__":
    logger.info(f"Scenes root: {SCENES_ROOT}")
    logger.info(f"Device: {DEVICE}  |  trials per scene: {N_TRIALS}  |  max_shift: {MAX_SHIFT}")
    if SKIP_SCENES:
        logger.info(f"Skipping scenes: {SKIP_SCENES}")

    scene_dirs = sorted(
        d for d in SCENES_ROOT.iterdir()
        if d.is_dir() and d.name not in SKIP_SCENES
    )
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {SCENES_ROOT}")

    logger.info(f"Found {len(scene_dirs)} scene(s): {[d.name for d in scene_dirs]}")

    sync = Synchronizer(device=DEVICE)
    rng  = np.random.default_rng(SEED)

    scene_summaries = []
    for scene_dir in scene_dirs:
        summary = run_scene(scene_dir, sync, rng)
        if summary is not None:
            scene_summaries.append(summary)

    if not scene_summaries:
        logger.error("No usable scenes found — nothing to summarise.")
        sys.exit(1)

    all_mae      = [s["mae_mean"]  for s in scene_summaries]
    all_within_1 = [s["within_1"]  for s in scene_summaries]
    all_within_2 = [s["within_2"]  for s in scene_summaries]

    logger.info("\n" + "=" * 60)
    logger.info(f"AGGREGATE SUMMARY  ({len(scene_summaries)} scene(s)  ×  {N_TRIALS} trials each)")
    logger.info("")
    logger.info(f"  {'Scene':<30}  {'Cams':>4}  {'Pers':>4}  {'MAE mean':>9}  {'Within-1':>9}  {'Within-2':>9}")
    logger.info(f"  {'-'*30}  {'-'*4}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}")
    for s in scene_summaries:
        logger.info(
            f"  {s['scene']:<30}  {s['n_cameras']:>4}  {s['n_persons']:>4}  "
            f"{s['mae_mean']:>9.2f}  {s['within_1']*100:>8.1f}%  {s['within_2']*100:>8.1f}%"
        )
    logger.info(f"  {'-'*30}  {'-'*4}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}")
    logger.info(
        f"  {'MEAN':<30}  {'':>4}  {'':>4}  "
        f"{np.mean(all_mae):>9.2f}  {np.mean(all_within_1)*100:>8.1f}%  {np.mean(all_within_2)*100:>8.1f}%"
    )
