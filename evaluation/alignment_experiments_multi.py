"""
Alignment experiments on multiple scenes from a common root folder.

Iterates over every subdirectory of SCENES_ROOT, runs the same random-shift
alignment experiment as alignment_experiments.py on each scene that has at
least 2 cameras and at least one common person ID across all cameras, then
prints per-scene and aggregate summaries.

Scenes listed in SKIP_SCENES are silently skipped — use this to exclude
scenes where you know cross-view re-ID is unreliable.

Individual cameras can be excluded per scene via SKIP_CAMERAS without
dropping the whole scene (e.g. a camera with track-stealing artefacts).

Usage (GPU node):
    pixi run python evaluation/alignment_experiments_multi.py
    pixi run python evaluation/alignment_experiments_multi.py --scene Pavallion_002_plankjack
    pixi run python evaluation/alignment_experiments_multi.py --scene BBQ_001_juggle ParkingLot1_005_pushup3
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from synchronize_videos.synchronizer import Synchronizer
from utilities.body_data import load_person_smplx_pose, load_person_smplx_joints

VERBOSE     = False     # set True to see per-person DTW offsets and shift distributions
SCENES_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
# Scene folder names (relative to SCENES_ROOT) to skip entirely.
# Add scenes here when you know cross-view re-ID is bad on them.
SKIP_SCENES: list[str] = [
    "Pavallion_013_plankjack"
]
# When non-empty, only these scenes are evaluated (overrides SKIP_SCENES).
# Useful for focused diagnosis on specific failing cases.
ONLY_SCENES: list[str] = [
    "BBQ_001_guitar",
    "BBQ_001_juggle",
    "ParkingLot1_002_burpee3",
    "ParkingLot1_002_overfence1",
    "ParkingLot1_002_overfence2",
    "ParkingLot1_002_pushup1",
    "ParkingLot1_002_stretching1",
]
# Per-scene cameras to exclude (e.g. track-stealing artefacts).
# The scene is still evaluated with its remaining cameras.
SKIP_CAMERAS: dict[str, list[str]] = {
    "Pavallion_003_018_tossball": ["cam_06"],
    "ParkingLot2_008_pushup2": ["cam_03"],
    "ParkingLot2_014_takingphotos2": ["cam_01"],
}
N_TRIALS  = 10      # random-shift trials per scene (set to 10 for full runs)
MAX_SHIFT = 30     # maximum absolute shift (frames) — matches paper setting
SEED      = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_scene(
    scene_dir: Path,
    exclude_cameras: list[str] | None = None,
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Load joint rotation sequences for all cameras in a scene.

    Loads smplx_body_pose (T, 63), smplx_left_hand_pose (T, 45) and
    smplx_right_hand_pose (T, 45), concatenates them and reshapes to
    (T, 51, 3): 21 body + 15 left hand + 15 right hand joints, each as
    a 3D axis-angle vector. Global orientation is excluded because it
    depends on the camera frame.

    Confidence comes from pred_joint_confidence[:, 1:52], which covers
    the same 51 joints in SMPL-X joint ordering (skipping root at index 0).

    Parameters
    ----------
    exclude_cameras : list of camera directory names to skip for this scene.

    Returns
    -------
    cam_data : {cam_id: {person_id: (rotations T×51×3, conf T×51)}}
    """
    exclude_cameras = exclude_cameras or []
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    for cam_dir in sorted(scene_dir.iterdir()):
        body_dir = cam_dir / "body_data"
        if not cam_dir.is_dir() or not body_dir.exists():
            continue
        if cam_dir.name in exclude_cameras:
            logger.info(f"  {cam_dir.name}: skipped (in SKIP_CAMERAS)")
            continue

        persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            result = load_person_smplx_pose(npz_path)
            if result is None:
                logger.warning(f"{npz_path}: missing pose params, skipping")
                continue
            persons[pid] = result

        if persons:
            cam_data[cam_dir.name] = persons
            logger.info(f"  {cam_dir.name}: {len(persons)} person(s), "
                        f"T={next(iter(persons.values()))[0].shape[0]}")

    return cam_data


def load_scene_joints(
    scene_dir: Path,
    exclude_cameras: list[str] | None = None,
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Same as load_scene but loads SMPL-X FK joint positions instead of rotations.

    Returns cam_data with (T, 51, 3) root-relative positions and (T, 51) confidence.
    """
    exclude_cameras = exclude_cameras or []
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    for cam_dir in sorted(scene_dir.iterdir()):
        body_dir = cam_dir / "body_data"
        if not cam_dir.is_dir() or not body_dir.exists():
            continue
        if cam_dir.name in exclude_cameras:
            continue

        persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            result = load_person_smplx_joints(npz_path)
            if result is None:
                continue
            persons[pid] = result

        if persons:
            cam_data[cam_dir.name] = persons

    return cam_data


def common_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Return person IDs present in every camera."""
    sets = [set(persons.keys()) for persons in cam_data.values()]
    return sorted(set.intersection(*sets))


def apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    end_cuts: dict[str, int],
    pids: list[int],
    min_overlap: int = 100,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]] | None:
    """Slice sequences to simulate temporal offsets and random end times.

    Each camera starts at a different offset and ends at a different frame,
    simulating realistic asynchronous recordings with no common start or end.
    Returns None if any camera ends up with fewer than min_overlap frames.
    """
    cam_ids = list(shifts.keys())
    max_s   = max(shifts.values())
    logger.info(f"  shift_spread={max_s - min(shifts.values())}")

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []

    for cam_id in cam_ids:
        s  = max_s - shifts[cam_id]
        ec = end_cuts[cam_id]
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            T   = rotations.shape[0]
            end = T - ec if ec > 0 else T
            remaining = end - s
            if remaining < min_overlap:
                logger.warning(
                    f"  {cam_id} has only {remaining} frames after shift "
                    f"(need ≥{min_overlap}) — skipping sync"
                )
                return None
            per_person_joints.append(rotations[s:end].to(DEVICE))
            per_person_confs .append(conf     [s:end].to(DEVICE))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)

    return joints_list, confs_list


def _analyze_distributional_pair(
    cam_i: str,
    cam_j: str,
    log_p: torch.Tensor,
    true_k: int,
) -> None:
    """Log diagnostic info for one pairwise log-distribution vs the true offset."""
    n  = len(log_p)
    S  = (n - 1) // 2
    finite = log_p.isfinite()
    n_valid = int(finite.sum().item())

    probs = log_p.exp()
    probs[~finite] = 0.0

    # top-5 peaks
    top_k   = min(5, n_valid)
    top_vals, top_idx = probs.topk(top_k)
    map_est = int(top_idx[0].item()) - S
    peaks_str = "  ".join(
        f"k={int(top_idx[i].item()) - S:+d}(p={top_vals[i].item():.3f})"
        for i in range(top_k)
    )

    # normalised entropy
    p_fin = log_p[finite].exp()
    entropy = -(p_fin * log_p[finite]).sum().item()
    norm_entropy = entropy / math.log(n_valid) if n_valid > 1 else 0.0

    # probability ratio between 1st and 2nd peak
    gap_str = f"{top_vals[0].item() / max(top_vals[1].item(), 1e-9):.2f}x" if top_k > 1 else "∞"

    # true offset rank and cost-scale analysis
    # With adaptive temperature, log_p is in normalised units (std(log_p) ≈ 1).
    # Report log-prob advantage directly instead of back-converting to radians.
    ki_true = int(true_k) + S
    if 0 <= ki_true < n and finite[ki_true]:
        true_prob = probs[ki_true].item()
        rank = int((probs > true_prob).sum().item()) + 1
        logp_adv = log_p[int(top_idx[0].item())].item() - log_p[ki_true].item()
        true_str = f"k={true_k:+d}(rank=#{rank}  logp_adv={logp_adv:+.3f})"
    else:
        true_str = f"k={true_k:+d}(no signal)"

    # std(log_p) ≈ 1/temperature_multiplier by construction with adaptive temp
    cost_std = log_p[finite].float().std().item()

    flag = " <-- WRONG" if abs(map_est - true_k) > 1.5 else ""
    logger.info(
        f"    ({cam_i}→{cam_j}): MAP={map_est:+d}  true={true_k:+d}"
        f"  logp_std={cost_std:.3f}  entropy={norm_entropy:.3f}  gap={gap_str}{flag}"
        f"\n      peaks: [{peaks_str}]  true: [{true_str}]"
    )


def run_trial(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    pids: list[int],
    true_shifts: dict[str, int],
    end_cuts: dict[str, int],
    sync: Synchronizer,
) -> dict:
    """Run one alignment experiment and return a result dict."""
    cam_ids = list(true_shifts.keys())

    result_shifts = apply_shifts(cam_data, true_shifts, end_cuts, pids)
    if result_shifts is None:
        return None
    joints_list, confs_list = result_shifts

    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    weights    = sync.cycle_consistency_weights(offset_mat)
    estimated  = sync.estimate_initial_times(offset_mat, weights)

    true_t = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
    true_t = true_t - true_t.min()

    K = len(cam_ids)
    if VERBOSE:
        logger.info("  Pairwise offset matrix (estimated vs. true):")
        for i in range(K):
            for j in range(i + 1, K):
                true_pair = int(true_shifts[cam_ids[j]] - true_shifts[cam_ids[i]])
                if isinstance(offset_mat, torch.Tensor):
                    est_pair = offset_mat[i, j].item()
                    err_pair = abs(est_pair - true_pair)
                    flag = " <-- WRONG" if err_pair > 1.5 else ""
                    logger.info(
                        f"    ({cam_ids[i]} → {cam_ids[j]}): "
                        f"estimated={est_pair:+.1f}  true={true_pair:+d}  "
                        f"pairwise_err={err_pair:.1f}  weight={weights[i, j].item():.3f}{flag}"
                    )
                elif isinstance(offset_mat[i][j], torch.Tensor):
                    _analyze_distributional_pair(cam_ids[i], cam_ids[j], offset_mat[i][j], true_pair)
                else:
                    est_pair = float(offset_mat[i][j][0][0]) if offset_mat[i][j] else 0.0
                    err_pair = abs(est_pair - true_pair)
                    flag = " <-- WRONG" if err_pair > 1.5 else ""
                    logger.info(
                        f"    ({cam_ids[i]} → {cam_ids[j]}): "
                        f"estimated={est_pair:+.1f}  true={true_pair:+d}  "
                        f"pairwise_err={err_pair:.1f}  weight=N/A{flag}"
                    )

    errors = (estimated.cpu() - true_t).abs()
    mae    = errors.mean().item()

    return {
        "true_shifts": {c: true_shifts[c] for c in cam_ids},
        "true_times":  true_t.tolist(),
        "estimated":   estimated.cpu().tolist(),
        "errors":      errors.tolist(),
        "mae":         mae,
        "median_ae":   errors.median().item(),
        "within_half": (errors <= 0.5).float().mean().item(),
        "within_1":    (errors <= 1).float().mean().item(),
        "within_2":    (errors <= 2).float().mean().item(),
    }


def run_scene(
    scene_dir: Path,
    sync: Synchronizer,
    rng: np.random.Generator,
) -> dict | None:
    """Run N_TRIALS experiments for one scene.

    Returns a summary dict, or None if the scene is not usable.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Scene: {scene_dir.name}")

    exclude_cams = SKIP_CAMERAS.get(scene_dir.name, [])
    if exclude_cams:
        logger.info(f"  Excluding cameras: {exclude_cams}")
    cam_data = load_scene(scene_dir, exclude_cameras=exclude_cams)
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
        end_cuts    = {cam_id: int(e) for cam_id, e in zip(cam_ids, rng.integers(0, MAX_SHIFT + 1, size=len(cam_ids)).tolist())}

        logger.info(f"\n  ── Trial {trial + 1}/{N_TRIALS}  true shifts: {true_shifts}  end_cuts: {end_cuts}")

        result = run_trial(cam_data, pids, true_shifts, end_cuts, sync)
        if result is None:
            logger.warning(f"  Trial {trial + 1} skipped (insufficient frames after shift)")
            continue
        results.append(result)

        for cam_id, true_t, est, err in zip(cam_ids, result["true_times"], result["estimated"], result["errors"]):
            logger.info(f"     {cam_id}: true={true_t:+.0f}  "
                        f"estimated={est:+.1f}  error={err:.1f}")
        logger.info(f"     MAE={result['mae']:.2f}  "
                    f"within-1={result['within_1']*100:.0f}%  "
                    f"within-2={result['within_2']*100:.0f}%")

    if not results:
        logger.warning(f"  All trials skipped for {scene_dir.name} — no usable results")
        return None

    all_mae        = [r["mae"]         for r in results]
    all_median_ae  = [r["median_ae"]   for r in results]
    all_within_half= [r["within_half"] for r in results]
    all_within_1   = [r["within_1"]    for r in results]
    all_within_2   = [r["within_2"]    for r in results]
    all_spreads  = [
        max(r["true_shifts"].values()) - min(r["true_shifts"].values())
        for r in results
    ]

    # AUC: pool all per-camera errors across every trial, integrate CDF (identical to VisualSync)
    all_errors = np.concatenate([np.array(r["errors"]) for r in results])

    def _auc(threshold_frames, n=1000):
        ts = np.linspace(0, threshold_frames, n)
        return float(np.mean([(all_errors <= t).mean() for t in ts]))

    auc_100ms = _auc(100 / 1000 * 15)   # 1.5 frames at 15 fps
    auc_500ms = _auc(500 / 1000 * 15)   # 7.5 frames at 15 fps

    logger.info(f"\n  SUMMARY — {scene_dir.name}  ({N_TRIALS} trials)")
    logger.info(f"    Shift spread    mean={np.mean(all_spreads):.1f}  median={np.median(all_spreads):.1f}  max={np.max(all_spreads):.1f}")
    logger.info(f"    MAE (frames)    mean={np.mean(all_mae):.2f}  median={np.median(all_mae):.2f}  max={np.max(all_mae):.2f}")
    logger.info(f"    MedAE (frames)  mean={np.mean(all_median_ae):.2f}")
    logger.info(f"    Within 0.5fr    {np.mean(all_within_half)*100:.1f}%")
    logger.info(f"    Within 1fr      {np.mean(all_within_1)*100:.1f}%")
    logger.info(f"    Within 2fr      {np.mean(all_within_2)*100:.1f}%")
    logger.info(f"    AUC@100ms       {auc_100ms*100:.1f}%")
    logger.info(f"    AUC@500ms       {auc_500ms*100:.1f}%")

    return {
        "scene":           scene_dir.name,
        "n_cameras":       len(cam_ids),
        "n_persons":       len(pids),
        "spread_mean":     float(np.mean(all_spreads)),
        "mae_mean":        float(np.mean(all_mae)),
        "mae_median":      float(np.median(all_mae)),
        "mae_max":         float(np.max(all_mae)),
        "median_ae_mean":  float(np.mean(all_median_ae)),
        "within_half":     float(np.mean(all_within_half)),
        "within_1":        float(np.mean(all_within_1)),
        "within_2":        float(np.mean(all_within_2)),
        "auc_100ms":       auc_100ms,
        "auc_500ms":       auc_500ms,
        "trial_results":   results,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", nargs="+", metavar="SCENE",
                        help="One or more scene names to evaluate (default: all)")
    parser.add_argument("--max-shift", type=int, default=MAX_SHIFT,
                        help="Maximum absolute shift in frames (default: %(default)s)")
    args = parser.parse_args()

    MAX_SHIFT = args.max_shift
    cli_scenes: list[str] = args.scene or []
    effective_only = cli_scenes or ONLY_SCENES

    logger.info(f"Scenes root: {SCENES_ROOT}")
    logger.info(f"Device: {DEVICE}  |  trials per scene: {N_TRIALS}  |  max_shift: {MAX_SHIFT}")
    if effective_only:
        logger.info(f"Running only scenes: {effective_only}")
    elif SKIP_SCENES:
        logger.info(f"Skipping scenes: {SKIP_SCENES}")

    scene_dirs = sorted(
        d for d in SCENES_ROOT.iterdir()
        if d.is_dir() and (
            d.name in effective_only if effective_only
            else d.name not in SKIP_SCENES
        )
    )
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {SCENES_ROOT}")

    logger.info(f"Found {len(scene_dirs)} scene(s): {[d.name for d in scene_dirs]}")

    sync = Synchronizer(method="cross_corr", use_acceleration_weights=False, device=DEVICE, min_overlap=100, max_shift=MAX_SHIFT, verbose=VERBOSE)
    rng  = np.random.default_rng(SEED)

    scene_summaries = []
    for scene_dir in scene_dirs:
        summary = run_scene(scene_dir, sync, rng)
        if summary is not None:
            scene_summaries.append(summary)

    if not scene_summaries:
        logger.error("No usable scenes found — nothing to summarise.")
        sys.exit(1)

    all_spread     = [s["spread_mean"]     for s in scene_summaries]
    all_mae        = [s["mae_mean"]        for s in scene_summaries]
    all_median_ae  = [s["median_ae_mean"]  for s in scene_summaries]
    all_within_half= [s["within_half"]     for s in scene_summaries]
    all_within_1   = [s["within_1"]        for s in scene_summaries]
    all_within_2   = [s["within_2"]        for s in scene_summaries]

    logger.info("\n" + "=" * 60)
    logger.info(f"AGGREGATE SUMMARY  ({len(scene_summaries)} scene(s)  ×  {N_TRIALS} trials each)")
    logger.info("")

    # ── per-scene table (frames) ──────────────────────────────────────────
    hdr = f"  {'Scene':<35}  {'Cams':>4}  {'Pers':>4}  {'Spread':>7}  {'MAE':>6}  {'MedAE':>6}  {'W-0.5':>7}  {'W-1':>7}  {'W-2':>7}  {'AUC@100':>8}  {'AUC@500':>8}"
    sep = f"  {'-'*35}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}"
    logger.info("  All values in frames; AUC thresholds at 15 fps (100ms=1.5fr, 500ms=7.5fr)")
    logger.info(hdr)
    logger.info(sep)
    for s in scene_summaries:
        row = (
            f"  {s['scene']:<35}  {s['n_cameras']:>4}  {s['n_persons']:>4}  "
            f"{s['spread_mean']:>7.1f}  "
            f"{s['mae_mean']:>6.2f}  {s['median_ae_mean']:>6.2f}  "
            f"{s['within_half']*100:>6.1f}%  {s['within_1']*100:>6.1f}%  {s['within_2']*100:>6.1f}%  "
            f"{s['auc_100ms']*100:>7.1f}%  {s['auc_500ms']*100:>7.1f}%"
        )
        logger.info(row)
    logger.info(sep)
    all_auc_100 = [s["auc_100ms"] for s in scene_summaries]
    all_auc_500 = [s["auc_500ms"] for s in scene_summaries]
    mean_row = (
        f"  {'MEAN':<35}  {'':>4}  {'':>4}  "
        f"{np.mean(all_spread):>7.1f}  "
        f"{np.mean(all_mae):>6.2f}  {np.mean(all_median_ae):>6.2f}  "
        f"{np.mean(all_within_half)*100:>6.1f}%  {np.mean(all_within_1)*100:>6.1f}%  {np.mean(all_within_2)*100:>6.1f}%  "
        f"{np.mean(all_auc_100)*100:>7.1f}%  {np.mean(all_auc_500)*100:>7.1f}%"
    )
    logger.info(mean_row)

    # ── ms conversion ─────────────────────────────────────────────────────
    for fps in [15, 30]:
        ms = 1000.0 / fps
        logger.info("")
        logger.info(f"  ── Assuming {fps} fps  (1 frame = {ms:.1f} ms) ──")
        logger.info(
            f"    MAE            {np.mean(all_mae)*ms:>7.1f} ms"
        )
        logger.info(
            f"    MedAE          {np.mean(all_median_ae)*ms:>7.1f} ms"
        )
        logger.info(
            f"    Within {0.5*ms:.0f} ms   {np.mean(all_within_half)*100:>6.1f}%   (≤0.5 fr)"
        )
        logger.info(
            f"    Within {1*ms:.0f} ms    {np.mean(all_within_1)*100:>6.1f}%   (≤1 fr)"
        )
        logger.info(
            f"    Within {2*ms:.0f} ms    {np.mean(all_within_2)*100:>6.1f}%   (≤2 fr)"
        )
