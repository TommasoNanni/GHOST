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
# The scenes to evaluate, and the ghost person id(s) to synchronise in each.
#
# This is the single source of truth: the keys select the scenes, the values
# select the people.  Both used to be inferred — scenes by an exclusion list,
# people by intersecting the person ids over all cameras — but that intersection
# is a cross-view-ReID product.  It answers "which id survived in every view",
# not "which person is the annotated subject", so it silently drops a subject
# that one camera labelled differently, and silently keeps a bystander that
# happened to be tracked everywhere.
#
# The ids below were verified against RICH ground truth by
# scripts/verify_sync_scenes_gt.py: the GT root reprojects onto these tracks in
# every calibrated camera of every scene, at 0.13-0.52 body-sizes (reject
# threshold 0.8).  Re-run it after any change to segmentation or cross-view ReID.
SCENE_PIDS: dict[str, list[int]] = {
    "BBQ_001_guitar":              [1],
    "BBQ_001_juggle":              [1],
    "ParkingLot1_002_burpee3":     [1],
    "ParkingLot1_002_overfence1":  [1],
    "ParkingLot1_002_overfence2":  [1],
    "ParkingLot1_002_pushup1":     [1],
    "ParkingLot1_002_stretching1": [1],
    "ParkingLot1_004_burpeejump1": [1],
    "ParkingLot1_005_overfence1":  [1],
    "ParkingLot2_008_pushup1":     [1],
    "ParkingLot2_008_overfence1":  [1],
    "ParkingLot2_014_pushup2":     [1],
    "ParkingLot1_005_burpeejump2": [1],
    "ParkingLot2_008_burpeejump1": [1],
    "ParkingLot2_014_burpeejump1": [1],
    "ParkingLot2_015_overfence1":  [1],
    "ParkingLot2_016_stretching1": [1],
    "Pavallion_003_plankjack":     [1],
}
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


def _load_anchored(npz_path: Path, loader) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load one person and re-anchor its sequence to absolute frame 0.

    The loaders in utilities.body_data already fill interior detection gaps, but
    they anchor the result at frame_indices[0] — each camera's *own* first
    detection. RICH is hardware synced, so the absolute frame index is true time:
    a camera whose person is first detected 27 frames later than the others would
    otherwise sit 27 frames early in its array. apply_shifts slices by array
    index, so that becomes a real offset on top of the injected one, which the
    ground truth does not model — the solver then recovers injected+pre-existing
    (correct) and is scored against injected alone.

    Left-padding by frame_indices[0] with zero pose and zero confidence makes
    array index == absolute frame in every camera, so the injected shift is the
    only offset present. Zero confidence keeps the padding out of the DTW cost,
    exactly as the loaders' own gap filling does.
    """
    result = loader(npz_path)
    if result is None:
        return None
    seq, conf = result
    with np.load(str(npz_path)) as d:
        if "frame_indices" not in d.files:
            raise KeyError(f"{npz_path}: no frame_indices — cannot anchor to absolute time")
        start = int(d["frame_indices"].astype(int).min())
    if start > 0:
        seq  = torch.cat([seq .new_zeros((start, *seq .shape[1:])), seq ], dim=0)
        conf = torch.cat([conf.new_zeros((start, *conf.shape[1:])), conf], dim=0)
    return seq, conf


def _pad_to_common_length(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    """Right-pad every sequence to the longest one, in place.

    Cameras stop detecting at different frames too. Once the sequences are
    anchored to absolute frame 0, padding the tails to a common length means
    end_cuts are measured from a common absolute end rather than from each
    camera's own last detection. Padding is zero-confidence, so it is ignored.
    """
    lengths = [s.shape[0] for persons in cam_data.values() for s, _ in persons.values()]
    if not lengths:
        return
    T_max = max(lengths)
    for persons in cam_data.values():
        for pid, (seq, conf) in persons.items():
            pad = T_max - seq.shape[0]
            if pad > 0:
                seq  = torch.cat([seq,  seq .new_zeros((pad, *seq .shape[1:]))], dim=0)
                conf = torch.cat([conf, conf.new_zeros((pad, *conf.shape[1:]))], dim=0)
                persons[pid] = (seq, conf)


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
            result = _load_anchored(npz_path, load_person_smplx_pose)
            if result is None:
                logger.warning(f"{npz_path}: missing pose params, skipping")
                continue
            persons[pid] = result

        if persons:
            cam_data[cam_dir.name] = persons
            logger.info(f"  {cam_dir.name}: {len(persons)} person(s), "
                        f"T={next(iter(persons.values()))[0].shape[0]}")

    _pad_to_common_length(cam_data)
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
            result = _load_anchored(npz_path, load_person_smplx_joints)
            if result is None:
                continue
            persons[pid] = result

        if persons:
            cam_data[cam_dir.name] = persons

    _pad_to_common_length(cam_data)
    return cam_data


def select_persons(scene_name: str, cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Return the person IDs to synchronise for this scene, from SCENE_PIDS.

    Every id must be present in every camera.  A missing id means the scene is
    broken — cross-view ReID labelled the subject differently in that camera, or
    never detected it — not that the scene has one person fewer, so this raises
    rather than quietly evaluating whoever is left.
    """
    try:
        pids = SCENE_PIDS[scene_name]
    except KeyError:
        raise KeyError(
            f"{scene_name}: no SCENE_PIDS entry. Add the GT-verified person id(s) "
            f"— see scripts/verify_sync_scenes_gt.py — before evaluating this scene."
        ) from None
    if not pids:
        raise ValueError(f"{scene_name}: SCENE_PIDS entry is empty")

    for cam_id, persons in cam_data.items():
        missing = [p for p in pids if p not in persons]
        if missing:
            raise KeyError(
                f"{scene_name}/{cam_id}: person(s) {missing} absent — this camera "
                f"has {sorted(persons)}. Fix the ids (scripts/gt_reid_rich.py) or "
                f"drop the camera via SKIP_CAMERAS."
            )
    return list(pids)


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


def pooled_stats(E: np.ndarray) -> dict | None:
    """All table metrics as statistics of one pooled per-camera error set.

    Ported from run_benchmark.pooled_stats (the VisualSync-faithful benchmark) so
    that both tables report the same estimator over the same quantity. Averaging
    per-trial or per-scene summaries instead would reweight scenes with unequal
    camera counts, and would make MedAE a median-of-medians rather than a median.

    AUC@tau closed form: mean over thresholds t in [0,tau] of P(E<=t)
    == 1 - mean(min(E,tau))/tau (exact integral, no linspace discretization).
    """
    if len(E) == 0:
        return None
    auc = lambda tau: 1.0 - float(np.minimum(E, tau).mean()) / tau  # noqa: E731
    return {"mae": float(E.mean()), "median_ae": float(np.median(E)),
            "within_half": float((E <= 0.5).mean()),
            "within_1": float((E <= 1).mean()),
            "within_2": float((E <= 2).mean()),
            "auc_100ms": auc(100 / 1000 * 15), "auc_500ms": auc(500 / 1000 * 15)}


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

    # Global offsets are only defined up to a constant, so anchor both series to
    # zero before differencing (run_benchmark.run_trial does the same). true_t is
    # already anchored above; the solver's output is not.
    est = estimated.cpu()
    if torch.isnan(est).any():
        return None          # isolated camera(s): the trial has no full result
    est = est - est.min()
    errors = (est - true_t).abs()

    # Only the raw per-camera errors are returned: every table metric is a
    # statistic of those errors pooled across trials (see pooled_stats), never an
    # average of per-trial summaries.
    return {
        "true_shifts": {c: true_shifts[c] for c in cam_ids},
        "true_times":  true_t.tolist(),
        "estimated":   est.tolist(),
        "errors":      errors.tolist(),
        "mae":         errors.mean().item(),   # per-trial log line only
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
        raise RuntimeError(
            f"{scene_dir.name}: need ≥2 cameras with body_data, found {len(cam_data)}"
        )

    pids = select_persons(scene_dir.name, cam_data)

    cam_ids = list(cam_data.keys())
    logger.info(f"  Cameras: {cam_ids}")
    logger.info(f"  Persons: {pids}")

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
        logger.info(f"     MAE={result['mae']:.2f}")

    if not results:
        logger.warning(f"  All trials skipped for {scene_dir.name} — no usable results")
        return None

    all_spreads = [
        max(r["true_shifts"].values()) - min(r["true_shifts"].values())
        for r in results
    ]

    # Pool the per-camera errors of every solved trial, then take every metric
    # off that one vector. Trials that produced no result are excluded here and
    # reported as coverage (solved/N_TRIALS) instead of inflating the error.
    errors = np.concatenate([np.array(r["errors"]) for r in results])
    stats  = pooled_stats(errors)

    logger.info(f"\n  SUMMARY — {scene_dir.name}  (solved {len(results)}/{N_TRIALS} trials)")
    logger.info(f"    Shift spread    mean={np.mean(all_spreads):.1f}  median={np.median(all_spreads):.1f}  max={np.max(all_spreads):.1f}")
    logger.info(f"    MAE (frames)    {stats['mae']:.2f}")
    logger.info(f"    MedAE (frames)  {stats['median_ae']:.2f}")
    logger.info(f"    Within 0.5fr    {stats['within_half']*100:.1f}%")
    logger.info(f"    Within 1fr      {stats['within_1']*100:.1f}%")
    logger.info(f"    Within 2fr      {stats['within_2']*100:.1f}%")
    logger.info(f"    AUC@100ms       {stats['auc_100ms']*100:.1f}%")
    logger.info(f"    AUC@500ms       {stats['auc_500ms']*100:.1f}%")

    return {
        "scene":         scene_dir.name,
        "n_cameras":     len(cam_ids),
        "n_persons":     len(pids),
        "solved":        len(results),
        "n_trials":      N_TRIALS,
        "spread_mean":   float(np.mean(all_spreads)),
        "errors":        errors,
        **stats,
        "trial_results": results,
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
    scene_names = cli_scenes or sorted(SCENE_PIDS)

    logger.info(f"Scenes root: {SCENES_ROOT}")
    logger.info(f"Device: {DEVICE}  |  trials per scene: {N_TRIALS}  |  max_shift: {MAX_SHIFT}")

    unknown = [s for s in scene_names if s not in SCENE_PIDS]
    if unknown:
        raise KeyError(
            f"no SCENE_PIDS entry for: {unknown}. Add the GT-verified person id(s) "
            f"— see scripts/verify_sync_scenes_gt.py — before evaluating them."
        )
    if not scene_names:
        raise RuntimeError("SCENE_PIDS is empty — nothing to evaluate")

    scene_dirs = []
    for name in scene_names:
        d = SCENES_ROOT / name
        if not d.is_dir():
            raise FileNotFoundError(f"scene directory missing: {d}")
        scene_dirs.append(d)

    logger.info(f"Evaluating {len(scene_dirs)} scene(s): {scene_names}")

    sync = Synchronizer(use_acceleration_weights=False, device=DEVICE, min_overlap=100, max_shift=MAX_SHIFT, verbose=VERBOSE)
    rng  = np.random.default_rng(SEED)

    scene_summaries = []
    for scene_dir in scene_dirs:
        summary = run_scene(scene_dir, sync, rng)
        if summary is not None:
            scene_summaries.append(summary)

    if not scene_summaries:
        logger.error("No usable scenes found — nothing to summarise.")
        sys.exit(1)

    # TOTAL pools the per-camera errors of every solved trial of every scene, so
    # scenes are weighted by cameras x solved trials. Averaging the per-scene
    # numbers instead would give a 6-camera scene the same weight as an 8-camera
    # one; run_benchmark pools the same way.
    all_spread = [s["spread_mean"] for s in scene_summaries]
    errors     = np.concatenate([s["errors"] for s in scene_summaries])
    total      = pooled_stats(errors)
    n_solved   = sum(s["solved"]   for s in scene_summaries)
    n_total    = sum(s["n_trials"] for s in scene_summaries)

    logger.info("\n" + "=" * 60)
    logger.info(f"AGGREGATE SUMMARY  ({len(scene_summaries)} scene(s)  ×  {N_TRIALS} trials each)")
    logger.info("  metrics pooled over per-camera errors of solved trials")
    logger.info("")

    # ── per-scene table (frames) ──────────────────────────────────────────
    hdr = f"  {'Scene':<35}  {'Cams':>4}  {'Pers':>4}  {'Solved':>7}  {'Spread':>7}  {'MAE':>6}  {'MedAE':>6}  {'W-0.5':>7}  {'W-1':>7}  {'W-2':>7}  {'AUC@100':>8}  {'AUC@500':>8}"
    sep = f"  {'-'*35}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}"
    logger.info("  All values in frames; AUC thresholds at 15 fps (100ms=1.5fr, 500ms=7.5fr)")
    logger.info(hdr)
    logger.info(sep)
    for s in scene_summaries:
        cov = f"{s['solved']}/{s['n_trials']}"
        row = (
            f"  {s['scene']:<35}  {s['n_cameras']:>4}  {s['n_persons']:>4}  "
            f"{cov:>7}  {s['spread_mean']:>7.1f}  "
            f"{s['mae']:>6.2f}  {s['median_ae']:>6.2f}  "
            f"{s['within_half']*100:>6.1f}%  {s['within_1']*100:>6.1f}%  {s['within_2']*100:>6.1f}%  "
            f"{s['auc_100ms']*100:>7.1f}%  {s['auc_500ms']*100:>7.1f}%"
        )
        logger.info(row)
    logger.info(sep)
    total_row = (
        f"  {'TOTAL':<35}  {'':>4}  {'':>4}  "
        f"{f'{n_solved}/{n_total}':>7}  {np.mean(all_spread):>7.1f}  "
        f"{total['mae']:>6.2f}  {total['median_ae']:>6.2f}  "
        f"{total['within_half']*100:>6.1f}%  {total['within_1']*100:>6.1f}%  {total['within_2']*100:>6.1f}%  "
        f"{total['auc_100ms']*100:>7.1f}%  {total['auc_500ms']*100:>7.1f}%"
    )
    logger.info(total_row)
    logger.info(f"  Coverage: {n_solved / n_total * 100:.1f}%  "
                f"({len(errors)} pooled per-camera errors)")

    # ── ms conversion ─────────────────────────────────────────────────────
    for fps in [15, 30]:
        ms = 1000.0 / fps
        logger.info("")
        logger.info(f"  ── Assuming {fps} fps  (1 frame = {ms:.1f} ms) ──")
        logger.info(
            f"    MAE            {total['mae']*ms:>7.1f} ms"
        )
        logger.info(
            f"    MedAE          {total['median_ae']*ms:>7.1f} ms"
        )
        logger.info(
            f"    Within {0.5*ms:.0f} ms   {total['within_half']*100:>6.1f}%   (≤0.5 fr)"
        )
        logger.info(
            f"    Within {1*ms:.0f} ms    {total['within_1']*100:>6.1f}%   (≤1 fr)"
        )
        logger.info(
            f"    Within {2*ms:.0f} ms    {total['within_2']*100:>6.1f}%   (≤2 fr)"
        )
