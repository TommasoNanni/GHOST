"""
Alignment experiments on a 14-scene EgoHumans subset (2 scenes per activity).

Same random-shift alignment experiment as alignment_experiments_multi.py (RICH),
run on the EgoHumans exo cameras. EgoHumans scenes are nested one level deeper
than RICH ones — <root>/<activity>/<scene>/<cam>/ — so scenes are keyed by
"<activity>/<scene>" throughout.

Body tracks are read from body_data_clean/, not body_data/: that directory is
the output of scripts/build_clean_body_data.py, i.e. the manual within-view ops
plus the manual cross-view re-ID, so person_<id>.npz means the same real person
in every camera of the scene. Raw body_data/ ids are per-camera and would make
the DTW compare two different people.

Usage (GPU node):
    pixi run python -m evaluation.alignment_experiments_multi_egohumans
    pixi run python -m evaluation.alignment_experiments_multi_egohumans --scene 03_fencing/005_fencing
    pixi run python -m evaluation.alignment_experiments_multi_egohumans --scene 005_fencing 007_tennis
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
SCENES_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
# The scenes to evaluate, and the ghost person id(s) to synchronise in each.
#
# 14 scenes: 2 per activity, picked as the two scenes of each activity with the
# longest span that is covered in every camera by every listed person (so a
# 30-frame shift plus a 30-frame end cut still leaves >= 100 overlapping frames).
# All 14 have the full 4 exo cameras.
#
# The ids are the UNION of the globally-consistent body_data_clean ids over the
# cameras of the scene, not the intersection.  estimate_couple_offset scores each
# pair of cameras independently and skips any person it cannot score there (a
# zero-confidence person makes every cost entry +inf, so p_scores comes out empty
# and that person is dropped for that pair alone; the combined cost is the mean
# over the persons that did score, so the other pairs are unaffected).  Listing
# only the intersection would therefore throw away real evidence: a person seen
# in 3 of 4 cameras can still align the 3 pairs among those cameras.
#
# Cameras that lack a listed person get a zero pose / zero confidence sequence
# (see load_scene), which is what triggers the per-pair skip above.
#
# Unlike RICH, these ids are not a cross-view-ReID product — they come from
# manual_reid.json via scripts/build_clean_body_data.py.
SCENE_PIDS: dict[str, list[int]] = {
    "01_tagging/007_tagging":         [1, 2, 3, 4],
    "01_tagging/011_tagging":         [1, 2, 3, 4],
    "02_lego/001_legoassemble":       [1, 2, 3],
    "02_lego/003_legoassemble":       [1, 2, 3],
    "03_fencing/005_fencing":         [1, 2, 3],
    "03_fencing/006_fencing":         [1, 2, 3],
    "04_basketball/001_basketball":   [1, 2, 3, 4],
    "04_basketball/011_basketball":   [1, 2, 3, 4],
    "05_volleyball/004_volleyball":   [1, 2, 3, 4],
    "05_volleyball/011_volleyball":   [1, 2, 3, 4],
    "06_badminton/022_badminton":     [1, 2, 3, 4],
    "06_badminton/031_badminton":     [1, 2, 3, 4],
    "07_tennis/007_tennis":           [1, 2],
    "07_tennis/012_tennis":           [1, 2],
}
# Per-scene cameras to exclude (e.g. track-stealing artefacts).
# The scene is still evaluated with its remaining cameras.
SKIP_CAMERAS: dict[str, list[str]] = {}
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
    detection. The EgoHumans exo GoPros are hardware synced, so the absolute
    frame index is true time:
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


def _fill_missing_persons(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    pids: list[int],
) -> dict[int, int]:
    """Give every camera an entry for every requested person, in place.

    A person purged from one view (or never detected there) is inserted as a zero
    pose / zero confidence sequence rather than omitted. estimate_couple_offset
    takes index-aligned person lists of equal length from both cameras, so the
    alternative is dropping that person from the whole scene.

    Zero confidence makes w = conf1 * conf2 vanish, so every entry of that
    person's cost matrix is +inf and the person is skipped in exactly the pairs
    that involve this camera — the pairs between two cameras that both see them
    still use them at full weight.

    Returns {pid: number of cameras that really observed it}, for logging.
    """
    seen = {pid: sum(pid in persons for persons in cam_data.values()) for pid in pids}
    T_ref = max(
        (s.shape[0] for persons in cam_data.values() for s, _ in persons.values()),
        default=0,
    )
    shapes = next(
        ((s.shape[1:], c.shape[1:]) for persons in cam_data.values() for s, c in persons.values()),
        None,
    )
    if shapes is None:
        return seen
    seq_shape, conf_shape = shapes
    for persons in cam_data.values():
        for pid in pids:
            if pid not in persons:
                persons[pid] = (torch.zeros((T_ref, *seq_shape)), torch.zeros((T_ref, *conf_shape)))
    return seen


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
        body_dir = cam_dir / "body_data_clean"
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
        body_dir = cam_dir / "body_data_clean"
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


def scene_key(scene_dir: Path) -> str:
    """Identify a scene by "<activity>/<scene>" — its path relative to SCENES_ROOT.

    EgoHumans scene names are unique across activities in practice, but the on-disk
    layout is nested, and the activity is what makes a row of the table readable.
    """
    return f"{scene_dir.parent.name}/{scene_dir.name}"


def select_persons(scene_name: str, cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Return the person IDs to synchronise for this scene, from SCENE_PIDS.

    Every id must be observed by at least two cameras.  One camera is not an
    error of bookkeeping but a person who can score no pair at all: the offset is
    only ever estimated between two views, so a track seen once contributes
    nothing and would only add an all-+inf person to every pair.
    """
    try:
        pids = SCENE_PIDS[scene_name]
    except KeyError:
        raise KeyError(
            f"{scene_name}: no SCENE_PIDS entry. Add the body_data_clean person "
            f"id(s) that survive in every camera before evaluating this scene."
        ) from None
    if not pids:
        raise ValueError(f"{scene_name}: SCENE_PIDS entry is empty")

    seen = {pid: sum(pid in persons for persons in cam_data.values()) for pid in pids}
    lonely = [pid for pid, n in seen.items() if n < 2]
    if lonely:
        raise KeyError(
            f"{scene_name}: person(s) {lonely} observed by fewer than 2 cameras "
            f"({ {p: seen[p] for p in lonely} }) — they can score no camera pair. "
            f"Drop them from SCENE_PIDS, or fix manual_reid.json and re-run "
            f"scripts/build_clean_body_data.py."
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
    key = scene_key(scene_dir)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Scene: {key}")

    exclude_cams = SKIP_CAMERAS.get(key, [])
    if exclude_cams:
        logger.info(f"  Excluding cameras: {exclude_cams}")
    cam_data = load_scene(scene_dir, exclude_cameras=exclude_cams)
    if len(cam_data) < 2:
        raise RuntimeError(
            f"{key}: need ≥2 cameras with body_data_clean, found {len(cam_data)}"
        )

    pids = select_persons(key, cam_data)
    seen = _fill_missing_persons(cam_data, pids)

    cam_ids = list(cam_data.keys())
    logger.info(f"  Cameras: {cam_ids}")
    logger.info(f"  Persons: {pids}  (cameras observing each: "
                f"{ {p: seen[p] for p in pids} } of {len(cam_ids)})")
    partial = [p for p in pids if seen[p] < len(cam_ids)]
    if partial:
        logger.info(f"  Person(s) {partial} zero-padded where unobserved — they score "
                    f"only the pairs whose two cameras both see them")

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
        logger.warning(f"  All trials skipped for {key} — no usable results")
        return None

    # Mean pairwise offset: the average |t_i - t_j| over all K(K-1)/2 camera pairs,
    # i.e. the size of the alignment problem the solver actually faces.  The spread
    # (max - min) was the old column, but it is an extremum over the same draws — it
    # grows with K at fixed difficulty and ignores every pair but the extreme one.
    # run_benchmark's MeanΔ is this quantity, so the tables are comparable.
    all_deltas = [
        float(np.mean([
            abs(v[i] - v[j])
            for v in [list(r["true_shifts"].values())]
            for i in range(len(v)) for j in range(i + 1, len(v))
        ]))
        for r in results
    ]

    # Pool the per-camera errors of every solved trial, then take every metric
    # off that one vector. Trials that produced no result are excluded here and
    # reported as coverage (solved/N_TRIALS) instead of inflating the error.
    errors = np.concatenate([np.array(r["errors"]) for r in results])
    stats  = pooled_stats(errors)

    logger.info(f"\n  SUMMARY — {key}  (solved {len(results)}/{N_TRIALS} trials)")
    logger.info(f"    Mean pairwise Δ {np.mean(all_deltas):.1f}  (median={np.median(all_deltas):.1f}  max={np.max(all_deltas):.1f})")
    logger.info(f"    MAE (frames)    {stats['mae']:.2f}")
    logger.info(f"    MedAE (frames)  {stats['median_ae']:.2f}")
    logger.info(f"    Within 0.5fr    {stats['within_half']*100:.1f}%")
    logger.info(f"    Within 1fr      {stats['within_1']*100:.1f}%")
    logger.info(f"    Within 2fr      {stats['within_2']*100:.1f}%")
    logger.info(f"    AUC@100ms       {stats['auc_100ms']*100:.1f}%")
    logger.info(f"    AUC@500ms       {stats['auc_500ms']*100:.1f}%")

    return {
        "scene":         key,
        "n_cameras":     len(cam_ids),
        "n_persons":     len(pids),
        "solved":        len(results),
        "n_trials":      N_TRIALS,
        "mean_delta":    float(np.mean(all_deltas)),
        "errors":        errors,
        **stats,
        "trial_results": results,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", nargs="+", metavar="SCENE",
                        help="One or more scenes to evaluate, as '<activity>/<scene>' or "
                             "a bare scene name (default: all)")
    parser.add_argument("--max-shift", type=int, default=MAX_SHIFT,
                        help="Maximum absolute shift in frames (default: %(default)s)")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Random-shift trials per scene (default: %(default)s)")
    args = parser.parse_args()

    MAX_SHIFT = args.max_shift
    N_TRIALS  = args.trials
    # A bare "005_fencing" resolves to its "<activity>/<scene>" key; ambiguity is
    # an error rather than a silent pick.
    by_bare: dict[str, list[str]] = {}
    for k in SCENE_PIDS:
        by_bare.setdefault(k.split("/")[-1], []).append(k)

    def resolve(name: str) -> str:
        if name in SCENE_PIDS:
            return name
        matches = by_bare.get(name.split("/")[-1], [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"{name}: ambiguous, matches {matches}")
        return name          # unknown — reported by the check below

    cli_scenes: list[str] = [resolve(n) for n in (args.scene or [])]
    scene_names = cli_scenes or sorted(SCENE_PIDS)

    logger.info(f"Scenes root: {SCENES_ROOT}")
    logger.info(f"Device: {DEVICE}  |  trials per scene: {N_TRIALS}  |  max_shift: {MAX_SHIFT}")

    unknown = [s for s in scene_names if s not in SCENE_PIDS]
    if unknown:
        raise KeyError(
            f"no SCENE_PIDS entry for: {unknown}. Add the body_data_clean person "
            f"id(s) that survive in every camera before evaluating them."
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
    all_delta  = [s["mean_delta"] for s in scene_summaries]
    errors     = np.concatenate([s["errors"] for s in scene_summaries])
    total      = pooled_stats(errors)
    n_solved   = sum(s["solved"]   for s in scene_summaries)
    n_total    = sum(s["n_trials"] for s in scene_summaries)

    logger.info("\n" + "=" * 60)
    logger.info(f"AGGREGATE SUMMARY  ({len(scene_summaries)} scene(s)  ×  {N_TRIALS} trials each)")
    logger.info("  metrics pooled over per-camera errors of solved trials")
    logger.info("")

    # ── per-scene table (frames) ──────────────────────────────────────────
    hdr = f"  {'Scene':<32}  {'Cams':>4}  {'Pers':>4}  {'Solved':>7}  {'MeanΔ':>7}  {'MAE':>6}  {'MedAE':>6}  {'W-0.5':>7}  {'W-1':>7}  {'W-2':>7}  {'AUC@100':>8}  {'AUC@500':>8}"
    sep = f"  {'-'*32}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}"
    logger.info("  All values in frames; AUC thresholds at 15 fps (100ms=1.5fr, 500ms=7.5fr)")
    logger.info(hdr)
    logger.info(sep)
    for s in scene_summaries:
        cov = f"{s['solved']}/{s['n_trials']}"
        row = (
            f"  {s['scene']:<32}  {s['n_cameras']:>4}  {s['n_persons']:>4}  "
            f"{cov:>7}  {s['mean_delta']:>7.1f}  "
            f"{s['mae']:>6.2f}  {s['median_ae']:>6.2f}  "
            f"{s['within_half']*100:>6.1f}%  {s['within_1']*100:>6.1f}%  {s['within_2']*100:>6.1f}%  "
            f"{s['auc_100ms']*100:>7.1f}%  {s['auc_500ms']*100:>7.1f}%"
        )
        logger.info(row)
    logger.info(sep)
    total_row = (
        f"  {'TOTAL':<32}  {'':>4}  {'':>4}  "
        f"{f'{n_solved}/{n_total}':>7}  {np.mean(all_delta):>7.1f}  "
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
