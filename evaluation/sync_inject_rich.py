"""STEP 1 of the through-sync RICH evaluation: inject desync, estimate it, rebuild.

For each scene: draw a random per-camera temporal desync (start shift + end
cut, like evaluation/alignment_experiments_multi.py), run the Synchronizer to
ESTIMATE it, then re-run VGGT + MapAnything on the CENTERED-CROP images
selected by the ESTIMATED (not true) alignment, and re-window every camera's
body_data to match. Writes one self-contained scene directory per trial:

    <sync_output_root>/<scene>/trial<k>/
        <cam_id>/body_data/person_*.npz   (windowed + relabeled)
        vggt_cameras_centered.npz         (rerun on the estimated alignment)
        vggt_depth_centered.npz
        mapanything_scale_baseline.npy    (rerun on the same alignment)
        sync_meta.json                    (true/estimated shifts, real_frame_anchor)

Images come from <rich_root>/centered_<gt_split>/<scene>/<cam>/ -- the
principal-point-recentered crop that production runs VGGT + MapAnything on
(utilities/center_images.py), NOT the raw scan images. This matters: SAM3D
pred_keypoints_2d are in raw (uncropped) source pixels, and BodyPlacer's
_orig_to_vggt only removes that offset correctly if the VGGT camera really
was calibrated on the centered crop -- feeding it raw frames instead would
silently reintroduce the reprojection bias PP-cropping exists to remove
(the crop is a pure translation, no resize -- center_images.py writes
img[y1:y2, x1:x2] at source resolution; VGGT's own internal downscale to its
~512 working resolution is unaffected either way and handled entirely by
run_vggt.py's own original_coords/original_size bookkeeping). That mount is
a squashfuse job (centered_<split>.sqsh) -- point --rich_root/--centered_root
at wherever the caller already mounted it; this script never touches the
.sqsh itself.

GT is never touched here -- that only happens in STEP 2
(evaluation/evaluate_rich_sync.py), which reads sync_meta.json to know how to
map this trial's local frame labels back to real RICH frame numbers for GT
lookup, and runs fusion + placement + CHROMM metrics against it.

Usage
-----
    python evaluation/sync_inject_rich.py \\
        --ghost_output_root /path/to/ghost_outputs/rich_test \\
        --rich_root         /path/to/rich \\
        --sync_output_root  /path/to/scratch/rich_sync_eval \\
        --vggt_weights      /path/to/vggt_omega.pt \\
        [--centered_root /path/to/mounted/centered_test] \\
        [--max_scenes N] [--max_shift 45] [--n_trials 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configuration import CONFIG
from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.run_vggt import VGGTPreprocessor
from synchronize_videos.synchronizer import Synchronizer
from utilities.body_data import load_person_smplx_pose

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Sync injection -- semantics
# ---------------------------------------------------------------------------
#
# RICH cameras are hardware genlocked, so raw absolute frame index F is a
# real, shared clock -- frame F in every camera is the same instant. We
# simulate a desynced capture by windowing each camera's stream at a
# different raw start index, feed those windows to the Synchronizer, and get
# back an ESTIMATE of the per-camera shift. That estimate (not the true
# shift) then decides which raw frames get read for VGGT + MapAnything +
# body_data aggregation, exactly as a real deployed synchronizer would drive
# the rest of the pipeline.


def _load_anchored(npz_path: Path, loader) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load one person and re-anchor its sequence to absolute frame 0.

    The loaders in utilities.body_data anchor the result at frame_indices[0]
    -- each camera's own first detection. RICH is hardware synced, so the
    absolute frame index is true time; left-padding by frame_indices[0] with
    zero pose/confidence makes array index == absolute frame in every camera,
    so an injected shift is the only offset present. Copied verbatim from
    evaluation/alignment_experiments_multi.py (the bug-fixed reference).
    """
    result = loader(npz_path)
    if result is None:
        return None
    seq, conf = result
    with np.load(str(npz_path)) as d:
        if "frame_indices" not in d.files:
            raise KeyError(f"{npz_path}: no frame_indices -- cannot anchor to absolute time")
        start = int(d["frame_indices"].astype(int).min())
    if start > 0:
        seq  = torch.cat([seq .new_zeros((start, *seq .shape[1:])), seq ], dim=0)
        conf = torch.cat([conf.new_zeros((start, *conf.shape[1:])), conf], dim=0)
    return seq, conf


def _pad_to_common_length(cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]) -> None:
    """Right-pad every sequence to the longest one, in place. Zero-confidence, so ignored by DTW cost."""
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


def _common_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    sets = [set(persons.keys()) for persons in cam_data.values()]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    end_cuts: dict[str, int],
    pids: list[int],
    device: str,
    min_overlap: int = 100,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]] | None:
    """Slice pose sequences to simulate temporal offsets + random end times.

    Copied from evaluation/alignment_experiments_multi.py::apply_shifts.
    Returns None if any camera ends up with fewer than min_overlap frames.
    """
    cam_ids = list(shifts.keys())
    max_s   = max(shifts.values())

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
                    f"(need >={min_overlap}) -- skipping trial"
                )
                return None
            per_person_joints.append(rotations[s:end].to(device))
            per_person_confs .append(conf     [s:end].to(device))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)

    return joints_list, confs_list


def _list_camera_frames(cam_raw_dir: Path) -> list[Path]:
    """Sorted raw image files for one camera, mirroring Video.frames_home's fallback.

    Prefers <cam_raw_dir>/frames/ (already migrated by an earlier pipeline
    run) and falls back to images sitting directly in <cam_raw_dir>/. No
    Video/RichDataset construction, so no writes and no squashfuse resize
    path -- see [[rich-sqsh-frames-issue]].

    Returns [] (not an error) if cam_raw_dir doesn't exist at all -- some
    cameras have body_data in ghost_output_root but no calibration, so
    utilities/center_images.py never wrote a centered-crop dir for them
    (same class of gap as [[missing_camera_handling]]'s cam_10). Caller is
    responsible for treating an empty result as "drop this camera", not
    "scene is broken".
    """
    if not cam_raw_dir.is_dir():
        return []
    frames_sub = cam_raw_dir / "frames"
    if frames_sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in frames_sub.iterdir()):
        d = frames_sub
    else:
        d = cam_raw_dir
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _window_and_relabel_npz(data: dict[str, np.ndarray], s_hat: int, T_hat: int) -> dict[str, np.ndarray] | None:
    """Keep frames whose raw frame_indices fall in [s_hat, s_hat+T_hat), relabel to [0, T_hat).

    Every per-frame array (leading dim == len(frame_indices)) is sliced by
    the same boolean mask; anything else is copied through unchanged.
    Returns None if no frame of this person survives the window.
    """
    if "frame_indices" not in data:
        return None
    fi = data["frame_indices"].astype(int)
    mask = (fi >= s_hat) & (fi < s_hat + T_hat)
    if not mask.any():
        return None
    n = len(fi)
    out: dict[str, np.ndarray] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[mask]
        else:
            out[k] = v
    out["frame_indices"] = (fi[mask] - s_hat).astype(data["frame_indices"].dtype)
    return out


class SyncTrialResult:
    """Everything downstream needs from one (scene, trial) sync injection."""

    def __init__(self, cam_ids, true_shifts, shift_hat, sync_errors, real_frame_anchor, T_hat):
        self.cam_ids = cam_ids
        self.true_shifts = true_shifts
        self.shift_hat = shift_hat
        self.sync_errors = sync_errors              # {cam_id: |shift_hat - true_shift|}
        self.real_frame_anchor = real_frame_anchor   # real_frame = local_label + real_frame_anchor
        self.T_hat = T_hat


def run_sync_trial(
    cam_pose_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    pids: list[int],
    cam_ids: list[str],
    raw_frame_counts: dict[str, int],
    sync: Synchronizer,
    rng: np.random.Generator,
    max_shift: int,
    min_overlap: int,
    device: str,
) -> tuple[SyncTrialResult, dict[str, int]] | None:
    """Draw one random desync trial, estimate it, and derive the corrected raw-frame window.

    Returns (SyncTrialResult, s_hat) where s_hat[cam] is the per-camera raw
    frame index to read at local reconstruction time 0 (local time t reads
    raw index s_hat[cam]+t, for t in [0, T_hat)) -- built by resampling each
    camera's own TRUE (fixed) desynced window at the ESTIMATED shift, per the
    module docstring. None if the trial couldn't be solved (insufficient
    overlap or isolated camera).
    """
    ref_cam = cam_ids[0]
    raw_shifts  = [0] + rng.integers(-max_shift, max_shift + 1, size=len(cam_ids) - 1).tolist()
    true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
    end_cuts    = {c: int(e) for c, e in zip(
        cam_ids, rng.integers(0, max_shift + 1, size=len(cam_ids)).tolist()
    )}
    logger.info(f"  true shifts: {true_shifts}  end_cuts: {end_cuts}")

    result = apply_shifts(cam_pose_data, true_shifts, end_cuts, pids, device, min_overlap)
    if result is None:
        return None
    joints_list, confs_list = result

    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    weights    = sync.cycle_consistency_weights(offset_mat)
    estimated  = sync.estimate_initial_times(offset_mat, weights)
    if torch.isnan(estimated).any():
        logger.warning("  sync solver returned NaN (isolated camera?) -- skipping trial")
        return None

    # Re-anchor to the reference camera (true_shifts[ref] == 0 exactly, by
    # construction), NOT to min(estimated) -- so shift_hat shares the SAME
    # real-world zero-point as true_shifts, and real_frame_anchor below is
    # correct even when the estimate is imperfect. Using the Synchronizer's
    # own min-anchor here would silently offset every GT lookup (in step 2)
    # by whatever constant separates the two anchors.
    est = estimated.cpu()
    ref_idx = cam_ids.index(ref_cam)
    shift_hat_f = est - est[ref_idx]
    shift_hat = {c: int(round(v)) for c, v in zip(cam_ids, shift_hat_f.tolist())}

    sync_errors = {c: abs(shift_hat[c] - true_shifts[c]) for c in cam_ids}
    logger.info(f"  shift_hat:   {shift_hat}  |err|={sync_errors}")

    max_s = max(true_shifts.values())
    T_common_raw = min(raw_frame_counts[c] for c in cam_ids)

    # Camera c's TRUE observed window: what's physically available given the
    # INJECTED (true) desync -- fixed, independent of the estimate. This is
    # the same s/end apply_shifts used above for the sync-input pose window,
    # just re-derived here for raw image indices.
    s_true   = {c: max(0, min(max_s - true_shifts[c], T_common_raw)) for c in cam_ids}
    end_true = {c: max(0, min(T_common_raw - end_cuts[c] if end_cuts[c] > 0 else T_common_raw,
                              T_common_raw)) for c in cam_ids}
    avail_len = {c: end_true[c] - s_true[c] for c in cam_ids}
    if any(avail_len[c] < min_overlap for c in cam_ids):
        logger.warning(f"  true window too short for camera(s) "
                        f"{[c for c in cam_ids if avail_len[c] < min_overlap]} -- skipping trial")
        return None

    # To CORRECT the desync, a deployed system resamples WITHIN each camera's
    # own (fixed, truly-desynced) observed stream at local index
    # (t + shift_hat[c]) -- its estimate applied to what's actually recorded.
    # This is NOT the same as reapplying apply_shifts' "ANCHOR - shift[c]"
    # formula with shift_hat swapped in for true_shift: that reproduces the
    # SAME desynced window (since when shift_hat==true_shift, ANCHOR-shift_hat
    # == s_true, i.e. zero correction applied) -- the bug this replaces.
    # raw_file(c, t) = s_true[c] + shift_hat[c] + t; valid t needs
    # 0 <= t+shift_hat[c] < avail_len[c] for every camera, i.e. the
    # intersection of each camera's own valid t-range.
    lower = {c: -shift_hat[c] for c in cam_ids}
    upper = {c: avail_len[c] - shift_hat[c] for c in cam_ids}
    T0 = max(lower.values())
    T1 = min(upper.values())
    T_hat = T1 - T0
    if T_hat < min_overlap:
        logger.warning(
            f"  corrected window too short (T_hat={T_hat} < {min_overlap}) -- "
            f"sync estimate too far off within max_shift={max_shift} -- skipping trial"
        )
        return None

    s_hat = {c: s_true[c] + shift_hat[c] + T0 for c in cam_ids}
    # real_frame(t') = raw_file(ref, t') = s_true[ref] + shift_hat[ref] + T0 + t'
    #                = max_s + T0 + t'  (shift_hat[ref] == true_shifts[ref] == 0
    #                  exactly, by the ref-anchoring above -- the ref camera is
    #                  never actually desynced, so its own footage IS ground
    #                  truth time regardless of estimation error elsewhere).
    real_frame_anchor = max_s + T0

    trial = SyncTrialResult(
        cam_ids=cam_ids, true_shifts=true_shifts, shift_hat=shift_hat,
        sync_errors=sync_errors, real_frame_anchor=real_frame_anchor, T_hat=T_hat,
    )
    return trial, s_hat


def build_synced_scene(
    orig_scene_dir: Path,
    scene_raw_dir: Path,
    trial_scene_dir: Path,
    cam_ids: list[str],
    s_hat: dict[str, int],
    T_hat: int,
    preprocessor: VGGTPreprocessor,
    vggt_devices: list[str],
    ma_estimator: MapAnythingScaleEstimator | None,
) -> None:
    """Write the windowed+relabeled body_data and re-run VGGT (+ MapAnything) for one trial.

    cam_ids must be sorted (it becomes the VGGT camera_names axis, and
    BodyPlacer/MapAnythingScaleEstimator positionally align their own sorted
    directory listings against that axis) -- every camera directory is
    created even if a camera ends up with zero people this trial, or that
    positional alignment breaks.
    """
    trial_scene_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[list[Path]] = []
    for t in range(T_hat):
        frame_paths.append([
            _list_camera_frames(scene_raw_dir / c)[s_hat[c] + t] for c in cam_ids
        ])

    for c in cam_ids:
        out_body_dir = trial_scene_dir / c / "body_data"
        out_body_dir.mkdir(parents=True, exist_ok=True)
        src_body_dir = orig_scene_dir / c / "body_data"
        for npz_path in sorted(src_body_dir.glob("person_*.npz")):
            data = dict(np.load(npz_path, allow_pickle=False))
            windowed = _window_and_relabel_npz(data, s_hat[c], T_hat)
            if windowed is not None:
                np.savez_compressed(out_body_dir / npz_path.name, **windowed)

    preprocessor.process_scene(
        frame_paths=frame_paths, camera_names=cam_ids,
        output_dir=trial_scene_dir, devices=vggt_devices,
    )

    if ma_estimator is not None:
        # MapAnythingScaleEstimator.process_scene wants a real directory
        # (img_root/<cam_name>/*.jpg, sorted, positionally matching the vggt
        # npz's T axis) -- it can't take an explicit frame_paths list like
        # VGGTPreprocessor. Materialize the SAME windowed selection as a
        # symlink tree, named by local index so sort order == VGGT's frame
        # order regardless of the source files' own naming.
        images_dir = trial_scene_dir / "_ma_images"
        for ci, c in enumerate(cam_ids):
            cam_img_dir = images_dir / c
            cam_img_dir.mkdir(parents=True, exist_ok=True)
            for t in range(T_hat):
                src = frame_paths[t][ci]
                dst = cam_img_dir / f"{t:06d}{src.suffix.lower()}"
                # A prior failed attempt at this trial may have left a symlink
                # pointing into ITS squashfuse mount, since torn down -- that
                # reads as dangling (dst.exists() follows the link and is
                # False), but the link inode is still there, so a plain
                # symlink_to() hits EEXIST. Always replace rather than
                # skip-if-exists: this directory is trial-local scratch,
                # rebuilding it is cheap and the only correct option once a
                # stale link is possible.
                dst.unlink(missing_ok=True)
                dst.symlink_to(src)
        ma_estimator.process_scene(scene_dir=trial_scene_dir, img_root=images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STEP 1: inject desync, estimate it, rerun VGGT + MapAnything "
                    "on the estimated alignment. Writes self-contained trial dirs; "
                    "run evaluation/evaluate_rich_sync.py on the output for metrics."
    )
    parser.add_argument("--ghost_output_root", required=True, type=Path,
                        help="Existing ghost test outputs (source of un-windowed body_data).")
    parser.add_argument("--rich_root",         required=True, type=Path,
                        help="RICH dataset root. Used only to locate the centered-crop image "
                             "tree (<rich_root>/centered_<gt_split>/) unless --centered_root "
                             "overrides it.")
    parser.add_argument("--centered_root",     type=Path, default=None,
                        help="Root of the centered-crop images (<scene>/<cam>/*.jpg), i.e. a "
                             "mounted centered_<split>.sqsh -- see utilities/center_images.py. "
                             "Defaults to <rich_root>/centered_<gt_split>. This script never "
                             "mounts a .sqsh itself; mount it first and pass the mount point.")
    parser.add_argument("--gt_split",          default="test",
                        help="Only used to build the default --centered_root path.")
    parser.add_argument("--sync_output_root",  required=True, type=Path,
                        help="Destination for the freshly-built synced-trial scene dirs. "
                             "Never the same tree as --ghost_output_root.")
    parser.add_argument("--vggt_weights",      type=str, default=None,
                        help="VGGT-Omega weights (.pt). CONFIG.data.vggt_omega_checkpoint if omitted.")
    parser.add_argument("--vggt_devices",      type=str, nargs="+", default=None,
                        help="CUDA device strings for VGGT. Defaults to all visible GPUs.")
    parser.add_argument("--skip_mapanything",  action="store_true", default=False,
                        help="Skip the MapAnything rerun (evaluate_rich_sync.py will then fall "
                             "back to triangulated scale, like evaluate_rich_median.py does when "
                             "the scale file is absent).")
    parser.add_argument("--mapanything_device", type=str, default=None,
                        help="CUDA device for MapAnything (defaults to first VGGT device).")
    parser.add_argument("--mapanything_batch_size", type=int, default=8)
    parser.add_argument("--max_scenes",        type=int, default=None,
                        help="Limit to the first N scenes (debugging).")
    parser.add_argument("--scenes",            default="",
                        help="Comma-separated scene names to evaluate (default: all).")
    parser.add_argument("--skip_scenes",       default="")
    parser.add_argument("--max_shift",         type=int, default=45,
                        help="Max absolute injected shift and end-cut, in frames.")
    parser.add_argument("--n_trials",          type=int, default=1,
                        help="Random desync trials per scene.")
    parser.add_argument("--min_overlap",       type=int, default=100)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--sync_device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    vggt_weights = args.vggt_weights or CONFIG.data.vggt_omega_checkpoint
    if args.vggt_devices:
        vggt_devices = args.vggt_devices
    elif torch.cuda.is_available():
        vggt_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        vggt_devices = ["cpu"]
    logger.info(f"VGGT weights: {vggt_weights}  devices: {vggt_devices}")
    preprocessor = VGGTPreprocessor(weights=vggt_weights, device=vggt_devices[0])

    if args.skip_mapanything:
        ma_estimator = None
    else:
        ma_device = args.mapanything_device or vggt_devices[0]
        ma_estimator = MapAnythingScaleEstimator(
            device=ma_device, batch_size=args.mapanything_batch_size, scale_from="baselines",
        )
        logger.info(f"MapAnything: device={ma_device}  batch_size={args.mapanything_batch_size}")

    centered_root = args.centered_root or (args.rich_root / f"centered_{args.gt_split}")
    logger.info(f"Reading images from {centered_root} (centered-crop tree)")

    sync = Synchronizer(device=args.sync_device, min_overlap=args.min_overlap, max_shift=args.max_shift)
    rng  = np.random.default_rng(args.seed)

    skip_scenes: set[str] = {s.strip() for s in args.skip_scenes.split(",") if s.strip()}
    only_scenes: set[str] = {s.strip() for s in args.scenes.split(",") if s.strip()}

    scenes = sorted(
        d for d in args.ghost_output_root.iterdir()
        if d.is_dir() and any((d / c.name / "body_data").exists() for c in d.iterdir() if c.is_dir())
        and d.name not in skip_scenes
        and (not only_scenes or d.name in only_scenes)
    )
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    logger.info(f"Found {len(scenes)} scene(s).")

    n_trials_attempted = 0
    n_trials_built = 0
    # {scene_name: reason} -- a scene lands here iff it ends this run with ZERO
    # built trials (not synchronizable, or missing prerequisites). Recomputed
    # fresh every run (not merged with a prior run's file): a scene that only
    # failed because of a since-fixed bug should stop being "skipped" the
    # moment it starts succeeding, with no stale entry to clean up by hand.
    skipped_scenes: dict[str, str] = {}

    def _write_skipped():
        with open(args.sync_output_root / "skipped_scenes.json", "w") as f:
            json.dump(skipped_scenes, f, indent=2)

    args.sync_output_root.mkdir(parents=True, exist_ok=True)

    for orig_scene_dir in scenes:
        scene_name = orig_scene_dir.name

        # Resume fast-path: every trial dir for this scene already has
        # sync_meta.json (a prior, possibly killed, job run built it) -- skip
        # the (cheap but non-trivial) setup below entirely.
        done_trials = {
            k for k in range(args.n_trials)
            if (args.sync_output_root / scene_name / f"trial{k}" / "sync_meta.json").exists()
        }
        if len(done_trials) == args.n_trials:
            logger.info(f"{scene_name}: all {args.n_trials} trial(s) already built -- skipping")
            n_trials_built += args.n_trials
            continue

        scene_raw_dir = centered_root / scene_name
        if not scene_raw_dir.is_dir():
            logger.warning(f"{scene_name}: no centered-crop images at {scene_raw_dir} -- skipping")
            skipped_scenes[scene_name] = f"no centered-crop images at {scene_raw_dir}"
            _write_skipped()
            continue

        cam_dirs = sorted(d for d in orig_scene_dir.iterdir()
                          if d.is_dir() and (d / "body_data").is_dir())

        # Some cameras have body_data (production ran SAM3D on them) but no
        # calibration, so utilities/center_images.py never produced a
        # centered-crop dir for them (same gap as [[missing_camera_handling]]'s
        # cam_10) -- drop those cameras rather than fail/skip the whole scene,
        # matching how load_scene_body_data/BodyPlacer already drop cameras
        # absent from the VGGT npz.
        uncalibrated = [d.name for d in cam_dirs if not (scene_raw_dir / d.name).is_dir()]
        if uncalibrated:
            logger.warning(f"{scene_name}: dropping camera(s) with no centered-crop images "
                            f"(no calibration): {uncalibrated}")
            cam_dirs = [d for d in cam_dirs if d.name not in uncalibrated]

        cam_ids = [d.name for d in cam_dirs]
        if len(cam_ids) < 2:
            logger.warning(f"{scene_name}: fewer than 2 calibrated cameras with body_data -- skipping")
            skipped_scenes[scene_name] = f"fewer than 2 calibrated cameras with body_data ({len(cam_ids)})"
            _write_skipped()
            continue

        raw_frame_counts = {c: len(_list_camera_frames(scene_raw_dir / c)) for c in cam_ids}
        if any(n == 0 for n in raw_frame_counts.values()):
            missing = [c for c, n in raw_frame_counts.items() if n == 0]
            logger.warning(f"{scene_name}: no centered-crop frames for camera(s) {missing} -- skipping")
            skipped_scenes[scene_name] = f"no centered-crop frames for camera(s) {missing}"
            _write_skipped()
            continue

        cam_pose_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        for cam_dir in cam_dirs:
            persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(npz_path.stem.split("_")[1])
                result = _load_anchored(npz_path, load_person_smplx_pose)
                if result is not None:
                    persons[pid] = result
            cam_pose_data[cam_dir.name] = persons
        _pad_to_common_length(cam_pose_data)
        pids = _common_persons(cam_pose_data)
        if not pids:
            logger.warning(f"{scene_name}: no person id common to every camera -- skipping")
            skipped_scenes[scene_name] = "no person id common to every camera"
            _write_skipped()
            continue

        logger.info(f"\n{'='*60}\nScene: {scene_name}  cameras={cam_ids}  sync persons={pids}")
        if done_trials:
            logger.info(f"  Resuming: {len(done_trials)}/{args.n_trials} trial(s) already built "
                        f"({sorted(done_trials)}), building the rest")

        last_failure_reason = "sync could not be solved"
        n_built_this_scene = len(done_trials)

        for trial in range(args.n_trials):
            if trial in done_trials:
                continue
            logger.info(f"\n-- Trial {trial + 1}/{args.n_trials} --")
            n_trials_attempted += 1
            outcome = run_sync_trial(
                cam_pose_data, pids, cam_ids, raw_frame_counts, sync, rng,
                args.max_shift, args.min_overlap, args.sync_device,
            )
            if outcome is None:
                last_failure_reason = (
                    f"sync solver could not align cameras within max_shift={args.max_shift} "
                    f"/ min_overlap={args.min_overlap}"
                )
                continue
            sync_trial, s_hat = outcome

            trial_scene_dir = args.sync_output_root / scene_name / f"trial{trial}"
            try:
                build_synced_scene(
                    orig_scene_dir, scene_raw_dir, trial_scene_dir, cam_ids,
                    s_hat, sync_trial.T_hat, preprocessor, vggt_devices, ma_estimator,
                )
            except Exception as e:
                logger.error(f"  VGGT/MapAnything rerun failed for {scene_name}/trial{trial}: {e}", exc_info=True)
                last_failure_reason = f"VGGT/MapAnything rerun failed: {e}"
                continue

            meta = {
                "scene": scene_name,
                "trial": trial,
                "cam_ids": sync_trial.cam_ids,
                "true_shifts": sync_trial.true_shifts,
                "shift_hat": sync_trial.shift_hat,
                "sync_errors": sync_trial.sync_errors,
                "real_frame_anchor": sync_trial.real_frame_anchor,
                "T_hat": sync_trial.T_hat,
                "max_shift": args.max_shift,
                "min_overlap": args.min_overlap,
                "mapanything": ma_estimator is not None,
            }
            with open(trial_scene_dir / "sync_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            n_trials_built += 1
            n_built_this_scene += 1
            logger.info(f"  Built {trial_scene_dir}  (sync_mae={np.mean(list(sync_trial.sync_errors.values())):.2f} frames)")

        # A scene that never got a single trial built (this run + any prior,
        # resumed run) could not be synchronized at all -- record it so step 2
        # can report it explicitly instead of it just being silently absent.
        if n_built_this_scene == 0:
            skipped_scenes[scene_name] = last_failure_reason
        elif scene_name in skipped_scenes:
            del skipped_scenes[scene_name]   # partial success this run: no longer fully skipped
        _write_skipped()

    logger.info(f"\n{'='*60}\nDone. {n_trials_built} trial(s) built ({n_trials_attempted} sync attempts "
                f"this run) under {args.sync_output_root}  ({len(skipped_scenes)} scene(s) fully skipped)")


if __name__ == "__main__":
    main()
