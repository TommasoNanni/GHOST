"""STEP 0 of the through-sync EgoHumans evaluation: inject desync, estimate it, PARTITION.

Loads every EgoHumans scene that has clean body tracks + a VGGT reconstruction,
draws a random per-camera temporal desync (start shift + end cut, same protocol
as evaluation/alignment_experiments_multi_egohumans.py), runs the Synchronizer
to ESTIMATE it, and writes one manifest describing what the rest of the
evaluation has to do per scene. It touches no images and no GPU-heavy model, so
it runs over all 132 annotated scenes in minutes.

The point of the manifest is the partition it induces:

  * EXACT scenes (shift_hat == true_shift for every camera). The corrected read
    index is  s_true[c] + shift_hat[c] + T0 + t = max_s + T0 + t  -- identical in
    every camera, because s_true[c] = max_s - true_shift[c] cancels the estimate.
    So the frames a deployed system would line up are EXACTLY a contiguous slice
    of the original, correctly-synced stream. The existing per-scene
    vggt_cameras_centered.npz / vggt_depth_centered.npz / mapanything_scale_baseline.npy
    are therefore already valid on that window -- no images, no VGGT rerun, no
    MapAnything rerun. Evaluation is the normal median-fusion eval restricted to
    real frames [real_frame_anchor, real_frame_anchor + T_hat).

  * INEXACT scenes (some camera off by >=1 frame). Each camera reads a DIFFERENT
    real instant, so the multi-view reconstruction genuinely changes and VGGT +
    MapAnything must be re-run on the mis-aligned image selection. Only these
    scenes need their images downloaded.

Nothing here is random-access dependent on the images: the raw stream length
comes from the VGGT npz's own T axis (which was built as min-over-cameras of the
frame count at pipeline time), so the partition can be computed while the image
data is still offline.

Frame bookkeeping
-----------------
_load_anchored left-pads every track by frame_indices.min(), so array index ==
frame label in whatever base body_data_clean uses. `frame_label_min` is recorded
per scene so the downstream steps can convert a frame label to a positional
index into the VGGT T axis without re-deriving the convention.

Usage
-----
    pixi run python evaluation/sync_dryrun_egohumans.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new \\
        --gt_root    /iopsstor/scratch/cscs/tnanni/egohumans_gt_full \\
        --out        eval_egohumans/sync_manifest_s75.json \\
        --max_shift  75
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

from synchronize_videos.synchronizer import Synchronizer
from utilities.body_data import load_person_smplx_pose

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track loading -- copied from evaluation/alignment_experiments_multi_egohumans.py
# ---------------------------------------------------------------------------

def _load_anchored(npz_path: Path, loader) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Load one person and re-anchor its sequence to absolute frame 0.

    The loaders in utilities.body_data fill interior detection gaps but anchor at
    frame_indices[0] -- each camera's own first detection. The EgoHumans exo
    GoPros are hardware synced, so the absolute frame index is true time; a
    camera whose person is first detected 27 frames late would otherwise sit 27
    frames early in its array and that would become a real offset on top of the
    injected one. Left-padding by frame_indices.min() with zero pose and zero
    confidence makes array index == frame label. Zero confidence keeps the
    padding out of the DTW cost, exactly as the loaders' own gap filling does.

    Returns (seq, conf, frame_label_min).
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
    return seq, conf, start


def _fill_missing_persons(cam_data: dict[str, dict[int, tuple]], pids: list[int]) -> dict[int, int]:
    """Give every camera a zero pose / zero confidence entry for the pids it lacks.

    A zero-confidence person makes every cost entry +inf for that pair, so
    estimate_couple_offset drops that person for that pair alone and the
    remaining pairs are unaffected -- which is why the pid set is the UNION over
    cameras, not the intersection: a person seen in 3 of 4 cameras still aligns
    the 3 pairs among those cameras. Returns {pid: n_cameras_that_really_saw_it}.
    """
    seen = {pid: sum(pid in persons for persons in cam_data.values()) for pid in pids}
    ref_seq = ref_conf = None
    for persons in cam_data.values():
        for seq, conf in persons.values():
            ref_seq, ref_conf = seq, conf
            break
        if ref_seq is not None:
            break
    if ref_seq is None:
        return seen
    T_ref = ref_seq.shape[0]
    for persons in cam_data.values():
        for pid in pids:
            if pid not in persons:
                persons[pid] = (torch.zeros((T_ref, *ref_seq.shape[1:])),
                                torch.zeros((T_ref, *ref_conf.shape[1:])))
    return seen


def _pad_to_common_length(cam_data: dict[str, dict[int, tuple]]) -> None:
    """Right-pad every sequence to the longest one, in place. Zero confidence => ignored by DTW."""
    lengths = [s.shape[0] for persons in cam_data.values() for s, _ in persons.values()]
    if not lengths:
        return
    T_max = max(lengths)
    for persons in cam_data.values():
        for pid, (seq, conf) in persons.items():
            pad = T_max - seq.shape[0]
            if pad > 0:
                persons[pid] = (torch.cat([seq,  seq .new_zeros((pad, *seq .shape[1:]))], dim=0),
                                torch.cat([conf, conf.new_zeros((pad, *conf.shape[1:]))], dim=0))


def load_scene(scene_dir: Path) -> tuple[dict[str, dict[int, tuple]], int]:
    """{cam_id: {pid: (rot T×51×3, conf T×51)}} from body_data_clean, plus frame_label_min.

    body_data_clean (not body_data): those ids come from manual_reid.json via
    scripts/build_clean_body_data.py, so person_<id> is the same real person in
    every camera. Raw body_data ids are per-camera and would make the DTW
    compare two different people.
    """
    cam_data: dict[str, dict[int, tuple]] = {}
    label_mins: list[int] = []
    for cam_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
        body_dir = cam_dir / "body_data_clean"
        if not body_dir.is_dir():
            continue
        persons: dict[int, tuple] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            try:
                pid = int(npz_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            result = _load_anchored(npz_path, load_person_smplx_pose)
            if result is None:
                logger.warning(f"  {npz_path.name}: missing pose params, skipping")
                continue
            seq, conf, label_min = result
            persons[pid] = (seq, conf)
            label_mins.append(label_min)
        if persons:
            cam_data[cam_dir.name] = persons
    _pad_to_common_length(cam_data)
    return cam_data, (min(label_mins) if label_mins else 0)


def select_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Union of body_data_clean ids, minus those seen by fewer than 2 cameras.

    The offset is only ever estimated between two views, so a track seen once
    can score no pair and would only add an all-+inf person to every pair.
    """
    counts: dict[int, int] = {}
    for persons in cam_data.values():
        for pid in persons:
            counts[pid] = counts.get(pid, 0) + 1
    return sorted(pid for pid, n in counts.items() if n >= 2)


def apply_shifts(
    cam_data: dict[str, dict[int, tuple]],
    shifts: dict[str, int],
    end_cuts: dict[str, int],
    pids: list[int],
    device: str,
    min_overlap: int,
    base: int = 0,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]] | None:
    """Slice pose sequences to simulate temporal offsets + random end times.

    `base` is frame_label_min: _load_anchored left-pads every track so that array
    index == frame LABEL, and EgoHumans labels are 1-based (00001.jpg), so array
    rows [0, base) are padding for a frame that does not exist. Starting the
    window at `base` keeps the sync input on the same real frames the image
    selection will later use.

    Returns None if any camera ends up with fewer than min_overlap frames.
    """
    cam_ids = list(shifts.keys())
    max_s   = max(shifts.values())
    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []
    for cam_id in cam_ids:
        s  = base + max_s - shifts[cam_id]
        ec = end_cuts[cam_id]
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            T   = rotations.shape[0]
            end = T - ec if ec > 0 else T
            if end - s < min_overlap:
                logger.warning(f"  {cam_id}: only {end - s} frames after shift "
                               f"(need >={min_overlap}) -- skipping trial")
                return None
            per_person_joints.append(rotations[s:end].to(device))
            per_person_confs .append(conf     [s:end].to(device))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)
    return joints_list, confs_list


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_sync_trial(
    cam_data: dict[str, dict[int, tuple]],
    pids: list[int],
    cam_ids: list[str],
    T_common_raw: int,
    sync: Synchronizer,
    rng: np.random.Generator,
    max_shift: int,
    min_overlap: int,
    device: str,
    base: int,
) -> dict | None:
    """Draw one desync, estimate it, and derive the corrected raw-frame window.

    Frame algebra copied from evaluation/sync_inject_rich.py (the reference
    implementation that STEP 1/2 of the RICH through-sync eval runs on).
    """
    ref_cam     = cam_ids[0]
    raw_shifts  = [0] + rng.integers(-max_shift, max_shift + 1, size=len(cam_ids) - 1).tolist()
    true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
    end_cuts    = {c: int(e) for c, e in zip(
        cam_ids, rng.integers(0, max_shift + 1, size=len(cam_ids)).tolist())}

    result = apply_shifts(cam_data, true_shifts, end_cuts, pids, device, min_overlap, base)
    if result is None:
        return None
    joints_list, confs_list = result

    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    weights    = sync.cycle_consistency_weights(offset_mat)
    estimated  = sync.estimate_initial_times(offset_mat, weights)
    if torch.isnan(estimated).any():
        logger.warning("  sync solver returned NaN (isolated camera?) -- skipping trial")
        return None

    # Re-anchor to the reference camera (true_shifts[ref] == 0 by construction),
    # NOT to min(estimated): shift_hat must share the same real-world zero point
    # as true_shifts, or real_frame_anchor below offsets every GT lookup by
    # whatever constant separates the two anchors.
    est         = estimated.cpu()
    shift_hat_f = est - est[cam_ids.index(ref_cam)]
    shift_hat   = {c: int(round(v)) for c, v in zip(cam_ids, shift_hat_f.tolist())}
    sync_errors = {c: abs(shift_hat[c] - true_shifts[c]) for c in cam_ids}

    max_s = max(true_shifts.values())
    # Camera c's TRUE observed window: what is physically available given the
    # INJECTED desync -- fixed, independent of the estimate.
    s_true   = {c: base + max(0, min(max_s - true_shifts[c], T_common_raw)) for c in cam_ids}
    end_true = {c: base + max(0, min(T_common_raw - end_cuts[c] if end_cuts[c] > 0 else T_common_raw,
                                     T_common_raw)) for c in cam_ids}
    avail_len = {c: end_true[c] - s_true[c] for c in cam_ids}
    if any(avail_len[c] < min_overlap for c in cam_ids):
        logger.warning("  true window too short -- skipping trial")
        return None

    # A deployed system resamples WITHIN each camera's own truly-desynced stream
    # at local index t + shift_hat[c]:  raw(c, t) = s_true[c] + shift_hat[c] + t.
    lower = {c: -shift_hat[c] for c in cam_ids}
    upper = {c: avail_len[c] - shift_hat[c] for c in cam_ids}
    T0, T1 = max(lower.values()), min(upper.values())
    T_hat  = T1 - T0
    # NOTE: no min_overlap gate here, on purpose. T0/T1 are functions of shift_hat,
    # so rejecting on T_hat would drop precisely the scenes whose estimate was
    # WORST -- selection bias that flatters the result. Admission is decided by the
    # estimate-INDEPENDENT gates above (the injected window must hold min_overlap
    # frames); a poor estimate is then paid for in the metric and in a shorter
    # evaluated window, not by the scene disappearing. Trials below min_overlap are
    # flagged `short_window` so they can be reported separately.
    if T_hat <= 0:
        # Empty intersection: the estimate is so far off that no frame is common to
        # all cameras. That is a synchronisation FAILURE, not missing data -- report
        # it as such rather than letting it look like protocol attrition.
        logger.warning(f"  NO OVERLAP after correction (T_hat={T_hat}) -- "
                       f"sync failure, err={sync_errors}")
        return {"cam_ids": cam_ids, "pids": pids, "true_shifts": true_shifts,
                "end_cuts": end_cuts, "shift_hat": shift_hat, "sync_errors": sync_errors,
                "max_sync_error": max(sync_errors.values()),
                "mean_sync_error": float(np.mean(list(sync_errors.values()))),
                "exact": False, "sync_failure": True, "s_hat": {}, "T_hat": 0,
                "T_common_raw": int(T_common_raw), "real_frame_anchor": 0,
                "frame_label_min": int(base)}
    if T_hat < min_overlap:
        logger.warning(f"  short corrected window (T_hat={T_hat} < {min_overlap}) -- "
                       f"KEEPING, flagged short_window (err={sync_errors})")

    s_hat = {c: s_true[c] + shift_hat[c] + T0 for c in cam_ids}
    # real(t) = raw(ref, t) = max_s + T0 + t, since shift_hat[ref] == true_shifts[ref] == 0.
    real_frame_anchor = base + max_s + T0
    exact = all(v == 0 for v in sync_errors.values())
    # Sanity: an exact estimate must make every camera read the same real frame.
    if exact:
        assert len(set(s_hat.values())) == 1 and next(iter(s_hat.values())) == real_frame_anchor, \
            f"exact trial but s_hat disagrees: {s_hat} vs anchor {real_frame_anchor}"

    return {
        "cam_ids": cam_ids,
        "pids": pids,
        "true_shifts": true_shifts,
        "end_cuts": end_cuts,
        "shift_hat": shift_hat,
        "sync_errors": sync_errors,
        "max_sync_error": max(sync_errors.values()),
        "mean_sync_error": float(np.mean(list(sync_errors.values()))),
        "exact": bool(exact),
        "s_hat": s_hat,
        "T_hat": int(T_hat),
        "T_common_raw": int(T_common_raw),
        "real_frame_anchor": int(real_frame_anchor),
        "frame_label_min": int(base),
        "sync_failure": False,
        "short_window": bool(T_hat < min_overlap),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="STEP 0: inject + estimate desync over every EgoHumans scene, "
                    "and partition them into exact / inexact recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ghost_root", required=True, type=Path,
                   help="ghost_outputs/egohumans_new (<activity>/<scene>/<cam>/body_data_clean).")
    p.add_argument("--gt_root", type=Path, default=None,
                   help="egohumans_gt_full. If given, scenes without processed_data/smpl are skipped.")
    p.add_argument("--out", required=True, type=Path, help="Manifest JSON to write.")
    p.add_argument("--max_shift", type=int, default=75)
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument("--min_overlap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activity", default="", help="Comma-separated activity filter (e.g. 07_tennis).")
    p.add_argument("--scenes", default="", help="Comma-separated '<activity>/<scene>' filter.")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    only_act    = {a.strip() for a in args.activity.split(",") if a.strip()}
    only_scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}

    scene_dirs: list[Path] = []
    for act_dir in sorted(d for d in args.ghost_root.iterdir() if d.is_dir()):
        if only_act and act_dir.name not in only_act:
            continue
        for scene_dir in sorted(d for d in act_dir.iterdir() if d.is_dir()):
            key = f"{act_dir.name}/{scene_dir.name}"
            if only_scenes and key not in only_scenes:
                continue
            scene_dirs.append(scene_dir)
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]
    logger.info(f"{len(scene_dirs)} candidate scene(s) under {args.ghost_root}")

    sync = Synchronizer(device=args.device, min_overlap=args.min_overlap, max_shift=args.max_shift)
    rng  = np.random.default_rng(args.seed)

    manifest: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    for scene_dir in scene_dirs:
        key = f"{scene_dir.parent.name}/{scene_dir.name}"
        logger.info(f"\n=== {key} ===")

        vggt_path = scene_dir / "vggt_cameras_centered.npz"
        if not vggt_path.exists():
            skipped[key] = "no vggt_cameras_centered.npz"; logger.warning(f"  {skipped[key]}"); continue
        if not (scene_dir / "mapanything_scale_baseline.npy").exists():
            skipped[key] = "no mapanything_scale_baseline.npy"; logger.warning(f"  {skipped[key]}"); continue
        if args.gt_root is not None:
            smpl_dir = args.gt_root / scene_dir.parent.name / scene_dir.name / "processed_data" / "smpl"
            if not smpl_dir.is_dir() or not any(smpl_dir.glob("*.npy")):
                skipped[key] = "no GT processed_data/smpl"; logger.warning(f"  {skipped[key]}"); continue

        with np.load(str(vggt_path), allow_pickle=False) as v:
            T_common_raw = int(v["valid"].shape[0])
            vggt_cams = [n.decode() if isinstance(n, bytes) else str(n) for n in v["camera_names"]]

        cam_data, frame_label_min = load_scene(scene_dir)
        if len(cam_data) < 2:
            skipped[key] = f"only {len(cam_data)} camera(s) with body_data_clean"
            logger.warning(f"  {skipped[key]}"); continue

        pids = select_persons(cam_data)
        if not pids:
            skipped[key] = "no person seen by >=2 cameras"; logger.warning(f"  {skipped[key]}"); continue
        _fill_missing_persons(cam_data, pids)

        # Camera axis must be the VGGT one: downstream, extrinsics[:, k] is
        # positionally matched against sorted camera directories.
        cam_ids = [c for c in vggt_cams if c in cam_data]
        if len(cam_ids) < 2:
            skipped[key] = f"only {len(cam_ids)} camera(s) shared with the VGGT axis"
            logger.warning(f"  {skipped[key]}"); continue
        logger.info(f"  {len(cam_ids)} cam(s) {cam_ids}, pids={pids}, "
                    f"T_raw={T_common_raw}, frame_label_min={frame_label_min}")

        trials = []
        for k in range(args.n_trials):
            t = run_sync_trial(cam_data, pids, cam_ids, T_common_raw, sync, rng,
                               args.max_shift, args.min_overlap, args.device,
                               frame_label_min)
            if t is None:
                continue
            t["trial"] = k
            trials.append(t)
            logger.info(f"  trial {k}: {'EXACT' if t['exact'] else 'INEXACT'}  "
                        f"err={t['sync_errors']}  T_hat={t['T_hat']}  anchor={t['real_frame_anchor']}")
        if not trials:
            skipped[key] = "no trial solvable (overlap/NaN)"; logger.warning(f"  {skipped[key]}"); continue

        manifest[key] = {
            "scene_dir": str(scene_dir),
            "frame_label_min": int(frame_label_min),
            "vggt_camera_names": vggt_cams,
            "trials": trials,
        }

    short_trials   = [(k, t) for k, m in manifest.items() for t in m["trials"] if t.get("short_window")]
    failed_trials  = [(k, t) for k, m in manifest.items() for t in m["trials"] if t.get("sync_failure")]
    exact_trials   = [(k, t) for k, m in manifest.items() for t in m["trials"] if t["exact"]]
    inexact_trials = [(k, t) for k, m in manifest.items() for t in m["trials"] if not t["exact"]]
    inexact_scenes = sorted({k for k, _ in inexact_trials})

    out = {
        "config": {"max_shift": args.max_shift, "n_trials": args.n_trials,
                   "min_overlap": args.min_overlap, "seed": args.seed,
                   "ghost_root": str(args.ghost_root),
                   "gt_root": str(args.gt_root) if args.gt_root else None},
        "scenes": manifest,
        "skipped": skipped,
        "partition": {
            "n_scenes": len(manifest),
            "n_trials_exact": len(exact_trials),
            "n_trials_inexact": len(inexact_trials),
            "scenes_needing_images": inexact_scenes,
            "n_trials_short_window": len(short_trials),
            "n_trials_sync_failure": len(failed_trials),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    n_tr = len(exact_trials) + len(inexact_trials)
    logger.info(f"\n{'='*66}\nmax_shift={args.max_shift}  scenes={len(manifest)}  trials={n_tr}")
    logger.info(f"  EXACT   (reuse existing VGGT/MA, no images): {len(exact_trials)}")
    logger.info(f"  INEXACT (need images + VGGT/MA rerun)      : {len(inexact_trials)}")
    logger.info(f"  short window (<min_overlap, kept)          : {len(short_trials)}")
    logger.info(f"  SYNC FAILURES (no overlap after correction): {len(failed_trials)}")
    logger.info(f"  skipped scenes                             : {len(skipped)}")
    if inexact_trials:
        logger.info("\n  scenes needing images:")
        for k, t in sorted(inexact_trials, key=lambda kt: -kt[1]["max_sync_error"]):
            bad = {c: e for c, e in t["sync_errors"].items() if e}
            logger.info(f"    {k} trial{t['trial']}: max_err={t['max_sync_error']} frames  {bad}")
    logger.info(f"\nManifest -> {args.out}\n{'='*66}")


if __name__ == "__main__":
    main()
