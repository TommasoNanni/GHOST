"""PAIRED CONTROL for the through-sync EgoHumans evaluation: the true alignment.

Builds, for every INEXACT trial, a second trial directory identical in shape to
the one evaluation/sync_build_egohumans.py produced, except that every camera
reads the SAME real instant (s_hat = real_frame_anchor for all of them) instead
of its estimated one. Scoring the two against each other isolates the single
variable of interest -- the per-camera offset error -- with the scene, the
window, the frame count and the code path all held fixed.

Why this exists
---------------
The obvious control, "score the original scene directory restricted to the same
window", is NOT paired. It differs from the sync arm in two ways at once, and the
second one is not the one being measured:

  1. the alignment (what we want to vary), and
  2. the set of frames handed to BodyPlacer -- the original directory carries the
     scene's FULL tracks, so the placer sees the whole clip, while the sync arm's
     placer sees only the window.

Measured on 02_lego/002_legoassemble: fused pre-placement poses agree to 6.2 mm
between the two arms, yet the PLACED bodies differ by 2517 mm (a 1991 mm / 77 deg
rigid discrepancy with a 782 mm non-rigid residual). Nothing upstream of the
placer differs -- VGGT extrinsics agree to 0.008, depth to a 1.005 ratio, and the
MapAnything scale to 1.00x -- so that gap is the placer responding to a different
frame set, not to desync. A control that carries that confound cannot measure a
sync cost.

Cheap by construction: VGGT infers each timestep independently (run_vggt.py loops
over t and distributes frames round-robin across devices), so slicing the scene's
existing arrays over the window IS the true-alignment rerun -- bit-identical, no
GPU. Depth is not copied: it is optional and legacy in BodyPlacer (loaded at
placer.py:244, consumed only by the depth-rescaling path at :1262, never by
estimate_procrustes_dlt_mhr).

Usage
-----
    pixi run python evaluation/sync_control_egohumans.py \\
        --manifest      eval_egohumans/sync_manifest_s75.json \\
        --control_root  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_ctrl_s75

Then score it with the ordinary evaluator, pointing --sync_root at the control:
    ... evaluate_egohumans_sync.py --sync_root <control_root> --dump_dir <ctrl dumps>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def window_and_relabel(data: dict[str, np.ndarray], s_hat: int, T_hat: int,
                       base: int) -> dict[str, np.ndarray] | None:
    """Keep frames whose LABEL is in [s_hat, s_hat+T_hat), relabel to start at `base`.

    Copied verbatim from evaluation/sync_build_egohumans.py so the control is
    written by exactly the same rule as the arm it is compared against.
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
        out[k] = v[mask] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n) else v
    out["frame_indices"] = (fi[mask] - s_hat + base).astype(data["frame_indices"].dtype)
    return out


def slice_leading_axis(npz_path: Path, out_path: Path, r0: int, r1: int) -> None:
    """Copy an .npz, slicing every array whose leading axis is the frame axis.

    The frame axis is identified by length, taken from the longest array present:
    a scene's camera npz stores (T,K,...) arrays alongside (K,) camera_names, and
    only the former may be sliced. Arrays shorter than the window are an error,
    not something to pad -- silently short-changing the window is exactly the kind
    of failure this evaluation is trying not to have.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        T = max(v.shape[0] for v in z.values() if v.ndim >= 1)
        if r1 > T:
            raise ValueError(f"{npz_path.name}: window [{r0},{r1}) exceeds T={T}")
        out = {k: (v[r0:r1] if (v.ndim >= 1 and v.shape[0] == T) else v) for k, v in z.items()}
    np.savez_compressed(out_path, **out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--control_root", required=True, type=Path)
    p.add_argument("--ghost_root", type=Path, default=None)
    p.add_argument("--scenes", default="", help="Comma-separated '<activity>/<scene>' filter.")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    ghost_root = args.ghost_root or Path(manifest["config"]["ghost_root"])
    only = {s.strip() for s in args.scenes.split(",") if s.strip()}

    todo = []
    for key, entry in manifest["scenes"].items():
        if only and key not in only:
            continue
        for trial in entry["trials"]:
            if not trial["exact"] and not trial.get("sync_failure"):
                todo.append((key, entry, trial))
    if args.max_scenes:
        todo = todo[:args.max_scenes]
    logger.info(f"{len(todo)} inexact trial(s) to build a true-alignment control for")

    n_ok = n_skip = n_fail = 0
    for i, (key, entry, trial) in enumerate(todo, 1):
        activity, scene = key.split("/", 1)
        k = trial["trial"]
        base = entry["frame_label_min"]
        anchor, T_hat = trial["real_frame_anchor"], trial["T_hat"]
        src = ghost_root / activity / scene
        dst = args.control_root / activity / scene / f"trial{k}"
        head = f"[{i}/{len(todo)}] {key} trial{k} (err={trial['max_sync_error']})"

        if (dst / "sync_meta.json").exists() and not args.overwrite:
            logger.info(f"{head}: exists, skip"); n_skip += 1; continue

        try:
            dst.mkdir(parents=True, exist_ok=True)
            r0, r1 = anchor - base, anchor - base + T_hat

            # Cameras: slicing the scene's own arrays IS the true-alignment rerun.
            slice_leading_axis(src / "vggt_cameras_centered.npz",
                               dst / "vggt_cameras_centered.npz", r0, r1)

            # MapAnything scale: per-frame, same window.
            sc = np.load(src / "mapanything_scale_baseline.npy")
            if sc.ndim >= 1 and sc.shape[0] >= r1:
                sc = sc[r0:r1]
            np.save(dst / "mapanything_scale_baseline.npy", sc)

            # Bodies: EVERY camera reads the anchor -- that is the true alignment.
            for cam in trial["cam_ids"]:
                out_dir = dst / cam / "body_data_clean"
                out_dir.mkdir(parents=True, exist_ok=True)
                for npz_path in sorted((src / cam / "body_data_clean").glob("person_*.npz")):
                    data = dict(np.load(npz_path, allow_pickle=False))
                    w = window_and_relabel(data, anchor, T_hat, base)
                    if w is not None:
                        np.savez_compressed(out_dir / npz_path.name, **w)

            meta = dict(trial)
            meta.update({"scene": key, "frame_label_min": base, "control": True,
                         "s_hat": {c: anchor for c in trial["cam_ids"]},
                         "max_shift": manifest["config"]["max_shift"]})
            with open(dst / "sync_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.error(f"{head}: FAILED -- {e}"); n_fail += 1; continue

        n_ok += 1
        if i % 10 == 0 or i == len(todo):
            logger.info(f"{head}: built ({n_ok} ok, {n_skip} skipped, {n_fail} failed)")

    logger.info(f"\nDone. built={n_ok} skipped={n_skip} failed={n_fail} -> {args.control_root}")


if __name__ == "__main__":
    main()
