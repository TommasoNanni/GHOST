"""
Desynchronise a scene, recover the offsets, then reconstruct from the result.

The qualitative figures so far assume the cameras are already aligned.  This runs
the honest version end to end on one scene:

  1. load every camera's body tracks and apply a random per-camera shift,
     exactly as evaluation/alignment_experiments_multi_egohumans.py does;
  2. recover those shifts with the Synchronizer (DTW + cycle-consistency +
     global least squares) and compare against the truth;
  3. pick a frame.  If every recovered offset matches, the corrected frame set is
     bit-identical to the aligned data and the VGGT already on disk is valid.  If
     any camera is off by r frames, that camera really would contribute image
     f + r, so VGGT has to be re-run on that desynchronised tuple and written to
     a scratch directory instead;
  4. render the reconstruction from whichever VGGT that turned out to be.

Step 3 is the whole point: it is what makes the figure a claim about the
*pipeline* rather than about hand-aligned inputs.

The shift/recovery code is copied from the alignment experiment rather than
imported, so tuning this demo can never move a published sync number.

Usage
-----
    pixi run python scripts/sync_to_figure.py --frame 289
    pixi run python scripts/sync_to_figure.py --frame 289 --seed 7 --max-shift 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from synchronize_videos.synchronizer import Synchronizer
from utilities.body_data import load_person_smplx_pose

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
RICH_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test")
UNDIST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted")
RICH_IMG = Path("/users/tnanni/rich_centered_mnt")
INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Person ids the alignment experiments use.  EgoHumans tracks are the curated
# body_data_clean ones (manual cross-view ReID); RICH has a single annotated
# subject per scene, always id 1, read from body_data.
SCENE_PIDS = {"01_tagging/007_tagging": [1, 2, 3, 4]}
DATASETS = {
    "egohumans": {"root": GHOST_ROOT, "body": "body_data_clean"},
    "rich":      {"root": RICH_ROOT,  "body": "body_data"},
}


def _frame_path(scene: str, cam: str, frame: int, dataset: str) -> Path:
    if dataset == "rich":
        return RICH_IMG / scene / cam / f"{frame:05d}_{int(cam.split('_')[1]):02d}.jpg"
    act, seq = scene.split("/")
    return (UNDIST_ROOT / act / INNER / act / seq / "exo" / cam
            / "images_undistorted" / "frames" / f"{frame:05d}.jpg")


# ── loading (copied from the alignment experiment) ─────────────────────────
def _load_anchored(npz_path: Path):
    """Load a person's pose track re-anchored to absolute frame 0.

    load_person_smplx_pose already fills interior detection gaps, but it anchors
    index 0 at that camera's own first detection.  A camera whose person appears
    27 frames later would therefore sit 27 frames early in its array, and
    apply_shifts slices by array index — so that pre-existing offset would be
    added to the injected one and the solver would be scored against the wrong
    truth.  Left-padding by frame_indices[0] makes array index == absolute frame
    everywhere; the padding carries zero confidence so it stays out of the cost.
    """
    result = load_person_smplx_pose(str(npz_path))
    if result is None:
        return None
    seq, conf = result
    with np.load(str(npz_path)) as d:
        if "frame_indices" not in d.files:
            raise KeyError(f"{npz_path}: no frame_indices — cannot anchor to absolute time")
        start = int(d["frame_indices"].astype(int).min())
    if start > 0:
        seq = torch.cat([seq.new_zeros((start, *seq.shape[1:])), seq], dim=0)
        conf = torch.cat([conf.new_zeros((start, *conf.shape[1:])), conf], dim=0)
    return seq, conf


def load_scene(scene_dir: Path, body_dirname: str = "body_data_clean"):
    """{cam: {pid: (rotations T x 51 x 3, conf T x 51)}}, padded to one length."""
    cam_data: dict[str, dict[int, tuple]] = {}
    for cam_dir in sorted(scene_dir.iterdir()):
        body = cam_dir / body_dirname
        if not cam_dir.is_dir() or not body.exists():
            continue
        persons = {}
        for npz_path in sorted(body.glob("person_*.npz")):
            got = _load_anchored(npz_path)
            if got is not None:
                persons[int(npz_path.stem.split("_")[1])] = got
        if persons:
            cam_data[cam_dir.name] = persons
    T = max(r.shape[0] for cam in cam_data.values() for r, _ in cam.values())
    for cam in cam_data.values():
        for pid, (rot, conf) in list(cam.items()):
            if rot.shape[0] < T:
                pad_r = torch.zeros((T - rot.shape[0], *rot.shape[1:]), dtype=rot.dtype)
                pad_c = torch.zeros((T - conf.shape[0], *conf.shape[1:]), dtype=conf.dtype)
                cam[pid] = (torch.cat([rot, pad_r]), torch.cat([conf, pad_c]))
    return cam_data


def apply_shifts(cam_data, shifts, end_cuts, pids, min_overlap=100):
    """Slice each camera's tracks to simulate a different start and end time."""
    max_s = max(shifts.values())
    joints_list, confs_list = [], []
    for cam_id in shifts:
        s = max_s - shifts[cam_id]
        ec = end_cuts[cam_id]
        pj, pc = [], []
        for pid in pids:
            rot, conf = cam_data[cam_id][pid]
            end = rot.shape[0] - ec if ec > 0 else rot.shape[0]
            if end - s < min_overlap:
                logger.warning(f"  {cam_id}: only {end - s} frames after shift")
                return None
            pj.append(rot[s:end].to(DEVICE))
            pc.append(conf[s:end].to(DEVICE))
        joints_list.append(pj)
        confs_list.append(pc)
    return joints_list, confs_list


# ── VGGT on a desynchronised tuple ─────────────────────────────────────────
def run_vggt_on_tuple(scene: str, cams: list[str], frames: dict[str, int],
                      out_dir: Path, weights: str | None, dataset: str):
    """Run VGGT on one frame per camera, taken at whatever frame each really has.

    frame_paths[t][k] is exactly the contract VGGTPreprocessor expects, so a
    desynchronised tuple needs no new VGGT code — only a different set of paths.
    """
    from preprocessing.run_vggt import VGGTPreprocessor

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [[_frame_path(scene, c, frames[c], dataset) for c in cams]]
    for p in paths[0]:
        if not p.exists():
            raise FileNotFoundError(p)
    logger.info(f"  VGGT on the desynchronised tuple -> {out_dir}")
    for c in cams:
        logger.info(f"    {c}: frame {frames[c]:05d}")
    kw = {"vggt_weights": weights} if weights else {}
    pre = VGGTPreprocessor(**kw)
    pre.process_scene(frame_paths=paths, output_dir=out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["egohumans", "rich"], default="egohumans")
    ap.add_argument("--scene", default="01_tagging/007_tagging")
    ap.add_argument("--pids", type=int, nargs="+", default=None,
                    help="person ids to synchronise; defaults to the alignment "
                         "experiment's choice for the dataset")
    ap.add_argument("--frame", type=int, default=289,
                    help="absolute frame to reconstruct, in the aligned timeline")
    ap.add_argument("--max-shift", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-overlap", type=int, default=100)
    ap.add_argument("--vggt-out", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/sync_demo"))
    ap.add_argument("--vggt-weights", default=None)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    scene_dir = cfg["root"] / args.scene
    pids = args.pids or (SCENE_PIDS.get(args.scene) if args.dataset == "egohumans" else [1])
    if not pids:
        sys.exit(f"no person ids configured for {args.scene}; pass --pids")
    logger.info(f"  persons: {pids}")

    # ── 1. desynchronise ───────────────────────────────────────────────────
    logger.info(f"Loading {args.scene}")
    cam_data = load_scene(scene_dir, cfg["body"])
    cams = list(cam_data.keys())
    logger.info(f"  {len(cams)} cameras: {cams}")

    rng = np.random.default_rng(args.seed)
    raw = [0] + rng.integers(-args.max_shift, args.max_shift + 1,
                             size=len(cams) - 1).tolist()
    true_shifts = {c: int(s) for c, s in zip(cams, raw)}
    end_cuts = {c: int(e) for c, e in zip(
        cams, rng.integers(0, args.max_shift + 1, size=len(cams)).tolist())}
    logger.info(f"  injected shifts : {true_shifts}")
    logger.info(f"  injected end cuts: {end_cuts}")

    got = apply_shifts(cam_data, true_shifts, end_cuts, pids, args.min_overlap)
    if got is None:
        sys.exit("not enough overlap after shifting")
    joints_list, confs_list = got

    # ── 2. recover ─────────────────────────────────────────────────────────
    sync = Synchronizer(use_acceleration_weights=False, device=DEVICE,
                        min_overlap=args.min_overlap, max_shift=args.max_shift,
                        verbose=False)
    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    weights = sync.cycle_consistency_weights(offset_mat)
    est = sync.estimate_initial_times(offset_mat, weights).cpu()
    if torch.isnan(est).any():
        sys.exit("synchroniser returned NaN (isolated camera)")

    true_t = torch.tensor([true_shifts[c] for c in cams], dtype=torch.float32)
    true_t = true_t - true_t.min()
    est = est - est.min()
    resid = (est - true_t)

    logger.info("\n  camera      true   estimated   residual")
    for i, c in enumerate(cams):
        logger.info(f"  {c:10s} {true_t[i]:5.0f} {est[i]:11.2f} {resid[i]:10.2f}")
    mae = resid.abs().mean().item()
    logger.info(f"  MAE {mae:.3f} frames, max |residual| {resid.abs().max():.3f}")

    # ── 3. which VGGT applies ──────────────────────────────────────────────
    delta = {c: int(round(float(resid[i]))) for i, c in enumerate(cams)}
    exact = all(v == 0 for v in delta.values())

    if exact:
        logger.info("\n  every offset recovered exactly — the corrected frame set is "
                    "identical to the aligned data, so the VGGT already on disk applies.")
        logger.info(f"  reconstruct frame {args.frame} with:")
        logger.info(
            "    pixi run python -m visualize.paper_qualitative_egohumans "
            f"--scene {args.scene} --frame {args.frame} --cams cam01 cam08 "
            f"--pred fusion_outputs/007_tagging_f{args.frame}w20.npz --panels c "
            "--depth-cams all --bg balls --ball-radius 1 --gt-outline white "
            "--max-points 2000000")
        return

    logger.warning(f"\n  residual desynchronisation: {delta}")
    frames = {c: args.frame + delta[c] for c in cams}
    run_vggt_on_tuple(args.scene, cams, frames,
                      args.vggt_out / args.scene.split("/")[-1],
                      args.vggt_weights, args.dataset)
    logger.warning(
        "\n  VGGT was re-run on the desynchronised tuple, but the reconstruction is "
        "NOT rendered here: the bodies would also have to be read at the shifted "
        "frames, and the sign of the residual->frame mapping has never been "
        "exercised (the synchroniser has been exact on these scenes). Rendering it "
        "now would risk a figure that is wrong in a way nobody could see. Extend "
        "deliberately once a real non-zero case exists.")


if __name__ == "__main__":
    main()
