"""Verify the Synchronizer is fed the representation its cost function expects.

``Synchronizer._compute_cost_matrix`` converts its input with
``_axis_angle_to_rot_mat`` and scores an SO(3) geodesic, so it must be given
axis-angle ROTATIONS ``(T, 51, 3)``.  ``evaluation/alignment_experiments_multi.py``
does that (via ``utilities.body_data.load_person_smplx_pose``); ``main.py`` and
``scripts/sync_demo.py`` historically passed ``pred_keypoints_3d`` — MHR70 3D
POSITIONS — which the cost silently reinterpreted as rotation vectors.  The
resulting surface still correlated on fast motion but went flat on slow motion,
where ``estimate_couple_offset`` falls back to ``return 0.0`` and every camera
comes back with offset 0.

This script injects known per-camera delays and reports whether they are
recovered (residual ≈ 0).  Run it on a slow-motion scene (cooking) — that is
where the mis-fed representation fails and the correct one should succeed.

Usage
-----
    pixi run python scripts/verify_sync_representation.py \\
        --scene Gym_010_cooking1 --device cuda
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import tyro

from sync_demo import _camera_names, _estimate_offsets

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _load_tracks(scene_dir: Path, cams: list[str],
                 min_frames: int = 0) -> list[dict[int, dict]]:
    """tracks[k][pid] = arrays.  ``min_frames`` drops sparsely observed persons.

    A person can share an ID across cameras yet be observed in only a few hundred
    frames; its cross-correlation then prefers whatever shift makes its two
    observation windows coincide, which is a spurious offset, and a plain sum over
    persons lets that outvote a fully tracked subject.
    """
    tracks: list[dict[int, dict]] = []
    for cam in cams:
        per_pid: dict[int, dict] = {}
        sub = "body_data_clean" if (scene_dir / cam / "body_data_clean").is_dir() else "body_data"
        for npz_path in sorted((scene_dir / cam / sub).glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            with np.load(npz_path, allow_pickle=False) as d:
                arrays = {k: d[k] for k in d.files}
            if len(arrays["frame_indices"]) < min_frames:
                continue
            per_pid[pid] = arrays
        tracks.append(per_pid)
    return tracks


def main(
    scene:       str,
    scenes_root: Path = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"),
    delays:      tuple[int, ...] = (0, 10, 8, 4),
    max_shift:   int = 15,
    device:      str = "cuda",
    tol:         int = 3,
    min_frames:  int = 0,
) -> None:
    scene_dir = Path(scenes_root) / scene
    cams = _camera_names(scene_dir)
    tracks = _load_tracks(scene_dir, cams, min_frames)
    T_scene = int(max(int(p["frame_indices"].max())
                      for per in tracks for p in per.values())) + 1
    inj = np.array(list(delays)[:len(cams)], dtype=int)
    logger.info(f"{scene}: {len(cams)} cams {cams}, T={T_scene}, device={device}, "
                f"min_frames={min_frames}, pids={[sorted(t) for t in tracks]}")

    t0 = time.time()
    zero = np.array(_estimate_offsets(tracks, np.zeros(len(cams), int),
                                      T_scene, device, max_shift))
    logger.info(f"  delay=0   → est {list(map(int, zero))}   "
                f"(hardware-synced ⇒ want ≈0)   [{time.time()-t0:.0f}s]")

    t0 = time.time()
    est = np.array(_estimate_offsets(tracks, inj, T_scene, device, max_shift))
    resid = inj + est          # estimate_initial_times returns −δ
    ok = int(np.abs(resid).max()) <= tol
    logger.info(f"  inject {list(map(int, inj))} → est {list(map(int, est))}   "
                f"residual {list(map(int, resid))}   [{time.time()-t0:.0f}s]")
    logger.info(f"  RESULT: {'RECOVERED' if ok else 'FAILED'} "
                f"(max |residual| = {int(np.abs(resid).max())} frames, tol={tol})")


if __name__ == "__main__":
    tyro.cli(main)
