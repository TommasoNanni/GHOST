#!/usr/bin/env python
"""Run the REFERENCE EgoHumans alignment experiment on scenes where the
through-sync dry run reports large errors.

evaluation/sync_dryrun_egohumans.py re-implements the injection + estimation of
evaluation/alignment_experiments_multi_egohumans.py so it can run over all 133
scenes (the reference hardcodes SCENE_PIDS for 14). If the re-implementation has
a defect, the two disagree on the same scene under the same protocol; if the
algorithm simply struggles there, they fail together.

Nothing here re-implements anything: it imports the reference module and calls
its own run_scene(). The only things overridden are the module-level constants
that gate which scenes and shifts it will consider -- SCENE_PIDS (to admit a
scene the reference never curated), MAX_SHIFT and N_TRIALS. Person ids come from
the scene's own body_data_clean, which is what the dry run uses.

    pixi run python scripts/diag_sync_reference_vs_dryrun.py \\
        --scene 03_fencing/014_fencing --max-shift 75 --trials 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import evaluation.alignment_experiments_multi_egohumans as ref
from synchronize_videos.synchronizer import Synchronizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", nargs="+", default=["03_fencing/014_fencing"])
    ap.add_argument("--max-shift", type=int, default=75)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for key in args.scene:
        scene_dir = ref.SCENES_ROOT / key
        cam_data = ref.load_scene(scene_dir)
        counts: dict[int, int] = {}
        for persons in cam_data.values():
            for pid in persons:
                counts[pid] = counts.get(pid, 0) + 1
        pids = sorted(p for p, n in counts.items() if n >= 2)
        ref.SCENE_PIDS[key] = pids
        ref.MAX_SHIFT = args.max_shift
        ref.N_TRIALS = args.trials

        print(f"\n{'='*70}\nREFERENCE implementation: {key}  "
              f"max_shift={args.max_shift} trials={args.trials} pids={pids}\n{'='*70}", flush=True)
        sync = Synchronizer(use_acceleration_weights=False, device=ref.DEVICE,
                            min_overlap=100, max_shift=args.max_shift, verbose=False)
        rng = np.random.default_rng(args.seed)
        out = ref.run_scene(scene_dir, sync, rng)
        if out is None:
            print(f"{key}: reference returned None (no usable trials)")
            continue
        print(f"\n{key}: REFERENCE  solved={out['solved']}/{out['n_trials']}  "
              f"MAE={out['mae']:.2f}  MedAE={out['median_ae']:.2f}  "
              f"within1={out['within_1']*100:.0f}%  max_err={out['errors'].max():.1f}")


if __name__ == "__main__":
    main()
