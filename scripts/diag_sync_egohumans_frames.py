#!/usr/bin/env python
"""Localise a frame-mapping error in the through-sync EgoHumans evaluation.

For ONE inexact trial with a small sync error, score the same window twice:

  A) off the REBUILT trial dir, local labels, gt_offset = anchor - base
  B) off the ORIGINAL scene dir, real labels, gt_offset = 0

For a 1-frame error the two reconstructions are the same geometry (verified:
rebuilt VGGT extrinsics match the original to ~1e-2), so A and B must land within
a few mm of each other. A large gap means the trial-dir path mis-maps frames --
predictions scored against the wrong instant's GT -- which shows up as normal
PA (pose is right) but destroyed W-dagger/GA (position is wrong).

Run on the debug partition, not the login node: BodyPlacer is CPU-bound for
minutes and the login-node watchdog SIGKILLs it.

    sbatch --export=ALL,SCENE=02_lego/001_legoassemble bash_jobs/eval_egohumans_sync.sh   # (normal eval)
    # this diagnostic:
    srun -A a144 -p debug -t 00:30:00 --gpus-per-node=1 --mem=64G \\
        pixi run python scripts/diag_sync_egohumans_frames.py --scene 02_lego/001_legoassemble
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluate_egohumans_sync import eval_scene, scene_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="02_lego/001_legoassemble")
    ap.add_argument("--manifest", default="eval_egohumans/sync_manifest_s75.json")
    ap.add_argument("--sync_root",
                    default="/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_sync_s75")
    ap.add_argument("--gt_root", default="/iopsstor/scratch/cscs/tnanni/egohumans_gt_full")
    ap.add_argument("--smplx_model", default=str(_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl"))
    args = ap.parse_args()

    M = json.loads(Path(args.manifest).read_text())
    entry = M["scenes"][args.scene]
    trial = entry["trials"][0]
    base = entry["frame_label_min"]
    anchor, T_hat = trial["real_frame_anchor"], trial["T_hat"]
    activity, scene = args.scene.split("/", 1)
    gt_scene = Path(args.gt_root) / activity / scene
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{args.scene}: err={trial['sync_errors']} anchor={anchor} T_hat={T_hat} base={base}")
    print(f"  s_hat={trial['s_hat']}\n")

    cases = [
        ("A rebuilt trial dir (local labels)",
         Path(args.sync_root) / activity / scene / f"trial{trial['trial']}",
         anchor - base, base),
        ("B original scene dir (real labels)",
         Path(entry["scene_dir"]), 0, base),
    ]
    for label, ghost_scene, gt_offset, row_base in cases:
        d = eval_scene(ghost_scene, gt_scene, device, args.smplx_model,
                       window=(anchor, T_hat), gt_offset=gt_offset, row_base=row_base)
        if d is None:
            print(f"{label}: returned None"); continue
        w, ga, pa = scene_metrics(d["pred"], d["gt"], bool(d["have_world"]))
        print(f"{label}: gt_offset={gt_offset:4d} row_base={row_base} "
              f"n={int(d['n_scored']):4d}  W+={w*1000:9.1f}  GA={ga*1000:8.1f}  PA={pa*1000:6.1f} mm")


if __name__ == "__main__":
    main()
