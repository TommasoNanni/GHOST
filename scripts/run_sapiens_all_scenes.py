"""Run Sapiens 2D pose estimation on all scenes in a ghost output root.

Skips any camera that already has all sapiens_centered_kps_person_*.npz outputs.
With multiple GPUs, scenes are distributed round-robin across worker processes.

Usage:
    pixi run python scripts/run_sapiens_all_scenes.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich/centered_train \\
        --model_size        0.3b
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load sapiens_centered_preprocess without requiring scripts/ to be a package
_spec = importlib.util.spec_from_file_location(
    "sapiens_centered_preprocess",
    _REPO_ROOT / "scripts" / "sapiens_centered_preprocess.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_load_model    = _mod._load_model
process_camera = _mod.process_camera


def _scene_fully_done(scene_dir: Path) -> bool:
    cam_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and (d / "body_data").is_dir()]
    if not cam_dirs:
        return True
    for cam_dir in cam_dirs:
        pids = [int(p.stem.split("_")[1]) for p in (cam_dir / "body_data").glob("person_*.npz")]
        if not pids:
            continue
        if not all((cam_dir / f"sapiens_centered_kps_person_{pid}.npz").exists() for pid in pids):
            return False
    return True


def _worker(rank: int, scene_dirs: list[Path], rich_root: str,
            model_size: str, devices: list[str], overwrite: bool) -> None:
    """Process a round-robin slice of scenes on one GPU."""
    device = torch.device(devices[rank])
    my_scenes = scene_dirs[rank::len(devices)]
    print(f"[GPU {rank} / {devices[rank]}] {len(my_scenes)} scene(s) assigned", flush=True)

    model = _load_model(model_size, device)

    for scene_dir in my_scenes:
        if not overwrite and _scene_fully_done(scene_dir):
            print(f"[GPU {rank}] [{scene_dir.name}] already done — skipping", flush=True)
            continue

        print(f"[GPU {rank}] === {scene_dir.name} ===", flush=True)
        cam_dirs = sorted(d for d in scene_dir.iterdir()
                          if d.is_dir() and (d / "body_data").is_dir())
        for cam_dir in cam_dirs:
            process_camera(cam_dir, scene_dir.name, rich_root, model, device,
                           overwrite=overwrite)

    print(f"[GPU {rank}] Done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghost_output_root", required=True, type=Path)
    ap.add_argument("--rich_root",         required=True, type=str)
    ap.add_argument("--model_size",        default="0.3b",
                    choices=["0.3b", "0.6b", "1b", "2b"])
    ap.add_argument("--scenes",            nargs="+", default=None)
    ap.add_argument("--overwrite",         action="store_true")
    args = ap.parse_args()

    ghost_root = args.ghost_output_root.resolve()

    scene_dirs = sorted(d for d in ghost_root.iterdir() if d.is_dir())
    if args.scenes:
        scene_dirs = [d for d in scene_dirs if d.name in args.scenes]

    n_gpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu > 0 else ["cpu"]
    print(f"Found {len(scene_dirs)} scene(s) | {len(devices)} device(s): {devices}")

    if len(devices) == 1:
        _worker(0, scene_dirs, args.rich_root, args.model_size, devices, args.overwrite)
    else:
        mp.spawn(
            _worker,
            args=(scene_dirs, args.rich_root, args.model_size, devices, args.overwrite),
            nprocs=len(devices),
            join=True,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
