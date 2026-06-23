"""Prune FusionWithBetas checkpoints to PoseFusionModule format.

Reads best.pt and last.pt from checkpoints/fusion_module_latest/,
writes pruned versions to checkpoints/fusion_module/.

Removes all betas_aggregator.* keys and strips the pose_module. prefix
so the state dict matches PoseFusionModule directly.

Usage:
    python scripts/prune_checkpoint.py
"""

from pathlib import Path
import torch

SRC_DIR = Path("checkpoints/fusion_module_latest")
DST_DIR = Path("checkpoints/fusion_module")


def prune(src: Path, dst: Path) -> None:
    ckpt = torch.load(str(src), map_location="cpu")

    key = None
    for k in ("model", "model_state_dict"):
        if k in ckpt and isinstance(ckpt[k], dict):
            key = k
            break
    state = ckpt[key] if key else ckpt

    new_state = {}
    for k, v in state.items():
        if k.startswith("betas_aggregator."):
            continue
        new_key = k[len("pose_module."):] if k.startswith("pose_module.") else k
        new_state[new_key] = v

    if key:
        ckpt[key] = new_state
    else:
        ckpt = new_state

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(dst))
    n_removed = sum(1 for k in state if k.startswith("betas_aggregator."))
    print(f"  removed {n_removed} betas_aggregator keys, {len(new_state)} remaining -> {dst}")


if __name__ == "__main__":
    for name in ("best.pt", "last.pt"):
        src = SRC_DIR / name
        dst = DST_DIR / name
        if not src.exists():
            print(f"  skipping {src} (not found)")
            continue
        print(f"Pruning {src} ...")
        prune(src, dst)
