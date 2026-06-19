"""Prune a FusionWithBetas checkpoint to PoseFusionModule format.

Removes all betas_aggregator.* keys and strips the pose_module. prefix
so the state dict matches PoseFusionModule directly.

Usage:
    python scripts/prune_checkpoint.py <input.pt> <output.pt>
"""

import sys
from pathlib import Path
import torch


def prune(src: Path, dst: Path) -> None:
    ckpt = torch.load(str(src), map_location="cpu")

    # Locate the state dict — TrainerV2 saves under "model"
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

    torch.save(ckpt, str(dst))
    n_removed = sum(1 for k in state if k.startswith("betas_aggregator."))
    print(f"Pruned {n_removed} betas_aggregator keys.")
    print(f"Remaining keys: {len(new_state)}")
    print(f"Saved to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/prune_checkpoint.py <input.pt> <output.pt>")
        sys.exit(1)
    prune(Path(sys.argv[1]), Path(sys.argv[2]))
