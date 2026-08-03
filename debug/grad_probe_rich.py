"""Gradient-share probe for the R-family training objective — RICH train.

QUESTION
--------
In the running R2 configuration the loss VALUES split roughly 60:40 between
lambda_pose * L_pose (rotation, hands weighted 3x = 79% of the term's weight
mass) and L_joint (FK position, body-only). Loss values are only a proxy: the
FK Jacobian amplifies position gradients on proximal joints (a hip rotation
moves 8 leg joints with up-to-0.8 m lever arms), while the geodesic rotation
loss treats every joint independently. So: WHERE do the actual gradients go?

WHAT IS MEASURED
----------------
On a few real training batches, with the real model at its current R2
checkpoint, each loss term is backpropagated SEPARATELY:

1. PARAM SPACE — ||grad|| of lambda*L_pose vs lambda*L_joint over all model
   parameters, their ratio, and the cosine between the two gradient vectors
   (negative = the objectives actively conflict).

2. OUTPUT SPACE — grad of each term w.r.t. the model output pose_aggr
   (B, T, P, 54, 6), reduced to a per-joint-slot norm. Reported per group
   (body / hands / face, packed layout of loss_v2.PoseMSELoss) and per body
   joint, so the hip share of the position gradient is visible directly.

CONFIRMS the "rotation-dominated" reading: lambda*L_pose param-grad comparable
to or larger than L_joint's, with its output-space mass concentrated on hand
slots.  REFUTES it: L_joint gradients dominate once FK amplification is
accounted for.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/grad_probe_rich.py \
        --checkpoint checkpoints/fusion_r2/last.pt --n_scenes 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "human_body_prior"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration import CONFIG
from fusion.fusion_module_v2 import PoseFusionModule
from fusion.loss_v2 import JointPositionLoss, PoseMSELoss

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Load the training script by file path (a site-packages `scripts` package
# shadows the repo directory, so `import scripts.train_rich_v2` is unreliable).
_spec = importlib.util.spec_from_file_location(
    "train_rich_v2", _REPO / "scripts" / "train_rich_v2.py")
_trv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trv2)

_BODY_NAMES = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def _truncate_T(d: dict, T_full: int, T_max: int) -> dict:
    """Slice every tensor whose second dim equals the scene length."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == T_full:
            out[k] = v[:, :T_max]
        else:
            out[k] = v
    return out


def _flat_grads(loss: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="R2 gradient-share probe")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/fusion_r2/last.pt"))
    ap.add_argument("--n_scenes", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=256,
                    help="cap scene length (2x the temporal window by default)")
    ap.add_argument("--pose_hand_weight", type=float, default=3.0)
    ap.add_argument("--joint_body_only", type=int, default=1)
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/grad_probe_rich.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    lambda_pose  = float(CONFIG.fusion.loss.pose_mse_weight)
    lambda_joint = float(CONFIG.fusion.loss.joint_position_weight)

    # ── Model at the current R2 state ────────────────────────────────────────
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = CONFIG.fusion.architecture
    mc = state.get("model_config") or {}
    model = PoseFusionModule(
        embedding_dim=arch.embedding_dimension,
        num_heads=mc.get("num_heads", arch.num_heads),
        num_layers=arch.num_layers,
        max_temporal_len=arch.max_temporal_len,
        dropout=arch.dropout,
        temporal_window=mc.get("temporal_window", arch.temporal_window),
        kintree_mask_k=mc.get("kintree_mask_k",
                              getattr(arch, "kintree_mask_k", None)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.train()   # match the training regime (dropout active, seeded)
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"checkpoint epoch {state.get('epoch', '?')}  "
                f"model_config={mc}  lambda_pose={lambda_pose} "
                f"lambda_joint={lambda_joint}")

    # ── Data: first n train scenes, deterministic ────────────────────────────
    scenes = [s for s in sorted(_trv2.SCENES_ROOT.iterdir())
              if s.is_dir() and s.name not in _trv2.SKIP_SCENES]
    train_scenes, _ = _trv2._split_by_location(scenes, _trv2.NUM_VAL_SCENES)
    dps = _trv2.load_datapoints(train_scenes[:args.n_scenes])
    ds = _trv2.RICHFusionDataset(dps, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    loss_pose  = PoseMSELoss(hand_weight=args.pose_hand_weight)
    loss_joint = JointPositionLoss(body_only=bool(args.joint_body_only))

    rows = []
    for batch in loader:
        inputs, targets = batch
        scene = targets.get("scene_name", ["?"])
        scene = scene[0] if isinstance(scene, (list, tuple)) else scene

        T_full = int(inputs["pose"].shape[1])
        if T_full > args.max_frames:
            inputs  = _truncate_T(inputs, T_full, args.max_frames)
            targets = _truncate_T(targets, T_full, args.max_frames)

        model_inputs = {k: v for k, v in inputs.items()
                        if k in {"pose", "person_mask", "joint_mask"}}
        preds = model(**model_inputs)
        preds = preds if isinstance(preds, tuple) else (preds,)
        pose_aggr = preds[0]                      # (B, T, P, 54, 6)

        Lp = lambda_pose  * loss_pose(preds, targets)
        Lj = lambda_joint * loss_joint(preds, targets)

        # 1. param space
        gp = _flat_grads(Lp, params)
        gj = _flat_grads(Lj, params)
        cos = float(torch.nn.functional.cosine_similarity(gp, gj, dim=0))

        # 2. output space, per joint slot
        op = torch.autograd.grad(Lp, pose_aggr, retain_graph=True)[0]
        oj = torch.autograd.grad(Lj, pose_aggr, retain_graph=False)[0]
        per_slot_p = op.pow(2).sum(dim=(0, 1, 2, 4)).sqrt().detach().numpy()  # (54,)
        per_slot_j = oj.pow(2).sum(dim=(0, 1, 2, 4)).sqrt().detach().numpy()

        def grp(x):
            return {"body": float(np.linalg.norm(x[0:21])),
                    "hands": float(np.linalg.norm(x[21:51])),
                    "face": float(np.linalg.norm(x[51:54]))}

        row = {
            "scene": scene, "T": int(pose_aggr.shape[1]),
            "loss_pose_w": float(Lp), "loss_joint_w": float(Lj),
            "param_grad_pose": float(gp.norm()),
            "param_grad_joint": float(gj.norm()),
            "param_cosine": cos,
            "out_pose_by_group": grp(per_slot_p),
            "out_joint_by_group": grp(per_slot_j),
            "out_pose_per_slot": per_slot_p.tolist(),
            "out_joint_per_slot": per_slot_j.tolist(),
        }
        rows.append(row)
        r = row["param_grad_pose"] / max(row["param_grad_joint"], 1e-12)
        logger.info(
            f"{scene}  T={row['T']}  L_pose*λ={row['loss_pose_w']:.5f} "
            f"L_joint*λ={row['loss_joint_w']:.5f}  ‖g_pose‖/‖g_joint‖={r:.2f} "
            f"cos={cos:+.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    gp_m = float(np.mean([r["param_grad_pose"] for r in rows]))
    gj_m = float(np.mean([r["param_grad_joint"] for r in rows]))
    print("\n================ AGGREGATE ================")
    print(f"scenes: {len(rows)}   (model.train(), seeded dropout)")
    print(f"param-space  ‖grad λ·L_pose‖  mean : {gp_m:.3e}")
    print(f"param-space  ‖grad λ·L_joint‖ mean : {gj_m:.3e}")
    print(f"ratio pose/joint                   : {gp_m / max(gj_m, 1e-12):.2f}")
    print(f"cosine(g_pose, g_joint) mean       : "
          f"{np.mean([r['param_cosine'] for r in rows]):+.3f}")

    for tag, key in (("λ·L_pose", "out_pose_by_group"),
                     ("λ·L_joint", "out_joint_by_group")):
        g = {k: np.mean([r[key][k] for r in rows]) for k in ("body", "hands", "face")}
        tot = sum(g.values()) + 1e-12
        print(f"\noutput-space {tag}: " + "  ".join(
            f"{k}={v:.2e} ({100 * v / tot:.0f}%)" for k, v in g.items()))

    slot_j = np.mean([r["out_joint_per_slot"] for r in rows], axis=0)
    slot_p = np.mean([r["out_pose_per_slot"] for r in rows], axis=0)
    print("\nper-body-joint output-space grad norms (mean over scenes):")
    print(f"{'joint':<15} {'λ·L_joint':>10} {'λ·L_pose':>10}")
    for i, name in enumerate(_BODY_NAMES):
        print(f"{name:<15} {slot_j[i]:>10.2e} {slot_p[i]:>10.2e}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
