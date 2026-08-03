"""Verify the joint layout of every tensor the fusion module and its losses touch.

WHY THIS EXISTS
---------------
The pose tensor and the SMPL-X joint output do NOT use the same joint ordering,
and we have already shipped one silent 3-slot offset from assuming they did (the
`pred_joint_confidence` remap). Before adding per-joint loss weights or a
kinematic-tree attention mask, every index set must be derived from the VERIFIED
layout, not from a hardcoded list.

WHAT IT CHECKS
--------------
1. Shapes of the pose tensor going into and out of PoseFusionModule, and of the
   tensors JointPositionLoss / PoseMSELoss operate on.
2. The packed input layout, established EMPIRICALLY: rotate exactly one packed
   slot and observe which canonical SMPL-X output joints move. The moved set is
   the descendant set of the joint that slot drives.
3. Index sets (body / hands / face) in BOTH spaces, printed with joint names.
4. hop_dist over the kinematic tree, expressed in the layout the joint attention
   actually sees.

Run:
    pixi run python scripts/verify_joint_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utilities.smplx_utilities import _get_smplx_model, get_smplx_joints

_N = 55

# Canonical SMPL-X OUTPUT joint order (what model.joints returns).
CANONICAL = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2",
    "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1",
    "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1",
    "right_middle2", "right_middle3", "right_pinky1", "right_pinky2",
    "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1",
    "right_thumb2", "right_thumb3",
]

# PACKED input layout implied by get_smplx_joints' slicing of the (J*3) vector:
#   global_orient  = [0:3]      -> slot 0
#   body_pose      = [3:66]     -> slots 1..21
#   left_hand_pose = [66:111]   -> slots 22..36
#   right_hand_pose= [111:156]  -> slots 37..51
#   jaw_pose       = [156:159]  -> slot 52
#   leye_pose      = [159:162]  -> slot 53
#   reye_pose      = [162:165]  -> slot 54
PACKED = (
    ["pelvis"] + CANONICAL[1:22]
    + CANONICAL[25:40]        # left hand
    + CANONICAL[40:55]        # right hand
    + ["jaw", "left_eye_smplhf", "right_eye_smplhf"]
)
assert len(PACKED) == _N and len(CANONICAL) == _N


def identity_6d(*shape) -> torch.Tensor:
    e = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return e.expand(*shape, 6).contiguous()


def main() -> None:
    dev = torch.device("cpu")
    torch.manual_seed(0)

    print("=" * 78)
    print("1. TENSOR SHAPES AND WHAT EACH LOSS CONSUMES")
    print("=" * 78)
    print("  dataset  inputs['pose']    (B,T,K,P,55,6)   packed, INCLUDES root at slot 0")
    print("  dataset  targets['pose']   (B,T,P,55,6)     packed, INCLUDES root")
    print("  model    output pose_aggr  (B,T,P,54,6)     packed, root STRIPPED")
    print("           (fusion_module_v2.py:415 drops slot 0 when handed 55)")
    print("  PoseMSELoss   : operates on pose_aggr vs targets['pose'][...,1:,:]")
    print("                  -> PACKED order, root-stripped, J=54")
    print("  JointPositionLoss: FK output joints -> CANONICAL order, J=55 with root")
    print("                  body_only slices [..., :22, :]  (root + 21 body)")

    # ── 2. Empirical packed-slot -> moved canonical joints ───────────────────
    print("\n" + "=" * 78)
    print("2. EMPIRICAL LAYOUT — rotate one packed slot, see which canonical joints move")
    print("=" * 78)
    betas = torch.zeros(1, 1, 1, 10)
    base_pose = identity_6d(1, 1, 1, _N)                       # (1,1,1,55,6)
    with torch.no_grad():
        base = get_smplx_joints(base_pose, betas)[0, 0, 0, :_N].detach().numpy()

    # 40 deg about X as a 6D (first two rows of R)
    th = np.deg2rad(40.0)
    R = np.array([[1, 0, 0],
                  [0, np.cos(th), -np.sin(th)],
                  [0, np.sin(th),  np.cos(th)]], dtype=np.float32)
    rot6d = torch.tensor(np.concatenate([R[0], R[1]]), dtype=torch.float32)

    moved_by_slot = {}
    for j in range(_N):
        p = base_pose.clone()
        p[0, 0, 0, j] = rot6d
        with torch.no_grad():
            out = get_smplx_joints(p, betas)[0, 0, 0, :_N].detach().numpy()
        d = np.linalg.norm(out - base, axis=-1)
        moved_by_slot[j] = np.flatnonzero(d > 1e-3)            # >1 mm

    print(f"  {'packed':>6} {'expected name':<20} {'#moved':>7}  moved canonical joints")
    for j in list(range(25)) + list(range(50, 55)):
        mv = moved_by_slot[j]
        rng = (f"{mv.min()}..{mv.max()}" if len(mv) else "-")
        sample = ", ".join(CANONICAL[i] for i in mv[:3])
        print(f"  {j:>6} {PACKED[j]:<20} {len(mv):>7}  [{rng}] {sample}"
              + (" ..." if len(mv) > 3 else ""))

    # Decisive checks: which canonical family does each packed block drive?
    lh_moved = sorted({i for j in range(22, 37) for i in moved_by_slot[j]})
    rh_moved = sorted({i for j in range(37, 52) for i in moved_by_slot[j]})
    print(f"\n  packed 22..36 moves canonical {min(lh_moved)}..{max(lh_moved)} "
          f"({CANONICAL[min(lh_moved)]} .. {CANONICAL[max(lh_moved)]})")
    print(f"  packed 37..51 moves canonical {min(rh_moved)}..{max(rh_moved)} "
          f"({CANONICAL[min(rh_moved)]} .. {CANONICAL[max(rh_moved)]})")
    left_ok  = all(25 <= i <= 39 for i in lh_moved)
    right_ok = all(40 <= i <= 54 for i in rh_moved)
    print(f"  => packed hand block 22..51 == canonical 25..54 : "
          f"{'CONFIRMED' if (left_ok and right_ok) else 'MISMATCH'}")
    print(f"  => packed 52,53,54 = jaw,leye,reye ; canonical 22,23,24 = jaw,leye,reye")
    print(f"     LAYOUTS DIFFER: the pose tensor puts jaw/eyes LAST, the FK output")
    print(f"     puts them at 22..24. Any index set must state which space it is in.")

    # ── 3. Index sets in BOTH spaces ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("3. INDEX SETS  (packed = what PoseMSELoss and joint attention see)")
    print("=" * 78)
    packed_root  = [0]
    packed_body  = list(range(1, 22))
    packed_hands = list(range(22, 52))
    packed_face  = [52, 53, 54]
    for nm, idx in (("root", packed_root), ("body", packed_body),
                    ("hands", packed_hands), ("face", packed_face)):
        names = [PACKED[i] for i in idx]
        print(f"  packed {nm:<6} n={len(idx):2d}  {idx[0]}..{idx[-1]}  "
              f"{names[0]} .. {names[-1]}")

    print("\n  After the model strips the root, pose_aggr index = packed index - 1:")
    print(f"    body  -> 0..20   (21 joints)")
    print(f"    hands -> 21..50  (30 joints)")
    print(f"    face  -> 51..53  (3 joints)")
    print("  ^ THIS is the space PoseMSELoss weights must be built in.")
    print("  NOTE PoseMSELoss.body_only currently slices [..., :21, :] "
          "(loss_v2.py:76-77) = the 21 body joints, root already stripped. Correct.")

    print("\n  canonical (= FK output, what JointPositionLoss sees):")
    print(f"    root+body -> 0..21  (body_only slices [:22] -> correct)")
    print(f"    face      -> 22..24")
    print(f"    hands     -> 25..54")

    # ── 4. hop_dist in the attention's layout ────────────────────────────────
    print("\n" + "=" * 78)
    print("4. HOP DISTANCE over the kinematic tree, in pose_aggr (root-stripped) order")
    print("=" * 78)
    model = _get_smplx_model(1, dev, torch.float32)
    parents_can = np.asarray(model.parents.detach().cpu().numpy()).astype(int)[:_N]

    # STALENESS GUARD — fusion/fusion_module_v2.py hardcodes this table to build the
    # kintree attention mask without needing the SMPL-X files at construction time.
    # If body_models/ is ever swapped, the constant would silently diverge and the
    # mask would encode the wrong skeleton. Fail loudly here instead.
    from fusion.fusion_module_v2 import (
        _SMPLX_PARENTS_CANONICAL, _canonical_to_packed, _packed_hop_matrix)
    if not np.array_equal(np.array(_SMPLX_PARENTS_CANONICAL), parents_can):
        raise SystemExit(
            "FATAL: fusion_module_v2._SMPLX_PARENTS_CANONICAL no longer matches "
            f"model.parents.\n  hardcoded: {_SMPLX_PARENTS_CANONICAL}\n"
            f"  live     : {parents_can.tolist()}")
    print("  hardcoded _SMPLX_PARENTS_CANONICAL == live model.parents : OK")

    # The permutation and the FK slicing must agree: rotating packed slot j must
    # move exactly the canonical descendants of the joint that slot drives.
    c2p = _canonical_to_packed()
    p2c = {p: c for c, p in enumerate(c2p)}
    kids: list[list[int]] = [[] for _ in range(_N)]
    for c in range(1, _N):
        kids[parents_can[c]].append(c)

    def _desc(c: int) -> set[int]:
        out: set[int] = set()
        for ch in kids[c]:
            out.add(ch)
            out |= _desc(ch)
        return out

    bad = [j for j in range(_N) if set(moved_by_slot[j].tolist()) != _desc(p2c[j])]
    if bad:
        raise SystemExit(f"FATAL: packed->canonical mapping inconsistent at slots {bad}")
    print("  packed slot -> FK-moved joints == descendants(parents[perm]) : OK (all 55)")

    hop_mod = _packed_hop_matrix(54).numpy()

    # canonical index -> packed index
    can_to_packed = {}
    for p_idx, nm in enumerate(PACKED):
        can_to_packed[CANONICAL.index(nm)] = p_idx

    # adjacency in PACKED space, from the canonical parent table
    A = np.zeros((_N, _N), dtype=bool)
    for c in range(1, _N):
        pc = parents_can[c]
        if 0 <= pc < _N:
            A[can_to_packed[c], can_to_packed[pc]] = True
            A[can_to_packed[pc], can_to_packed[c]] = True

    INF = 10 ** 6
    hop = np.full((_N, _N), INF, dtype=int)
    np.fill_diagonal(hop, 0)
    hop[A] = 1
    for k in range(_N):                       # Floyd-Warshall
        hop = np.minimum(hop, hop[:, k, None] + hop[None, k, :])

    hop54 = hop[1:, 1:]                       # drop root -> pose_aggr space
    for k in (1, 2, 3):
        m = hop54 <= k
        np.fill_diagonal(m, True)
        rows_empty = int((~m).all(axis=1).sum())
        print(f"  k={k}: mask density {m.mean():6.2%}  "
              f"mean neighbours/joint {m.sum(1).mean():5.1f}  "
              f"all-False rows {rows_empty}")
    m2 = hop54 <= 2
    np.fill_diagonal(m2, True)
    print(f"\n  k=2 neighbours of a few joints (pose_aggr indices, names):")
    for j in (0, 1, 19, 20, 21):
        nb = np.flatnonzero(m2[j])
        print(f"    {j:>2} {PACKED[j+1]:<16} -> " +
              ", ".join(PACKED[i + 1] for i in nb[:9]) +
              (f" ... (+{len(nb)-9})" if len(nb) > 9 else ""))
    print(f"\n  hips: left_hip(pose_aggr 0) k=2 neighbours = "
          f"{[PACKED[i+1] for i in np.flatnonzero(m2[0])]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
