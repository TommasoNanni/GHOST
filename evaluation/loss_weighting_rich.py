"""Is the fusion module's per-joint error pattern an artifact of L_joint weighting? — RICH.

HYPOTHESIS
----------
`JointPositionLoss` (fusion/loss_v2.py:152) supervises the POSITIONS of all 55
SMPL-X joints after forward kinematics. A rotation error at joint j displaces
every descendant of j, so j's gradient scales with how many descendants it has.
The hand chains hold 30 of the 55 joints; the whole leg chain holds 8. So the
loss is LOUD on the trunk/arm/hand chain and QUIET on the legs, and the module
would be expected to optimise the former and neglect the latter — which is what
the per-joint decomposition showed (B/A rotation bias: hips 1.19/1.33, feet
0.25/0.37).

WHAT IS COMPUTED
----------------
Part 1 — root-relative FK position error per joint GROUP (legs / trunk / arms /
hands / face), for A chordal and B ghost, with GT betas.

Part 2 — descendant count per joint from the SMPL-X kinematic tree, against the
per-joint B/A rotation-bias ratio from the previous experiment. The hypothesis
predicts a NEGATIVE Spearman correlation: more descendants -> lower B/A -> the
module does relatively better exactly where the loss weights it more.

Part 3 — the fraction of total L_joint magnitude contributed by hand joints,
which is the quantitative argument for reweighting.

NOTE ON GLOBAL ORIENTATION
--------------------------
Global orient is set to IDENTITY for both prediction and GT. This is equivalent
to the earlier 2x2 experiment, which prepended the GT root: root-relative error
is invariant to a global rotation shared by both sides, since
||R(a - b)|| = ||a - b||. The loss itself relies on the same cancellation
(loss_v2.py:189-193).

Group indices are VERIFIED against the model's own kintree parents rather than
trusted; the joint names per group are printed as a check.

INPUT
-----
Cached fused poses from evaluation/temporal_smoothness_rich.py (no inference
pass). GT betas are not cached, so the datapoint is rebuilt for `shape` only.

Usage
-----
    pixi run python evaluation/loss_weighting_rich.py \\
        --cache_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/fused_cache/rich_test \\
        --rich_data_root /capstor/scratch/cscs/tnanni/datasets/rich/centered_test \\
        --rich_gt_dir    /capstor/scratch/cscs/tnanni/datasets/rich
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utilities.smplx_utilities import _get_smplx_model, get_smplx_joints

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_VARIANTS = ("chordal", "ghost")
_N_JOINTS = 55

# Proposed grouping (SMPL-X canonical indices) — verified against kintree below.
_GROUPS = {
    "legs":  [1, 2, 4, 5, 7, 8, 10, 11],
    "trunk": [3, 6, 9, 12, 15],
    "arms":  [13, 14, 16, 17, 18, 19, 20, 21],
    "face":  [22, 23, 24],
    "hands": list(range(25, 55)),
}

_SMPLX_JOINT_NAMES = [
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
assert len(_SMPLX_JOINT_NAMES) == _N_JOINTS


def _identity_6d(shape, dtype, device) -> torch.Tensor:
    e = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)
    return e.expand(*shape, 6).contiguous()


def smplx_parents(device: torch.device) -> np.ndarray:
    """Parent index per joint from the SMPL-X model's kintree, truncated to 55."""
    model = _get_smplx_model(1, device, torch.float32)
    par = np.asarray(model.parents.detach().cpu().numpy()).astype(int)
    return par[:_N_JOINTS]


def descendant_counts(parents: np.ndarray) -> np.ndarray:
    """Number of transitive descendants of every joint."""
    n = len(parents)
    children: list[list[int]] = [[] for _ in range(n)]
    for j in range(1, n):
        p = parents[j]
        if 0 <= p < n:
            children[p].append(j)
    counts = np.zeros(n, dtype=int)

    def walk(j: int) -> int:
        tot = 0
        for c in children[j]:
            tot += 1 + walk(c)
        counts[j] = tot
        return tot

    walk(0)
    return counts


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c > 1:
                m = inv == i
                r[m] = r[m].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


@torch.no_grad()
def fk_rootrel(pose54: np.ndarray, betas: torch.Tensor, device, chunk: int) -> np.ndarray:
    """(T,P,54,6) -> root-relative FK joints (T,P,55,3), identity global orient."""
    T, P = pose54.shape[0], pose54.shape[1]
    body = torch.from_numpy(pose54).to(device).float()[None]        # (1,T,P,54,6)
    root = _identity_6d((1, T, P, 1), body.dtype, device)
    full = torch.cat([root, body], dim=3)                           # (1,T,P,55,6)
    out = []
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        j = get_smplx_joints(full[:, t0:t1], betas[:, t0:t1])[..., :_N_JOINTS, :]
        out.append((j - j[..., :1, :])[0].cpu())
    return torch.cat(out, 0).numpy().astype(np.float64)             # (T,P,55,3)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Does L_joint's descendant weighting explain the per-joint error pattern?")
    ap.add_argument("--cache_dir", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/"
                                 "fused_cache/rich_test"))
    ap.add_argument("--ghost_output_root", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"))
    ap.add_argument("--rich_data_root", required=True, type=Path)
    ap.add_argument("--rich_gt_dir",    required=True, type=Path)
    ap.add_argument("--body_split",     default="test_body")
    ap.add_argument("--decomposition_json", type=Path,
                    default=Path("eval_explainability/error_decomposition_rich_10scenes.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk",  type=int, default=32)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", type=Path,
                    default=Path("eval_explainability/loss_weighting_rich.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    parents = smplx_parents(device)
    desc = descendant_counts(parents)

    # ── Verify the proposed grouping against the kintree ─────────────────────
    print(f"\n{'='*78}")
    print("JOINT GROUPS — verified against the SMPL-X kintree parents")
    print(f"{'='*78}")
    leg_root_ok = all(parents[j] in (0, *_GROUPS["legs"]) for j in _GROUPS["legs"])
    for g, idx in _GROUPS.items():
        names = [f"{i}:{_SMPLX_JOINT_NAMES[i]}" for i in idx]
        print(f"  {g:<6} n={len(idx):2d}  " + ", ".join(names[:8]) +
              (f", ... (+{len(names)-8} more)" if len(names) > 8 else ""))
    covered = sorted(j for idx in _GROUPS.values() for j in idx)
    missing = sorted(set(range(_N_JOINTS)) - set(covered) - {0})
    dup = len(covered) != len(set(covered))
    print(f"  coverage: {len(covered)}/54 non-root joints, duplicates={dup}, "
          f"unassigned={missing}")
    print(f"  legs form a connected chain from pelvis: {leg_root_ok}")
    print(f"  hands are descendants of the wrists: "
          f"{all(20 in _ancestors(j, parents) or 21 in _ancestors(j, parents) for j in _GROUPS['hands'])}")
    print(f"  descendants(left_wrist)={desc[20]}  descendants(right_wrist)={desc[21]}  "
          f"descendants(left_hip)={desc[1]}  descendants(right_hip)={desc[2]}")

    # ── Part 1: per-group root-relative FK error ─────────────────────────────
    from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    files = sorted(args.cache_dir.glob("*.npz"))
    if wanted:
        files = [f for f in files if f.stem in wanted]
    if not files:
        logger.error(f"no cached scenes in {args.cache_dir}")
        return

    per_joint_err = {v: np.zeros(_N_JOINTS) for v in _VARIANTS}   # sum of mm
    per_joint_sq  = {v: np.zeros(_N_JOINTS) for v in _VARIANTS}   # sum of squared m
    all_err       = {v: [] for v in _VARIANTS}                    # (N,55) mm, for medians
    n_frames = 0
    scenes_used = []

    for f in files:
        d = np.load(f, allow_pickle=False)
        valid = d["valid"].astype(bool)
        T, P = valid.shape
        try:
            dp = RICHFusionDatapoint(
                scene_dir=args.ghost_output_root / f.stem,
                rich_data_root=args.rich_data_root, rich_gt_dir=args.rich_gt_dir,
                body_split=args.body_split, restrict_to_gt_persons=True)
            loader = torch.utils.data.DataLoader(RICHFusionDataset([dp]), batch_size=1)
            _, targets = next(iter(loader))
            betas = targets["shape"].to(device).float()               # (1,T,P,10)
        except Exception as e:                                        # noqa: BLE001
            logger.warning(f"{f.stem}: skipped ({type(e).__name__}: {e})")
            continue
        if betas.shape[1] != T or betas.shape[2] != P:
            logger.warning(f"{f.stem}: cache/dataset shape mismatch "
                           f"{(T,P)} vs {tuple(betas.shape[1:3])} — skipped")
            continue

        gt = fk_rootrel(d["pose_gt"], betas, device, args.chunk)
        vt, vp = np.where(valid)
        n_frames += len(vt)
        for v in _VARIANTS:
            pr = fk_rootrel(d[f"pose_{v}"], betas, device, args.chunk)
            e = np.linalg.norm(pr[vt, vp] - gt[vt, vp], axis=-1)      # (N,55) metres
            per_joint_err[v] += e.sum(axis=0) * 1000.0
            per_joint_sq[v]  += (e ** 2).sum(axis=0)
            all_err[v].append((e * 1000.0).astype(np.float32))
        scenes_used.append(f.stem)
        logger.info(f"{f.stem:<40} frames={len(vt):5d}")

    if not scenes_used:
        logger.error("no scenes processed")
        return

    counts = {v: sum(a.shape[0] for a in all_err[v]) for v in _VARIANTS}
    assert len(set(counts.values())) == 1, f"A/B frame count mismatch: {counts}"
    N = counts["chordal"]
    err = {v: np.concatenate(all_err[v], axis=0) for v in _VARIANTS}   # (N,55) mm

    groups_out = {}
    print(f"\n{'='*78}")
    print("PART 1 — root-relative FK position error by joint group (mm, GT betas)")
    print(f"  scenes={len(scenes_used)}  person-frames={N:,} (identical for A and B)")
    print(f"{'='*78}")
    print(f"{'group':<8}{'n':>4}{'A mean':>10}{'A med':>9}{'B mean':>10}{'B med':>9}"
          f"{'B-A':>9}{'winner':>8}")
    print("-" * 78)
    for g, idx in _GROUPS.items():
        a = err["chordal"][:, idx]
        b = err["ghost"][:, idx]
        am, bm = float(a.mean()), float(b.mean())
        groups_out[g] = {
            "n_joints": len(idx), "indices": idx,
            "A_mean_mm": am, "A_median_mm": float(np.median(a)),
            "B_mean_mm": bm, "B_median_mm": float(np.median(b)),
            "B_minus_A_mm": bm - am,
        }
        print(f"{g:<8}{len(idx):>4}{am:>10.2f}{float(np.median(a)):>9.2f}"
              f"{bm:>10.2f}{float(np.median(b)):>9.2f}{bm-am:>+9.2f}"
              f"{('B' if bm < am else 'A'):>8}")
    a_all, b_all = err["chordal"][:, 1:], err["ghost"][:, 1:]
    print("-" * 78)
    print(f"{'ALL 54':<8}{54:>4}{a_all.mean():>10.2f}{np.median(a_all):>9.2f}"
          f"{b_all.mean():>10.2f}{np.median(b_all):>9.2f}"
          f"{b_all.mean()-a_all.mean():>+9.2f}")

    # ── Part 2: descendants vs B/A rotation bias ─────────────────────────────
    spear = None
    table = []
    if args.decomposition_json.exists():
        with open(args.decomposition_json) as fh:
            dec = json.load(fh)
        bias_a = dec["per_joint"]["chordal"]["bias"]
        bias_b = dec["per_joint"]["ghost"]["bias"]
        names  = dec["per_joint"]["chordal"]["joints"]
        xs, ys = [], []
        for k, nm in enumerate(names):                 # body joints 1..21
            j = k + 1
            ratio = bias_b[k] / bias_a[k] if bias_a[k] else float("nan")
            table.append({"joint": nm, "index": j, "descendants": int(desc[j]),
                          "bias_A_deg": bias_a[k], "bias_B_deg": bias_b[k],
                          "B_over_A": ratio})
            if np.isfinite(ratio):
                xs.append(desc[j]); ys.append(ratio)
        spear = spearman(np.array(xs), np.array(ys))
        print(f"\n{'='*78}")
        print("PART 2 — descendants vs B/A rotation bias (21 body joints)")
        print(f"{'='*78}")
        print(f"    {'joint':<16}{'idx':>4}{'desc':>6}{'bias A':>9}{'bias B':>9}{'B/A':>8}")
        for row in sorted(table, key=lambda r: -r["descendants"]):
            print(f"    {row['joint']:<16}{row['index']:>4}{row['descendants']:>6}"
                  f"{row['bias_A_deg']:>9.2f}{row['bias_B_deg']:>9.2f}{row['B_over_A']:>8.2f}")
        print(f"\n  Spearman(descendants, B/A bias) = {spear:+.3f}   "
              f"(hypothesis predicts NEGATIVE)")
    else:
        logger.warning(f"{args.decomposition_json} not found — skipping part 2")

    # ── Part 3: share of L_joint magnitude from the hands ────────────────────
    print(f"\n{'='*78}")
    print("PART 3 — share of L_joint magnitude by group (squared error, the loss's own reduction)")
    print(f"{'='*78}")
    share = {}
    print(f"{'group':<8}{'n':>4}{'A share':>10}{'B share':>10}")
    for g, idx in _GROUPS.items():
        sa = per_joint_sq["chordal"][idx].sum() / per_joint_sq["chordal"][1:].sum()
        sb = per_joint_sq["ghost"][idx].sum() / per_joint_sq["ghost"][1:].sum()
        share[g] = {"A": float(sa), "B": float(sb)}
        print(f"{g:<8}{len(idx):>4}{sa:>9.1%}{sb:>10.1%}")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "config": {"cache_dir": str(args.cache_dir), "n_scenes": len(scenes_used),
                       "scenes": scenes_used, "n_person_frames": int(N),
                       "betas": "GT", "global_orient": "identity (equivalent to GT: "
                                                       "root-relative error is rotation-invariant)"},
            "groups": groups_out,
            "all54": {"A_mean_mm": float(a_all.mean()), "B_mean_mm": float(b_all.mean()),
                      "B_minus_A_mm": float(b_all.mean() - a_all.mean())},
            "descendants": {_SMPLX_JOINT_NAMES[j]: int(desc[j]) for j in range(_N_JOINTS)},
            "descendant_vs_bias": table,
            "spearman_descendants_vs_bias_ratio": spear,
            "loss_share_by_group": share,
            "per_joint_mean_mm": {v: (per_joint_err[v] / N).tolist() for v in _VARIANTS},
        }, fh, indent=2)
    logger.info(f"wrote {args.out}")


def _ancestors(j: int, parents: np.ndarray) -> set[int]:
    out, cur = set(), j
    while 0 <= cur < len(parents) and parents[cur] >= 0:
        cur = int(parents[cur])
        out.add(cur)
    return out


if __name__ == "__main__":
    main()
