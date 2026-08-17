"""
Standalone ablation study of the temporal synchronization algorithm.

Reimplements the pairwise-offset estimation of
synchronize_videos/synchronizer.py (Synchronizer._compute_cost_matrix +
estimate_couple_offset) and its downstream combination
(estimate_offset_matrix, cycle_consistency_weights, estimate_initial_times)
as ONE shared pipeline with two injection points:

  1. per-joint distance (what is compared between two frames)
       - SO(3) geodesic distance on axis-angle rotations
           (full, no_confidence, mean_diagonal)
       - squared Euclidean distance on root-relative FK joint positions
           (root_relative)
  2. diagonal aggregation (how one diagonal collapses to a per-person score)
       - median over finite entries (full, no_confidence, root_relative)
       - mean   over finite entries (mean_diagonal)

no_confidence is an *input transform*, not a third code path: confidences are
binarized (1 where conf > 0, else 0), so structural zero-confidence entries —
detection-gap fill, absolute-frame anchoring pads, tail pads, and the
EgoHumans missing-person zero fill — stay uninformative, the den > 1e-4
degenerate-denominator guard keeps functioning, and the per-pair person skip
is preserved. Only the *weighting* of observed joints is ablated.

root_relative positions come from utilities.body_data.load_person_smplx_joints
(via the experiment modules' load_scene_joints): SMPL-X forward kinematics with
NEUTRAL betas (identical skeleton in every camera → no scale normalization
needed) and ZERO global orient (camera-agnostic pose space), root-relative,
joints 1-51 — the exact joint set and confidence slice of the rotation path.

mean_diagonal averages only the FINITE entries of a diagonal, under the same
n_finite >= min_overlap gate as full; a diagonal with too few finite entries
is dropped from that person's candidate set exactly as in full, so no inf
ever propagates into a mean.

Everything else is written once and shared by all four variants: the cost
normalization (Sum(w*d) / (Sum(w) + 1e-8))**2 with the den > 1e-4 inf guard,
the diagonal scan over k in [-2*max_shift, 2*max_shift], both min_overlap
gates, the union-of-ks cross-person mean, the argmin, the K x K antisymmetric
offset matrix, the cycle-consistency weights and the weighted least-squares
solve.

Nothing in the repo is modified: imports from the two alignment experiment
modules are read-only reuse of data loading, scene lists and metrics; the
original Synchronizer is imported only to validate the `full` variant
against it.

Validation (run this FIRST — the ablation is meaningless if it fails):
    pixi run python -m evaluation.sync_ablation --validate
compares the `full` variant against synchronize_videos.synchronizer on a few
scenes with identical trial draws; recovered pairwise offset matrices and
solved start times must match exactly. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────
# Injection point 1: per-joint distance
# ──────────────────────────────────────────────────────────────────────────

def _axis_angle_to_rot_mat(theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: (..., 3) axis-angle → (..., 3, 3) rotation matrices.

    Copied verbatim from Synchronizer._axis_angle_to_rot_mat so the `full`
    variant is bit-identical to the original.
    """
    angle = theta.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    axis  = theta / angle                                       # (..., 3)
    s, c  = torch.sin(angle), torch.cos(angle)                 # (..., 1)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    # skew-symmetric K
    K = torch.stack([
        zeros, -z,  y,
        z,  zeros, -x,
       -y,  x,  zeros,
    ], dim=-1).reshape(*theta.shape[:-1], 3, 3)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype).expand(*theta.shape[:-1], 3, 3)
    return I + s.unsqueeze(-1) * K + (1 - c).unsqueeze(-1) * (K @ K)


class GeodesicDistance:
    """SO(3) geodesic distance on axis-angle rotations.

    Row computation mirrors Synchronizer._compute_cost_matrix exactly:
    R2 is pre-converted once per sequence pair, each row converts one frame
    of sequence 1 and takes arccos((tr(R1ᵀR2) - 1)/2) with the same clamp.
    """
    key = "geodesic"

    @staticmethod
    def precompute(seq2: torch.Tensor) -> torch.Tensor:
        # (T2, J, 3) axis-angle → (T2, J, 3, 3)
        return _axis_angle_to_rot_mat(seq2)

    @staticmethod
    def row(frame1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        R1 = _axis_angle_to_rot_mat(frame1)                          # J x 3 x 3
        R_rel = R1.unsqueeze(0).transpose(-1, -2) @ R2               # T2 x J x 3 x 3
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)         # T2 x J
        return torch.arccos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))  # T2 x J (radians)


class SquaredEuclideanDistance:
    """Squared Euclidean distance on root-relative joint positions (metres²)."""
    key = "sq_euclid"

    @staticmethod
    def precompute(seq2: torch.Tensor) -> torch.Tensor:
        # (T2, J, 3) positions need no conversion
        return seq2

    @staticmethod
    def row(frame1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        return ((frame1.unsqueeze(0) - p2) ** 2).sum(dim=-1)         # T2 x J


DISTANCES = {d.key: d for d in (GeodesicDistance, SquaredEuclideanDistance)}


# ──────────────────────────────────────────────────────────────────────────
# Injection point 2: diagonal aggregation (over the FINITE entries only)
# ──────────────────────────────────────────────────────────────────────────

def _median_finite(vals: torch.Tensor) -> torch.Tensor:
    return vals.median()


def _mean_finite(vals: torch.Tensor) -> torch.Tensor:
    return vals.mean()


# ──────────────────────────────────────────────────────────────────────────
# Variant registry
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VariantSpec:
    distance: str        # "geodesic" | "sq_euclid" — injection point 1
    conf: str            # "raw" | "binarized"      — input transform (no_confidence)
    reduce: Callable     # injection point 2
    uses_positions: bool = False


VARIANT_SPECS: dict[str, VariantSpec] = {
    "full":          VariantSpec("geodesic",  "raw",       _median_finite),
    "no_confidence": VariantSpec("geodesic",  "binarized", _median_finite),
    "root_relative": VariantSpec("sq_euclid", "raw",       _median_finite, uses_positions=True),
    "mean_diagonal": VariantSpec("geodesic",  "raw",       _mean_finite),
}
VARIANTS = tuple(VARIANT_SPECS)


def _weight_pair(conf1: torch.Tensor, conf2: torch.Tensor, kind: str):
    if kind == "raw":
        return conf1, conf2
    if kind == "binarized":
        # Structural zeros (padding / gap fill / missing-person fill) must stay
        # zero: they mark "no observation", not low model confidence.
        return (conf1 > 0).float(), (conf2 > 0).float()
    raise ValueError(f"unknown confidence transform: {kind}")


# ──────────────────────────────────────────────────────────────────────────
# Shared pipeline
# ──────────────────────────────────────────────────────────────────────────

class AblationSynchronizer:
    """The Synchronizer pipeline with the two injection points above.

    All four variants share every line of this class; a variant only selects
    a distance object, a confidence transform and a diagonal reducer.
    Cost matrices are shared where the variants coincide: full and
    mean_diagonal read the SAME matrix (they differ only at the reducer), and
    no_confidence reuses the per-joint geodesic distances of the same row
    pass rather than recomputing them.
    """

    def __init__(
        self,
        device: str = DEVICE,
        q: int = 2,
        min_overlap: int = 100,
        max_shift: int | None = None,
    ):
        self.device = device
        self.q = q
        self.min_overlap = min_overlap
        self.max_shift = max_shift

    # ── cost matrices ────────────────────────────────────────────────────
    def _cost_matrices(
        self,
        seq1: torch.Tensor,           # T1 x J x 3  (rotations or positions)
        seq2: torch.Tensor,           # T2 x J x 3
        conf1: torch.Tensor,          # T1 x J
        conf2: torch.Tensor,          # T2 x J
        distance,                     # GeodesicDistance | SquaredEuclideanDistance
        conf_kinds: list[str],
    ) -> dict[str, torch.Tensor]:
        """One pass over the rows of the T1×T2 cost matrix, producing one
        matrix per requested confidence transform from the same per-joint
        distances.

        Per frame pair, per weighting:  C = (Σ w·d / (Σ w + 1e-8))**q,
        marked uninformative (+inf) when the denominator is ≤ 1e-4 — the
        exact structure of Synchronizer._compute_cost_matrix.
        """
        n  = seq1.shape[0]
        T2 = seq2.shape[0]
        weights = {kind: _weight_pair(conf1, conf2, kind) for kind in conf_kinds}
        costs = {kind: torch.zeros(n, T2, device=self.device) for kind in conf_kinds}

        ctx = distance.precompute(seq2)
        for i in range(n):
            diff = distance.row(seq1[i], ctx)                        # T2 x J
            for kind, (w1, w2) in weights.items():
                w = w1[i].unsqueeze(0) * w2                          # T2 x J
                num = (w * diff).sum(dim=-1)                          # T2
                den = w.sum(dim=-1) + 1e-8                            # T2
                valid = den > 1e-4
                costs[kind][i] = torch.where(
                    valid, (num / den) ** self.q, torch.full_like(num, float("inf"))
                )
        return costs

    # ── diagonal scan ────────────────────────────────────────────────────
    def _diagonal_scores(self, cost: torch.Tensor, reduce_fn: Callable) -> dict[int, float]:
        """Per-person diagonal scores, mirroring estimate_couple_offset.

        Both gates are kept: index overlap ≥ min_overlap AND ≥ min_overlap
        FINITE entries on the diagonal. The reducer sees only the finite
        entries, so a mean can never absorb an inf; a diagonal with too few
        finite entries is dropped from the candidate set, as in the original.
        """
        n, m = cost.shape
        k_lo = -(n - 1) if self.max_shift is None else max(-(n - 1), -2 * self.max_shift)
        k_hi = m        if self.max_shift is None else min(m,          2 * self.max_shift + 1)
        p_scores: dict[int, float] = {}
        for k in range(k_lo, k_hi):
            i0 = max(0, -k)
            i1 = min(n, m - k)
            if i1 - i0 < self.min_overlap:
                continue
            i_idx = torch.arange(i0, i1, device=cost.device)
            diag = cost[i_idx, i_idx + k]
            finite = torch.isfinite(diag)
            n_finite = int(finite.sum().item())
            if n_finite < self.min_overlap:
                continue
            p_scores[k] = reduce_fn(diag[finite]).item()
        return p_scores

    # ── cross-person combination ─────────────────────────────────────────
    @staticmethod
    def _combine_offset(per_person_costs: list[dict[int, float]]) -> float | None:
        """Union-of-ks mean across the persons that scored each k, then argmin.

        Copied structurally from estimate_couple_offset so tie-breaking of the
        argmin (dict insertion order from set iteration) is identical.
        """
        if not per_person_costs:
            return None
        common_ks = set().union(*[set(p.keys()) for p in per_person_costs])
        combined: dict[int, float] = {}
        for k in common_ks:
            vals = [p_scores[k] for p_scores in per_person_costs if k in p_scores]
            combined[k] = sum(vals) / len(vals)
        best_k = min(combined, key=combined.__getitem__)
        return float(best_k)

    # ── pairwise offsets, all variants at once ───────────────────────────
    def estimate_couple_offsets(
        self,
        rot1: list[torch.Tensor], rot2: list[torch.Tensor],     # P x (T x 51 x 3) rotations
        pos1: list[torch.Tensor] | None, pos2: list[torch.Tensor] | None,  # positions or None
        conf1: list[torch.Tensor], conf2: list[torch.Tensor],   # P x (T x 51)
        variants: tuple[str, ...],
    ) -> dict[str, float]:
        """Estimated offset (frames) of sequence 2 vs sequence 1, per variant."""
        P = len(rot1)
        assert P == len(rot2) == len(conf1) == len(conf2), \
            "Number of people must match across both videos and their confidences"
        assert P > 0, "Need at least one person"

        geo_kinds = sorted({VARIANT_SPECS[v].conf for v in variants
                            if VARIANT_SPECS[v].distance == "geodesic"})
        need_pos = any(VARIANT_SPECS[v].uses_positions for v in variants)
        if need_pos and (pos1 is None or pos2 is None):
            raise ValueError("root_relative requested but no position sequences given")

        per_person: dict[str, list[dict[int, float]]] = {v: [] for v in variants}
        for p in range(P):
            costs: dict[tuple[str, str], torch.Tensor] = {}
            if geo_kinds:
                geo = self._cost_matrices(rot1[p], rot2[p], conf1[p], conf2[p],
                                          GeodesicDistance, geo_kinds)
                for kind, c in geo.items():
                    costs[("geodesic", kind)] = c
            if need_pos:
                pos = self._cost_matrices(pos1[p], pos2[p], conf1[p], conf2[p],
                                          SquaredEuclideanDistance, ["raw"])
                costs[("sq_euclid", "raw")] = pos["raw"]

            for v in variants:
                spec = VARIANT_SPECS[v]
                scores = self._diagonal_scores(costs[(spec.distance, spec.conf)], spec.reduce)
                if scores:
                    per_person[v].append(scores)

        offsets: dict[str, float] = {}
        for v in variants:
            off = self._combine_offset(per_person[v])
            if off is None:
                logger.warning(f"    [{v}] no person had valid overlap — returning offset=0")
                off = 0.0
            offsets[v] = off
        return offsets

    def estimate_offset_matrices(
        self,
        rot_list: list[list[torch.Tensor]],
        pos_list: list[list[torch.Tensor]] | None,
        conf_list: list[list[torch.Tensor]],
        variants: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """K×K antisymmetric offset matrix per variant (all pairs share loads)."""
        K = len(rot_list)
        assert K == len(conf_list)
        mats = {v: torch.zeros((K, K), device=self.device) for v in variants}
        for i in range(K):
            for j in range(i + 1, K):
                offs = self.estimate_couple_offsets(
                    rot_list[i], rot_list[j],
                    None if pos_list is None else pos_list[i],
                    None if pos_list is None else pos_list[j],
                    conf_list[i], conf_list[j],
                    variants,
                )
                for v, off in offs.items():
                    mats[v][i, j] = off
                    mats[v][j, i] = -off
        return mats


# ──────────────────────────────────────────────────────────────────────────
# Downstream (shared, copied verbatim from Synchronizer)
# ──────────────────────────────────────────────────────────────────────────

def cycle_consistency_weights(offset_matrix: torch.Tensor) -> torch.Tensor:
    """Per-edge weights from cycle consistency — verbatim Synchronizer copy."""
    K = offset_matrix.shape[0]
    O = offset_matrix  # K×K
    residuals_3d = O.unsqueeze(2) + O.unsqueeze(0) - O.unsqueeze(1)
    idx = torch.arange(K, device=offset_matrix.device)
    mask = (idx.view(1, 1, K) != idx.view(K, 1, 1)) & \
           (idx.view(1, 1, K) != idx.view(1, K, 1))  # K×K×K
    mean_residual = (residuals_3d.abs() * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
    return 1.0 / (1.0 + mean_residual)  # K×K


def estimate_initial_times(
    offset_matrix: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted LSE solver — verbatim Synchronizer copy."""
    K = offset_matrix.shape[0]
    num_pairs = K * (K - 1) // 2
    A = torch.zeros(num_pairs, K - 1, device=offset_matrix.device)
    b = torch.zeros(num_pairs, device=offset_matrix.device)
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            w = weights[i, j].item() ** 0.5 if weights is not None else 1.0
            if j >= 1:
                A[count, j - 1] = w
            if i >= 1:
                A[count, i - 1] = -w
            b[count] = w * offset_matrix[i, j]
            count += 1

    sol = torch.linalg.lstsq(A, b).solution  # K-1

    initial_times = torch.zeros(K, device=offset_matrix.device)
    initial_times[1:] = sol  # t_0 fixed to 0, otherwise system is undetermined
    if initial_times.min() < 0:
        initial_times = initial_times - initial_times.min()
    return initial_times


# ──────────────────────────────────────────────────────────────────────────
# Scene loading / trial execution shared by the runner and the validation
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SceneBundle:
    key: str
    cam_ids: list[str]
    pids: list[int]
    rot: dict   # {cam: {pid: (rotations T×51×3, conf T×51)}}
    pos: dict | None  # same structure with FK positions, or None


def load_scene_bundle(mod, scene_dir: Path, key: str, need_pos: bool) -> SceneBundle:
    """Load one scene through the experiment module `mod` (read-only reuse).

    `mod` is evaluation.alignment_experiments_multi (RICH) or
    evaluation.alignment_experiments_multi_egohumans; both expose SKIP_CAMERAS,
    load_scene, load_scene_joints and select_persons with identical signatures.
    The EgoHumans module additionally has _fill_missing_persons (union pids).
    """
    exclude = mod.SKIP_CAMERAS.get(key, [])
    rot = mod.load_scene(scene_dir, exclude_cameras=exclude)
    if len(rot) < 2:
        raise RuntimeError(f"{key}: need ≥2 cameras with body data, found {len(rot)}")
    pids = mod.select_persons(key, rot)

    pos = None
    if need_pos:
        pos = mod.load_scene_joints(scene_dir, exclude_cameras=exclude)

    fill = getattr(mod, "_fill_missing_persons", None)
    if fill is not None:
        fill(rot, pids)
        if pos is not None:
            fill(pos, pids)

    if pos is not None:
        # Both loaders read the same npz keys and fill/anchor/pad identically —
        # verify rather than assume, since apply_shifts slices by array index.
        assert set(pos) == set(rot), f"{key}: camera sets differ between loaders"
        for cam in rot:
            for pid in pids:
                r_seq, r_conf = rot[cam][pid]
                p_seq, p_conf = pos[cam][pid]
                assert p_seq.shape[0] == r_seq.shape[0], \
                    f"{key}/{cam}/person_{pid}: T mismatch rot={r_seq.shape[0]} pos={p_seq.shape[0]}"
                assert torch.equal(p_conf, r_conf), \
                    f"{key}/{cam}/person_{pid}: confidence mismatch between loaders"

    return SceneBundle(key=key, cam_ids=list(rot.keys()), pids=pids, rot=rot, pos=pos)


def run_trial_all_variants(
    engine: AblationSynchronizer,
    mod,
    bundle: SceneBundle,
    true_shifts: dict[str, int],
    end_cuts: dict[str, int],
    variants: tuple[str, ...],
) -> dict[str, dict | None] | None:
    """One trial, all variants on the identical slices.

    Returns None on the protocol rejection (apply_shifts min_overlap), which
    is variant-independent; otherwise {variant: result | None}, where None
    marks a variant whose solve produced NaN (as in run_trial).
    """
    sliced_rot = mod.apply_shifts(bundle.rot, true_shifts, end_cuts, bundle.pids)
    if sliced_rot is None:
        return None
    joints_list, confs_list = sliced_rot

    pos_list = None
    if any(VARIANT_SPECS[v].uses_positions for v in variants):
        sliced_pos = mod.apply_shifts(bundle.pos, true_shifts, end_cuts, bundle.pids)
        # Same T per (cam, pid) as the rotation data (asserted at load time),
        # so the min_overlap outcome is identical.
        assert sliced_pos is not None
        pos_list = sliced_pos[0]

    mats = engine.estimate_offset_matrices(joints_list, pos_list, confs_list, variants)

    cam_ids = list(true_shifts.keys())
    true_t = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
    true_t = true_t - true_t.min()

    out: dict[str, dict | None] = {}
    for v in variants:
        off = mats[v]
        weights = cycle_consistency_weights(off)
        est = estimate_initial_times(off, weights).cpu()
        if torch.isnan(est).any():
            out[v] = None
            continue
        est = est - est.min()
        errors = (est - true_t).abs()
        out[v] = {
            "true_times": true_t.tolist(),
            "estimated":  est.tolist(),
            "errors":     errors.tolist(),
            "mae":        errors.mean().item(),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# Validation: `full` vs the original Synchronizer, exact match required
# ──────────────────────────────────────────────────────────────────────────

def validate(max_shift: int, trials: int, n_rich: int, n_ego: int) -> bool:
    """Compare the `full` variant against synchronize_videos.synchronizer on
    identical trial draws. Returns True iff every pairwise offset matrix and
    every solved start-time vector match exactly."""
    import evaluation.alignment_experiments_multi as rich_exp
    import evaluation.alignment_experiments_multi_egohumans as ego_exp
    from synchronize_videos.synchronizer import Synchronizer

    engine = AblationSynchronizer(device=DEVICE, min_overlap=100, max_shift=max_shift)
    orig = Synchronizer(use_acceleration_weights=False, device=DEVICE,
                        min_overlap=100, max_shift=max_shift, verbose=False)

    jobs = []
    for name in sorted(rich_exp.SCENE_PIDS)[:n_rich]:
        jobs.append((rich_exp, rich_exp.SCENES_ROOT / name, name))
    for name in sorted(ego_exp.SCENE_PIDS)[:n_ego]:
        jobs.append((ego_exp, ego_exp.SCENES_ROOT / name, name))

    rng = np.random.default_rng(42)
    all_match = True
    for mod, scene_dir, key in jobs:
        logger.info(f"\nValidating on {key}")
        bundle = load_scene_bundle(mod, scene_dir, key, need_pos=False)
        cam_ids = bundle.cam_ids
        for trial in range(trials):
            raw_shifts = [0] + rng.integers(-max_shift, max_shift + 1, size=len(cam_ids) - 1).tolist()
            true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
            end_cuts    = {c: int(e) for c, e in zip(cam_ids, rng.integers(0, max_shift + 1, size=len(cam_ids)).tolist())}

            sliced = mod.apply_shifts(bundle.rot, true_shifts, end_cuts, bundle.pids)
            if sliced is None:
                logger.info(f"  trial {trial + 1}: protocol-rejected on both sides — skipped")
                continue
            joints_list, confs_list = sliced

            off_orig = orig.estimate_offset_matrix(joints_list, confs_list)
            w_orig   = orig.cycle_consistency_weights(off_orig)
            t_orig   = orig.estimate_initial_times(off_orig, w_orig)

            mats  = engine.estimate_offset_matrices(joints_list, None, confs_list, ("full",))
            off_abl = mats["full"]
            w_abl = cycle_consistency_weights(off_abl)
            t_abl = estimate_initial_times(off_abl, w_abl)

            m_off = torch.equal(off_orig, off_abl)
            m_t   = torch.equal(t_orig, t_abl)
            status = "MATCH" if (m_off and m_t) else "MISMATCH"
            logger.info(f"  trial {trial + 1}: offsets={'ok' if m_off else 'DIFFER'}  "
                        f"times={'ok' if m_t else 'DIFFER'}  → {status}")
            if not (m_off and m_t):
                all_match = False
                K = len(cam_ids)
                for i in range(K):
                    for j in range(i + 1, K):
                        a, b = off_orig[i, j].item(), off_abl[i, j].item()
                        if a != b:
                            logger.error(f"    ({cam_ids[i]}→{cam_ids[j]}): original={a:+.1f}  ablation={b:+.1f}")
                logger.error(f"    times original: {t_orig.cpu().tolist()}")
                logger.error(f"    times ablation: {t_abl.cpu().tolist()}")
    return all_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--validate", action="store_true",
                        help="Compare the `full` variant against the original Synchronizer")
    parser.add_argument("--max-shift", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rich-scenes", type=int, default=2,
                        help="Number of RICH scenes to validate on (sorted order)")
    parser.add_argument("--ego-scenes", type=int, default=1,
                        help="Number of EgoHumans scenes to validate on (sorted order)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not args.validate:
        parser.error("this module only runs --validate; use sync_ablation_runner.py for the study")

    ok = validate(args.max_shift, args.trials, args.rich_scenes, args.ego_scenes)
    if ok:
        logger.info("\nVALIDATION PASSED — `full` matches the original Synchronizer exactly")
        sys.exit(0)
    logger.error("\nVALIDATION FAILED — do not trust the ablation until this is resolved")
    sys.exit(1)
