"""
Standalone ablation study of the temporal synchronization algorithm.

Reimplements the pairwise-offset estimation of
synchronize_videos/synchronizer.py (Synchronizer._compute_cost_matrix +
estimate_couple_offset) and its downstream combination
(estimate_offset_matrix, cycle_consistency_weights, estimate_initial_times)
as ONE shared pipeline with four injection points:

  1. per-joint distance (what is compared between two frames)
       - SO(3) geodesic distance on axis-angle rotations
           (full, no_confidence, mean_diagonal, no_cycle, ref_cam, body_only)
       - squared Euclidean distance on root-relative FK joint positions
           (root_relative)
  2. diagonal aggregation (how one diagonal collapses to a per-person score)
       - median over finite entries (all variants except mean_diagonal)
       - mean   over finite entries (mean_diagonal)
  3. joint set (which joints carry weight in the cost)
       - 21 body + 30 hand joints (all variants except body_only)
       - 21 body joints only      (body_only)
  4. global solve (how K(K-1)/2 pairwise offsets become K start times)
       - weighted LSE, cycle-consistency weights (all variants except below)
       - weighted LSE, uniform weights           (no_cycle)
       - reference camera, row 0 read directly   (ref_cam)

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

body_only is a second *input transform* on the same row pass: raw confidences
with joints 21-50 (both hands) zeroed. Because the cost is
Sum(w*d) / (Sum(w) + 1e-8), zeroing a joint's weight is algebraically exact
joint-subset selection, and the den > 1e-4 guard still marks frame pairs with
no usable body joint as uninformative.

no_cycle and ref_cam share the offset matrix of `full` bit-for-bit and differ
only in the global solve, forming a three-rung ladder that separates the two
effects:
  ref_cam  uses K-1 measurements (row 0 only), exactly determined, no weights
  no_cycle uses all K(K-1)/2 measurements, uniform weights
  full     uses all K(K-1)/2 measurements, cycle-consistency weights
ref_cam -> no_cycle isolates the gain of the global WLS redundancy;
no_cycle -> full isolates the gain of the cycle-consistency weighting. The
ref_cam reference is camera index 0 (loader order); every trial additionally
records the MAE obtained for each of the K possible references under
"ref_spread", so anchor sensitivity is measurable rather than assumed.

mean_diagonal averages only the FINITE entries of a diagonal, under the same
n_finite >= min_overlap gate as full; a diagonal with too few finite entries
is dropped from that person's candidate set exactly as in full, so no inf
ever propagates into a mean.

Everything else is written once and shared by all variants: the cost
normalization (Sum(w*d) / (Sum(w) + 1e-8))**2 with the den > 1e-4 inf guard,
the diagonal scan over k in [-2*max_shift, 2*max_shift], both min_overlap
gates, the union-of-ks cross-person mean, the argmin, the K x K antisymmetric
offset matrix, the cycle-consistency weights and the weighted least-squares
solve. Variants that agree on (distance, confidence transform, diagonal
reducer) compute their pairwise offsets ONCE and share the result, so the
three solve variants cost one matrix, not three.

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

# 21 body joints, then 15 + 15 hand joints, in the (T, 51) joint axis.
N_BODY_JOINTS = 21


@dataclass(frozen=True)
class VariantSpec:
    distance: str        # "geodesic" | "sq_euclid"           — injection point 1
    conf: str            # "raw" | "binarized" | "body_only"  — input transform (2, 3)
    reduce: Callable     # injection point 2
    solve: str = "wls_cycle"  # "wls_cycle" | "wls_plain" | "ref_cam" — point 4
    uses_positions: bool = False

    @property
    def matrix_sig(self) -> tuple:
        """Everything that determines the pairwise offset matrix.

        Variants with equal signatures produce bit-identical matrices and are
        computed once; the solve variants differ only downstream of it.
        """
        return (self.distance, self.conf, self.reduce)


VARIANT_SPECS: dict[str, VariantSpec] = {
    "full":          VariantSpec("geodesic",  "raw",       _median_finite),
    "no_confidence": VariantSpec("geodesic",  "binarized", _median_finite),
    "root_relative": VariantSpec("sq_euclid", "raw",       _median_finite, uses_positions=True),
    "mean_diagonal": VariantSpec("geodesic",  "raw",       _mean_finite),
    # Same offset matrix as `full`; only the global solve changes.
    "no_cycle":      VariantSpec("geodesic",  "raw",       _median_finite, solve="wls_plain"),
    "ref_cam":       VariantSpec("geodesic",  "raw",       _median_finite, solve="ref_cam"),
    # Body joints only (hands dropped from the cost).
    "body_only":     VariantSpec("geodesic",  "body_only", _median_finite),
}
VARIANTS = tuple(VARIANT_SPECS)


def _weight_pair(conf1: torch.Tensor, conf2: torch.Tensor, kind: str):
    if kind == "raw":
        return conf1, conf2
    if kind == "binarized":
        # Structural zeros (padding / gap fill / missing-person fill) must stay
        # zero: they mark "no observation", not low model confidence.
        return (conf1 > 0).float(), (conf2 > 0).float()
    if kind == "body_only":
        # Joint layout is 21 body + 15 left hand + 15 right hand (SMPL-X
        # indices 1-51). Zeroing the hand weights removes those joints from
        # both the numerator and the denominator of the cost, i.e. it is an
        # exact joint-subset selection rather than a re-weighting.
        mask = torch.zeros(conf1.shape[-1], device=conf1.device, dtype=conf1.dtype)
        mask[:N_BODY_JOINTS] = 1.0
        return conf1 * mask, conf2 * mask
    raise ValueError(f"unknown confidence transform: {kind}")


# ──────────────────────────────────────────────────────────────────────────
# Shared pipeline
# ──────────────────────────────────────────────────────────────────────────

class AblationSynchronizer:
    """The Synchronizer pipeline with the two injection points above.

    Every variant shares every line of this class; a variant only selects a
    distance object, a confidence transform and a diagonal reducer (the
    global solve happens downstream, in solve_times).
    Work is shared at two levels. Cost matrices: full, mean_diagonal,
    no_cycle and ref_cam read the SAME matrix, while no_confidence and
    body_only reuse the per-joint geodesic distances of the same row pass
    rather than recomputing them. Diagonal scans: variants with an equal
    VariantSpec.matrix_sig scan the diagonals once and share the offsets, so
    full, no_cycle and ref_cam cost exactly one pairwise estimation.
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

        # Variants with an equal matrix signature are estimated once.
        sig_of = {v: VARIANT_SPECS[v].matrix_sig for v in variants}
        uniq_sigs = list(dict.fromkeys(sig_of.values()))  # deterministic order
        per_person: dict[tuple, list[dict[int, float]]] = {sg: [] for sg in uniq_sigs}
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

            for sg in uniq_sigs:
                distance, conf_kind, reduce_fn = sg
                scores = self._diagonal_scores(costs[(distance, conf_kind)], reduce_fn)
                if scores:
                    per_person[sg].append(scores)

        sig_offsets: dict[tuple, float] = {}
        for sg in uniq_sigs:
            off = self._combine_offset(per_person[sg])
            if off is None:
                names = [v for v in variants if sig_of[v] == sg]
                logger.warning(f"    [{'/'.join(names)}] no person had valid overlap "
                               f"— returning offset=0")
                off = 0.0
            sig_offsets[sg] = off
        return {v: sig_offsets[sig_of[v]] for v in variants}

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


def reference_camera_times(offset_matrix: torch.Tensor, ref: int = 0) -> torch.Tensor:
    """Start times from ONE reference camera — no global solve.

    Keeps only the K-1 measurements that touch `ref` and reads them off
    directly: t_i = O[ref, i], t_ref = 0. That system is exactly determined,
    so no averaging happens and edge weights are meaningless — the ablation
    baseline that isolates what the over-determined WLS actually buys.
    The gauge fix mirrors estimate_initial_times.
    """
    times = offset_matrix[ref].clone()
    if times.min() < 0:
        times = times - times.min()
    return times


def solve_times(offset_matrix: torch.Tensor, mode: str) -> torch.Tensor:
    """Global solve, injection point 4."""
    if mode == "wls_cycle":
        return estimate_initial_times(offset_matrix,
                                      cycle_consistency_weights(offset_matrix))
    if mode == "wls_plain":
        return estimate_initial_times(offset_matrix, None)
    if mode == "ref_cam":
        return reference_camera_times(offset_matrix, ref=0)
    raise ValueError(f"unknown solve mode: {mode}")


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
        spec = VARIANT_SPECS[v]
        off = mats[v]
        est = solve_times(off, spec.solve).cpu()
        if torch.isnan(est).any():
            out[v] = None
            continue
        est = est - est.min()
        errors = (est - true_t).abs()
        entry = {
            "true_times": true_t.tolist(),
            "estimated":  est.tolist(),
            "errors":     errors.tolist(),
            "mae":        errors.mean().item(),
        }
        if spec.solve == "ref_cam":
            # Camera 0 is loader order, i.e. arbitrary. Record the MAE of every
            # possible anchor so the reported row can be checked for anchor
            # sensitivity instead of being assumed representative.
            spread = []
            for r in range(off.shape[0]):
                t_r = reference_camera_times(off, ref=r).cpu()
                t_r = t_r - t_r.min()
                spread.append((t_r - true_t).abs().mean().item())
            entry["ref_spread"] = spread
        out[v] = entry
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
