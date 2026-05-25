import logging
import math

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Synchronizer:
    """
    Class performing the temporal alignment of videos.
    Uses weighted DTW to find the optimal alignment path between pairs of
    joint sequences, extracts the temporal offset from the path, and solves
    a least-squares system to recover globally consistent start times.
    """

    def __init__(
        self,
        method: str = "dtw",
        only_overlap: bool = True,
        device: str = "cuda",
        q: int = 2,
        min_overlap: int = 100,
        max_shift: int | None = None,
        verbose: bool = False,
        n_candidates: int = 3,
        n_bp_iterations: int = 10,
        bp_sigma: float = 1.0,
        use_acceleration_weights: bool = False,
        temperature: float = 1.0,
        n_dist_bp_iterations: int = 5,
    ):
        self.method = method
        self.only_overlap = only_overlap
        self.device = device
        self.q = q
        self.min_overlap = min_overlap
        self.max_shift = max_shift
        self.verbose = verbose
        self.n_candidates = n_candidates
        self.n_bp_iterations = n_bp_iterations
        self.bp_sigma = bp_sigma
        self.use_acceleration_weights = use_acceleration_weights
        self.temperature = temperature
        self.n_dist_bp_iterations = n_dist_bp_iterations

    @staticmethod
    def _axis_angle_to_rot_mat(theta: torch.Tensor) -> torch.Tensor:
        """Rodrigues formula: (..., 3) axis-angle → (..., 3, 3) rotation matrices."""
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

    @staticmethod
    def _rot_mat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
        """SO(3) Log map: (..., 3, 3) rotation matrices → (..., 3) axis-angle vectors."""
        trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)                       # (...)
        angle = torch.arccos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))   # (...)
        # skew-symmetric part: (R - Rᵀ)/2 = sin(angle) * [axis]_×
        skew = (R - R.transpose(-1, -2)) / 2                                   # (..., 3, 3)
        axis_unnorm = torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1
        )                                                                       # (..., 3)
        axis = axis_unnorm / torch.sin(angle).clamp(min=1e-8).unsqueeze(-1)
        theta = angle.unsqueeze(-1) * axis
        # near-zero angle → Log(I) = 0
        return torch.where(angle.unsqueeze(-1) < 1e-6, torch.zeros_like(theta), theta)

    @staticmethod
    def _compute_accelerations(theta: torch.Tensor) -> torch.Tensor:
        """Geodesic angular acceleration magnitudes from an axis-angle sequence.

        Computes angular velocity as ω[t] = Log(R[t-1]ᵀ R[t]) (Lie algebra element),
        then angular acceleration as α[t] = ω[t] - ω[t-1], and returns ||α[t]||
        normalised to [0, 1] by the 95th percentile.  Returns all-ones (uniform)
        when the scene is effectively static (p95 < 1e-3 rad/frame²).

        theta  : T x J x 3
        returns: T x J
        """
        T, J = theta.shape[:2]
        R = Synchronizer._axis_angle_to_rot_mat(theta)          # T x J x 3 x 3
        # angular velocity between consecutive frames
        R_vel = R[:-1].transpose(-1, -2) @ R[1:]               # (T-1) x J x 3 x 3
        omega = Synchronizer._rot_mat_to_axis_angle(R_vel)      # (T-1) x J x 3
        # angular acceleration: difference of consecutive Lie-algebra velocities
        alpha = omega[1:] - omega[:-1]                          # (T-2) x J x 3
        mag_inner = alpha.norm(dim=-1)                          # (T-2) x J
        # pad boundaries by replicating the nearest interior value
        mag = torch.zeros(T, J, device=theta.device, dtype=theta.dtype)
        mag[1:-1] = mag_inner
        mag[0]    = mag[1]
        mag[-1]   = mag[-2]
        p95 = torch.quantile(mag.flatten(), 0.95)
        if p95 < 1e-3:          # static scene: do not amplify noise
            return torch.ones_like(mag)
        return (mag / p95).clamp(0.0, 1.0)

    def _compute_cost_matrix_positions(
        self,
        positions_1: torch.Tensor,   # T1 x J x 3  (3D joint positions, any space)
        positions_2: torch.Tensor,   # T2 x J x 3
        confidences_1: torch.Tensor, # T1 x J
        confidences_2: torch.Tensor, # T2 x J
    ) -> torch.Tensor:
        """Pairwise weighted L2 cost matrix on root-relative 3D joint positions (T1 x T2).

        Root joint (index 0) is subtracted per frame before computing distances,
        making the signal camera-agnostic.
        """
        pos1 = positions_1 - positions_1[:, 0:1, :]  # T1 x J x 3
        pos2 = positions_2 - positions_2[:, 0:1, :]  # T2 x J x 3
        T1, T2 = pos1.shape[0], pos2.shape[0]
        cost = torch.zeros(T1, T2, device=self.device)
        for i in range(T1):
            diff = pos1[i].unsqueeze(0) - pos2          # T2 x J x 3
            l2   = diff.norm(dim=-1)                    # T2 x J
            w    = confidences_1[i].unsqueeze(0) * confidences_2  # T2 x J
            num  = (w * l2).sum(dim=-1)
            den  = w.sum(dim=-1) + 1e-8
            cost[i] = torch.where(den > 1e-4, num / den, torch.full_like(num, float('inf')))
        return cost

    def _compute_cost_matrix(
        self,
        body_joints_1: torch.Tensor,  # T1 x J x 3
        body_joints_2: torch.Tensor,  # T2 x J x 3
        confidences_1: torch.Tensor,  # T1 x J
        confidences_2: torch.Tensor,  # T2 x J
    ) -> torch.Tensor:
        """T1×T2 pairwise cost matrix.

        joints_position: input tensors are 3D positions → L2 on root-relative coords.
        All other methods: input tensors are axis-angle rotations → SO(3) geodesic distance.
        """
        if self.method == "joints_position":
            return self._compute_cost_matrix_positions(
                body_joints_1, body_joints_2, confidences_1, confidences_2
            )
        n  = body_joints_1.shape[0]
        T2 = body_joints_2.shape[0]
        cost = torch.zeros(n, T2, device=self.device)

        # Pre-convert all T2 frames to rotation matrices: T2 x J x 3 x 3
        R2 = self._axis_angle_to_rot_mat(body_joints_2)

        if self.use_acceleration_weights:
            a1 = self._compute_accelerations(body_joints_1)  # T1 x J
            a2 = self._compute_accelerations(body_joints_2)  # T2 x J

        for i in range(n):
            # R1: J x 3 x 3  →  broadcast to  T2 x J x 3 x 3
            R1 = self._axis_angle_to_rot_mat(body_joints_1[i])           # J x 3 x 3
            R_rel  = R1.unsqueeze(0).transpose(-1, -2) @ R2              # T2 x J x 3 x 3
            trace  = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)        # T2 x J
            diff   = torch.arccos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))  # T2 x J  (radians)
            w = confidences_1[i].unsqueeze(0) * confidences_2            # T2 x J
            if self.use_acceleration_weights:
                w = w * a1[i].unsqueeze(0) * a2                          # T2 x J
            num = (w * diff).sum(dim=-1)                                  # T2
            den = w.sum(dim=-1) + 1e-8                                    # T2
            valid = den > 1e-4
            cost[i] = torch.where(valid, (num / den) ** self.q, torch.full_like(num, float('inf')))

        return cost  # T1 x T2

    @staticmethod
    def _dtw_accumulate(cost: torch.Tensor) -> torch.Tensor:
        """
        CUDA-parallel DTW accumulation using anti-diagonal wavefront.

        Cells (i, j) on the same anti-diagonal (i + j = d) depend only on
        anti-diagonals d-1 and d-2, so they can be computed in parallel.
        This reduces the Python loop from O(n*m) to O(n+m) iterations,
        each dispatching a single vectorised CUDA kernel over up to
        min(n, m) cells.
        """
        n, m = cost.shape
        dtw = cost.clone()

        for d in range(2, n + m - 1):
            # Cells on this anti-diagonal: i + j = d, with i >= 1 and j >= 1.
            i_start = max(1, d - m + 1)
            i_end   = min(d, n)
            if i_start >= i_end:
                continue
            i_idx = torch.arange(i_start, i_end, device=cost.device)
            j_idx = d - i_idx

            # True DTW: min of diagonal, vertical (i-1,j), horizontal (i,j-1).
            # Vertical/horizontal allow minor speed warping between cameras.
            diag  = dtw[i_idx - 1, j_idx - 1]
            vert  = dtw[i_idx - 1, j_idx]
            horiz = dtw[i_idx,     j_idx - 1]
            dtw[i_idx, j_idx] = cost[i_idx, j_idx] + torch.minimum(diag, torch.minimum(vert, horiz))

        return dtw

    @staticmethod
    def _dtw_backtrace(dtw: torch.Tensor) -> torch.Tensor:
        """Backtrace the optimal DTW path. Returns Px2 tensor of (i, j) indices."""
        n, m = dtw.shape
        dtw_cpu = dtw.detach().cpu()
        # Free end: find the endpoint (last row or last column) with the lowest
        # average cost per step to avoid bias toward shorter overlaps.
        last_row_avg = dtw_cpu[n - 1, :] / (torch.arange(m, device=dtw_cpu.device) + 1)
        last_col_avg = dtw_cpu[:, m - 1] / (torch.arange(n, device=dtw_cpu.device) + 1)

        best_row_j = int(last_row_avg.argmin().item())
        best_col_i = int(last_col_avg.argmin().item())
        if last_row_avg[best_row_j] <= last_col_avg[best_col_i]:
            i, j = n - 1, best_row_j
        else:
            i, j = best_col_i, m - 1
        path = [(i, j)]

        # Follow the minimum-cost predecessor at each step.
        while i > 0 and j > 0:
            diag  = dtw_cpu[i - 1, j - 1]
            vert  = dtw_cpu[i - 1, j]
            horiz = dtw_cpu[i,     j - 1]
            best  = min(diag, vert, horiz)
            if best == diag:
                i -= 1; j -= 1
            elif best == vert:
                i -= 1
            else:
                j -= 1
            path.append((i, j))
        # Drain remaining boundary
        while i > 0:
            i -= 1
            path.append((i, j))
        while j > 0:
            j -= 1
            path.append((i, j))

        return torch.tensor(path[::-1], device=dtw.device)

    def _cross_corr_offset(
        self,
        body_joints_1: torch.Tensor,  # T1 x J x 3
        body_joints_2: torch.Tensor,  # T2 x J x 3
        confidences_1: torch.Tensor,  # T1 x J
        confidences_2: torch.Tensor,  # T2 x J
    ) -> float:
        """
        Estimate temporal offset via cross-correlation with median cost.

        For each candidate offset k (seq_2 starts k frames after seq_1),
        extract the overlapping frames and compute the median per-frame cost.
        The offset with the lowest median is returned.

        offset > 0 → seq_2 starts later than seq_1.
        offset < 0 → seq_2 starts earlier than seq_1.
        """
        cost = self._compute_cost_matrix(
            body_joints_1, body_joints_2, confidences_1, confidences_2
        )
        n, m = cost.shape
        # k = j_start - i_start, i.e. cost[i, i+k] for valid i
        k_lo = -(n - 1) if self.max_shift is None else max(-(n - 1), -2 * self.max_shift)
        k_hi = m        if self.max_shift is None else min(m,          2 * self.max_shift + 1)
        scores = {}
        for k in range(k_lo, k_hi):
            i0 = max(0, -k)
            i1 = min(n, m - k)
            overlap = i1 - i0
            if overlap < self.min_overlap:
                continue
            i_idx = torch.arange(i0, i1, device=cost.device)
            scores[k] = cost[i_idx, i_idx + k].median().item()

        if not scores:
            logger.warning(
                f"    cross-corr: no offset candidate has ≥{self.min_overlap} frames of overlap "
                f"(n={n}, m={m}) — returning 0"
            )
            return 0.0

        sorted_offsets = sorted(scores.items(), key=lambda x: x[1])
        best_k, best_score = sorted_offsets[0]

        if self.verbose:
            top10 = sorted_offsets[:10]
            top10_str = "  ".join(
                f"k={k:+d}(×{s / best_score:.2f})" for k, s in top10
            )
            logger.debug(
                f"    cross-corr top-10: {top10_str}"
                f"  → best offset={best_k:+d}  median_cost={best_score:.4f}"
            )
        return float(best_k)

    def _estimate_single_person_offset(
        self,
        body_joints_1: torch.Tensor,  # T1 x J x 3
        body_joints_2: torch.Tensor,  # T2 x J x 3
        confidences_1: torch.Tensor,  # T1 x J
        confidences_2: torch.Tensor,  # T2 x J
    ) -> float:
        """
        Returns the estimated temporal offset (in frames) between sequence 2
        and sequence 1 for a single person, i.e.  offset ≈ t_start_2 - t_start_1.
        """
        assert body_joints_1.shape[1:] == body_joints_2.shape[1:], "Joint shapes must match"
        assert confidences_1.shape[1] == confidences_2.shape[1], "Joint count must match in confidences"
        assert body_joints_1.shape[0] == confidences_1.shape[0], "Frame count must match (seq 1)"
        assert body_joints_2.shape[0] == confidences_2.shape[0], "Frame count must match (seq 2)"

        if self.method == "cross_corr":
            return self._cross_corr_offset(
                body_joints_1, body_joints_2, confidences_1, confidences_2
            )

        cost = self._compute_cost_matrix(
            body_joints_1, body_joints_2, confidences_1, confidences_2
        )
        dtw  = self._dtw_accumulate(cost)
        path = self._dtw_backtrace(dtw)

        # Temporal offset = mode of (j - i) along the warping path.
        shifts = path[:, 1] - path[:, 0]
        offset = torch.mode(shifts).values.item()

        if self.verbose:
            unique, counts = torch.unique(shifts, return_counts=True)
            top_k = min(5, len(unique))
            top_idx = counts.topk(top_k).indices
            dist_str = "  ".join(
                f"{unique[i].item():+d}×{counts[i].item()}" for i in top_idx
            )
            logger.debug(f"    shift distribution (top-{top_k}): {dist_str}  → chosen={offset:+.0f}")

        return offset

    def estimate_couple_offset(
        self,
        body_joints_1: list[torch.Tensor],  # P elements, each T1 x J x 3
        body_joints_2: list[torch.Tensor],  # P elements, each T2 x J x 3
        confidences_1: list[torch.Tensor],  # P elements, each T1 x J
        confidences_2: list[torch.Tensor],  # P elements, each T2 x J
    ) -> float:
        """
        Returns the estimated temporal offset (in frames) between sequence 2
        and sequence 1, i.e.  offset ≈ t_start_2 - t_start_1.

        For cross_corr: sums the per-person median costs at each candidate
        offset k and picks the k that minimises the joint cost across all
        persons.  This avoids the winner-takes-all failure mode where one
        person's secondary peak consistently overrides the correct offset.
        For DTW: returns the median offset across all persons.
        """
        P = len(body_joints_1)
        assert P == len(body_joints_2) == len(confidences_1) == len(confidences_2), \
            "Number of people must match across both videos and their confidences"
        assert P > 0, "Need at least one person"

        if self.method == "cross_corr":
            # Build per-person cost matrices and accumulate a joint score for every
            # candidate offset k.  Winner-takes-all (one person's offset) breaks when
            # the dominant person has a secondary peak at the wrong k; summing across
            # all persons lets the correct k win even if one person is ambiguous.
            per_person_costs: list[dict[int, float]] = []
            for p in range(P):
                cost = self._compute_cost_matrix(
                    body_joints_1[p], body_joints_2[p],
                    confidences_1[p], confidences_2[p],
                )
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
                    p_scores[k] = cost[i_idx, i_idx + k].median().item()
                if not p_scores:
                    if self.verbose:
                        logger.debug(f"    person {p}: no valid overlap — skipping")
                    continue
                if self.verbose:
                    logger.debug(
                        f"    person {p}: best k={min(p_scores, key=p_scores.get):+d}"
                        f"  median_cost={min(p_scores.values()):.4f}"
                    )
                per_person_costs.append(p_scores)

            if not per_person_costs:
                logger.warning("    no person had valid overlap — returning offset=0")
                return 0.0

            # Sum median costs across all persons for each k that every person covers.
            common_ks = set(per_person_costs[0].keys())
            for p_scores in per_person_costs[1:]:
                common_ks &= set(p_scores.keys())

            if not common_ks:
                # Fall back to the union and treat missing persons as having cost=0
                common_ks = set().union(*[set(p.keys()) for p in per_person_costs])

            combined: dict[int, float] = {
                k: sum(p_scores.get(k, 0.0) for p_scores in per_person_costs)
                for k in common_ks
            }
            best_k = min(combined, key=combined.__getitem__)
            if self.verbose:
                sorted_combined = sorted(combined.items(), key=lambda x: x[1])
                top_n = sorted_combined[:10]
                top_str = "  ".join(
                    f"k={k:+d}(cost={c:.4f})" for k, c in top_n
                )
                logger.debug(f"    top-10 combined: {top_str}")
                logger.debug(f"    → chosen k={best_k:+d}  cost={combined[best_k]:.4f}")
            return float(best_k)

        per_person_offsets = []
        for p in range(P):
            off = self._estimate_single_person_offset(
                body_joints_1[p], body_joints_2[p],
                confidences_1[p], confidences_2[p],
            )
            if self.verbose:
                logger.debug(f"    person {p}: raw offset = {off:+.0f}")
            per_person_offsets.append(off)

        median_offset = torch.median(torch.tensor(per_person_offsets, dtype=torch.float32)).item()
        if self.verbose:
            logger.debug(f"    per-person offsets: {per_person_offsets}  → median = {median_offset:+.1f}")
        return median_offset

    def estimate_offset_matrix(
        self,
        body_joints_list: list[list[torch.Tensor]],  # K videos, each containing P person tensors of shape T_i x J x 3
        confidences_list: list[list[torch.Tensor]],  # K videos, each containing P person tensors of shape T_i x J
    ) -> torch.Tensor | list[list[list[tuple[int, float]]]]:
        """
        For dtw / cross_corr: returns a K×K antisymmetric float tensor of scalar offsets.
        For multi_cross_corr: returns a K×K nested list of (offset, cost) candidate lists.
        """
        K = len(body_joints_list)
        assert K == len(confidences_list), "Number of body joints and confidence sets must match"

        if self.method == "multi_cross_corr":
            return self.estimate_offset_matrix_multi(body_joints_list, confidences_list)

        if self.method == "distributional":
            return self._estimate_offset_matrix_distributional(body_joints_list, confidences_list)

        offset_matrix = torch.zeros((K, K), device=self.device)
        for i in range(K):
            for j in range(i + 1, K):
                off = self.estimate_couple_offset(
                    body_joints_list[i],
                    body_joints_list[j],
                    confidences_list[i],
                    confidences_list[j],
                )
                offset_matrix[i, j] = off
                offset_matrix[j, i] = -off  # antisymmetric

        return offset_matrix

    def cycle_consistency_weights(
        self,
        offset_matrix: torch.Tensor | list[list[list[tuple[int, float]]]],
    ) -> torch.Tensor | None:
        """
        Per-edge weights derived from cycle consistency.

        For dtw / cross_corr: standard computation on the scalar K×K tensor.
        For multi_cross_corr: builds a scalar matrix using the best (lowest-cost)
        candidate per pair, runs the same residual logic, and returns the weights.
        The weights are not used by belief_propagation_sync, so None is returned
        as a sentinel to keep the call site uniform.
        """
        if self.method == "multi_cross_corr":
            # BP does not use external weights, but we keep the call site identical.
            return None

        if self.method == "distributional":
            # Cycle-consistency weights on MAP argmaxes — same formula as cross_corr.
            # A pair whose MAP peak violates triangle constraints gets low weight,
            # catching both flat distributions AND confident-but-wrong alias peaks.
            K = len(offset_matrix)
            S = (offset_matrix[0][1].shape[0] - 1) // 2
            O = torch.zeros(K, K, device=self.device)
            for i in range(K):
                for j in range(K):
                    if i != j:
                        O[i, j] = float(offset_matrix[i][j].argmax().item() - S)
            residuals_3d = O.unsqueeze(2) + O.unsqueeze(0) - O.unsqueeze(1)
            idx = torch.arange(K, device=self.device)
            mask = (idx.view(1, 1, K) != idx.view(K, 1, 1)) & \
                   (idx.view(1, 1, K) != idx.view(1, K, 1))
            mean_residual = (residuals_3d.abs() * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
            return 1.0 / (1.0 + mean_residual)

        K = offset_matrix.shape[0]
        O = offset_matrix  # K×K

        # residuals_3d[i,j,k] = O[i,j] + O[j,k] - O[i,k]  (should be 0 by cycle consistency)
        residuals_3d = O.unsqueeze(2) + O.unsqueeze(0) - O.unsqueeze(1)

        # Mask out k==i and k==j (those triangles are degenerate)
        idx = torch.arange(K, device=self.device)
        mask = (idx.view(1, 1, K) != idx.view(K, 1, 1)) & \
               (idx.view(1, 1, K) != idx.view(1, K, 1))  # K×K×K

        mean_residual = (residuals_3d.abs() * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return 1.0 / (1.0 + mean_residual)  # K×K

    def estimate_initial_times(
        self,
        offset_matrix: torch.Tensor | list[list[list[tuple[int, float]]]],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        For dtw / cross_corr: weighted LSE solver (weights from cycle_consistency_weights).
        For multi_cross_corr: discrete Min-Sum BP solver (weights argument is ignored).
        """
        if self.method == "multi_cross_corr":
            return self.belief_propagation_sync(offset_matrix)

        if self.method == "distributional":
            return self._distributional_bp(offset_matrix, weights)

        K = offset_matrix.shape[0]
        num_pairs = K * (K - 1) // 2
        A = torch.zeros(num_pairs, K - 1, device=self.device)
        b = torch.zeros(num_pairs, device=self.device)
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

        sol = torch.linalg.lstsq(A, b).solution  # K-1, use a LS approach to find initial times

        initial_times = torch.zeros(K, device=self.device)
        initial_times[1:] = sol # t_0 is fixed to 0, otherwise system is undetermined
        if initial_times.min() < 0:
            initial_times = initial_times - initial_times.min() # Shift if any initial time is smaller than 0
        return initial_times

    # ------------------------------------------------------------------
    # multi_cross_corr: multi-hypothesis offset matrix + BP global solver
    # ------------------------------------------------------------------

    @staticmethod
    def _local_minima(costs: dict[int, float], n: int) -> list[tuple[int, float]]:
        """Return the top-n local minima from a {k: cost} dict, sorted by cost.

        A point k is a local minimum if its cost is ≤ both neighbours.
        Falls back to the global minimum when the curve is too short to have
        any interior local minimum.
        """
        ks = sorted(costs.keys())
        if len(ks) < 3:
            return sorted(costs.items(), key=lambda x: x[1])[:n]
        minima = []
        for idx, k in enumerate(ks):
            c = costs[k]
            left  = costs[ks[idx - 1]] if idx > 0          else float('inf')
            right = costs[ks[idx + 1]] if idx < len(ks)-1  else float('inf')
            if c <= left and c <= right:
                minima.append((k, c))
        if not minima:
            minima = [min(costs.items(), key=lambda x: x[1])]
        minima.sort(key=lambda x: x[1])
        return minima[:n]

    def estimate_couple_candidates(
        self,
        body_joints_1: list[torch.Tensor],  # P elements, each T1 x J x 3
        body_joints_2: list[torch.Tensor],  # P elements, each T2 x J x 3
        confidences_1: list[torch.Tensor],  # P elements, each T1 x J
        confidences_2: list[torch.Tensor],  # P elements, each T2 x J
        n_candidates: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Returns the top-n_candidates local minima of the combined cross-correlation
        cost curve as (offset, cost) pairs, where offset = t_start_2 - t_start_1.

        Persons are fused by summing their per-person median cost curves before
        peak detection, so the true minimum and alias peaks are all preserved.
        """
        n_candidates = n_candidates or self.n_candidates
        combined: dict[int, float] = {}
        for p in range(len(body_joints_1)):
            cost = self._compute_cost_matrix(
                body_joints_1[p], body_joints_2[p],
                confidences_1[p], confidences_2[p],
            )
            n, m = cost.shape
            k_lo = -(n-1) if self.max_shift is None else max(-(n-1), -2*self.max_shift)
            k_hi = m       if self.max_shift is None else min(m,       2*self.max_shift+1)
            for k in range(k_lo, k_hi):
                i0 = max(0, -k)
                i1 = min(n, m - k)
                if i1 - i0 < self.min_overlap:
                    continue
                i_idx = torch.arange(i0, i1, device=cost.device)
                combined[k] = combined.get(k, 0.0) + cost[i_idx, i_idx + k].median().item()

        if not combined:
            logger.warning("    multi_cross_corr: no valid overlap — returning [(0, inf)]")
            return [(0, float('inf'))]

        cands = self._local_minima(combined, n_candidates)
        if self.verbose:
            logger.debug(f"    multi_cross_corr candidates: {cands}")
        return cands

    def estimate_offset_matrix_multi(
        self,
        body_joints_list: list[list[torch.Tensor]],
        confidences_list: list[list[torch.Tensor]],
        n_candidates: int | None = None,
    ) -> list[list[list[tuple[int, float]]]]:
        """
        Returns a K×K nested list where each cell [i][j] holds up to n_candidates
        (offset, cost) pairs meaning t_start_j - t_start_i.  The matrix is
        antisymmetric: candidates[j][i] has negated offsets.
        """
        n_candidates = n_candidates or self.n_candidates
        K = len(body_joints_list)
        cands = [[[] for _ in range(K)] for _ in range(K)]
        for i in range(K):
            for j in range(i + 1, K):
                c_ij = self.estimate_couple_candidates(
                    body_joints_list[i], body_joints_list[j],
                    confidences_list[i], confidences_list[j],
                    n_candidates=n_candidates,
                )
                cands[i][j] = c_ij
                cands[j][i] = [(-k, c) for k, c in c_ij]
        return cands

    def belief_propagation_sync(
        self,
        candidates: list[list[list[tuple[int, float]]]],
        n_iterations: int | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """
        Discrete Min-Sum (log-domain Max-Product) Belief Propagation over the
        camera network modelled as a pairwise MRF.

        State space:
          - Camera 0 is fixed at t_0 = 0 (single state).
          - Camera j ≥ 1: states are the candidate absolute start times derived
            from the direct pair (0, j), i.e. {k | (k, cost) ∈ candidates[0][j]}.

        Unary cost   (theta_j[s]):  cross-corr cost of candidate s for camera j.
        Pairwise cost(theta_ij[s_i, s_j]):
            min_{(c,_) in candidates[i][j]}  |s_j - s_i - c| / sigma
        i.e. how much the assignment (s_i, s_j) violates the nearest candidate
        offset for edge (i,j).

        Returns a K-vector of estimated absolute start times (non-negative).
        """
        n_iterations = n_iterations or self.n_bp_iterations
        sigma        = sigma        or self.bp_sigma
        K = len(candidates)

        # --- state spaces ---
        state_values: list[list[float]] = [[0.0]]
        for j in range(1, K):
            c0j = candidates[0][j]
            state_values.append([float(k) for k, _ in c0j] if c0j else [0.0])

        # --- unary costs (lower = more likely) ---
        unary: list[torch.Tensor] = [torch.zeros(1, device=self.device)]
        for j in range(1, K):
            c0j = candidates[0][j]
            if c0j:
                unary.append(torch.tensor([c for _, c in c0j], dtype=torch.float32, device=self.device))
            else:
                unary.append(torch.zeros(1, device=self.device))

        # --- precompute pairwise cost tensors  (Ni x Nj) ---
        pw: dict[tuple[int,int], torch.Tensor] = {}
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                si = torch.tensor(state_values[i], device=self.device)   # Ni
                sj = torch.tensor(state_values[j], device=self.device)   # Nj
                c_ij = candidates[i][j]
                if c_ij:
                    offsets = torch.tensor([float(k) for k, _ in c_ij], device=self.device)  # Nc
                    # diff[ni, nj] = s_j[nj] - s_i[ni]
                    diff = sj.unsqueeze(0) - si.unsqueeze(1)             # Ni x Nj
                    dev  = (diff.unsqueeze(2) - offsets.view(1, 1, -1)).abs()  # Ni x Nj x Nc
                    pw[(i, j)] = dev.min(dim=2).values / sigma           # Ni x Nj
                else:
                    pw[(i, j)] = torch.zeros(len(state_values[i]), len(state_values[j]), device=self.device)

        # --- initialise messages to 0 (uniform in log domain) ---
        msgs: dict[tuple[int,int], torch.Tensor] = {
            (i, j): torch.zeros(len(state_values[j]), device=self.device)
            for i in range(K) for j in range(K) if i != j
        }

        # --- loopy Min-Sum BP ---
        for _ in range(n_iterations):
            new_msgs: dict[tuple[int,int], torch.Tensor] = {}
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    # sum of incoming messages to i, excluding the one from j
                    incoming = unary[i].clone()
                    for k in range(K):
                        if k != i and k != j:
                            incoming = incoming + msgs[(k, i)]
                    # msg[i→j][s_j] = min_{s_i}( incoming[s_i] + pw[i,j][s_i, s_j] )
                    total   = incoming.unsqueeze(1) + pw[(i, j)]         # Ni x Nj
                    new_msg = total.min(dim=0).values                    # Nj
                    new_msg = new_msg - new_msg.min()                    # normalise
                    new_msgs[(i, j)] = new_msg
            msgs = new_msgs

        # --- MAP beliefs ---
        result = torch.zeros(K, device=self.device)
        for j in range(K):
            belief = unary[j].clone()
            for i in range(K):
                if i != j:
                    belief = belief + msgs[(i, j)]
            best = int(belief.argmin().item())
            result[j] = torch.tensor(state_values[j], device=self.device)[best]

        if result.min() < 0:
            result = result - result.min()
        return result

    def estimate_initial_times_multi(
        self,
        body_joints_list: list[list[torch.Tensor]],
        confidences_list: list[list[torch.Tensor]],
        n_candidates: int | None = None,
        n_bp_iterations: int | None = None,
        bp_sigma: float | None = None,
    ) -> torch.Tensor:
        """
        Full multi_cross_corr pipeline:
          1. Build K×K multi-hypothesis offset matrix (top-N local minima per pair).
          2. Run discrete Min-Sum BP to find globally consistent start times.
        Returns a K-vector of absolute start times.
        """
        candidates = self.estimate_offset_matrix_multi(
            body_joints_list, confidences_list, n_candidates=n_candidates
        )
        return self.belief_propagation_sync(
            candidates, n_iterations=n_bp_iterations, sigma=bp_sigma
        )

    # ------------------------------------------------------------------
    # distributional: softmax-distribution edge model + convolutional BP
    # ------------------------------------------------------------------

    def _compute_pair_log_distribution(
        self,
        cost_matrix: torch.Tensor,  # T1 x T2
        S: int,
    ) -> torch.Tensor:
        """Convert a T1×T2 cost matrix into a log-probability distribution over
        shifts k ∈ [-S, S] (output shape [2S+1]).

        For each shift k the diagonal slice cost[i, i+k] is extracted; its median
        becomes the score. Scores are converted to log-probs via softmax(-score/τ).
        Shifts with insufficient overlap get log-prob = -inf (zero probability).
        When no shift has valid overlap the distribution is returned as uniform
        (maximum uncertainty, no information).
        """
        T1, T2 = cost_matrix.shape
        n = 2 * S + 1
        valid_ki, scores = [], []

        for ki, k in enumerate(range(-S, S + 1)):
            i0 = max(0, -k)
            i1 = min(T1, T2 - k)
            if i1 - i0 < self.min_overlap:
                continue
            i_idx = torch.arange(i0, i1, device=self.device)
            score = cost_matrix[i_idx, i_idx + k].median()
            if score.isfinite():
                valid_ki.append(ki)
                scores.append(score)

        log_p = torch.full((n,), float('-inf'), device=self.device)
        if not valid_ki:                              # fully uncertain: return uniform
            log_p.fill_(-math.log(n))
            return log_p

        valid_idx  = torch.tensor(valid_ki, device=self.device)
        score_t    = torch.stack(scores)

        # Adaptive temperature: scale with cost-curve std so signal/temp is constant
        # regardless of absolute cost level.  self.temperature acts as a multiplier
        # (target signal/temp ratio = 1/self.temperature, e.g. 0.1 → SNR=10).
        s_std = score_t.std().item()
        temperature = max(s_std * self.temperature, 1e-6)

        if self.verbose:
            s_min  = score_t.min().item()
            s_max  = score_t.max().item()
            s_mean = score_t.mean().item()
            best_k = int(valid_idx[score_t.argmin()].item()) - S
            logger.debug(
                f"        cost curve: min={s_min:.4f}(k={best_k:+d})  mean={s_mean:.4f}"
                f"  std={s_std:.4f}  range={s_max - s_min:.4f}  n={len(score_t)}"
                f"  (adaptive_temp={temperature:.4f}  signal/temp={s_std / temperature:.3f})"
            )

        raw        = -score_t / temperature            # unnormalised log-probs
        log_p[valid_idx] = raw - torch.logsumexp(raw, dim=0)
        return log_p

    def _log_dist_summary(self, log_p: torch.Tensor, label: str = "") -> None:
        """Debug-log top peaks and entropy of a log-probability distribution over shifts."""
        n = len(log_p)
        S = (n - 1) // 2
        finite = log_p.isfinite()
        n_valid = int(finite.sum().item())
        if n_valid == 0:
            logger.debug(f"      {label}: all -inf (no valid shifts)")
            return
        probs = log_p.exp()
        probs[~finite] = 0.0
        top_k = min(5, n_valid)
        top_vals, top_idx = probs.topk(top_k)
        peaks_str = "  ".join(
            f"k={int(top_idx[i].item()) - S:+d}(p={top_vals[i].item():.3f})"
            for i in range(top_k)
        )
        p_fin = log_p[finite].exp()
        entropy = -(p_fin * log_p[finite]).sum().item()
        norm_entropy = entropy / math.log(n_valid) if n_valid > 1 else 0.0
        gap = top_vals[0].item() / max(top_vals[1].item(), 1e-9) if top_k > 1 else float('inf')
        logger.debug(
            f"      {label}: [{peaks_str}]  entropy={norm_entropy:.3f}  "
            f"gap={gap:.2f}x  n_valid={n_valid}"
        )

    def _entropy_weight(self, log_p: torch.Tensor) -> float:
        """Reliability scalar in [0, 1] derived from the entropy of a log-distribution.
        0 = uniform (no information), 1 = delta (perfect certainty).
        """
        finite = log_p.isfinite()
        n_valid = finite.sum().item()
        if n_valid <= 1:
            return 1.0
        p          = log_p[finite].exp()
        entropy    = -(p * log_p[finite]).sum().item()
        max_entropy = math.log(n_valid)
        return float(max(0.0, 1.0 - entropy / max_entropy))

    @staticmethod
    def _fft_convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Linear convolution of two 1-D tensors via zero-padded FFT.
        Output length = len(a) + len(b) - 1.
        """
        n     = len(a) + len(b) - 1
        n_fft = 1 << (n - 1).bit_length()            # next power of 2
        A = torch.fft.rfft(a.float(), n=n_fft)
        B = torch.fft.rfft(b.float(), n=n_fft)
        return torch.fft.irfft(A * B, n=n_fft)[:n]

    def _estimate_offset_matrix_distributional(
        self,
        body_joints_list: list[list[torch.Tensor]],
        confidences_list: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        """Returns a K×K table of log-probability distributions (each [2S+1]).

        For each camera pair (i,j) and each person p:
          1. Compute the T_i×T_j geodesic cost matrix.
          2. Convert it to a log-distribution over shifts [-S, S].
        Then fuse across persons by summing log-probs (= multiplying probabilities)
        and renormalising — a person with no valid overlap contributes a uniform
        distribution which cancels out after renormalisation.
        Antisymmetry: log_dists[j][i] is the reversed version of log_dists[i][j].
        """
        K = len(body_joints_list)
        # S must cover the largest possible pairwise offset between any two cameras.
        # max_shift only bounds the simulation shift; the true pairwise offset between
        # two cameras can be up to 2*max_shift. Use the full sequence length to be safe.
        S = max(joints[0].shape[0] for joints in body_joints_list)
        n = 2 * S + 1

        log_dists = [[torch.full((n,), -math.log(n), device=self.device)
                      for _ in range(K)] for _ in range(K)]

        for i in range(K):
            for j in range(i + 1, K):
                P = len(body_joints_list[i])
                # Accumulate log-probs across persons (starts at log(1)=0 per shift)
                log_p_fused = torch.zeros(n, device=self.device)
                for p in range(P):
                    cost = self._compute_cost_matrix(
                        body_joints_list[i][p], body_joints_list[j][p],
                        confidences_list[i][p],  confidences_list[j][p],
                    )
                    log_p_fused = log_p_fused + self._compute_pair_log_distribution(cost, S)

                # Renormalise the fused distribution
                finite = log_p_fused.isfinite()
                if finite.any():
                    log_p_fused[finite] -= torch.logsumexp(log_p_fused[finite], dim=0)
                    log_p_fused[~finite] = float('-inf')
                else:
                    log_p_fused.fill_(-math.log(n))

                log_dists[i][j] = log_p_fused
                log_dists[j][i] = log_p_fused.flip(0)   # P_ji(k) = P_ij(-k)

                if self.verbose:
                    self._log_dist_summary(log_p_fused, f"pairwise ({i}→{j})")

        return log_dists

    def _distributional_bp(
        self,
        log_dists: list[list[torch.Tensor]],  # K×K, each [2S+1] in log-space
        weights: torch.Tensor,                 # K×K entropy-based edge weights
    ) -> torch.Tensor:
        """Convolutional Belief Propagation in probability space via FFT.

        Each camera node maintains a 1-D probability distribution over its
        absolute start time on a discrete timeline t ∈ [-S, S].
        Camera 0 is fixed as a delta at t=0 throughout.

        Message from i to j:
            M_{i→j}(t) = Σ_k P_ij(k) · T_i(t - k)   [linear convolution]
        Belief update for j (log-space, weighted):
            log T_j(t) = Σ_{i≠j}  w_ij · log M_{i→j}(t)
        then renormalised to sum to 1.
        """
        K   = len(log_dists)
        n   = log_dists[0][1].shape[0]       # 2S+1
        S   = (n - 1) // 2

        # --- initialise beliefs ---
        # Camera 0: delta at index S (= time 0), fixed throughout.
        # Camera j≥1: warm-start from the direct pairwise distribution with camera 0.
        # This avoids the CDF-shape boundary artifact that arises when convolving a
        # uniform distribution through the linear convolution crop: conv(uniform, P)
        # produces a step-function message (partial sums of P), not a flat one, which
        # pushes all beliefs toward one side of the timeline in the first iteration.
        log_T = [torch.full((n,), float('-inf'), device=self.device) for _ in range(K)]
        log_T[0][S] = 0.0
        for j in range(1, K):
            log_T[j] = log_dists[0][j].clone()

        if self.verbose:
            logger.debug("    [BP] initial beliefs (warm-start from cam 0):")
            for j in range(K):
                self._log_dist_summary(log_T[j], f"cam {j}")

        for _ in range(self.n_dist_bp_iterations):
            new_log_T = [log_T[0].clone()]   # camera 0 stays fixed

            for j in range(1, K):
                log_belief = torch.zeros(n, device=self.device)

                for i in range(K):
                    if i == j:
                        continue
                    # Message: M_{i→j} = conv(T_i, P_ij), crop centre n elements
                    T_i  = log_T[i].exp()                    # [n]
                    P_ij = log_dists[i][j].exp()             # [n]
                    conv = self._fft_convolve(T_i, P_ij)     # [2n-1]
                    M    = conv[S: S + n].clamp(min=1e-30)   # [n], crop centre
                    M    = M / M.sum()                        # normalise: removes boundary taper

                    log_belief = log_belief + weights[i, j].item() * M.log()

                # Renormalise
                finite = log_belief.isfinite()
                if finite.any():
                    log_belief[~finite] = float('-inf')
                    log_belief[finite] -= torch.logsumexp(log_belief[finite], dim=0)
                else:
                    log_belief.fill_(-math.log(n))

                new_log_T.append(log_belief)

            log_T = new_log_T

        if self.verbose:
            logger.debug("    [BP] final beliefs (after all iterations):")
            for j in range(K):
                self._log_dist_summary(log_T[j], f"cam {j}")

        # --- MAP decoding ---
        result = torch.zeros(K, device=self.device)
        for j in range(K):
            result[j] = float(log_T[j].argmax().item() - S)

        if result.min() < 0:
            result = result - result.min()
        return result
