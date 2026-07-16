import logging

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Synchronizer:
    """
    Class performing the temporal alignment of videos.
    Uses cross-correlation with median cost to find the optimal offset
    between pairs of joint sequences, then solves a least-squares system
    to recover globally consistent start times.
    """

    def __init__(
        self,
        device: str = "cuda",
        q: int = 2,
        min_overlap: int = 100,
        max_shift: int | None = None,
        verbose: bool = False,
        use_acceleration_weights: bool = False,
    ):
        self.device = device
        self.q = q
        self.min_overlap = min_overlap
        self.max_shift = max_shift
        self.verbose = verbose
        self.use_acceleration_weights = use_acceleration_weights

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

    def _compute_cost_matrix(
        self,
        body_joints_1: torch.Tensor,  # T1 x J x 3
        body_joints_2: torch.Tensor,  # T2 x J x 3
        confidences_1: torch.Tensor,  # T1 x J
        confidences_2: torch.Tensor,  # T2 x J
    ) -> torch.Tensor:
        """T1×T2 pairwise cost matrix.

        Input tensors are axis-angle rotations → SO(3) geodesic distance.
        """
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

        Sums the per-person median costs at each candidate offset k and picks
        the k that minimises the joint cost across all persons. This avoids
        the winner-takes-all failure mode where one person's secondary peak
        consistently overrides the correct offset.
        """
        P = len(body_joints_1)
        assert P == len(body_joints_2) == len(confidences_1) == len(confidences_2), \
            "Number of people must match across both videos and their confidences"
        assert P > 0, "Need at least one person"

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

    def estimate_offset_matrix(
        self,
        body_joints_list: list[list[torch.Tensor]],  # K videos, each containing P person tensors of shape T_i x J x 3
        confidences_list: list[list[torch.Tensor]],  # K videos, each containing P person tensors of shape T_i x J
    ) -> torch.Tensor:
        """Returns a K×K antisymmetric float tensor of scalar offsets."""
        K = len(body_joints_list)
        assert K == len(confidences_list), "Number of body joints and confidence sets must match"

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
        offset_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Per-edge weights derived from cycle consistency on the scalar K×K tensor."""
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
        offset_matrix: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted LSE solver (weights from cycle_consistency_weights)."""
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
