import logging

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
        verbose: bool = False,
    ):
        self.method = method
        self.only_overlap = only_overlap
        self.device = device
        self.q = q
        self.min_overlap = min_overlap
        self.verbose = verbose

    def _compute_cost_matrix(
        self,
        body_joints_1: torch.Tensor,  # T1 x J x 3
        body_joints_2: torch.Tensor,  # T2 x J x 3
        confidences_1: torch.Tensor,  # T1 x J
        confidences_2: torch.Tensor,  # T2 x J
    ) -> torch.Tensor:
        """Vectorised pairwise weighted distance matrix (T1 x T2)."""
        n = body_joints_1.shape[0]
        cost = torch.zeros(n, body_joints_2.shape[0], device=self.device)
        # Process one row at a time to avoid T1*T2*J*3 memory blow-up
        for i in range(n):
            diff = torch.norm(body_joints_1[i].unsqueeze(0) - body_joints_2, dim=-1)  # ||1 x J x 3 - T2 x J x 3|| -> T2 x J 
            w = confidences_1[i].unsqueeze(0) * confidences_2                         # 1 x J * T2 x J -> T2 x J 
            num = torch.sum(w * diff, dim=-1)                                         # sum(T2 x J) -> T2
            den = torch.sum(w, dim=-1) + 1e-8                                         # T2
            valid = den > 1e-4                                                          # T2
            cost[i] = torch.where(valid, (num / den) ** self.q, torch.full_like(num, float('inf')))  # T2
        return cost # T1 x T2

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
            # Cells on this anti-diagonal: i + j = d
            # Skip d=1 (only boundary cells) — they stay at cost[i,j] (free start).
            i_start = max(1, d - m + 1)  # i >= 1 and j >= 1 enforces interior only
            i_end = min(d, n)             # j = d - i >= 1  →  i <= d - 1
            if i_start >= i_end:
                continue
            i_idx = torch.arange(i_start, i_end, device=cost.device)
            j_idx = d - i_idx  # all >= 1 by construction

            # Pure diagonal update: same frame rate means no time warping.
            # Horizontal/vertical steps are not allowed.
            dtw[i_idx, j_idx] = cost[i_idx, j_idx] + dtw[i_idx - 1, j_idx - 1]

        return dtw

    @staticmethod
    def _dtw_backtrace(dtw: torch.Tensor) -> torch.Tensor:
        """Backtrace the optimal DTW path. Returns Px2 tensor of (i, j) indices."""
        n, m = dtw.shape
        dtw_cpu = dtw.detach().cpu()
        # Free end: find the endpoint (last row or last column) with the lowest
        # *average* cost per step to avoid bias toward shorter overlaps.
        # Path length for cell (i, j) = min(i, j) + 1.
        last_row_j   = torch.arange(m, device=dtw_cpu.device)
        last_row_len = torch.minimum(torch.tensor(n - 1), last_row_j) + 1
        last_row_avg = dtw_cpu[n - 1, :] / last_row_len

        last_col_i   = torch.arange(n, device=dtw_cpu.device)
        last_col_len = torch.minimum(last_col_i, torch.tensor(m - 1)) + 1
        last_col_avg = dtw_cpu[:, m - 1] / last_col_len

        best_row_j = int(last_row_avg.argmin().item())
        best_col_i = int(last_col_avg.argmin().item())
        if last_row_avg[best_row_j] <= last_col_avg[best_col_i]:
            i, j = n - 1, best_row_j
        else:
            i, j = best_col_i, m - 1
        path = [(i, j)]

        # Pure diagonal: always step (i-1, j-1) until a boundary is reached.
        while i > 0 and j > 0:
            i -= 1
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
        # k ranges from -(n-1) to (m-1)
        scores = {}
        for k in range(-(n - 1), m):
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
                p_scores: dict[int, float] = {}
                for k in range(-(n - 1), m):
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
                logger.debug(f"    joint offset={best_k:+d}  combined_cost={combined[best_k]:.4f}")
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
    ) -> torch.Tensor:

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

    def cycle_consistency_weights(self, offset_matrix: torch.Tensor) -> torch.Tensor:
        """
        Per-edge weights derived from cycle consistency.

        For each pair (i,j), the residual is the mean absolute cycle error
        across all third cameras k:
            residual(i,j) = mean_{k≠i,j} |offset(i,j) + offset(j,k) - offset(i,k)|

        A perfectly consistent edge has residual=0 and weight=1.
        An inconsistent edge has large residual and weight→0.
        weight = 1 / (1 + residual)
        """
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
        offset_matrix: torch.Tensor,        # K x K  (antisymmetric)
        weights: torch.Tensor | None = None, # K x K  per-edge weights, e.g. from cycle_consistency_weights
    ) -> torch.Tensor:
        """
        Solve for start times t_0 … t_{K-1} from pairwise offsets via weighted LSE.
        Fixes t_0 = 0 and solves for the remaining K-1 variables.
        Each pairwise equation is scaled by its weight so unreliable edges contribute less.
        """
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
