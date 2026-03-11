import logging

import torch

logger = logging.getLogger(__name__)


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
    ):
        self.method = method
        self.only_overlap = only_overlap
        self.device = device
        self.q = q

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
            cost[i] = (num / den) ** self.q                                           # T2
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
        INF = float("inf")

        for d in range(1, n + m - 1):
            # Cells on this anti-diagonal: i + j = d
            i_start = max(0, d - m + 1)
            i_end = min(d + 1, n)
            i_idx = torch.arange(i_start, i_end, device=cost.device) # Rows covered by this ant-diagonal
            j_idx = d - i_idx                                        # Corresponding columns

            # Gather the three predecessor values (diagonal, vertical, horizontal)
            # All predecessors lie on anti-diagonals d-1 or d-2, already computed.
            size = len(i_idx)
            candidates = torch.full((3, size), INF, device=cost.device)

            # (i-1, j-1) — diagonal
            mask_diag = (i_idx > 0) & (j_idx > 0)
            if mask_diag.any():
                candidates[0, mask_diag] = dtw[i_idx[mask_diag] - 1, j_idx[mask_diag] - 1]

            # (i-1, j) — vertical (insertion)
            mask_vert = i_idx > 0
            if mask_vert.any():
                candidates[1, mask_vert] = dtw[i_idx[mask_vert] - 1, j_idx[mask_vert]]

            # (i, j-1) — horizontal (deletion)
            mask_horiz = j_idx > 0
            if mask_horiz.any():
                candidates[2, mask_horiz] = dtw[i_idx[mask_horiz], j_idx[mask_horiz] - 1]

            dtw[i_idx, j_idx] += candidates.min(dim=0).values

        return dtw

    @staticmethod
    def _dtw_backtrace(dtw: torch.Tensor) -> torch.Tensor:
        """Backtrace the optimal DTW path. Returns Px2 tensor of (i, j) indices."""
        n, m = dtw.shape
        i, j = n - 1, m - 1
        path = [(i, j)]
        # Pull values to CPU once for fast scalar access
        dtw_cpu = dtw.detach().cpu()
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                vals = (
                    dtw_cpu[i - 1, j - 1].item(),
                    dtw_cpu[i - 1, j].item(),
                    dtw_cpu[i, j - 1].item(),
                )
                best = vals.index(min(vals))
                if best == 0:
                    i -= 1
                    j -= 1
                elif best == 1:
                    i -= 1
                else:
                    j -= 1
            path.append((i, j))
        return torch.tensor(path[::-1], device=dtw.device) # Reverse it so that it is useful for later

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

        cost = self._compute_cost_matrix(
            body_joints_1, body_joints_2, confidences_1, confidences_2
        )
        dtw = self._dtw_accumulate(cost)
        path = self._dtw_backtrace(dtw)  # P x 2

        # Temporal offset = mode of (j - i) along the warping path.
        shifts = path[:, 1] - path[:, 0]  # integer shifts between the frames according to the best path
        offset = torch.mode(shifts).values.item()

        # Log the distribution of shifts to diagnose ambiguity
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

        Computes a per-person DTW offset and returns the median across people,
        which is robust to outliers from mismatched or poorly tracked individuals.
        """
        P = len(body_joints_1)
        assert P == len(body_joints_2) == len(confidences_1) == len(confidences_2), \
            "Number of people must match across both videos and their confidences"
        assert P > 0, "Need at least one person"

        per_person_offsets = []
        for p in range(P):
            off = self._estimate_single_person_offset(
                body_joints_1[p], body_joints_2[p],
                confidences_1[p], confidences_2[p],
            )
            logger.debug(f"    person {p}: raw DTW offset = {off:+.0f}")
            per_person_offsets.append(off)

        # Use median for robustness against outlier persons
        median_offset = torch.median(torch.tensor(per_person_offsets, dtype=torch.float32)).item()
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

    def estimate_initial_times(
        self,
        offset_matrix: torch.Tensor,  # K x K  (antisymmetric)
    ) -> torch.Tensor:
        """
        Solve for start times t_0 … t_{K-1} from pairwise offsets via LSE.
        Fixes t_0 = 0 and solves for the remaining K-1 variables.
        """
        K = offset_matrix.shape[0]
        num_pairs = K * (K - 1) // 2
        A = torch.zeros(num_pairs, K - 1, device=self.device)
        b = torch.zeros(num_pairs, device=self.device)
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                if j >= 1:
                    A[count, j - 1] = 1.0
                if i >= 1:
                    A[count, i - 1] = -1.0
                b[count] = offset_matrix[i, j]
                count += 1

        sol = torch.linalg.lstsq(A, b).solution  # K-1, use a LS approach to find initial times

        initial_times = torch.zeros(K, device=self.device)
        initial_times[1:] = sol # t_0 is fixed to 0, otherwise system is undetermined
        if initial_times.min() < 0:
            initial_times = initial_times - initial_times.min() # Shift if any initial time is smaller than 0
        return initial_times
