"""
Mapper: unified SMPLX vertex regressor for MHR70 keypoints.

For each MHR70 keypoint k used in placer.py (bones + DLT), precomputes a row
R[k] ∈ ℝ^{10475} such that:

    smplx_equivalent[k] = R[k] @ smplx_verts_posed

Two types of MHR70 keypoints (determined from head_pose.keypoint_mapping in the
SAM3D checkpoint):

  • Vertex-based (elbow, ankle): keypoint_mapping weight is 100% mesh vertices.
    R[k] is derived by composing the vertex weights with smplx2mhr_mapping.npz,
    which maps each MHR vertex to a face+barycentric on the SMPLX mesh.

  • Joint-based (hip, knee, wrist, neck): keypoint_mapping weight is 100% a single
    pymomentum skeleton joint. R[k] = J_regressor[smplx_joint_idx], i.e. the SMPLX
    joint centre expressed as a linear function of posed vertices.

Usage
-----
    mapper = Mapper.build(ckpt_path, smplx_pkl, smplx2mhr_path)
    mapper.save("mappings/mhr_smplx_regressor.npy")

    # at inference (placer.py), with posed SMPLX vertices (T, 10475, 3):
    kp_smplx = mapper.map(smplx_verts)   # (T, N_kps, 3)

    # index a specific MHR70 keypoint:
    l_ankle_smplx = kp_smplx[:, mapper.index(13)]
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

# ── MHR70 keypoints covered by this mapper ───────────────────────────────────
# Ordered list used for the regressor rows.
KP_ORDER: list[int] = [7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]

KP_NAMES: dict[int, str] = {
    7:  "L-elbow",
    8:  "R-elbow",
    9:  "L-hip",
    10: "R-hip",
    11: "L-knee",
    12: "R-knee",
    13: "L-ankle",
    14: "R-ankle",
    41: "R-wrist",
    62: "L-wrist",
    69: "neck",
}

# For joint-based KPs: which SMPLX joint index (from J_regressor) to use.
_MHR70_TO_SMPLX_JOINT: dict[int, int] = {
    9:  1,   # L-hip
    10: 2,   # R-hip
    11: 4,   # L-knee
    12: 5,   # R-knee
    41: 21,  # R-wrist
    62: 20,  # L-wrist
    69: 12,  # neck
}

# Vertex-based MHR70 KPs: weight comes 100% from MHR mesh vertices.
_VERTEX_KPS: frozenset[int] = frozenset({7, 8, 13, 14})

N_SMPLX_VERTS = 10475
N_MHR_VERTS   = 18439


class Mapper:
    """Regressor from posed SMPLX vertices to MHR70-equivalent landmark positions."""

    def __init__(self, R: np.ndarray) -> None:
        """
        Args:
            R: float32 array (N_kps, N_SMPLX_VERTS) — the precomputed regressor.
        """
        assert R.shape == (len(KP_ORDER), N_SMPLX_VERTS), (
            f"Expected R shape ({len(KP_ORDER)}, {N_SMPLX_VERTS}), got {R.shape}"
        )
        self._R = R.astype(np.float32)
        self._index: dict[int, int] = {k: i for i, k in enumerate(KP_ORDER)}

    # ── public interface ───────────────────────────────────────────────────────

    def map(self, smplx_verts: np.ndarray) -> np.ndarray:
        """
        Args:
            smplx_verts: (10475, 3) or (T, 10475, 3) — posed SMPLX vertices.
        Returns:
            (N_kps, 3) or (T, N_kps, 3) — SMPLX-equivalent positions per KP.
        """
        if smplx_verts.ndim == 2:
            return self._R @ smplx_verts                             # (N_kps, 3)
        return np.einsum("kv,tvd->tkd", self._R, smplx_verts)       # (T, N_kps, 3)

    def index(self, mhr70_idx: int) -> int:
        """Row index into the map() output for a given MHR70 keypoint."""
        return self._index[mhr70_idx]

    def row(self, mhr70_idx: int) -> np.ndarray:
        """Return the regressor row for a single MHR70 keypoint (N_SMPLX_VERTS,)."""
        return self._R[self._index[mhr70_idx]]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        np.save(str(path), self._R)
        print(f"Mapper regressor saved to {path}  shape={self._R.shape}")

    @classmethod
    def load(cls, path: str | Path) -> "Mapper":
        R = np.load(str(path))
        return cls(R)

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        ckpt_path: str | Path,
        smplx_pkl: str | Path,
        smplx2mhr_path: str | Path,
        verbose: bool = True,
    ) -> "Mapper":
        """
        Build regressor from scratch.

        Args:
            ckpt_path:      path to SAM3D model.ckpt (head_pose.keypoint_mapping).
            smplx_pkl:      path to SMPLX_NEUTRAL.pkl.
            smplx2mhr_path: path to smplx2mhr_mapping.npz.
            verbose:        print progress.
        """
        if verbose:
            print("Building Mapper regressor...")

        # ── SMPLX geometry ─────────────────────────────────────────────────
        with open(smplx_pkl, "rb") as f:
            smplx_data = pickle.load(f, encoding="latin1")

        smplx_faces = smplx_data["f"]          # (20908, 3) int
        J_reg = smplx_data["J_regressor"]
        if hasattr(J_reg, "toarray"):
            J_reg = J_reg.toarray()
        J_reg = np.asarray(J_reg, dtype=np.float32)  # (55, 10475)

        # ── smplx2mhr mapping ──────────────────────────────────────────────
        m = np.load(smplx2mhr_path)
        tri_ids = m["triangle_ids"]   # (18439,): SMPLX face idx for each MHR vertex
        baryc   = m["baryc_coords"]   # (18439, 3)

        def _smplx_row_for_mhr_vert(mhr_v: int) -> np.ndarray:
            """ℝ^{10475}: SMPLX vertex weights for MHR vertex mhr_v."""
            face = smplx_faces[tri_ids[mhr_v]]   # (3,) SMPLX vertex indices
            r = np.zeros(N_SMPLX_VERTS, dtype=np.float64)
            for j, vi in enumerate(face):
                r[vi] += baryc[mhr_v, j]
            return r

        # ── SAM3D keypoint_mapping ─────────────────────────────────────────
        if verbose:
            print(f"  Loading keypoint_mapping from {ckpt_path} ...")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        km = ckpt["head_pose.keypoint_mapping"].numpy()  # (308, 18566)

        # ── assemble regressor ─────────────────────────────────────────────
        R = np.zeros((len(KP_ORDER), N_SMPLX_VERTS), dtype=np.float64)

        for i, mhr_idx in enumerate(KP_ORDER):
            name = KP_NAMES[mhr_idx]

            if mhr_idx in _VERTEX_KPS:
                # Surface landmark: compose MHR vertex weights with smplx2mhr
                v_part = km[mhr_idx, :N_MHR_VERTS]
                mask = np.abs(v_part) > 1e-6
                vert_indices = np.where(mask)[0]
                vert_weights = v_part[mask]
                for vi, w in zip(vert_indices, vert_weights):
                    R[i] += w * _smplx_row_for_mhr_vert(int(vi))
                if verbose:
                    print(f"  [{mhr_idx:2d}] {name:10s}  vertex-based "
                          f"({mask.sum()} MHR verts, weight_sum={vert_weights.sum():.4f})")
            else:
                # Skeletal joint: use J_regressor row for corresponding SMPLX joint
                smplx_j = _MHR70_TO_SMPLX_JOINT[mhr_idx]
                R[i] = J_reg[smplx_j]
                if verbose:
                    print(f"  [{mhr_idx:2d}] {name:10s}  joint-based "
                          f"(SMPLX joint {smplx_j})")

        return cls(R.astype(np.float32))


# ── quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "MHR"))

    CKPT = ("/users/tnanni/.cache/huggingface/hub/"
            "models--facebook--sam-3d-body-dinov3/snapshots/"
            "11aaa346c7204874a1cbafe3d39a979080b2c55a/model.ckpt")
    SMPLX_PKL    = ROOT / "body_models/SMPLX_NEUTRAL.pkl"
    SMPLX2MHR    = ROOT / "MHR/tools/mhr_smpl_conversion/assets/smplx2mhr_mapping.npz"
    SAVE_PATH    = ROOT / "mappings/mhr_smplx_regressor.npy"

    mapper = Mapper.build(CKPT, SMPLX_PKL, SMPLX2MHR)
    mapper.save(SAVE_PATH)

    # Sanity check: apply to SMPLX T-pose vertices and compare to FK joints
    import pickle as pkl
    with open(SMPLX_PKL, "rb") as f:
        sd = pkl.load(f, encoding="latin1")
    v_tpose = sd["v_template"]                    # (10475, 3) — T-pose, zero betas
    J_reg = sd["J_regressor"]
    if hasattr(J_reg, "toarray"):
        J_reg = J_reg.toarray()
    fk_joints = np.asarray(J_reg) @ v_tpose      # (55, 3)

    kp_smplx = mapper.map(v_tpose)               # (N_kps, 3)

    print("\nSanity check on SMPLX T-pose vertices:")
    print(f"{'idx':>4s}  {'name':12s}  {'mapper (cm)':>30s}  {'FK joint (cm)':>30s}  {'|Δ| cm':>8s}")
    print("-" * 100)
    for i, mhr_idx in enumerate(KP_ORDER):
        from mapper import _MHR70_TO_SMPLX_JOINT, _VERTEX_KPS
        tag = "surf" if mhr_idx in _VERTEX_KPS else "fk  "
        mp = np.round(kp_smplx[i] * 100, 1)
        if mhr_idx in _MHR70_TO_SMPLX_JOINT:
            fk = np.round(fk_joints[_MHR70_TO_SMPLX_JOINT[mhr_idx]] * 100, 1)
            diff = np.linalg.norm(kp_smplx[i] - fk_joints[_MHR70_TO_SMPLX_JOINT[mhr_idx]]) * 100
        else:
            fk = np.array([float("nan")] * 3)
            diff = float("nan")
        print(f"[{mhr_idx:2d}]  {KP_NAMES[mhr_idx]:12s} [{tag}]  {str(mp):>30s}  {str(fk):>30s}  {diff:>8.2f}")
