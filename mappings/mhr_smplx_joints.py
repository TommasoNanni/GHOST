"""
Dry-run calibration: compute offsets between MHR70 keypoints and SMPLX joints in T-pose.

MHR70 keypoints are of two types (from keypoint_mapping in SAM3D checkpoint):
  - Joint-based (hip, knee, wrist, shoulder, neck): 100% from a single pymomentum
    skeleton joint. Both MHR and SMPLX define these as skeletal joint centers but
    from different skeleton models → systematic offset in body-local frame.
  - Vertex-based (elbow, ankle): weighted sum of MHR mesh vertices (surface landmark).
    The corresponding SMPLX point is computed via smplx2mhr_mapping.npz, which maps
    each MHR vertex to a face+barycentric on the SMPLX mesh.

Output: per-keypoint offset vector (in SMPLX meters, pelvis-aligned frame) that can
be applied at inference to get apples-to-apples comparisons.
"""

import sys
import numpy as np
import pickle
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MHR"))
import pymomentum.geometry as g

# ── Keypoint-mapping derived constants ────────────────────────────────────────
# MHR70 idx → single pymomentum joint index (weight=1.0)
MHR70_TO_PYM_JOINT = {
    5:  75,         # L-shoulder
    6:  39,         # R-shoulder
    9:   2,         # L-hip  (l_upleg)
    10: 18,         # R-hip  (r_upleg)
    11:  3,         # L-knee (l_lowleg)
    12: 19,         # R-knee (r_lowleg)
    41: 42,         # R-wrist
    62: 78,         # L-wrist
}
MHR70_NECK_JOINTS = (110, 111)  # neck = average of joints 110+111

# SMPLX joint indices for the same anatomical landmarks
MHR70_TO_SMPLX_JOINT = {
    5:  16,  # L-shoulder
    6:  17,  # R-shoulder
    9:   1,  # L-hip
    10:  2,  # R-hip
    11:  4,  # L-knee
    12:  5,  # R-knee
    41: 21,  # R-wrist
    62: 20,  # L-wrist
    69: 12,  # neck
}
JOINT_NAMES = {
    5: "L-shoulder", 6: "R-shoulder",
    9: "L-hip",      10: "R-hip",
    11: "L-knee",    12: "R-knee",
    41: "R-wrist",   62: "L-wrist",
    69: "neck",
}

# Vertex-based MHR70 keypoints (surface landmarks)
VERTEX_KP_NAMES = {7: "L-elbow", 8: "R-elbow", 13: "L-ankle", 14: "R-ankle"}
VERTEX_KP_SMPLX = {7: 18, 8: 19, 13: 7, 14: 8}  # SMPLX FK joint for comparison


# ─── 1. Load keypoint_mapping from SAM3D checkpoint ──────────────────────────
CKPT = ("/users/tnanni/.cache/huggingface/hub/"
        "models--facebook--sam-3d-body-dinov3/snapshots/"
        "11aaa346c7204874a1cbafe3d39a979080b2c55a/model.ckpt")

print("Loading SAM3D checkpoint...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
km = ckpt["head_pose.keypoint_mapping"].numpy()  # (308, 18566)
N_VERTS = 18439

vertex_kp_weights: dict[int, tuple[np.ndarray, np.ndarray]] = {}
for mhr_idx in VERTEX_KP_NAMES:
    v_part = km[mhr_idx, :N_VERTS]
    mask = np.abs(v_part) > 1e-6
    vertex_kp_weights[mhr_idx] = (np.where(mask)[0], v_part[mask])
    print(f"  [{mhr_idx:2d}] {VERTEX_KP_NAMES[mhr_idx]:10s}: {mask.sum()} vertices, "
          f"weight sum={v_part[mask].sum():.4f}")


# ─── 2. SMPLX T-pose joint positions ─────────────────────────────────────────
print("\nLoading SMPLX T-pose...")
with open(ROOT / "body_models/SMPLX_NEUTRAL.pkl", "rb") as f:
    smplx_data = pickle.load(f, encoding="latin1")

smplx_verts = smplx_data["v_template"]          # (10475, 3)
smplx_faces = smplx_data["f"]                   # (20908, 3)
J_reg = smplx_data["J_regressor"]
if hasattr(J_reg, "toarray"):
    J_reg = J_reg.toarray()
smplx_joints = J_reg @ smplx_verts              # (55, 3)

smplx_pelvis = (smplx_joints[1] + smplx_joints[2]) / 2  # midpoint L/R hip


# ─── 3. MHR T-pose joint positions via pymomentum ────────────────────────────
print("Loading MHR T-pose...")
ch = g.Character.load_fbx(
    str(ROOT / "MHR/assets/lod0.fbx"),
    str(ROOT / "MHR/assets/compact_v6_1.model"),
    load_blendshapes=False,
)
sk = ch.skeleton
parents = np.array([max(j.parent, 0) for j in sk.joints], dtype=np.int32)
offsets = np.array([j.translation_offset for j in sk.joints], dtype=np.float64)
mhr_joints = g.model_parameters_to_positions(ch, np.zeros(ch.parameter_transform.size),
                                              parents, offsets)  # (127, 3)

mhr_pelvis = (mhr_joints[2] + mhr_joints[18]) / 2  # midpoint l_upleg / r_upleg

# Scale: match hip-to-hip width
smplx_hip_w = np.linalg.norm(smplx_joints[1] - smplx_joints[2])
mhr_hip_w   = np.linalg.norm(mhr_joints[2]   - mhr_joints[18])
scale = smplx_hip_w / mhr_hip_w   # MHR units → SMPLX metres

print(f"  MHR hip width : {mhr_hip_w:.4f} units  →  {mhr_hip_w*scale*100:.2f} cm")
print(f"  SMPLX hip width: {smplx_hip_w*100:.2f} cm")
print(f"  Scale (MHR→SMPLX): {scale:.6f} m/unit")


# ─── 4. smplx2mhr_mapping: MHR vertex → SMPLX surface point ─────────────────
print("\nLoading smplx2mhr_mapping...")
mapping = np.load(ROOT / "MHR/tools/mhr_smpl_conversion/assets/smplx2mhr_mapping.npz")
tri_ids = mapping["triangle_ids"]   # (18439,): SMPLX face idx for each MHR vertex
baryc   = mapping["baryc_coords"]   # (18439, 3)


def mhr_vert_to_smplx_surface(mhr_v: int) -> np.ndarray:
    """Return SMPLX surface point (in SMPLX metres) for MHR vertex index mhr_v."""
    face = smplx_faces[tri_ids[mhr_v]]           # (3,) vertex indices
    return baryc[mhr_v] @ smplx_verts[face]      # (3,)


def to_smplx_frame(pos_smplx: np.ndarray) -> np.ndarray:
    return pos_smplx - smplx_pelvis

def mhr_to_smplx_frame(pos_mhr: np.ndarray) -> np.ndarray:
    return (pos_mhr - mhr_pelvis) * scale


# ─── 5. Joint-based offsets ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("JOINT-BASED KEYPOINTS  (pymomentum joint  vs  SMPLX FK joint)")
print("=" * 72)
print(f"{'':4s} {'Name':12s} {'MHR pos (cm)':>30s}  {'SMPLX pos (cm)':>30s}  {'|offset| cm':>10s}")
print("-" * 100)

joint_offsets: dict[int, np.ndarray] = {}

for mhr_idx, name in JOINT_NAMES.items():
    smplx_idx = MHR70_TO_SMPLX_JOINT[mhr_idx]

    if mhr_idx == 69:
        mhr_pos_raw = (mhr_joints[MHR70_NECK_JOINTS[0]] + mhr_joints[MHR70_NECK_JOINTS[1]]) / 2
    else:
        mhr_pos_raw = mhr_joints[MHR70_TO_PYM_JOINT[mhr_idx]]

    mhr_pos_aligned   = mhr_to_smplx_frame(mhr_pos_raw)
    smplx_pos_aligned = to_smplx_frame(smplx_joints[smplx_idx])

    offset = smplx_pos_aligned - mhr_pos_aligned
    dist   = np.linalg.norm(offset) * 100

    joint_offsets[mhr_idx] = offset  # in SMPLX metres, pelvis-centred

    mp = np.round(mhr_pos_aligned * 100, 1)
    sp = np.round(smplx_pos_aligned * 100, 1)
    print(f"[{mhr_idx:2d}] {name:12s}  MHR={mp}  SMPLX={sp}  |Δ|={dist:.2f}cm  dir={np.round(offset*100,1)}")


# ─── 6. Vertex-based offsets (elbow, ankle) ───────────────────────────────────
print("\n" + "=" * 72)
print("VERTEX-BASED KEYPOINTS  (MHR surface cluster  vs  SMPLX surface & FK)")
print("=" * 72)

vertex_offsets: dict[int, np.ndarray] = {}

for mhr_idx, name in VERTEX_KP_NAMES.items():
    vert_indices, vert_weights = vertex_kp_weights[mhr_idx]
    smplx_fk_idx = VERTEX_KP_SMPLX[mhr_idx]

    # SMPLX surface point corresponding to this MHR vertex cluster
    smplx_surf = np.zeros(3)
    for vi, w in zip(vert_indices, vert_weights):
        smplx_surf += w * mhr_vert_to_smplx_surface(vi)

    smplx_surf_aligned = to_smplx_frame(smplx_surf)
    smplx_fk_aligned   = to_smplx_frame(smplx_joints[smplx_fk_idx])

    # Offset: how far is the SMPLX FK joint from the true surface landmark
    offset_surf_to_fk = smplx_fk_aligned - smplx_surf_aligned
    dist_surf_to_fk   = np.linalg.norm(offset_surf_to_fk) * 100

    vertex_offsets[mhr_idx] = offset_surf_to_fk

    print(f"\n[{mhr_idx:2d}] {name}")
    print(f"     SMPLX surface point (from MHR cluster): {np.round(smplx_surf_aligned*100,1)} cm")
    print(f"     SMPLX FK joint:                         {np.round(smplx_fk_aligned*100,1)} cm")
    print(f"     surface → FK offset:  |Δ|={dist_surf_to_fk:.2f}cm  dir={np.round(offset_surf_to_fk*100,1)} cm")

# ─── 7. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY: offsets to add to SMPLX FK joint to match MHR70 landmark")
print("(in SMPLX metres, pelvis-centred frame)")
print("=" * 72)
for mhr_idx, name in {**JOINT_NAMES, **VERTEX_KP_NAMES}.items():
    if mhr_idx in joint_offsets:
        off = joint_offsets[mhr_idx]
        tag = "joint"
    else:
        off = -vertex_offsets[mhr_idx]  # negate: FK→surface
        tag = "vertex"
    print(f"  [{mhr_idx:2d}] {name:12s} ({tag:6s}): {np.round(off*100,2)} cm  "
          f"|Δ|={np.linalg.norm(off)*100:.2f}cm")
