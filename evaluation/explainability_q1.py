"""Explainability Q1 — does the fusion module select views, or just average?

Corrupt one camera's pose by an exact geodesic angle `delta`, sweep delta, and
measure how far the FUSED OUTPUT moves. That curve is Tukey's finite-sample
sensitivity curve (the finite-K analogue of Hampel's influence function; with
K = 4-8 cameras the asymptotic influence function does not apply).

Sensitivity is output DISPLACEMENT, not error, so no ground truth is needed:

    S(delta) = geodesic_angle( f(clean), f(camera k0 corrupted by delta) )

Read the shape against two reference estimators run on the same inputs:

    straight line, slope ~1/K   -> the model is averaging          (Q1 negative)
    rises then flattens         -> learned robustness              (Q1 positive)
    rises then descends         -> it rejects bad views            (strongest)

Note on the textbook criterion: gross-error sensitivity (sup of S) is useless
here, because SO(3) is compact so S <= 180 degrees for EVERY estimator,
including the plain average. Shape versus the references is the honest test.

Loaders are copied from evaluation/evaluate_rich.py rather than imported, so
this script pulls in no BodyPlacer/pymomentum dependency and cannot drift if
that file is refactored. The tensors it builds are byte-identical to the ones
the production evaluation feeds the model.

Usage
-----
    pixi run python evaluation/explainability_q1.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --checkpoint        checkpoints/fusion_module/best.pt \\
        --max_scenes 10 --out eval_explainability/q1_rich.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.fusion_module_v2 import PoseFusionModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Sweep stops at 80 degrees on purpose. SO(3) is compact, so past ~90 degrees a
# corruption stops being "further away" and starts folding back: a plain chordal
# mean measured at 160 degrees MOVES LESS than at 80 (verified: 9.6 vs 17.3 deg),
# which would masquerade as outlier rejection. Everything below 90 is monotone
# and interpretable.
_DEFAULT_DELTAS = [0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0]


# ---------------------------------------------------------------------------
# Rotation helpers (6-D convention = first two ROWS of the matrix, matching
# _aa_to_6d in evaluate_rich.py and convert_pose in data/fusion_dataset.py)
# ---------------------------------------------------------------------------

def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3). Gram-Schmidt on the two rows."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_sixd(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6). First two rows."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def geodesic_deg(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Angle in degrees between two batches of rotations, (...,3,3) -> (...)."""
    tr = torch.einsum("...ij,...ij->...", A, B)          # trace(A^T B)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _hat(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric cross-product matrix."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def exp_map(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vector -> (..., 3, 3) rotation matrix (Rodrigues)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u = w / theta
    U = _hat(u)
    th = theta[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
    return I + torch.sin(th) * U + (1.0 - torch.cos(th)) * (U @ U)


def log_map(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) rotation vector."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    denom = (2.0 * torch.sin(theta)).clamp(min=1e-8)[..., None]
    small = (theta < 1e-6)[..., None]
    return torch.where(small, 0.5 * axis, axis / denom * theta[..., None])


# ---------------------------------------------------------------------------
# Reference estimators — both map K rotations -> 1 rotation
# ---------------------------------------------------------------------------

def chordal_mean(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Closed-form chordal (L2) rotation average.

    R : (N, K, 3, 3)   w : (N, K) visibility weights (0/1)
    -> (N, 3, 3)
    """
    M = (R * w[..., None, None]).sum(dim=1)                 # (N,3,3)
    U, _, Vt = torch.linalg.svd(M.double())
    d = torch.linalg.det(U @ Vt)
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return (U @ D @ Vt).to(R.dtype)


def geodesic_median(R: torch.Tensor, w: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Geodesic (L1) median via Weiszfeld in the tangent space.

    Weights are 1/distance, so a camera that drifts far away loses influence —
    this is exactly why the median's sensitivity curve saturates.
    """
    m = chordal_mean(R, w)
    for _ in range(iters):
        v = log_map(m[:, None].transpose(-2, -1) @ R)       # (N,K,3) tangent vectors
        d = v.norm(dim=-1).clamp(min=1e-6)                  # (N,K)
        ww = w / d
        denom = ww.sum(dim=1, keepdim=True).clamp(min=1e-12)
        step = (ww[..., None] * v).sum(dim=1) / denom       # (N,3)
        m = m @ exp_map(step)
    return m


# ---------------------------------------------------------------------------
# Data loading — copied verbatim from evaluation/evaluate_rich.py
# ---------------------------------------------------------------------------

def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> 6D (..., 6), first two rows of the matrix."""
    shape = aa.shape[:-1]
    try:
        mats = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()
    except Exception:
        return np.zeros(shape + (6,), dtype=np.float32)
    sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)
    return sixd.reshape(shape + (6,)).astype(np.float32)


def load_scene_body_data(scene_dir: Path) -> tuple[list[Path], list[dict[int, dict]]]:
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())

    # Drop cameras absent from the VGGT npz, matching BodyPlacer._cam_dirs.
    _npz = scene_dir / "vggt_cameras_centered.npz"
    if _npz.exists():
        _z = np.load(_npz, allow_pickle=True)
        _names = {(n.decode() if isinstance(n, bytes) else str(n))
                  for n in _z["camera_names"]}
        cam_dirs = [d for d in cam_dirs if d.name in _names]

    raw: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_persons: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            data = np.load(npz_path, allow_pickle=False)
            cam_persons[pid] = {k: data[k] for k in data.files}
        raw.append(cam_persons)
    return cam_dirs, raw


def build_fusion_tensors(raw, num_joints: int = 55):
    """-> pose (1,T,K,P,54,6), mask (1,T,K,P). Same packing as the production eval."""
    all_pids = sorted({pid for cam in raw for pid in cam})
    all_frames = sorted({int(fi) for cam in raw for pd in cam.values()
                         for fi in pd["frame_indices"]})
    if not all_pids or not all_frames:
        raise RuntimeError("No person data found.")

    frame_start = all_frames[0]
    T = all_frames[-1] + 1 - frame_start
    K, P = len(raw), len(all_pids)
    J = num_joints - 1
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    pose_arr = np.zeros((T, K, P, J, 6), dtype=np.float32)
    mask_arr = np.zeros((T, K, P), dtype=np.float32)

    for k, cam in enumerate(raw):
        for pid, pdata in cam.items():
            p = pid_to_slot[pid]
            fi = pdata["frame_indices"].astype(int)
            go = pdata.get("smplx_global_orient")
            bp = pdata.get("smplx_body_pose")
            if go is None or bp is None:
                continue
            lh = pdata.get("smplx_left_hand_pose")
            rh = pdata.get("smplx_right_hand_pose")
            for local_t, global_t in enumerate(fi):
                t = int(global_t) - frame_start
                if t < 0 or t >= T:
                    continue
                parts = [go[local_t].reshape(1, 3), bp[local_t].reshape(21, 3)]
                if lh is not None:
                    parts.append(lh[local_t].reshape(15, 3))
                if rh is not None:
                    parts.append(rh[local_t].reshape(15, 3))
                aa = np.concatenate(parts, axis=0)
                if aa.shape[0] < num_joints:
                    aa = np.concatenate(
                        [aa, np.zeros((num_joints - aa.shape[0], 3), dtype=np.float32)], 0)
                pose_arr[t, k, p] = _aa_to_6d(aa)[1:]
                mask_arr[t, k, p] = 1.0

    return (torch.from_numpy(pose_arr).unsqueeze(0),
            torch.from_numpy(mask_arr).unsqueeze(0))


def load_fusion_model(checkpoint: Path, device: torch.device,
                      kintree_k: int | None = None) -> PoseFusionModule:
    """Rebuild the module from a checkpoint.

    kintree_mask_k is a NON-PERSISTENT buffer, so it is absent from state_dict:
    a checkpoint alone cannot say whether the model was trained with the
    kinematic-tree attention mask. Newer checkpoints record it in model_config;
    older ones (anything before that fix, including the first R2 runs) do not, so
    `kintree_k` overrides explicitly. Getting this wrong silently evaluates a
    masked-trained model with its mask removed.
    """
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    emb_dim = state["joint_id_embedding.weight"].shape[1]
    n_joints = state["joint_id_embedding.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("layers.") and k.endswith(".ff.norm.weight"))
    max_tlen = state["temporal_pe.pe"].shape[0]
    cfg = ckpt.get("model_config") or {}
    k = kintree_k if kintree_k is not None else cfg.get("kintree_mask_k")
    if k is not None and k < 0:
        k = None
    model = PoseFusionModule(embedding_dim=emb_dim, num_layers=n_layers,
                             num_joints=n_joints, max_temporal_len=max_tlen,
                             num_heads=cfg.get("num_heads", 8),
                             temporal_window=cfg.get("temporal_window", 128),
                             kintree_mask_k=k).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(
        f"loaded {checkpoint.name}: emb={emb_dim} layers={n_layers} joints={n_joints} "
        f"kintree_mask_k={k}"
        + ("  (from --kintree_k)" if kintree_k is not None
           else "  (from model_config)" if "kintree_mask_k" in cfg
           else "  (ABSENT from checkpoint -> assuming UNMASKED)")
    )
    return model


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

def corrupt_camera(pose: torch.Tensor, k0: int, delta_deg: float,
                   generator: torch.Generator) -> torch.Tensor:
    """Rotate camera k0's every joint by exactly `delta_deg` about a random axis.

    pose : (1, T, K, P, J, 6)  ->  copy with camera k0 corrupted.
    A fresh axis is drawn per (t, p, j); the magnitude is exact by construction,
    which is why we compose a rotation rather than adding noise.
    """
    if delta_deg == 0.0:
        return pose
    out = pose.clone()
    sl = out[:, :, k0]                                   # (1,T,P,J,6)
    R = sixd_to_matrix(sl)                               # (1,T,P,J,3,3)

    axis = torch.randn(*R.shape[:-2], 3, generator=generator,
                       device=pose.device, dtype=pose.dtype)
    axis = torch.nn.functional.normalize(axis, dim=-1)
    E = exp_map(axis * np.deg2rad(delta_deg))            # (1,T,P,J,3,3)

    out[:, :, k0] = matrix_to_sixd(R @ E)
    return out


# ---------------------------------------------------------------------------
# Per-scene sensitivity curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def scene_sensitivity(model, pose, mask, deltas, device, seed=0):
    """Return {'model'|'mean'|'median': {delta: mean displacement in degrees}}.

    Only slots seen by >= 2 cameras are scored: with a single camera there is
    nothing to fuse and the displacement is meaningless.
    """
    pose, mask = pose.to(device), mask.to(device)
    _, T, K, P, J, _ = pose.shape

    # (T,P,J) slots with at least 2 visible cameras
    n_vis = mask[0].sum(dim=1)                                   # (T,P)
    scored = (n_vis >= 2)                                        # (T,P)
    if not scored.any():
        return None
    scored_j = scored[..., None].expand(T, P, J)                 # (T,P,J)

    w = mask[0].permute(0, 2, 1).reshape(T * P, K)               # (T*P,K)

    def _refs(p):
        """Chordal mean and geodesic median of the K cameras, per (t,p,j)."""
        R = sixd_to_matrix(p[0]).permute(0, 2, 3, 1, 4, 5)        # (T,P,J,K,3,3)
        R = R.reshape(T * P * J, K, 3, 3)
        ww = w[:, None, :].expand(T * P, J, K).reshape(T * P * J, K)
        return (chordal_mean(R, ww).reshape(T, P, J, 3, 3),
                geodesic_median(R, ww).reshape(T, P, J, 3, 3))

    Y0_model = sixd_to_matrix(model(pose, mask)[0])               # (T,P,J,3,3)
    Y0_mean, Y0_med = _refs(pose)

    out = {"model": {}, "mean": {}, "median": {}}
    for delta in deltas:
        acc = {"model": [], "mean": [], "median": []}
        for k0 in range(K):
            if mask[0, :, k0].sum() == 0:                        # camera never present
                continue
            gen = torch.Generator(device=device).manual_seed(seed + k0)
            pd = corrupt_camera(pose, k0, delta, gen)

            Yd_model = sixd_to_matrix(model(pd, mask)[0])
            Yd_mean, Yd_med = _refs(pd)

            for key, a, b in (("model", Y0_model, Yd_model),
                              ("mean", Y0_mean, Yd_mean),
                              ("median", Y0_med, Yd_med)):
                d = geodesic_deg(a, b)[scored_j]
                acc[key].append(d.mean().item())

        for key in out:
            out[key][delta] = float(np.mean(acc[key])) if acc[key] else float("nan")
    return out


# ---------------------------------------------------------------------------
# Curve summaries
# ---------------------------------------------------------------------------

@torch.no_grad()
def prep_scene(dp, device) -> dict | None:
    """Load one scene's tensors and the GT-scoring masks.

    Joints 1..54 are body-relative rotations, so GT comparison needs no world
    alignment. Scoring masks cover all 54 packed slots and the 21 body joints
    alone (slots 0..20 after the root is dropped); hand slots are noisy and
    slots 51..53 are zero padding.
    """
    from data.fusion_dataset import RICHFusionDataset

    loader = torch.utils.data.DataLoader(RICHFusionDataset([dp]), batch_size=1)
    for inputs, targets in loader:
        pose = inputs["pose"].to(device).float()            # (1,T,K,P,55,6)
        pmask = inputs["person_mask"].to(device).float()    # (1,T,K,P)
        gt = targets["pose"].to(device).float()             # (1,T,P,55,6)
        gt_valid = targets["gt_valid"].to(device).bool()    # (1,T,P)

        _, T, K, P, _, _ = pose.shape
        R_gt = sixd_to_matrix(gt[0])[:, :, 1:]              # (T,P,54,3,3)
        J = R_gt.shape[2]

        keep = gt_valid[0] & (pmask[0].sum(dim=1) >= 2)     # (T,P)
        if not keep.any():
            return None
        sel = keep[..., None].expand(T, P, J)
        body = torch.zeros(J, dtype=torch.bool, device=device)
        body[:21] = True
        return {"pose": pose, "pmask": pmask, "R_gt": R_gt, "sel": sel,
                "sel_body": sel & body, "T": T, "K": K, "P": P, "J": J}
    return None


@torch.no_grad()
def mpjre(R: torch.Tensor, s: dict) -> dict[str, float]:
    d = geodesic_deg(R, s["R_gt"])
    return {"all": d[s["sel"]].mean().item(), "body": d[s["sel_body"]].mean().item()}


@torch.no_grad()
def score_scene_vs_gt(model, s: dict) -> dict[str, dict[str, float]]:
    """MPJRE for model / chordal mean / geodesic median. Answers 'is it just a mean?'."""
    T, K, P, J = s["T"], s["K"], s["P"], s["J"]
    Rin = sixd_to_matrix(s["pose"][0][..., 1:, :])          # (T,K,P,54,3,3)
    Rin = Rin.permute(0, 2, 3, 1, 4, 5).reshape(-1, K, 3, 3)
    w = s["pmask"][0].permute(0, 2, 1)[:, :, None, :].expand(T, P, J, K).reshape(-1, K)
    return {
        "model": mpjre(sixd_to_matrix(model(s["pose"], s["pmask"])[0]), s),
        "mean": mpjre(chordal_mean(Rin, w).reshape(T, P, J, 3, 3), s),
        "median": mpjre(geodesic_median(Rin, w).reshape(T, P, J, 3, 3), s),
    }


# SMPL-X body kinematic tree, joints 0..21 (parent of each). Used to test whether
# the learned joint coupling matches real skeletal structure.
_SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
_N_BODY = 21          # packed slots 0..20 == SMPL-X joints 1..21 (root dropped)


def _tree_hops() -> np.ndarray:
    """(21,21) hop distance between SMPL-X body joints 1..21 along the kinematic tree."""
    n = len(_SMPLX_PARENTS)
    inf = 10**6
    d = np.full((n, n), inf, dtype=np.int64)
    np.fill_diagonal(d, 0)
    for j, p in enumerate(_SMPLX_PARENTS):
        if p >= 0:
            d[j, p] = d[p, j] = 1
    for k in range(n):                                    # Floyd-Warshall
        d = np.minimum(d, d[:, k, None] + d[None, k, :])
    return d[1:, 1:]                                      # drop root -> packed slots 0..20


def corrupt_joint(pose: torch.Tensor, j0: int, delta_deg: float,
                  generator: torch.Generator) -> torch.Tensor:
    """Rotate joint j0 by exactly `delta_deg` in EVERY camera (fresh axis per t,k,p)."""
    out = pose.clone()
    sl = out[..., j0, :]                                  # (1,T,K,P,6)
    R = sixd_to_matrix(sl)
    axis = torch.randn(*R.shape[:-2], 3, generator=generator,
                       device=pose.device, dtype=pose.dtype)
    axis = torch.nn.functional.normalize(axis, dim=-1)
    out[..., j0, :] = matrix_to_sixd(R @ exp_map(axis * np.deg2rad(delta_deg)))
    return out


@torch.no_grad()
def joint_influence(model, pose, mask, delta, device, seed=0) -> np.ndarray | None:
    """(J,J) causal influence: row j0 = how far each output joint moves when input
    joint j0 is corrupted by `delta` in every camera."""
    pose, mask = pose.to(device), mask.to(device)
    _, T, K, P, J, _ = pose.shape
    scored = (mask[0].sum(dim=1) >= 2)                    # (T,P)
    if not scored.any():
        return None
    sel = scored[..., None].expand(T, P, J)

    Y0 = sixd_to_matrix(model(pose, mask)[0])             # (T,P,J,3,3)
    M = np.zeros((J, J), dtype=np.float32)
    for j0 in range(J):
        gen = torch.Generator(device=device).manual_seed(seed + j0)
        Yd = sixd_to_matrix(model(corrupt_joint(pose, j0, delta, gen), mask)[0])
        d = geodesic_deg(Y0, Yd)                          # (T,P,J)
        d = torch.where(sel, d, torch.zeros_like(d))
        M[j0] = (d.sum(dim=(0, 1)) / scored.sum().clamp(min=1)).cpu().numpy()
    return M


def analyse_joint_matrix(M: np.ndarray) -> dict:
    """Locality and skeletal structure of the influence matrix (body joints only)."""
    B = M[:_N_BODY, :_N_BODY].astype(np.float64)
    row = B.sum(axis=1)
    diag = np.diag(B)
    self_frac = float(np.mean(diag / np.maximum(row, 1e-12)))

    off = ~np.eye(_N_BODY, dtype=bool)
    hops = _tree_hops()
    infl = B[off]
    hp = hops[off].astype(np.float64)

    # row-normalised off-diagonal mass, so scenes/joints with big totals don't dominate
    Bn = B / np.maximum(row[:, None], 1e-12)
    adj = hops == 1
    adj_mass = float(Bn[adj & off].sum() / max(Bn[off].sum(), 1e-12))
    chance = float(adj[off].sum() / off.sum())

    # Spearman via rank correlation, no scipy dependency
    def _rank(x):
        o = x.argsort()
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(len(x))
        return r
    ri, rh = _rank(infl), _rank(hp)
    rho = float(np.corrcoef(ri, rh)[0, 1])

    return {"self_fraction": self_frac, "adjacent_mass": adj_mass,
            "adjacent_chance": chance, "adjacent_lift": adj_mass / max(chance, 1e-12),
            "spearman_influence_vs_hops": rho}


# Packed slots 0..20 == SMPL-X body joints 1..21.
_BODY_NAMES = ["L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2",
               "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot", "neck",
               "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder",
               "L_elbow", "R_elbow", "L_wrist", "R_wrist"]
_MIRROR = [(0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20)]
_CHAINS = {"L_leg": [0, 3, 6, 9], "R_leg": [1, 4, 7, 10], "spine": [2, 5, 8, 11, 14],
           "L_arm": [12, 15, 17, 19], "R_arm": [13, 16, 18, 20]}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(v):
        o = v.argsort(); r = np.empty_like(o, dtype=np.float64); r[o] = np.arange(len(v))
        return r
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def correlate_structures(M: np.ndarray) -> dict:
    """Test COMPETING structural hypotheses against the joint-influence matrix.

    A single correlation against the kinematic tree only tests one hypothesis. If the
    coupling is organised by something else (mirror symmetry, limb membership), that
    would be real structure too — and would show up here instead.
    """
    B = M[:_N_BODY, :_N_BODY].astype(np.float64)
    row = B.sum(axis=1, keepdims=True)
    Bn = B / np.maximum(row, 1e-12)                     # row-normalised
    off = ~np.eye(_N_BODY, dtype=bool)
    hops = _tree_hops()
    infl = Bn[off]

    mirror = np.zeros((_N_BODY, _N_BODY), dtype=bool)
    for a, b in _MIRROR:
        mirror[a, b] = mirror[b, a] = True

    chain = np.zeros((_N_BODY, _N_BODY), dtype=bool)
    for idx in _CHAINS.values():
        for a in idx:
            for b in idx:
                if a != b:
                    chain[a, b] = True

    def _lift(mask):
        m = mask & off
        if not m.any():
            return float("nan"), float("nan")
        sel = Bn[m].sum() / max(Bn[off].sum(), 1e-12)
        return float(sel), float(sel / (m.sum() / off.sum()))

    adj_mass, adj_lift = _lift(hops == 1)
    mir_mass, mir_lift = _lift(mirror)
    chn_mass, chn_lift = _lift(chain)

    # influence grouped by hop distance
    by_hop = {}
    for h in range(1, int(hops[off].max()) + 1):
        m = (hops == h) & off
        if m.any():
            by_hop[h] = float(Bn[m].mean())

    return {
        "spearman_vs_hops": _spearman(infl, hops[off].astype(np.float64)),
        "pearson_vs_hops": float(np.corrcoef(infl, hops[off].astype(np.float64))[0, 1]),
        "adjacent": {"mass": adj_mass, "lift": adj_lift},
        "mirror": {"mass": mir_mass, "lift": mir_lift},
        "same_chain": {"mass": chn_mass, "lift": chn_lift},
        "influence_by_hop": by_hop,
        "matrix_symmetry": float(np.corrcoef(B[off], B.T[off])[0, 1]),
        "self_fraction": float(np.mean(np.diag(B) / np.maximum(row[:, 0], 1e-12))),
    }


def print_structure_report(M: np.ndarray, n_scenes: int) -> dict:
    a = correlate_structures(M)
    B = M[:_N_BODY, :_N_BODY]
    Bn = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-12)
    hops = _tree_hops()

    print(f"\n{'='*72}")
    print(f"Q2 structure correlations — {n_scenes} scenes")
    print(f"{'='*72}")
    print(f"  self influence fraction        {a['self_fraction']:.3f}")
    print(f"  matrix symmetry  corr(M,M^T)   {a['matrix_symmetry']:+.3f}")
    print(f"  Spearman influence vs hops     {a['spearman_vs_hops']:+.3f}")
    print(f"  Pearson  influence vs hops     {a['pearson_vs_hops']:+.3f}")
    print(f"\n  {'hypothesis':<16}{'mass':>10}{'lift vs chance':>18}")
    for k in ("adjacent", "mirror", "same_chain"):
        print(f"  {k:<16}{a[k]['mass']:>10.3f}{a[k]['lift']:>18.2f}x")
    print(f"\n  mean off-diagonal influence by tree distance:")
    for h, v in sorted(a["influence_by_hop"].items()):
        print(f"    {h} hop{'s' if h > 1 else ' '}  {v:.5f}")

    # Rank-1 test: if the off-diagonal is well approximated by outer(a, b), then the
    # influence of i on j depends only on "how much i leaks" times "how susceptible j
    # is" — no pairwise structure whatsoever. Genuine (even non-skeletal) coupling
    # would need higher rank.
    O = B.copy().astype(np.float64)
    np.fill_diagonal(O, 0.0)
    sv = np.linalg.svd(O, compute_uv=False)
    energy = float(sv[0] ** 2 / max((sv ** 2).sum(), 1e-12))
    print(f"\n  rank-1 energy of off-diagonal   {energy:.3f}"
          "   (1.0 = influence(i,j) = leak(i) x susceptibility(j), no pairing)")

    out_infl = O.sum(axis=1)
    in_susc = O.sum(axis=0)
    order = np.argsort(-in_susc)
    print(f"\n  per-joint profile (most susceptible first):")
    print(f"    {'joint':>11}{'susceptibility':>16}{'outgoing':>11}{'self':>9}")
    for i in order[:8]:
        print(f"    {_BODY_NAMES[i]:>11}{in_susc[i]:>16.4f}{out_infl[i]:>11.4f}"
              f"{B[i, i]:>9.3f}")

    print(f"\n  strongest off-diagonal couplings (row-normalised):")
    off = ~np.eye(_N_BODY, dtype=bool)
    flat = [(Bn[i, j], i, j) for i in range(_N_BODY) for j in range(_N_BODY) if off[i, j]]
    for v, i, j in sorted(flat, reverse=True)[:10]:
        print(f"    {_BODY_NAMES[i]:>11} -> {_BODY_NAMES[j]:<11} {v:.4f}"
              f"   ({hops[i, j]} hops)")
    print("=" * 72)
    return a


def set_temporal_window(model, W: int) -> None:
    """Truncate every layer's temporal attention window (read at forward time)."""
    for layer in model.layers:
        layer.temporal_attn.temporal_window = W


@torch.no_grad()
def temporal_sweep(model, s: dict, windows: list[int]) -> dict[int, dict[str, float]]:
    """MPJRE vs temporal window. W=0 leaves only the diagonal unmasked, i.e. each
    frame attends to itself alone — temporal modelling fully disabled."""
    out = {}
    for W in windows:
        set_temporal_window(model, W)
        out[W] = mpjre(sixd_to_matrix(model(s["pose"], s["pmask"])[0]), s)
    return out


def summarise(curve: dict[float, float], K: int) -> dict[str, float]:
    """Two wraparound-robust shape statistics.

    slope_x_K : local sensitivity times K. A plain average shifts by delta/K, so
                this is ~1.0 for pure averaging and below 1.0 if the model already
                discounts the outlier at small corruptions.

    linearity : S(delta_max) divided by the straight line through the origin with
                the measured initial slope. 1.0 = never stops caring (averaging),
                well below 1.0 = saturates (robust). Preferred over "sup of S"
                because it cannot be faked by SO(3) wraparound.
    """
    ds = sorted(curve)
    slope = curve[ds[1]] / ds[1] if len(ds) > 1 and ds[1] > 0 else float("nan")
    d_max = ds[-1]
    linear_extrap = slope * d_max
    return {
        "slope_at_0": slope,
        "slope_x_K": slope * K,
        "at_max": curve[d_max],
        "linearity": curve[d_max] / linear_extrap if linear_extrap > 1e-9 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_output_root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--deltas", default=",".join(str(d) for d in _DEFAULT_DELTAS),
                    help="comma-separated corruption angles in degrees")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--scenes", default="", help="comma-separated scene names")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="truncate T to bound runtime (K+1 forwards per delta)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kintree_k", type=int, default=None,
                    help="Hop radius of the kinematic-tree attention mask the "
                         "checkpoint was TRAINED with. Negative = no mask. Omit to "
                         "read model_config; checkpoints written before that field "
                         "existed (e.g. early R2) MUST pass it explicitly.")
    ap.add_argument("--out", type=Path, default=Path("eval_explainability/q1.json"))
    ap.add_argument("--mode", choices=["sensitivity", "accuracy", "temporal", "joint"],
                    default="sensitivity",
                    help="sensitivity: corruption sweep (no GT). "
                         "accuracy: MPJRE of model/mean/median vs GT. "
                         "temporal: MPJRE vs truncated temporal window (Q3). "
                         "joint: causal joint-to-joint influence matrix (Q2, no GT).")
    ap.add_argument("--joint_delta", type=float, default=40.0,
                    help="joint mode: corruption angle applied to one joint")
    ap.add_argument("--from_json", type=Path, default=None,
                    help="joint mode: re-analyse a saved influence matrix (no GPU)")
    ap.add_argument("--windows", default="0,1,2,4,8,16,32,64,128",
                    help="temporal mode: comma-separated attention half-widths")
    ap.add_argument("--rich_data_root", type=Path, default=None,
                    help="accuracy mode: RICH split root, e.g. .../rich/centered_test")
    ap.add_argument("--rich_gt_dir", type=Path, default=None,
                    help="accuracy mode: RICH GT dir, e.g. .../datasets/rich")
    ap.add_argument("--body_split", default="test_body",
                    help="accuracy mode: GT body split")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    device = torch.device(args.device)
    model = load_fusion_model(args.checkpoint, device, kintree_k=args.kintree_k)

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scene_dirs = sorted(d for d in args.ghost_output_root.iterdir() if d.is_dir())
    if wanted:
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]

    if args.mode in ("accuracy", "temporal"):
        if args.rich_data_root is None or args.rich_gt_dir is None:
            ap.error(f"--mode {args.mode} needs --rich_data_root and --rich_gt_dir")
        from data.fusion_dataset import RICHFusionDatapoint

        windows = [int(x) for x in args.windows.split(",") if x.strip()]
        rows = []
        for sd in scene_dirs:
            try:
                dp = RICHFusionDatapoint(
                    scene_dir=sd, rich_data_root=args.rich_data_root,
                    rich_gt_dir=args.rich_gt_dir, body_split=args.body_split,
                    restrict_to_gt_persons=True,
                )
                if dp.num_frames == 0 or not dp.has_gt:
                    logger.warning(f"{sd.name}: skipped (no frames / no GT)")
                    continue
                s = prep_scene(dp, device)
                if s is None:
                    continue
                if args.mode == "accuracy":
                    r = score_scene_vs_gt(model, s)
                    logger.info(f"{sd.name}  body MPJRE  model={r['model']['body']:.2f}  "
                                f"mean={r['mean']['body']:.2f}  median={r['median']['body']:.2f}")
                else:
                    sw = temporal_sweep(model, s, windows)
                    r = {str(W): sw[W] for W in windows}
                    logger.info(f"{sd.name}  body MPJRE by W  "
                                + "  ".join(f"{W}:{sw[W]['body']:.2f}" for W in windows))
            except Exception as e:
                logger.warning(f"{sd.name}: skipped ({e})")
                continue
            rows.append({"scene": sd.name, **r})

        if not rows:
            logger.error("no scenes scored")
            return

        print(f"\n{'='*64}")
        if args.mode == "accuracy":
            print(f"MPJRE vs GT (degrees, lower better) — {len(rows)} scenes")
            print(f"{'='*64}")
            print(f"{'estimator':>12}{'body(21)':>14}{'all(54)':>12}")
            for k in ("model", "mean", "median"):
                print(f"{k:>12}"
                      f"{float(np.mean([r[k]['body'] for r in rows])):>14.2f}"
                      f"{float(np.mean([r[k]['all'] for r in rows])):>12.2f}")
        else:
            print(f"MPJRE vs temporal window — {len(rows)} scenes")
            print(f"{'='*64}")
            print(f"{'window':>10}{'body(21)':>14}{'all(54)':>12}{'vs W=0':>12}")
            base = float(np.mean([r[str(windows[0])]["body"] for r in rows]))
            for W in windows:
                b = float(np.mean([r[str(W)]["body"] for r in rows]))
                a = float(np.mean([r[str(W)]["all"] for r in rows]))
                print(f"{W:>10}{b:>14.2f}{a:>12.2f}{b - base:>+12.2f}")
        print("=" * 64)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": args.mode, "windows": windows, "per_scene": rows},
                      f, indent=2)
        logger.info(f"wrote {args.out}")
        return

    if args.mode == "joint" and args.from_json is not None:
        d = json.load(open(args.from_json))
        print_structure_report(np.array(d["matrix"]), d.get("n_scenes", 0))
        return

    if args.mode == "joint":
        mats = []
        for sd in scene_dirs:
            try:
                _, raw = load_scene_body_data(sd)
                pose, mask = build_fusion_tensors(raw)
            except Exception as e:
                logger.warning(f"{sd.name}: skipped ({e})")
                continue
            if args.max_frames:
                pose, mask = pose[:, :args.max_frames], mask[:, :args.max_frames]
            M = joint_influence(model, pose, mask, args.joint_delta, device, args.seed)
            if M is None:
                continue
            mats.append(M)
            a = analyse_joint_matrix(M)
            logger.info(f"{sd.name}  self={a['self_fraction']:.2f}  "
                        f"adj_lift={a['adjacent_lift']:.2f}  "
                        f"rho={a['spearman_influence_vs_hops']:+.2f}")

        if not mats:
            logger.error("no scenes evaluated")
            return
        M = np.mean(mats, axis=0)
        a = analyse_joint_matrix(M)
        print(f"\n{'='*68}")
        print(f"Q2 joint influence — {len(mats)} scenes, corruption {args.joint_delta:g} deg")
        print(f"{'='*68}")
        print(f"  self influence fraction   {a['self_fraction']:.3f}"
              "   (1.0 = joints fully independent)")
        print(f"  mass on tree neighbours   {a['adjacent_mass']:.3f}"
              f"   (chance {a['adjacent_chance']:.3f}, lift {a['adjacent_lift']:.2f}x)")
        print(f"  Spearman influence↔hops   {a['spearman_influence_vs_hops']:+.3f}"
              "   (negative = follows skeleton)")
        print("=" * 68)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "joint", "delta": args.joint_delta,
                       "n_scenes": len(mats), "summary": a,
                       "matrix": M.tolist()}, f, indent=2)
        logger.info(f"wrote {args.out}")
        return

    per_scene, K_seen = [], []
    for sd in scene_dirs:
        try:
            _, raw = load_scene_body_data(sd)
            pose, mask = build_fusion_tensors(raw)
        except Exception as e:
            logger.warning(f"{sd.name}: skipped ({e})")
            continue
        if args.max_frames:
            pose, mask = pose[:, :args.max_frames], mask[:, :args.max_frames]

        K = pose.shape[2]
        res = scene_sensitivity(model, pose, mask, deltas, device, seed=args.seed)
        if res is None:
            logger.warning(f"{sd.name}: skipped (no slot with >=2 cameras)")
            continue
        per_scene.append({"scene": sd.name, "K": K, "curves": res})
        K_seen.append(K)
        logger.info(f"{sd.name}  K={K}  "
                    + "  ".join(f"{d:g}deg:{res['model'][d]:.2f}" for d in deltas))

    if not per_scene:
        logger.error("no scenes evaluated")
        return

    # Average each curve across scenes.
    agg = {k: {d: float(np.mean([s["curves"][k][d] for s in per_scene]))
               for d in deltas} for k in ("model", "mean", "median")}
    K_mean = float(np.mean(K_seen))
    summ = {k: summarise(agg[k], K_mean) for k in agg}

    print(f"\n{'='*72}")
    print(f"Q1 sensitivity curve — {len(per_scene)} scenes, mean K = {K_mean:.1f}")
    print(f"{'='*72}")
    print(f"{'delta':>8}" + "".join(f"{k:>12}" for k in ("model", "mean", "median")))
    for d in deltas:
        print(f"{d:>8.0f}" + "".join(f"{agg[k][d]:>12.2f}" for k in
                                     ("model", "mean", "median")))
    print("-" * 72)
    cols = ("model", "mean", "median")
    print(f"{'slope*K':>8}" + "".join(f"{summ[k]['slope_x_K']:>12.2f}" for k in cols)
          + "   ~1.0 = averaging")
    print(f"{'linearity':>8}" + "".join(f"{summ[k]['linearity']:>12.2f}" for k in cols)
          + "   1.0 = never saturates")
    print("=" * 72)
    lin_model, lin_mean, lin_med = (summ[k]["linearity"] for k in cols)
    if lin_model <= lin_med + 0.5 * (lin_mean - lin_med):
        print("VERDICT: model saturates like the median -> learned view discounting")
    else:
        print("VERDICT: model tracks the mean -> behaves like averaging")
    print("=" * 72)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"deltas": deltas, "n_scenes": len(per_scene), "K_mean": K_mean,
                   "aggregate": agg, "summary": summ, "per_scene": per_scene}, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
