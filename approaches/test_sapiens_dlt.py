"""Test Sapiens-based DLT triangulation for body placement.

Uses VGGT cameras (same as M1/M2 in eval_placer_trans.py) and fused pose from the
fusion model, but replaces MHR70 pred_keypoints_2d with Sapiens Goliath keypoints
for DLT triangulation.

Goliath body joints 0-14 (no wrists in body range):
    0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear
    5:l_shoulder 6:r_shoulder 7:l_elbow 8:r_elbow
    9:l_hip 10:r_hip 11:l_knee 12:r_knee 13:l_ankle 14:r_ankle
Wrists: 41=right_wrist (last of right hand 21-41), 62=left_wrist (last of left hand 42-62)

Mapping used (_GOLIATH_SMPLX_ALIGN):
    SMPL-X 1  → Goliath 9   (left hip)
    SMPL-X 2  → Goliath 10  (right hip)
    SMPL-X 4  → Goliath 11  (left knee)
    SMPL-X 5  → Goliath 12  (right knee)
    SMPL-X 7  → Goliath 13  (left ankle)
    SMPL-X 8  → Goliath 14  (right ankle)
    SMPL-X 18 → Goliath 7   (left elbow)
    SMPL-X 19 → Goliath 8   (right elbow)
    SMPL-X 20 → Goliath 62  (left wrist, end of left hand)
    SMPL-X 21 → Goliath 41  (right wrist, end of right hand)

Usage:
    pixi run python approaches/test_sapiens_dlt.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        --checkpoint /users/tnanni/ghost/checkpoints/fusion_module_latest/best.pt
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

# Goliath 308: body j0-14 has no wrists; wrists are end of hand ranges
_GOLIATH_SMPLX_ALIGN: dict[int, int] = {
    1:  9,   # l_hip
    2:  10,  # r_hip
    4:  11,  # l_knee
    5:  12,  # r_knee
    7:  13,  # l_ankle
    8:  14,  # r_ankle
    18: 7,   # l_elbow
    19: 8,   # r_elbow
    20: 62,  # l_wrist (left hand last joint)
    21: 41,  # r_wrist (right hand last joint)
}

# Bony extremities only — most likely to coincide with SMPL-X joint centers
_GOLIATH_SMPLX_DISTAL: dict[int, int] = {
    7:  13,  # l_ankle
    8:  14,  # r_ankle
    18: 7,   # l_elbow
    19: 8,   # r_elbow
    20: 62,  # l_wrist (left hand last joint)
    21: 41,  # r_wrist (right hand last joint)
}

# Import fusion helpers from eval_placer_trans (avoids code duplication)
_EVAL_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_SCRIPTS))
from eval_placer_trans import (
    _load_fusion_model,
    _run_fusion_fwd,
    _6d_to_aa,
    load_pred_betas as _load_pred_betas,
    load_gt_trans,
    load_gt_global_orient,
    load_gt_betas,
    correct_gt_pelvis,
)
import eval_placer_trans as _eval_mod

_EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EVAL_ROOT / "evaluation"))
from evaluate_on_rich_test import (
    _sim3_align,
    metric_wa_mpjpe,
    metric_w_mpjpe,
    metric_ga_mpjpe,
    metric_pa_mpjpe,
    metric_rte,
    _BODY_JOINT_IDX,
)
from scipy.optimize import minimize as _scipy_minimize
from scipy.signal import savgol_filter
from fusion.placer import _6d_to_aa_batch

_BODY_J = len(_BODY_JOINT_IDX)


def _geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos = np.clip((np.trace(R_gt.T @ R_pred) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def _scene_to_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


_RICH_CAM_W = 4112  # XML calibration full-resolution width
_RICH_CAM_H = 3008  # XML calibration full-resolution height


def _build_gt_cameras(
    placer: BodyPlacer,
    scene_name: str,
    rich_root: Path,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Build (T,K,3,4) GT extrinsics and (T,K,3,3) GT intrinsics in VGGT image space.

    Extrinsics are re-rooted to cam_dirs[0] (same as M3 in evaluate_on_rich_test.py).
    Intrinsics are scaled from original full-res XML space to VGGT crop space.
    Returns (None, None) if calibration files are missing.
    """
    location  = _scene_to_location(scene_name)
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None, None

    # Load all XML calibrations indexed by camera number
    raw_exts:  dict[int, np.ndarray] = {}
    raw_intrs: dict[int, np.ndarray] = {}
    for xml_path in sorted(calib_dir.glob("*.xml")):
        idx = int(re.search(r"\d+", xml_path.stem).group())
        tree = ET.parse(xml_path)
        root = tree.getroot()
        cam_node  = root.find("CameraMatrix")
        intr_node = root.find("Intrinsics")
        if cam_node is not None:
            vals = list(map(float, cam_node.find("data").text.split()))
            raw_exts[idx] = np.array(vals, dtype=np.float64).reshape(3, 4)
        if intr_node is not None:
            vals = list(map(float, intr_node.find("data").text.split()))
            raw_intrs[idx] = np.array(vals, dtype=np.float64).reshape(3, 3)

    vnames = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
    ref_idx = int(re.search(r"\d+", vnames[0]).group())
    if ref_idx not in raw_exts:
        return None, None

    E0  = raw_exts[ref_idx]
    R0  = E0[:3, :3];  t0 = E0[:3, 3]

    T_, K_ = placer.T, len(vnames)
    gt_exts  = np.zeros((T_, K_, 3, 4), dtype=np.float32)
    gt_intrs = np.zeros((T_, K_, 3, 3), dtype=np.float32)
    gt_intrs[:, :, 2, 2] = 1.0

    for ki, cn in enumerate(vnames):
        m = re.search(r"\d+", cn)
        if not m:
            continue
        gidx = int(m.group())
        if gidx not in raw_exts:
            continue
        Ek = raw_exts[gidx]
        Rk = Ek[:3, :3] @ R0.T
        tk = Ek[:3, 3] - Ek[:3, :3] @ R0.T @ t0
        gt_exts[:, ki, :, :3] = Rk.astype(np.float32)
        gt_exts[:, ki, :,  3] = tk.astype(np.float32)

        if gidx not in raw_intrs:
            continue
        K_orig = raw_intrs[gidx]
        fx_o, fy_o = K_orig[0, 0], K_orig[1, 1]
        cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]
        for t in range(T_):
            x1, y1, x2, y2 = placer.original_coords[t, ki]
            # Divide by full-res calibration size (4112×3008), not the resized VGGT input.
            # K_orig is in 4112-space; VGGT crop [x1,y1,x2,y2] is within W_vggt×H_vggt.
            # u_vggt = x1 + u_4112 * (x2-x1)/4112  →  fx_vggt = fx_o*(x2-x1)/4112
            sx = (x2 - x1) / _RICH_CAM_W
            sy = (y2 - y1) / _RICH_CAM_H
            gt_intrs[t, ki, 0, 0] = fx_o * sx
            gt_intrs[t, ki, 1, 1] = fy_o * sy
            gt_intrs[t, ki, 0, 2] = cx_o * sx + x1
            gt_intrs[t, ki, 1, 2] = cy_o * sy + y1
            gt_intrs[t, ki, 2, 2] = 1.0

    return gt_exts, gt_intrs


def _load_gt(scene_name: str, rich_root: Path, body_split: str):
    """Return gt_body_data[pid][frame_idx] = {transl, global_orient (aa), body_pose (aa), betas}.

    Also returns gt_betas[pid] (10,) for pelvis correction.
    """
    gt_root = rich_root / body_split / scene_name
    gt_body_data: dict[int, dict[int, dict]] = {}
    gt_betas:     dict[int, np.ndarray]      = {}
    if not gt_root.is_dir():
        return gt_body_data, gt_betas
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            raw_betas = d.get("betas") if d.get("betas") is not None else d.get("smplx_betas")
            betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)[:10] if raw_betas is not None \
                    else np.zeros(10, dtype=np.float32)
            if pid not in gt_betas:
                gt_betas[pid] = betas
            body_pose_raw = d.get("body_pose")
            body_pose = np.asarray(body_pose_raw, dtype=np.float32).reshape(63) \
                        if body_pose_raw is not None else np.zeros(63, dtype=np.float32)
            orient_raw = d.get("global_orient")
            orient_aa = np.asarray(orient_raw, dtype=np.float32).squeeze() \
                        if orient_raw is not None else np.zeros(3, dtype=np.float32)
            transl = np.asarray(d["transl"], dtype=np.float32).squeeze()
            gt_body_data.setdefault(pid, {})[frame_idx] = {
                "transl":        transl,
                "global_orient": orient_aa,
                "body_pose":     body_pose,
                "betas":         betas,
            }
    return gt_body_data, gt_betas


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_sapiens(cam_dirs: list[Path]) -> list[dict[int, dict]]:
    """Per camera: {pid: {local_t: dict[global_t→local_t], kps: (T,308,3)}}."""
    result = []
    for cam_dir in cam_dirs:
        cam_map: dict[int, dict] = {}
        body_dir = cam_dir / "body_data"
        if body_dir.is_dir():
            for npz_path in sorted(body_dir.glob("person_*.npz")):
                pid     = int(npz_path.stem.split("_")[1])
                sp_path = cam_dir / f"sapiens_centered_kps_person_{pid}.npz"
                if not sp_path.exists():
                    continue
                bd = np.load(npz_path, allow_pickle=False)
                sp = np.load(sp_path)
                fi = bd["frame_indices"].astype(int)
                cam_map[pid] = {
                    "local_t": {int(g): int(l) for l, g in enumerate(fi)},
                    "kps":     sp["keypoints"],  # (T, 308, 3) [x, y, conf]
                }
        result.append(cam_map)
    return result


# ---------------------------------------------------------------------------
# Procrustes + DLT with Sapiens kps + VGGT cameras
# ---------------------------------------------------------------------------

def run_sapiens_procrustes(
    placer:        BodyPlacer,
    scale:         float | np.ndarray,
    sapiens_data:  list[dict[int, dict]],
    pred_betas:    dict[int, np.ndarray],
    fused_pose_by_pid: dict[int, np.ndarray],   # {pid: (T_scene, 54, 6)}
    frame_start:   int,
    conf_thr:      float = 0.3,
    min_cams:      int   = 2,
    min_joints:    int   = 3,
    joint_map:     dict[int, int] | None = None,
    sim3:          bool  = False,
    sin_weight:    bool  = False,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]], dict[int, dict[int, float]]]:
    """DLT-triangulate Sapiens joints with VGGT cameras, then SE(3) or Sim(3) Procrustes.

    When sim3=True, also solves for a per-frame scale s using the Umeyama algorithm:
        s * R @ J_can[j] + t ≈ joint_world[j]

    Returns:
        translations : {pid: {frame_idx: pelvis_world (3,)}}
        orientations : {pid: {frame_idx: R (3,3)}}
        proc_scales  : {pid: {frame_idx: s (float)}}  — 1.0 for every frame if sim3=False
    """
    if joint_map is None:
        joint_map = _GOLIATH_SMPLX_ALIGN
    smplx_joints = sorted(joint_map)
    zero_orient  = np.zeros((1, 3), dtype=np.float32)

    all_pids: set[int] = set()
    for cm in sapiens_data:
        all_pids.update(cm.keys())

    translations: dict[int, dict[int, np.ndarray]] = {}
    orientations: dict[int, dict[int, np.ndarray]] = {}
    proc_scales:  dict[int, dict[int, float]]      = {}

    for pid in sorted(all_pids):
        betas = pred_betas.get(pid, np.zeros(10, dtype=np.float32))

        all_frames: set[int] = set()
        for cm in sapiens_data:
            if pid in cm:
                all_frames.update(cm[pid]["local_t"].keys())

        trans_out:  dict[int, np.ndarray] = {}
        orient_out: dict[int, np.ndarray] = {}
        scale_out:  dict[int, float]      = {}

        for global_t in sorted(all_frames):
            vggt_t = global_t - frame_start
            if vggt_t < 0 or vggt_t >= placer.T:
                continue

            s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

            # ── Step 1: DLT-triangulate each joint across cameras ─────────
            joint_world: dict[int, np.ndarray] = {}
            for smplx_j in smplx_joints:
                goliath_j = joint_map[smplx_j]
                obs:     list[tuple[float, float]] = []
                pmats:   list[np.ndarray]          = []
                weights: list[float]               = []

                for k, cm in enumerate(sapiens_data):
                    if pid not in cm:
                        continue
                    if global_t not in cm[pid]["local_t"]:
                        continue
                    if not placer.cam_valid[vggt_t, k]:
                        continue

                    lt  = cm[pid]["local_t"][global_t]
                    kps = cm[pid]["kps"][lt]   # (308, 3)
                    x, y, conf = float(kps[goliath_j, 0]), float(kps[goliath_j, 1]), float(kps[goliath_j, 2])
                    if conf < conf_thr:
                        continue

                    # Convert original pixel coords → VGGT image space
                    oc       = placer.original_coords[vggt_t, k]
                    os_      = placer.original_size[vggt_t, k]
                    W_orig, H_orig = float(os_[0]), float(os_[1])
                    u, v = placer._orig_to_vggt(np.array([x, y]), oc, W_orig, H_orig)
                    if not placer._in_bounds(u, v, oc[2], oc[3]):
                        continue

                    intr = placer.intrinsics[vggt_t, k].astype(np.float64)
                    ext  = placer.extrinsics[vggt_t, k].astype(np.float64).copy()
                    ext[:3, 3] *= s
                    pmats.append(intr @ ext)
                    obs.append((u, v))
                    # Angular weight: sin of angle between cam0 and cam_k optical axes.
                    # cam0 z-axis in world = [0,0,1]; cam_k z-axis = R_k[2,:] so
                    # cos(θ) = R_k[2,2] and sin(θ) = sqrt(1 - R_k[2,2]²).
                    if sin_weight and k > 0:
                        cos_a = float(np.clip(ext[2, 2], -1.0, 1.0))
                        ang_w = float(np.sqrt(max(1.0 - cos_a ** 2, 0.0))) ** 10
                    else:
                        ang_w = 1.0
                    weights.append(conf * ang_w)

                if len(obs) >= min_cams:
                    joint_world[smplx_j] = placer._triangulate_dlt(obs, pmats, weights)

            if len(joint_world) < min_joints:
                continue

            # ── Step 2: FK with fused body_pose ──────────────────────────
            if pid not in fused_pose_by_pid:
                continue
            fused_arr = fused_pose_by_pid[pid]          # (T_scene, 54, 6)
            t_local   = global_t - frame_start
            if not (0 <= t_local < len(fused_arr)):
                continue
            body_pose_frame = _6d_to_aa(fused_arr[t_local, :21]).reshape(63)

            J_can = placer._smplx_fk(
                betas[np.newaxis],
                body_pose_frame[np.newaxis],
                zero_orient,
            )[0]  # (55, 3) — canonical joints, zero global_orient

            # ── Step 3: SE(3) or Sim(3) Procrustes ───────────────────────
            vis = sorted(joint_world)
            A   = np.stack([joint_world[j] for j in vis], axis=0).astype(np.float64)
            B   = np.stack([J_can[j]        for j in vis], axis=0).astype(np.float64)

            A_mean = A.mean(0);  B_mean = B.mean(0)
            B_c = B - B_mean
            H   = B_c.T @ (A - A_mean)
            U, S_vals, Vt = np.linalg.svd(H)
            d_sign = np.linalg.det(Vt.T @ U.T)
            R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)

            if sim3:
                # Umeyama scale: s = tr(D·Σ) / σ²_B
                s_proc = float(np.dot(S_vals, [1.0, 1.0, d_sign]) / np.sum(B_c ** 2))
            else:
                s_proc = 1.0

            t = (A_mean - s_proc * R.astype(np.float64) @ B_mean).astype(np.float32)

            # pelvis_world = s * R @ J_can[0] + t
            pelvis_world = (s_proc * R.astype(np.float64) @ J_can[0].astype(np.float64) + t).astype(np.float32)
            trans_out[global_t]  = pelvis_world
            orient_out[global_t] = R
            scale_out[global_t]  = s_proc

        if trans_out:
            translations[pid] = trans_out
            orientations[pid] = orient_out
            proc_scales[pid]  = scale_out

    return translations, orientations, proc_scales


# ---------------------------------------------------------------------------
# Cam0-ray lateral + multi-cam depth placement
# ---------------------------------------------------------------------------

def run_cam0_lateral_dlt(
    placer:            BodyPlacer,
    scale:             float | np.ndarray,
    sapiens_data:      list[dict[int, dict]],
    pred_betas:        dict[int, np.ndarray],
    fused_pose_by_pid: dict[int, np.ndarray],
    frame_start:       int,
    cam0_intrs:        np.ndarray,   # (T, 3, 3) — hybrid-PP intrinsics for cam0 ray
    conf_thr:          float = 0.3,
    min_joints:        int   = 3,    # min observations from cams 1–K-1
    joint_map:         dict[int, int] | None = None,
    huber_px:          float = 20.0,
) -> tuple[dict, dict, dict]:
    """Asymmetric root placement: cam0 ray pins lateral (xy); cams 1+ pin depth (z).

    For each frame we constrain root_world = t0 * d0, where d0 is the unit
    direction from cam0 through the hip midpoint (using hybrid-PP intrinsics).
    We then jointly optimise (R ∈ SO(3), t0 > 0) by minimising the 2D Huber
    reprojection error in cameras 1..K-1 only.

    Returns the same (translations, orientations, proc_scales) triple as
    run_sapiens_procrustes.
    """
    if joint_map is None:
        joint_map = _GOLIATH_SMPLX_ALIGN
    smplx_joints = sorted(joint_map)
    zero_orient  = np.zeros((1, 3), dtype=np.float32)
    delta = float(huber_px)

    all_pids: set[int] = set()
    for cm in sapiens_data:
        all_pids.update(cm.keys())

    translations: dict[int, dict[int, np.ndarray]] = {}
    orientations: dict[int, dict[int, np.ndarray]] = {}
    proc_scales:  dict[int, dict[int, float]]      = {}

    for pid in sorted(all_pids):
        betas = pred_betas.get(pid, np.zeros(10, dtype=np.float32))

        all_frames: set[int] = set()
        for cm in sapiens_data:
            if pid in cm:
                all_frames.update(cm[pid]["local_t"].keys())

        trans_out:  dict[int, np.ndarray] = {}
        orient_out: dict[int, np.ndarray] = {}
        scale_out:  dict[int, float]      = {}

        prev_x: np.ndarray | None = None  # warm-start: [rv0, rv1, rv2, t0]

        for global_t in sorted(all_frames):
            vggt_t = global_t - frame_start
            if vggt_t < 0 or vggt_t >= placer.T:
                continue

            s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

            # ── Step 1: cam0 hip ray ──────────────────────────────────────
            K0 = cam0_intrs[vggt_t].astype(np.float64)
            fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]

            hip_uvs: list[tuple[float, float]] = []
            cm0 = sapiens_data[0]
            if pid in cm0 and global_t in cm0[pid]["local_t"]:
                lt0  = cm0[pid]["local_t"][global_t]
                kps0 = cm0[pid]["kps"][lt0]
                oc0  = placer.original_coords[vggt_t, 0]
                os0  = placer.original_size[vggt_t, 0]
                for gj in (9, 10):   # l_hip, r_hip
                    xg, yg, cg = float(kps0[gj, 0]), float(kps0[gj, 1]), float(kps0[gj, 2])
                    if cg < conf_thr:
                        continue
                    u_c, v_c = placer._orig_to_vggt(np.array([xg, yg]), oc0, float(os0[0]), float(os0[1]))
                    if placer._in_bounds(u_c, v_c, oc0[2], oc0[3]):
                        hip_uvs.append((u_c, v_c))

            if hip_uvs:
                u_hip = float(np.mean([uv[0] for uv in hip_uvs]))
                v_hip = float(np.mean([uv[1] for uv in hip_uvs]))
            else:
                u_hip, v_hip = cx0, cy0   # fall back to principal point

            # Unit ray direction in world frame (cam0 IS world origin)
            d0 = np.array([(u_hip - cx0) / fx0, (v_hip - cy0) / fy0, 1.0])
            d0 /= np.linalg.norm(d0)

            # ── Step 2: FK ────────────────────────────────────────────────
            if pid not in fused_pose_by_pid:
                continue
            fused_arr = fused_pose_by_pid[pid]
            t_local   = global_t - frame_start
            if not (0 <= t_local < len(fused_arr)):
                continue
            body_pose_frame = _6d_to_aa(fused_arr[t_local, :21]).reshape(63)
            J_can = placer._smplx_fk(betas[np.newaxis], body_pose_frame[np.newaxis], zero_orient)[0]
            J_rel = (J_can - J_can[0]).astype(np.float64)   # (55, 3)

            # ── Step 3: observations from cams 1..K-1 ────────────────────
            obs_J:  list[int]         = []
            obs_u:  list[float]       = []
            obs_v:  list[float]       = []
            obs_w:  list[float]       = []
            obs_fx: list[float]       = []
            obs_fy: list[float]       = []
            obs_cx: list[float]       = []
            obs_cy: list[float]       = []
            obs_Rk: list[np.ndarray]  = []
            obs_tk: list[np.ndarray]  = []

            for k in range(1, len(sapiens_data)):
                cm = sapiens_data[k]
                if pid not in cm or global_t not in cm[pid]["local_t"]:
                    continue
                if not placer.cam_valid[vggt_t, k]:
                    continue
                lt  = cm[pid]["local_t"][global_t]
                kps = cm[pid]["kps"][lt]
                intr = placer.intrinsics[vggt_t, k].astype(np.float64)
                ext  = placer.extrinsics[vggt_t, k].astype(np.float64).copy()
                ext[:, 3] *= s
                Rk = ext[:3, :3]
                tk = ext[:3, 3]
                oc = placer.original_coords[vggt_t, k]
                os_ = placer.original_size[vggt_t, k]
                for smplx_j in smplx_joints:
                    gj = joint_map[smplx_j]
                    xg, yg, cg = float(kps[gj, 0]), float(kps[gj, 1]), float(kps[gj, 2])
                    if cg < conf_thr:
                        continue
                    u_c, v_c = placer._orig_to_vggt(np.array([xg, yg]), oc, float(os_[0]), float(os_[1]))
                    if not placer._in_bounds(u_c, v_c, oc[2], oc[3]):
                        continue
                    obs_J.append(smplx_j)
                    obs_u.append(u_c);   obs_v.append(v_c);  obs_w.append(cg)
                    obs_fx.append(intr[0, 0]);  obs_fy.append(intr[1, 1])
                    obs_cx.append(intr[0, 2]);  obs_cy.append(intr[1, 2])
                    obs_Rk.append(Rk);  obs_tk.append(tk)

            if len(obs_J) < min_joints:
                continue

            # Pre-build vectorised arrays (constant during optimisation)
            J_rel_obs = J_rel[obs_J]                   # (N, 3)
            u_arr  = np.array(obs_u,  dtype=np.float64)
            v_arr  = np.array(obs_v,  dtype=np.float64)
            w_arr  = np.array(obs_w,  dtype=np.float64)
            fx_arr = np.array(obs_fx, dtype=np.float64)
            fy_arr = np.array(obs_fy, dtype=np.float64)
            cx_arr = np.array(obs_cx, dtype=np.float64)
            cy_arr = np.array(obs_cy, dtype=np.float64)
            Rk_arr = np.stack(obs_Rk)                  # (N, 3, 3)
            tk_arr = np.stack(obs_tk)                   # (N, 3)

            def loss_fn(x):
                R_mat   = SciR.from_rotvec(x[:3]).as_matrix()
                P_world = (R_mat @ J_rel_obs.T).T + x[3] * d0   # (N, 3)
                P_cam   = np.einsum('nij,nj->ni', Rk_arr, P_world) + tk_arr
                Z       = np.maximum(P_cam[:, 2], 1e-2)
                u_p     = fx_arr * P_cam[:, 0] / Z + cx_arr
                v_p     = fy_arr * P_cam[:, 1] / Z + cy_arr
                r       = np.sqrt((u_p - u_arr) ** 2 + (v_p - v_arr) ** 2 + 1e-12)
                h       = np.where(r < delta, 0.5 * r ** 2, delta * r - 0.5 * delta ** 2)
                return float(w_arr @ h)

            # Initialisation: warm-start or two-candidate first frame
            bounds = [(None, None), (None, None), (None, None), (0.5, None)]
            if prev_x is not None:
                candidates = [prev_x]
            else:
                candidates = [
                    np.array([0.0, 0.0, 0.0,    3.0]),
                    np.array([0.0, np.pi, 0.0,  3.0]),
                ]

            best_x, best_loss = None, np.inf
            for x0 in candidates:
                res = _scipy_minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds)
                if res.fun < best_loss:
                    best_loss, best_x = res.fun, res.x.copy()

            prev_x = best_x
            R_opt        = SciR.from_rotvec(best_x[:3]).as_matrix().astype(np.float32)
            pelvis_world = (best_x[3] * d0).astype(np.float32)

            trans_out[global_t]  = pelvis_world
            orient_out[global_t] = R_opt
            scale_out[global_t]  = 1.0

        if trans_out:
            translations[pid] = trans_out
            orientations[pid] = orient_out
            proc_scales[pid]  = scale_out

    return translations, orientations, proc_scales


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_translations_sg(
    pred_trans: dict[int, dict[int, np.ndarray]],
    window:     int,
    polyorder:  int = 2,
) -> dict[int, dict[int, np.ndarray]]:
    """Apply Savitzky-Golay filter to per-pid root translation trajectory.

    Frames are treated as equally-spaced (minor gaps are ignored).
    window must be odd and > polyorder.
    """
    smoothed: dict[int, dict[int, np.ndarray]] = {}
    for pid, frames in pred_trans.items():
        sorted_frames = sorted(frames)
        traj = np.stack([frames[f] for f in sorted_frames])  # (N, 3)
        if len(traj) < window:
            smoothed[pid] = dict(frames)
            continue
        traj_s = savgol_filter(traj, window_length=window, polyorder=polyorder, axis=0)
        smoothed[pid] = {f: traj_s[i].astype(np.float32)
                         for i, f in enumerate(sorted_frames)}
    return smoothed


def smooth_extrinsics_sg(
    extrinsics: np.ndarray,   # (T, K, 3, 4)
    window:     int,
    polyorder:  int = 2,
) -> np.ndarray:
    """Smooth camera extrinsics over T with Savitzky-Golay.

    Rotation (3×3): smooth each of the 9 entries independently, then
    re-orthogonalise via SVD to get a valid rotation matrix.
    Translation (3,): smoothed directly.

    For static cameras (RICH) VGGT produces slightly different [R|t] per
    frame due to per-frame estimation noise.  Smoothing collapses that noise
    before triangulation, reducing per-frame orientation drift in Procrustes.
    """
    T, K = extrinsics.shape[:2]
    out  = extrinsics.copy()
    if T < window:
        return out
    for k in range(K):
        # Smooth translation column directly
        out[:, k, :, 3] = savgol_filter(
            extrinsics[:, k, :, 3], window_length=window, polyorder=polyorder, axis=0
        )
        # Smooth rotation entries and re-orthogonalise
        R_smooth = savgol_filter(
            extrinsics[:, k, :, :3], window_length=window, polyorder=polyorder, axis=0
        )  # (T, 3, 3) — may no longer be orthogonal
        for t in range(T):
            U, _, Vt = np.linalg.svd(R_smooth[t])
            d_sign = np.linalg.det(U @ Vt)
            out[t, k, :, :3] = U @ np.diag([1.0, 1.0, d_sign]) @ Vt
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_eval_arrays(
    pred_trans:        dict[int, dict[int, np.ndarray]],
    pred_orient:       dict[int, dict[int, np.ndarray]],
    gt_body_data:      dict[int, dict[int, dict]],
    gt_betas:          dict[int, np.ndarray],
    fused_pose_by_pid: dict[int, np.ndarray],
    frame_start:       int,
    pred_betas:        dict[int, np.ndarray],
    placer:            BodyPlacer,
    proc_scales:       dict[int, dict[int, float]] | None = None,
    pid_map:           dict[int, int] | None = None,   # ghost_pid → gt_pid override
):
    """Build pred/gt arrays once. Returns arrays needed for score_from_arrays().

    pred_joints_rel: joint positions relative to root (unchanged by smoothing).
    pred_roots:      per-frame root (pelvis) world positions.
    gt_joints, gt_roots: ground truth.
    all_oerrs:       orientation errors (unchanged by smoothing).
    frame_start_t, T, P: array metadata.
    """
    gt_pids = sorted(gt_body_data)
    zero_orient = np.zeros((1, 3), dtype=np.float32)

    all_frames: set[int] = set()
    for frames in pred_trans.values():
        all_frames.update(frames)
    if not all_frames:
        return None
    frame_start_t = min(all_frames)
    T = max(all_frames) - frame_start_t + 1
    P = len(pred_trans)

    pred_joints_rel = np.full((T, P, _BODY_J, 3), np.nan, dtype=np.float32)
    pred_roots      = np.full((T, P, 3),          np.nan, dtype=np.float32)
    gt_joints       = np.full((T, P, _BODY_J, 3), np.nan, dtype=np.float32)
    gt_roots        = np.full((T, P, 3),          np.nan, dtype=np.float32)
    all_oerrs: list[float] = []

    for slot, ghost_pid in enumerate(sorted(pred_trans)):
        if pid_map and ghost_pid in pid_map:
            gt_pid = pid_map[ghost_pid]
        else:
            gt_pid = min(gt_pids, key=lambda p: abs(p - ghost_pid))
        betas    = pred_betas.get(ghost_pid, np.zeros(10, dtype=np.float32))
        fused_pp = fused_pose_by_pid.get(ghost_pid)

        for global_t, pelvis_world in sorted(pred_trans[ghost_pid].items()):
            R_mat   = pred_orient.get(ghost_pid, {}).get(global_t)
            t_rel   = int(global_t) - frame_start_t
            t_fused = int(global_t) - frame_start
            if R_mat is None or fused_pp is None:
                continue
            if not (0 <= t_rel < T) or not (0 <= t_fused < len(fused_pp)):
                continue
            bp_aa = _6d_to_aa_batch(fused_pp[t_fused, :21]).reshape(63)
            J_can = placer._smplx_fk(betas[np.newaxis], bp_aa[np.newaxis], zero_orient)[0]
            s_proc = (proc_scales or {}).get(ghost_pid, {}).get(global_t, 1.0)
            # joints relative to root — unchanged by smoothing
            pred_joints_rel[t_rel, slot] = (
                s_proc * (R_mat @ (J_can - J_can[0]).T).T
            )[_BODY_JOINT_IDX]
            pred_roots[t_rel, slot] = pelvis_world   # J_world[0] = pelvis_world

        for frame_idx, params in gt_body_data.get(gt_pid, {}).items():
            t_rel = int(frame_idx) - frame_start_t
            if not (0 <= t_rel < T):
                continue
            J_gt = placer._smplx_fk(
                params["betas"][np.newaxis],
                params["body_pose"][np.newaxis],
                params["global_orient"][np.newaxis],
            )[0] + params["transl"]
            gt_joints[t_rel, slot] = J_gt[_BODY_JOINT_IDX]
            gt_roots[t_rel,  slot] = J_gt[0]

        for f in sorted(pred_trans[ghost_pid]):
            R_pred = pred_orient.get(ghost_pid, {}).get(f)
            params = gt_body_data.get(gt_pid, {}).get(f, {})
            if R_pred is not None and "global_orient" in params:
                R_gt = SciR.from_rotvec(params["global_orient"].astype(np.float64)).as_matrix()
                all_oerrs.append(_geodesic_deg(R_pred, R_gt.astype(np.float32)))

        e_raw = np.array([
            float(np.linalg.norm(pred_roots[f - frame_start_t, slot] - gt_roots[f - frame_start_t, slot]))
            for f in sorted(pred_trans[ghost_pid])
            if f in gt_body_data.get(gt_pid, {})
            and np.isfinite(pred_roots[f - frame_start_t, slot]).all()
            and np.isfinite(gt_roots[f - frame_start_t, slot]).all()
        ])
        o = np.array(all_oerrs)
        print(f"  pid {ghost_pid} (GT {gt_pid})  N={len(e_raw)}", end="")
        if len(e_raw): print(f"  root mean={e_raw.mean()*100:.1f}cm", end="")
        if len(o):     print(f"  orient mean={o.mean():.1f}°", end="")
        print()

    return pred_joints_rel, pred_roots, gt_joints, gt_roots, frame_start_t, T, P, all_oerrs


def score_from_arrays(
    pred_joints_rel: np.ndarray,   # (T, P, J, 3) — relative to root, fixed
    pred_roots:      np.ndarray,   # (T, P, 3)    — possibly smoothed
    gt_joints:       np.ndarray,   # (T, P, J, 3)
    gt_roots:        np.ndarray,   # (T, P, 3)
    all_oerrs:       list[float],
    label:           str = "",
    verbose:         bool = True,
) -> dict:
    pred_joints = pred_joints_rel + pred_roots[:, :, np.newaxis, :]
    valid = (
        np.isfinite(pred_joints).all((-2, -1)) &
        np.isfinite(gt_joints).all((-2, -1))
    )
    wa  = metric_wa_mpjpe(pred_joints, gt_joints, valid)
    w   = metric_w_mpjpe( pred_joints, gt_joints, valid)
    ga  = metric_ga_mpjpe(pred_joints, gt_joints, valid)
    pa  = metric_pa_mpjpe(pred_joints, gt_joints, valid)
    rte = metric_rte(pred_roots, gt_roots)

    valid_r = np.isfinite(pred_roots).all(-1) & np.isfinite(gt_roots).all(-1)
    root_errs = np.linalg.norm(pred_roots - gt_roots, axis=-1)[valid_r]
    o = np.array(all_oerrs)

    if verbose:
        print(f"\n  {label}  N_frames={valid.any(-1).sum()}")
        print(f"    WA={wa:.1f}mm  W={w:.1f}mm  GA={ga:.1f}mm  PA={pa:.1f}mm  RTE={rte*100:.1f}cm")
        print(f"    root  mean={root_errs.mean()*100:.1f}cm  median={np.median(root_errs)*100:.1f}cm")
        if len(o): print(f"    orient  mean={o.mean():.1f}°  median={np.median(o):.1f}°")

    return dict(WA=wa, W=w, GA=ga, PA=pa, RTE=rte,
                root_mean=root_errs.mean(), root_median=float(np.median(root_errs)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",   required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--checkpoint",  required=True, type=Path,
                    help="Fusion model checkpoint (.pt)")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--conf_thr",    type=float, default=0.3)
    ap.add_argument("--min_cams",    type=int,   default=2)
    ap.add_argument("--min_joints",  type=int,   default=3)
    ap.add_argument("--body_split",  default="train_body")
    ap.add_argument("--gt_cams",     action="store_true",
                    help="Use GT calibration cameras instead of VGGT (like M3)")
    ap.add_argument("--hybrid_pp",   action="store_true",
                    help="Keep VGGT fx/fy + extrinsics but replace cx/cy with RICH GT PP")
    ap.add_argument("--rich_k",      action="store_true",
                    help="Use full RICH GT K (fx,fy,cx,cy scaled to VGGT crop) + VGGT extrinsics + MapAnything scale")
    ap.add_argument("--rich_ext_pp", action="store_true",
                    help="RICH extrinsics (scale=1) + VGGT fx/fy + RICH cx/cy")
    ap.add_argument("--distal_only", action="store_true",
                    help="Use only ankles+elbows+wrists for Procrustes (skip hips+knees)")
    ap.add_argument("--cam0_lateral", action="store_true",
                    help="Cam0 ray (VGGT f + RICH PP) pins lateral; cams 1-7 reprojection pins depth")
    ap.add_argument("--combine_xy_z", action="store_true",
                    help="xy from cam0-ray fit + z from DLT+PP-fixed triangulation (best of both)")
    ap.add_argument("--gt_pid", type=int, default=0,
                    help="Force GT person ID for ghost pid 1 (0 = nearest-id auto-match)")
    ap.add_argument("--exclude_cams", nargs="+", default=[],
                    help="Camera names to exclude from DLT (e.g. --exclude_cams cam_10)")
    ap.add_argument("--sim3", action="store_true",
                    help="Fit Sim(3) Procrustes (R, t, scale) instead of SE(3)")
    ap.add_argument("--sin_weight", action="store_true",
                    help="Weight DLT observations by sin(angle between cam0 and cam_k optical axes)")
    ap.add_argument("--smooth_cameras", type=int, default=0,
                    help="Savitzky-Golay window for smoothing VGGT extrinsics over T before triangulation (0=off, must be odd)")
    ap.add_argument("--gt_betas", action="store_true",
                    help="Use GT betas (from RICH pkl) instead of SAM3D pred betas for FK in Procrustes — isolates betas as error source")
    args = ap.parse_args()

    import torch
    device     = torch.device(args.device)
    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    rich_root  = args.rich_root.resolve()

    print(f"Scene: {scene_name}")

    # ── BodyPlacer ────────────────────────────────────────────────────────────
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(
        scene_output_dir = str(scene_dir),
        smplx_model_path = _smplx_arg,
    )
    cam_dirs = placer._cam_dirs
    print(f"Cameras: {[d.name for d in cam_dirs]}  T={placer.T}")

    # ── Scale: MapAnything first, triangulated fallback (same as M1) ─────────
    pred_betas = _load_pred_betas(list(cam_dirs))
    scale = placer.load_mapanything_scale()
    if scale is not None:
        print(f"Scale: MapAnything  median={float(np.median(scale)):.4f} m/VGGT-unit")
    else:
        scale = placer.estimate_scale_triangulated(
            all_pids          = set(pred_betas),
            pred_betas_by_pid = pred_betas,
        )
        print(f"Scale: triangulated  median={float(np.median(scale)):.4f} m/VGGT-unit")

    # ── Fusion model → fused pose ─────────────────────────────────────────────
    print("Loading fusion model …")
    fusion_model = _load_fusion_model(args.checkpoint, device)
    fwd = _run_fusion_fwd(list(cam_dirs), fusion_model, device)
    if fwd is None:
        print("ERROR: fusion forward pass failed"); return
    fused_pose_arr, _, fwd_pids, frame_start = fwd
    # fused_pose_arr: (T_scene, P, 54, 6)
    pid_to_slot = {pid: i for i, pid in enumerate(fwd_pids)}
    fused_pose_by_pid: dict[int, np.ndarray] = {
        pid: fused_pose_arr[:, slot]      # (T_scene, 54, 6)
        for pid, slot in pid_to_slot.items()
    }
    print(f"Fused pose: {len(fused_pose_by_pid)} pids, frame_start={frame_start}")

    # ── Sapiens keypoints ─────────────────────────────────────────────────────
    sapiens_data = _load_sapiens(list(cam_dirs))
    n_pairs = sum(len(cm) for cm in sapiens_data)
    print(f"Sapiens kps: {n_pairs} (cam, pid) pairs loaded")

    # ── GT cameras (optional) ─────────────────────────────────────────────────
    orig_extrinsics = placer.extrinsics.copy()
    orig_intrinsics = placer.intrinsics.copy()
    orig_cam_valid  = placer.cam_valid.copy()
    cam0_intrs_for_ray: np.ndarray | None = None
    if args.rich_ext_pp:
        gt_cam_exts, gt_cam_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_cam_exts is None:
            print("ERROR: GT cameras not found — aborting"); return
        # RICH extrinsics (metric, scale=1) + VGGT fx/fy + RICH cx/cy
        hybrid_intrs = gt_cam_intrs.copy()
        hybrid_intrs[:, :, 0, 0] = orig_intrinsics[:, :, 0, 0]  # VGGT fx
        hybrid_intrs[:, :, 1, 1] = orig_intrinsics[:, :, 1, 1]  # VGGT fy
        placer.extrinsics = gt_cam_exts
        placer.intrinsics = hybrid_intrs
        filled = np.any(gt_cam_exts != 0, axis=(0, 2, 3))
        placer.cam_valid = orig_cam_valid & filled[np.newaxis, :]
        scale = np.ones(placer.T, dtype=np.float32)
        dfx = float(np.abs(hybrid_intrs[:,:,0,0] - gt_cam_intrs[:,:,0,0]).mean())
        dfy = float(np.abs(hybrid_intrs[:,:,1,1] - gt_cam_intrs[:,:,1,1]).mean())
        dcx = float(np.abs(hybrid_intrs[:,:,0,2] - orig_intrinsics[:,:,0,2]).mean())
        dcy = float(np.abs(hybrid_intrs[:,:,1,2] - orig_intrinsics[:,:,1,2]).mean())
        print(f"RICH ext + VGGT fx/fy + RICH cx/cy  |Δfx|={dfx:.1f}px  |Δfy|={dfy:.1f}px  |Δcx|={dcx:.1f}px  |Δcy|={dcy:.1f}px  (scale=1.0)")
    elif args.gt_cams:
        gt_cam_exts, gt_cam_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_cam_exts is None:
            print("ERROR: GT cameras not found — aborting"); return
        placer.extrinsics = gt_cam_exts
        placer.intrinsics = gt_cam_intrs
        filled = np.any(gt_cam_exts != 0, axis=(0, 2, 3))
        placer.cam_valid = orig_cam_valid & filled[np.newaxis, :]
        scale = np.ones(placer.T, dtype=np.float32)
        print("GT cameras loaded  (scale forced to 1.0)")
    elif args.hybrid_pp or args.rich_k:
        _, gt_cam_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_cam_intrs is None:
            print("ERROR: GT cameras not found — aborting"); return
        if args.rich_k:
            # Full RICH GT K (fx, fy, cx, cy) scaled to VGGT crop space.
            # _build_gt_cameras already applies: fx_vggt = fx_rich * crop_w/4112, etc.
            placer.intrinsics = gt_cam_intrs
            dfx = float(np.abs(gt_cam_intrs[:,:,0,0] - orig_intrinsics[:,:,0,0]).mean())
            dfy = float(np.abs(gt_cam_intrs[:,:,1,1] - orig_intrinsics[:,:,1,1]).mean())
            dcx = float(np.abs(gt_cam_intrs[:,:,0,2] - orig_intrinsics[:,:,0,2]).mean())
            dcy = float(np.abs(gt_cam_intrs[:,:,1,2] - orig_intrinsics[:,:,1,2]).mean())
            print(f"Full RICH K: |Δfx|={dfx:.1f}px  |Δfy|={dfy:.1f}px  |Δcx|={dcx:.1f}px  |Δcy|={dcy:.1f}px")
        else:
            # Keep VGGT fx/fy; replace only cx,cy with RICH GT PP scaled to crop.
            hybrid_intrs = orig_intrinsics.copy()
            hybrid_intrs[:, :, 0, 2] = gt_cam_intrs[:, :, 0, 2]
            hybrid_intrs[:, :, 1, 2] = gt_cam_intrs[:, :, 1, 2]
            placer.intrinsics = hybrid_intrs
            dcx = float(np.abs(hybrid_intrs[:,:,0,2] - orig_intrinsics[:,:,0,2]).mean())
            dcy = float(np.abs(hybrid_intrs[:,:,1,2] - orig_intrinsics[:,:,1,2]).mean())
            print(f"Hybrid PP: VGGT fx/fy + RICH cx/cy  |Δcx|={dcx:.1f}px  |Δcy|={dcy:.1f}px")
    elif args.cam0_lateral:
        _, gt_cam_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_cam_intrs is None:
            print("ERROR: GT cameras not found — aborting"); return
        # Build hybrid cam0 intrinsics: VGGT fx/fy + RICH cx/cy (for ray only)
        cam0_hybrid = orig_intrinsics[:, 0].copy()       # (T, 3, 3)
        cam0_hybrid[:, 0, 2] = gt_cam_intrs[:, 0, 0, 2] # RICH cx
        cam0_hybrid[:, 1, 2] = gt_cam_intrs[:, 0, 1, 2] # RICH cy
        cam0_intrs_for_ray = cam0_hybrid
        dcx = float(np.abs(gt_cam_intrs[:, 0, 0, 2] - orig_intrinsics[:, 0, 0, 2]).mean())
        dcy = float(np.abs(gt_cam_intrs[:, 0, 1, 2] - orig_intrinsics[:, 0, 1, 2]).mean())
        print(f"Cam0 ray: VGGT fx/fy + RICH cx/cy  |Δcx|={dcx:.1f}px  |Δcy|={dcy:.1f}px")
        # placer cameras unchanged: VGGT extrinsics + MapAnything scale for cams 1-7
    elif args.combine_xy_z:
        _, gt_cam_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_cam_intrs is None:
            print("ERROR: GT cameras not found — aborting"); return
        # Hybrid PP for all cameras (same as --hybrid_pp) — gives best DLT depth
        hybrid_intrs = orig_intrinsics.copy()
        hybrid_intrs[:, :, 0, 2] = gt_cam_intrs[:, :, 0, 2]
        hybrid_intrs[:, :, 1, 2] = gt_cam_intrs[:, :, 1, 2]
        placer.intrinsics = hybrid_intrs
        # cam0 ray uses the same hybrid-PP intrinsics
        cam0_intrs_for_ray = hybrid_intrs[:, 0].copy()   # (T, 3, 3)
        dcx = float(np.abs(hybrid_intrs[:,:,0,2] - orig_intrinsics[:,:,0,2]).mean())
        dcy = float(np.abs(hybrid_intrs[:,:,1,2] - orig_intrinsics[:,:,1,2]).mean())
        print(f"Combine xy+z: hybrid PP applied  |Δcx|={dcx:.1f}px  |Δcy|={dcy:.1f}px")

    # ── Exclude cameras ───────────────────────────────────────────────────────
    _vnames = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
    if args.exclude_cams:
        excl = set(args.exclude_cams)
        for ki, cn in enumerate(_vnames):
            if cn in excl:
                placer.cam_valid[:, ki] = False
                print(f"Excluded camera: {cn}")

    # ── Smooth camera extrinsics ──────────────────────────────────────────────
    if args.smooth_cameras > 0:
        w = args.smooth_cameras
        if w % 2 == 0:
            w += 1
            print(f"smooth_cameras: window rounded up to {w} (must be odd)")
        placer.extrinsics = smooth_extrinsics_sg(placer.extrinsics, window=w)
        print(f"Extrinsics smoothed: window={w}")

    # ── Intrinsics table ──────────────────────────────────────────────────────
    print(f"\n  {'Camera':<10} {'fx':>8} {'fy':>8} {'cx':>8} {'cy':>8}  (median over T frames)")
    for ki, cn in enumerate(_vnames):
        intr = placer.intrinsics[:, ki]   # (T, 3, 3)
        v_intr = orig_intrinsics[:, ki]   # (T, 3, 3) original VGGT
        fx  = float(np.median(intr[:, 0, 0]));  fy  = float(np.median(intr[:, 1, 1]))
        cx  = float(np.median(intr[:, 0, 2]));  cy  = float(np.median(intr[:, 1, 2]))
        cx0 = float(np.median(v_intr[:, 0, 2])); cy0 = float(np.median(v_intr[:, 1, 2]))
        dcx = cx - cx0;  dcy = cy - cy0
        excl_tag = " [EXCL]" if cn in set(args.exclude_cams) else ""
        print(f"  {cn:<10} {fx:>8.1f} {fy:>8.1f} {cx:>8.1f} {cy:>8.1f}"
              f"   Δcx={dcx:+.1f}  Δcy={dcy:+.1f}{excl_tag}")
    print()

    # ── GT ────────────────────────────────────────────────────────────────────
    gt_body_data, gt_betas = _load_gt(scene_name, rich_root, args.body_split)
    print(f"GT: {len(gt_body_data)} persons\n")

    # Optionally replace pred betas with GT betas to isolate betas as error source
    betas_for_fk = gt_betas if args.gt_betas else pred_betas
    if args.gt_betas:
        print("Using GT betas for FK (oracle beta test)")

    # ── Run ───────────────────────────────────────────────────────────────────
    joint_map = _GOLIATH_SMPLX_DISTAL if args.distal_only else _GOLIATH_SMPLX_ALIGN
    jnames = "ankles+elbows+wrists" if args.distal_only else "all (hips+knees+ankles+elbows+wrists)"
    common_kwargs = dict(
        placer             = placer,
        scale              = scale,
        sapiens_data       = sapiens_data,
        pred_betas         = betas_for_fk,
        fused_pose_by_pid  = fused_pose_by_pid,
        frame_start        = frame_start,
        conf_thr           = args.conf_thr,
        min_cams           = args.min_cams,
        min_joints         = args.min_joints,
        joint_map          = joint_map,
        sin_weight         = args.sin_weight,
    )

    # ── Placement ────────────────────────────────────────────────────────────
    if args.cam0_lateral:
        pred_trans, pred_orient, proc_scales = run_cam0_lateral_dlt(
            placer             = placer,
            scale              = scale,
            sapiens_data       = sapiens_data,
            pred_betas         = pred_betas,
            fused_pose_by_pid  = fused_pose_by_pid,
            frame_start        = frame_start,
            cam0_intrs         = cam0_intrs_for_ray,
            conf_thr           = args.conf_thr,
            min_joints         = args.min_joints,
            joint_map          = joint_map,
        )
    elif args.combine_xy_z:
        # DLT+Procrustes gives best z and orientation; cam0_lateral gives best xy
        pred_trans_dlt, pred_orient_dlt, proc_scales_dlt = run_sapiens_procrustes(
            **common_kwargs, sim3=False
        )
        pred_trans_cam0, _, _ = run_cam0_lateral_dlt(
            placer             = placer,
            scale              = scale,
            sapiens_data       = sapiens_data,
            pred_betas         = pred_betas,
            fused_pose_by_pid  = fused_pose_by_pid,
            frame_start        = frame_start,
            cam0_intrs         = cam0_intrs_for_ray,
            conf_thr           = args.conf_thr,
            min_joints         = args.min_joints,
            joint_map          = joint_map,
        )
        # Assemble combined root: xy from cam0_lateral, z from DLT
        pred_trans = {}
        for pid, dlt_frames in pred_trans_dlt.items():
            combined: dict[int, np.ndarray] = {}
            cam0_frames = pred_trans_cam0.get(pid, {})
            for frame, dlt_root in dlt_frames.items():
                if frame in cam0_frames:
                    r = cam0_frames[frame]
                    combined[frame] = np.array([r[0], r[1], dlt_root[2]], dtype=np.float32)
                else:
                    combined[frame] = dlt_root   # fall back to DLT if cam0 ray unavailable
            pred_trans[pid] = combined
        pred_orient  = pred_orient_dlt
        proc_scales  = proc_scales_dlt
    else:
        pred_trans, pred_orient, proc_scales = run_sapiens_procrustes(**common_kwargs, sim3=args.sim3)
    total = sum(len(v) for v in pred_trans.values())
    print(f"  {len(pred_trans)} pids, {total} frames")

    # ── Build eval arrays once (FK is expensive, do it once) ─────────────────
    pid_map = {1: args.gt_pid} if args.gt_pid else None
    arrays = build_eval_arrays(
        pred_trans        = pred_trans,
        pred_orient       = pred_orient,
        gt_body_data      = gt_body_data,
        gt_betas          = gt_betas,
        fused_pose_by_pid = fused_pose_by_pid,
        frame_start       = frame_start,
        pred_betas        = betas_for_fk,
        placer            = placer,
        proc_scales       = proc_scales,
        pid_map           = pid_map,
    )
    if arrays is None:
        print("ERROR: no predictions — nothing to evaluate"); return
    pred_joints_rel, pred_roots, gt_joints, gt_roots, frame_start_t, T, P, all_oerrs = arrays

    # ── No smoothing baseline ─────────────────────────────────────────────────
    m0 = score_from_arrays(pred_joints_rel, pred_roots, gt_joints, gt_roots,
                           all_oerrs, label="no smoothing")
    # Procrustes scale report (sim3 mode)
    if args.sim3 and proc_scales:
        all_s = [s for pid_s in proc_scales.values() for s in pid_s.values()]
        arr_s = np.array(all_s)
        print(f"    Procrustes scale  mean={arr_s.mean():.4f}  median={np.median(arr_s):.4f}"
              f"  std={arr_s.std():.4f}  min={arr_s.min():.4f}  max={arr_s.max():.4f}")

    # Depth / lateral breakdown (cam0 = world origin → z = depth axis)
    valid_r = np.isfinite(pred_roots).all(-1) & np.isfinite(gt_roots).all(-1)
    if valid_r.any():
        delta_r = (pred_roots - gt_roots)[valid_r]
        _dx  = float(np.abs(delta_r[:, 0]).mean()) * 100
        _dy  = float(np.abs(delta_r[:, 1]).mean()) * 100
        _dz  = float(np.abs(delta_r[:, 2]).mean()) * 100
        _dxy = float(np.linalg.norm(delta_r[:, :2], axis=-1).mean()) * 100
        print(f"    x={_dx:.1f}cm  y={_dy:.1f}cm  depth(z)={_dz:.1f}cm  lateral(xy)={_dxy:.1f}cm")

    # ── Savitzky-Golay sweep ──────────────────────────────────────────────────
    # Only pred_roots changes per window; pred_joints_rel is reused as-is.
    pid_slots = {ghost_pid: slot for slot, ghost_pid in enumerate(sorted(pred_trans))}

    print(f"\n  {'Window':>8}  {'WA(mm)':>8}  {'W(mm)':>8}  "
          f"{'RTE(cm)':>8}  {'root(cm)':>9}  {'ΔRTE':>7}  {'ΔW':>7}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}")
    for win in [5, 7, 9, 11, 13, 15]:
        pt_s = smooth_translations_sg(pred_trans, window=win, polyorder=2)
        pr_smooth = np.full_like(pred_roots, np.nan)
        for ghost_pid, slot in pid_slots.items():
            if ghost_pid not in pt_s:
                continue
            for gf, root in pt_s[ghost_pid].items():
                t_rel = int(gf) - frame_start_t
                if 0 <= t_rel < T:
                    pr_smooth[t_rel, slot] = root
        m = score_from_arrays(pred_joints_rel, pr_smooth, gt_joints, gt_roots,
                              all_oerrs, label=f"win={win}", verbose=False)
        d_rte = (m["RTE"] - m0["RTE"]) * 100
        d_w   =  m["W"]  - m0["W"]
        print(f"  {win:>8}  {m['WA']:>8.1f}  {m['W']:>8.1f}  "
              f"{m['RTE']*100:>8.2f}  {m['root_mean']*100:>9.1f}  "
              f"{d_rte:>+7.2f}  {d_w:>+7.1f}")


if __name__ == "__main__":
    main()
