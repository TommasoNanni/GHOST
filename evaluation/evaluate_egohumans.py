#!/usr/bin/env python
"""Evaluate the ghost pipeline on EgoHumans, CHROMM single-frame protocol.

Metrics (metres, per the CHROMM paper "single-frame setting" on EgoHumans):
  W-MPJPE†  — world MPJPE after aligning ONLY the camera poses with SE(3)
              (no scale). Per-frame.  Needs GT cameras -> world frame.
  GA-MPJPE  — a single Sim(3) over ALL persons jointly, per frame
              (group-aligned; keeps inter-person geometry).
  PA-MPJPE  — per-person Procrustes (Sim3) per frame (local pose only).

Every frame is treated as a standalone scene: NO temporal attention in the
fusion module (frames go in the batch dim, temporal length 1).

Predictions come from ``<cam>/body_data_clean/person_<GLOBAL>.npz`` — already
within-view-ops'd and globally re-id'd, so global person id N == GT aria0N.
SMPL-X body pose -> SMPL-X FK verts -> SMPL verts (smplx2smpl) -> COCO-17
(J_regressor_coco), matching the GT ``poses3d`` COCO-17 joints.

World frame = aria01 SLAM world (GT ``poses3d`` is native there; the GT exo
cameras from ``images.txt`` (colmap world) are transformed into it via
``colmap_from_aria['aria01']^-1``).

Two stages so the 1.5 h walltime cannot lose work:
  Stage A (GPU): per scene, dump ``<dump_dir>/<scene>.npz`` =
      {pred_coco (T,P,17,3) in aria world, gt_coco (T,P,17,3), valid (T,P)}.
      Scenes with an existing dump are skipped -> relaunch until all done.
  Stage B (CPU): ``--metrics_only`` loads every dump and prints the table.

Usage
-----
  # single scene (prints metrics inline; needs a GT dir with poses3d + colmap)
  pixi run python evaluation/evaluate_egohumans.py \
      --ghost_root /iopsstor/.../egohumans_new/06_badminton \
      --gt_root    <mnt>/.../camera_ready/06_badminton \
      --checkpoint checkpoints/fusion_module/best.pt \
      --scene 031_badminton --dump_dir eval_egohumans/dumps

  # full run (resumable), then aggregate
  pixi run python evaluation/evaluate_egohumans.py --ghost_root ... --gt_root ... \
      --checkpoint ... --dump_dir eval_egohumans/dumps
  pixi run python evaluation/evaluate_egohumans.py --metrics_only \
      --dump_dir eval_egohumans/dumps
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.fusion_module_v2 import PoseFusionModule
from fusion.placer import BodyPlacer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_egohumans")

_NUM_JOINTS = 55                       # SMPL-X joints fed to fusion (root + 54)
# COCO-17: 0 nose,1/2 eyes,3/4 ears,5/6 shoulders,7/8 elbows,9/10 wrists,
#          11/12 hips,13/14 knees,15/16 ankles. Score the 12 limb joints only
#          (face joints have no clean SMPL-X correspondence).
_LIMB_COCO = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
_SMPL_EVAL = list(range(24))          # all 24 SMPL joints (GT is SMPL; see smpl24_regressor)
_COCO_REG = None


def coco_regressor() -> np.ndarray:
    """(17, 10475): SMPL-X verts -> COCO-17 via smplx2smpl then J_regressor_coco."""
    global _COCO_REG
    if _COCO_REG is None:
        bm = _REPO_ROOT / "body_models"
        jr = np.load(bm / "coco" / "J_regressor_coco.npy")                 # (17, 6890)
        mtx = pickle.load(open(bm / "smplx2smpl.pkl", "rb"))["matrix"]      # (6890, 10475)
        _COCO_REG = (jr @ mtx).astype(np.float64)
    return _COCO_REG


_SMPL_REG = None
def smpl24_regressor() -> np.ndarray:
    """(24, 10475): SMPL-X verts -> SMPL-24 joints via smplx2smpl then SMPL J_regressor.

    GT is SMPL (EgoHumans processed_data/smpl), so we score in SMPL-24 joint space:
    our SMPL-X FK verts -> SMPL verts (smplx2smpl) -> 24 SMPL joints (SMPL J_regressor),
    matching the GT smpl ``joints[:24]``.
    """
    global _SMPL_REG
    if _SMPL_REG is None:
        import scipy.sparse as _sp
        bm = _REPO_ROOT / "body_models"
        jr = pickle.load(open(bm / "smpl" / "SMPL_NEUTRAL.pkl", "rb"),
                         encoding="latin1")["J_regressor"]                  # (24, 6890) sparse
        jr = jr.toarray() if _sp.issparse(jr) else np.asarray(jr)
        mtx = pickle.load(open(bm / "smplx2smpl.pkl", "rb"))["matrix"]      # (6890, 10475)
        _SMPL_REG = (jr @ mtx).astype(np.float64)
    return _SMPL_REG


# ── geometry helpers (copied — this is a standalone eval) ──────────────────
def se3_align(src: np.ndarray, dst: np.ndarray):
    """Kabsch SE(3) (no scale): R,t minimising ||R@src+t-dst||. src,dst (N,3)."""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, dc - R @ sc


def sim3_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return pred aligned to gt by Sim(3). pred,gt (N,3)."""
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    s = np.sqrt((g0 ** 2).sum() / ((p0 ** 2).sum() + 1e-12))
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return s * p0 @ R.T + gc


def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """(J,3) axis-angle -> (J,6) = first two ROWS of R (training convention)."""
    from scipy.spatial.transform import Rotation as SciR
    Rm = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()      # (J,3,3)
    return Rm[:, :2, :].reshape(-1, 6)                        # first two rows


def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    """(J,6) -> (J,3) axis-angle; 6 = first two ROWS of R (Gram-Schmidt)."""
    from scipy.spatial.transform import Rotation as SciR
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    Rm = np.stack([b1, b2, b3], axis=1)                      # rows are b1,b2,b3
    return SciR.from_matrix(Rm).as_rotvec()


def load_fusion_model(ckpt: Path, device) -> PoseFusionModule:
    """Load PoseFusionModule, inferring architecture from the checkpoint."""
    c = torch.load(ckpt, map_location=device)
    state = c.get("model_state_dict", c.get("model", c))
    emb = state["joint_id_embedding.weight"].shape[1]
    n_joints = state["joint_id_embedding.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("layers.") and k.endswith(".ff.norm.weight"))
    max_tlen = state["temporal_pe.pe"].shape[0]
    model = PoseFusionModule(embedding_dim=emb, num_layers=n_layers,
                             num_joints=n_joints, max_temporal_len=max_tlen).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"fusion ckpt: emb={emb} layers={n_layers} joints={n_joints} maxT={max_tlen}")
    return model


# ── GT loading (EgoHumans) ─────────────────────────────────────────────────
def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _colmap_to_aria(gt_scene: Path):
    """4x4 mapping colmap-world -> aria01-world, or None if pkl missing."""
    p = gt_scene / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    if not p.exists():
        return None
    d = _load_pkl(p)
    T_aria_to_colmap = np.asarray(d["aria01"], dtype=np.float64)   # aria01 -> colmap
    return np.linalg.inv(T_aria_to_colmap)


def _gt_exo_cameras_aria(gt_scene: Path, T_c2a):
    """{cam_name: centre (3,) in aria world} from images.txt exo entries."""
    imgs = gt_scene / "colmap" / "workplace" / "images.txt"
    out: dict[str, np.ndarray] = {}
    from scipy.spatial.transform import Rotation as SciR
    with open(imgs) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:      # image lines only (skip POINTS2D lines)
                continue
            try:
                qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
            except ValueError:
                continue
            name = parts[9]
            cam = name.split("/")[0]
            if not cam.startswith("cam") or cam in out:
                continue
            R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()   # world->cam
            C = -R.T @ np.array([tx, ty, tz])                  # centre, colmap world
            if T_c2a is not None:
                C = (T_c2a[:3, :3] @ C) + T_c2a[:3, 3]         # -> aria world
            out[cam] = C
    return out


def load_gt_scene(gt_scene: Path):
    """Return (frames, gt_by_frame, cam_pos_aria, have_world).

    frames        : sorted list of GT frame indices (1-based, from smpl files)
    gt_by_frame   : {frame: {aria_id: (24,3) SMPL joints, aria world}}
    cam_pos_aria  : {cam_name: (3,) centre in aria world}  (empty if no pkl)
    have_world    : True if the colmap_from_aria pkl was found (=> W-MPJPE† ok)
    """
    smpl_dir = gt_scene / "processed_data" / "smpl"
    if not smpl_dir.is_dir():
        raise FileNotFoundError(f"no processed_data/smpl in {gt_scene}")
    T_c2a = _colmap_to_aria(gt_scene)
    cam_pos = _gt_exo_cameras_aria(gt_scene, T_c2a) if T_c2a is not None else {}
    frames, gt_by_frame = [], {}
    for f in sorted(smpl_dir.glob("*.npy")):
        try:
            fi = int(f.stem)
        except ValueError:
            continue
        arr = np.load(str(f), allow_pickle=True)
        d = arr.item() if arr.dtype == object and arr.shape == () else arr
        if not isinstance(d, dict):
            continue
        people = {}
        for aid, params in d.items():
            if not isinstance(params, dict) or "joints" not in params:
                continue
            j = np.asarray(params["joints"], dtype=np.float64)   # (45,3) SMPL joints, aria world
            if j.shape[0] >= 24 and j.shape[1] >= 3:
                people[aid] = j[:24, :3]
        if people:
            frames.append(fi)
            gt_by_frame[fi] = people
    return sorted(frames), gt_by_frame, cam_pos, (T_c2a is not None)


# ── prediction: fusion (per-frame) -> placer -> FK -> COCO ─────────────────
def _load_clean_tracks(ghost_scene: Path, cam_names, valid_mask, pids):
    """{cam_idx: {pid: {frame: entry}}} from body_data_clean."""
    per_cam = {}
    for k, cam in enumerate(cam_names):
        if not valid_mask[k]:
            per_cam[k] = {}
            continue
        bd = ghost_scene / cam / "body_data_clean"
        cam_map = {}
        for pid in pids:
            f = bd / f"person_{pid}.npz"
            if not f.exists():
                continue
            d = np.load(str(f), allow_pickle=False)
            fi = d["frame_indices"].astype(int)
            frames_map = {}
            for t, gfr in enumerate(fi):
                e = {"go": d["smplx_global_orient"][t].reshape(3),
                     "bp": d["smplx_body_pose"][t].reshape(63)}
                if "smplx_left_hand_pose" in d.files:
                    e["lh"] = d["smplx_left_hand_pose"][t].reshape(-1, 3)
                if "smplx_right_hand_pose" in d.files:
                    e["rh"] = d["smplx_right_hand_pose"][t].reshape(-1, 3)
                if "smplx_betas" in d.files:
                    e["betas"] = d["smplx_betas"][t][:10]
                frames_map[int(gfr)] = e
            if frames_map:
                cam_map[pid] = frames_map
        per_cam[k] = cam_map
    return per_cam


def _pack_pose(entry) -> np.ndarray:
    parts = [entry["go"].reshape(1, 3), entry["bp"].reshape(21, 3)]
    parts.append(entry.get("lh", np.zeros((15, 3))).reshape(15, 3))
    parts.append(entry.get("rh", np.zeros((15, 3))).reshape(15, 3))
    aa = np.concatenate(parts, 0).astype(np.float64)
    if aa.shape[0] < _NUM_JOINTS:
        aa = np.concatenate([aa, np.zeros((_NUM_JOINTS - aa.shape[0], 3))], 0)
    return _aa_to_6d(aa)[1:]            # (54,6) root excluded


def predict_scene(ghost_scene: Path, frames, pids, fusion_model, device,
                  smplx_arg, scale_mode="pred", temporal=False):
    """Return pred_coco {pid: {frame: (17,3) coco in aria world}} + R_align,t_align.

    R_align,t_align map ghost-metric(vggt cam-0) -> aria world (None if no cams).
    """
    vggt = np.load(ghost_scene / "vggt_cameras_centered.npz", allow_pickle=False)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    valid_mask = vggt["valid"][0]
    extrinsics = vggt["extrinsics"][0]
    valid_full = vggt["valid"]            # (T,K) per-frame camera validity
    extrinsics_full = vggt["extrinsics"]  # (T,K,3,4) per-frame extrinsics

    per_cam = _load_clean_tracks(ghost_scene, cam_names, valid_mask, pids)
    frames_set = set(int(f) for f in frames)
    K, P = len(cam_names), len(pids)
    pid_slot = {p: i for i, p in enumerate(pids)}
    fmin, fmax = frames[0], frames[-1]
    T = fmax - fmin + 1

    # dense per-frame tensors (T,1,K,P,54,6) — temporal length 1 => no temporal attn
    pose = np.zeros((T, 1, K, P, 54, 6), dtype=np.float32)
    mask = np.zeros((T, 1, K, P), dtype=np.float32)
    betas_acc = {p: [] for p in pids}
    for k, cam_map in per_cam.items():
        for pid, fm in cam_map.items():
            s = pid_slot[pid]
            for gfr, e in fm.items():
                if fmin <= gfr <= fmax:
                    t = gfr - fmin
                    pose[t, 0, k, s] = _pack_pose(e)
                    mask[t, 0, k, s] = 1.0
                    if "betas" in e:
                        betas_acc[pid].append(e["betas"])
    betas_by_pid = {p: (np.mean(v, 0).astype(np.float32) if v else np.zeros(10, np.float32))
                    for p, v in betas_acc.items()}

    with torch.no_grad():
        if temporal:
            # natural mode: one sequence (1,T,K,P,...) -> temporal attention on
            seq = pose.transpose(1, 0, 2, 3, 4, 5)         # (1,T,K,P,54,6)
            msq = mask.transpose(1, 0, 2, 3)               # (1,T,K,P)
            chunks = []
            for t0 in range(0, T, 512):
                pt = torch.from_numpy(seq[:, t0:t0 + 512]).to(device)
                mt = torch.from_numpy(msq[:, t0:t0 + 512]).to(device)
                chunks.append(fusion_model(pt, mt)[0].cpu().numpy())   # (t,P,54,6)
            fused = np.concatenate(chunks, 0)
        else:
            # Per-frame protocol, but the fusion model is OOD at temporal length 1
            # (trained with window 128; see evaluate_egoexo.py, PA 153 vs 125).
            # Replicate each frame TEMPORAL_PAD times along the temporal axis and
            # read back frame 0 — no cross-frame information is used.
            TEMPORAL_PAD = 32
            chunks = []
            bs = max(1, 256 // TEMPORAL_PAD)
            for t0 in range(0, T, bs):
                pt = torch.from_numpy(pose[t0:t0 + bs]).repeat(1, TEMPORAL_PAD, 1, 1, 1, 1).to(device)
                mt = torch.from_numpy(mask[t0:t0 + bs]).repeat(1, TEMPORAL_PAD, 1, 1).to(device)
                chunks.append(fusion_model(pt, mt)[:, 0].cpu().numpy())   # (b,P,54,6)
            fused = np.concatenate(chunks, 0)              # (T,P,54,6)

    # BodyPlacer hard-codes cam/body_data; give it a symlinked view whose
    # body_data points at body_data_clean (global ids, ops applied). Identity
    # remap since clean already uses global ids.
    view = _clean_scene_view(ghost_scene, cam_names)
    try:
        placer = BodyPlacer(view, smplx_arg, crop_meta_path=None)
        fused_pose_by_pid = {p: fused[:, pid_slot[p]] for p in pids}

        if scale_mode == "triangulated":
            try:
                scale = placer.estimate_scale_triangulated(
                    fused_pose_by_pid=fused_pose_by_pid, frame_start=fmin)
            except Exception:
                scale = placer.load_mapanything_scale()
        elif scale_mode == "baseline":
            scale = placer.load_mapanything_scale(filename="mapanything_scale_baseline.npy", smooth="median")
            if scale is None:
                raise RuntimeError("scale_mode=baseline but mapanything_scale_baseline.npy missing/mismatched")
        elif scale_mode == "human":
            try:
                scale = placer.estimate_scale_human_reference(frame_start=fmin)
            except Exception:
                scale = placer.load_mapanything_scale()
        else:
            scale = placer.load_mapanything_scale()
        if scale is None:
            scale = np.ones(placer.T, dtype=np.float32)
        # W† camera alignment must use the SAME scale the placer placed with
        ma_scale = float(np.median(np.asarray(scale)))

        trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
            scale=scale, all_pids=set(pids), pred_betas_by_pid=betas_by_pid,
            fused_pose_by_pid=fused_pose_by_pid, frame_start=fmin)

        reg = smpl24_regressor()
        pred_coco = {p: {} for p in pids}
        for pid in pids:
            s = pid_slot[pid]
            betas_p = betas_by_pid[pid][np.newaxis]
            for gfr in frames:
                pw = trans_dict.get(pid, {}).get(gfr)
                R_m = orient_dict.get(pid, {}).get(gfr)
                if pw is None or R_m is None:
                    continue
                t = gfr - fmin
                bp_aa = _6d_to_aa(fused[t, s, :21]).reshape(63)
                J55, verts = placer._smplx_fk(betas_p, bp_aa[np.newaxis],
                                              np.zeros((1, 3), np.float32), return_verts=True)
                pelvis = J55[0].astype(np.float64)[0]
                smpl_can = reg @ verts[0].astype(np.float64)         # (24,3) canonical SMPL joints
                pred_coco[pid][gfr] = (R_m @ (smpl_can - pelvis).T).T + pw   # ghost-metric

        # Raw single-view baseline: canonical SMPL-24 from each cam's own SMPL-X
        # pose (no fusion, no placement — PA-comparable only).
        raw_coco = {p: {} for p in pids}          # {pid: {frame: {cam_idx: (24,3)}}}
        for k, cam_map in per_cam.items():
            for pid, fm in cam_map.items():
                betas_p = betas_by_pid[pid][np.newaxis]
                for gfr, e in fm.items():
                    if gfr not in frames_set:
                        continue
                    _, verts = placer._smplx_fk(betas_p, e["bp"].reshape(1, 63),
                                                np.zeros((1, 3), np.float32), return_verts=True)
                    raw_coco[pid].setdefault(gfr, {})[k] = reg @ verts[0].astype(np.float64)
    finally:
        import shutil
        shutil.rmtree(view, ignore_errors=True)
    return pred_coco, raw_coco, cam_names, extrinsics_full, valid_full, ma_scale


def _clean_scene_view(ghost_scene: Path, cam_names) -> Path:
    """Temp scene dir: root vggt/scale files symlinked; each cam/body_data ->
    the real cam/body_data_clean. Lets BodyPlacer read the clean tracks."""
    import tempfile, os
    view = Path(tempfile.mkdtemp(prefix="cleanview_"))
    for f in ghost_scene.iterdir():
        if f.is_file():
            os.symlink(f, view / f.name)
    for cam in cam_names:
        clean = ghost_scene / cam / "body_data_clean"
        if clean.is_dir():
            (view / cam).mkdir(parents=True, exist_ok=True)
            os.symlink(clean, view / cam / "body_data")
    return view


# ── per-scene orchestration ────────────────────────────────────────────────
def eval_scene(ghost_scene: Path, gt_scene: Path, fusion_model, device, smplx_arg,
               scale_mode="pred", temporal=False):
    """Return dump dict {pred (T,P,17,3) aria, gt, valid (T,P), have_world} or None."""
    if not (ghost_scene / "vggt_cameras_centered.npz").exists():
        logger.warning(f"{ghost_scene.name}: no vggt cameras, skip"); return None
    frames, gt_by_frame, cam_pos_aria, have_world = load_gt_scene(gt_scene)
    if not frames:
        logger.warning(f"{ghost_scene.name}: no GT frames, skip"); return None

    aria_ids = sorted({a for fr in gt_by_frame.values() for a in fr})   # e.g. aria01..04
    pid_of_aria = {a: int(a.replace("aria", "")) for a in aria_ids}     # aria0N -> N
    pids = sorted(pid_of_aria.values())

    pred_coco, raw_coco, cam_names, extrinsics_full, valid_full, ma_scale = predict_scene(
        ghost_scene, frames, pids, fusion_model, device, smplx_arg, scale_mode, temporal)

    # Per-frame SE(3): ghost-metric cameras (this frame) -> aria-world GT cameras.
    # VGGT cameras are estimated per frame, so a single global SE(3) would dump
    # per-frame camera jitter into W†. CHROMM's single-frame protocol aligns the
    # camera poses per frame (GA/PA are per-frame); we fit SE(3) per frame for W†
    # too (still no scaling). Frames with <2 valid cams fall back to a global
    # (median-camera) SE(3).
    fmin = frames[0]
    Textr = extrinsics_full.shape[0]

    def _frame_se3(t):
        pc, gc = [], []
        for k, cam in enumerate(cam_names):
            if 0 <= t < Textr and valid_full[t, k] and cam in cam_pos_aria:
                Rk, tk = extrinsics_full[t, k, :, :3], extrinsics_full[t, k, :, 3]
                pc.append((-Rk.T @ tk) * ma_scale)
                gc.append(cam_pos_aria[cam])
        return se3_align(np.stack(pc), np.stack(gc)) if len(pc) >= 2 else None

    R_glob = t_glob = None
    if have_world and cam_pos_aria:
        pcg, gcg = [], []
        for k, cam in enumerate(cam_names):
            if cam not in cam_pos_aria:
                continue
            ctrs = [-(extrinsics_full[tt, k, :, :3].T @ extrinsics_full[tt, k, :, 3]) * ma_scale
                    for tt in range(Textr) if valid_full[tt, k]]
            if ctrs:
                pcg.append(np.median(ctrs, 0)); gcg.append(cam_pos_aria[cam])
        if len(pcg) >= 2:
            R_glob, t_glob = se3_align(np.stack(pcg), np.stack(gcg))

    # assemble (T,P,17,3): pred aligned per-frame into aria world; gt in aria world
    P = len(pids); F = len(frames)
    fidx = {fr: i for i, fr in enumerate(frames)}
    pred = np.full((F, P, 24, 3), np.nan)
    gt = np.full((F, P, 24, 3), np.nan)
    for fr in frames:
        i = fidx[fr]
        RA = None
        if have_world and cam_pos_aria:
            RA = _frame_se3(fr - fmin)
            if RA is None and R_glob is not None:
                RA = (R_glob, t_glob)
        for si, pid in enumerate(pids):
            aid = f"aria{pid:02d}"
            if aid in gt_by_frame[fr]:
                gt[i, si] = gt_by_frame[fr][aid]
            j = pred_coco[pid].get(fr)
            if j is None:
                continue
            pred[i, si] = (j @ RA[0].T + RA[1]) if RA is not None else j
    # raw single-view canonical poses: (T,P,K,17,3), NaN where absent
    Kn = len(cam_names)
    raw = np.full((F, P, Kn, 24, 3), np.nan)
    for si, pid in enumerate(pids):
        for fr, camj in raw_coco[pid].items():
            for k, j in camj.items():
                raw[fidx[fr], si, k] = j
    return {"pred": pred.astype(np.float32), "gt": gt.astype(np.float32),
            "raw": raw.astype(np.float32),
            "have_world": have_world, "pids": np.array(pids), "frames": np.array(frames)}


# ── metrics over a dump ────────────────────────────────────────────────────
def scene_metrics(pred, gt, have_world):
    """W†/GA/PA in metres over the 12 limb joints; per-frame single-frame protocol."""
    L = _SMPL_EVAL
    T, P = pred.shape[:2]
    valid = np.isfinite(pred[..., L, :]).all((-1, -2)) & np.isfinite(gt[..., L, :]).all((-1, -2))
    w, ga, pa = [], [], []
    for t in range(T):
        ps = [p for p in range(P) if valid[t, p]]
        if not ps:
            continue
        pr = pred[t][ps][:, L]      # (n,12,3)
        gtt = gt[t][ps][:, L]
        if have_world:
            w.append(np.linalg.norm(pr - gtt, axis=-1).mean())          # already SE3-cam-aligned
        # GA: single Sim3 over all persons jointly, this frame
        a = sim3_align(pr.reshape(-1, 3), gtt.reshape(-1, 3)).reshape(pr.shape)
        ga.append(np.linalg.norm(a - gtt, axis=-1).mean())
        # PA: per person
        for i in range(len(ps)):
            a = sim3_align(pr[i], gtt[i])
            pa.append(np.linalg.norm(a - gtt[i], axis=-1).mean())
    return (float(np.mean(w)) if w else float("nan"),
            float(np.mean(ga)) if ga else float("nan"),
            float(np.mean(pa)) if pa else float("nan"))


def raw_pa(raw, gt):
    """Single-view PA baselines, same protocol as fused PA (mean over frames/persons).
    Returns (best_view_pa, median_view_pa): per (frame,person), Procrustes-align each
    camera's raw pose to GT, then take best / median across cameras."""
    L = _SMPL_EVAL
    T, P, K = raw.shape[:3]
    best, med = [], []
    for t in range(T):
        for p in range(P):
            if not np.isfinite(gt[t, p, L]).all():
                continue
            errs = []
            for k in range(K):
                if not np.isfinite(raw[t, p, k, L]).all():
                    continue
                a = sim3_align(raw[t, p, k, L], gt[t, p, L])
                errs.append(np.linalg.norm(a - gt[t, p, L], axis=-1).mean())
            if errs:
                best.append(min(errs)); med.append(float(np.median(errs)))
    return (float(np.mean(best)) if best else float("nan"),
            float(np.mean(med)) if med else float("nan"))


def aggregate(dump_dir: Path):
    rows = []
    for f in sorted(dump_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        w, ga, pa = scene_metrics(d["pred"], d["gt"], bool(d["have_world"]))
        rows.append((f.stem, w, ga, pa))
        extra = ""
        if "raw" in d.files:
            b, m = raw_pa(d["raw"], d["gt"])
            extra = f"   | single-view PA: best={b*1000:.1f} median={m*1000:.1f}"
        print(f"  {f.stem:24s}  W†={w*1000:7.1f}  GA={ga*1000:7.1f}  PA={pa*1000:7.1f} mm{extra}")
    if not rows:
        print("no dumps"); return
    W = np.array([r[1] for r in rows]); G = np.array([r[2] for r in rows]); A = np.array([r[3] for r in rows])
    print("\n=== AGGREGATE (metres · mm) — CHROMM EgoHumans: W†0.51 GA0.15 PA0.05 ===")
    print(f"  scenes: {len(rows)}  (W† on {np.isfinite(W).sum()})")
    print(f"  W-MPJPE†  mean {np.nanmean(W):.3f} m  ({np.nanmean(W)*1000:.1f} mm)")
    print(f"  GA-MPJPE  mean {np.nanmean(G):.3f} m  ({np.nanmean(G)*1000:.1f} mm)")
    print(f"  PA-MPJPE  mean {np.nanmean(A):.3f} m  ({np.nanmean(A)*1000:.1f} mm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root")
    ap.add_argument("--gt_root", help="camera_ready/<activity> dir with per-scene GT")
    ap.add_argument("--checkpoint")
    ap.add_argument("--smplx_model", default=str(_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl"))
    ap.add_argument("--scale", choices=["pred", "triangulated", "baseline", "human"], default="pred")
    ap.add_argument("--temporal", action="store_true", help="use temporal fusion (default: per-frame)")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--dump_dir", default="eval_egohumans/dumps")
    ap.add_argument("--metrics_only", action="store_true", help="Stage B: aggregate dumps")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir); dump_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics_only:
        aggregate(dump_dir); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model = load_fusion_model(Path(args.checkpoint), device)
    smplx_arg = args.smplx_model
    ghost_root, gt_root = Path(args.ghost_root), Path(args.gt_root)

    scenes = [args.scene] if args.scene else sorted(
        p.name for p in ghost_root.iterdir()
        if p.is_dir() and any(c.is_dir() and c.name.startswith("cam") for c in p.iterdir()))
    scenes = [s for s in scenes if (gt_root / s).exists()]

    for scene in scenes:
        out = dump_dir / f"{scene}.npz"
        if out.exists() and not args.overwrite:
            logger.info(f"{scene}: dump exists, skip"); continue
        try:
            d = eval_scene(ghost_root / scene, gt_root / scene, fusion_model, device,
                           smplx_arg, args.scale, args.temporal)
        except Exception as e:
            logger.warning(f"{scene}: FAILED — {e}"); continue
        if d is None:
            continue
        np.savez_compressed(out, **d)
        w, ga, pa = scene_metrics(d["pred"], d["gt"], d["have_world"])
        logger.info(f"{scene}: dumped  W†={w*1000:.1f} GA={ga*1000:.1f} PA={pa*1000:.1f} mm")

    if args.scene:
        print("\n--- single-scene metrics ---")
        aggregate(dump_dir)


if __name__ == "__main__":
    main()
