"""Silhouette-based body placement optimization test.

For each frame: pre-compute SMPL-X canonical vertices (mean pose/shape across cameras,
zero global_orient/transl), then optimize (R_world, t_world) with Adam to minimize
soft silhouette IoU loss across K cameras. Uses GT XML cameras (metric, no scale issue).

Usage:
    pixi run python -m scripts.test_silhouette_place [scene_name] [--stride N] [--n-frames N]
"""
from __future__ import annotations
import sys, time, gc, pickle, re, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as SciR
import xml.etree.ElementTree as ET
import smplx as smplx_lib
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer, SoftSilhouetteShader,
    MeshRenderer, BlendParams,
)
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils import cameras_from_opencv_projection

GHOST_OUT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_ROOT  = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
SMPLX_PATH = Path("/users/tnanni/ghost/body_models")
# Original image size for BBQ / most RICH scenes derived from GT intrinsics
# (cx≈1945→W≈3890, cy≈1473→H≈2946); masks are at 1440×1053
_GT_W_ORIG, _GT_H_ORIG = 3890, 2946
_MASK_W,    _MASK_H    = 1440, 1053

RENDER_H = 256
RENDER_W = 256
N_ITER   = 150
LR       = 5e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt_cameras(scene_name: str) -> list[np.ndarray]:
    """List of (3,4) float64 [R|t] camera-from-world, metres."""
    location  = re.match(r"^(.+?)_\d{3}", scene_name).group(1)
    calib_dir = RICH_ROOT / "scan_calibration" / location / "calibration"
    exts = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        node = ET.parse(xml_path).getroot().find("CameraMatrix")
        if node is None: continue
        vals = list(map(float, node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return exts


def load_gt_intrinsics(scene_name: str) -> list[np.ndarray]:
    """List of (3,3) float64 intrinsics for original resolution."""
    location  = re.match(r"^(.+?)_\d{3}", scene_name).group(1)
    calib_dir = RICH_ROOT / "scan_calibration" / location / "calibration"
    intrs = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        node = ET.parse(xml_path).getroot().find("Intrinsics")
        if node is None: continue
        vals = list(map(float, node.find("data").text.split()))
        intrs.append(np.array(vals, dtype=np.float64).reshape(3, 3))
    return intrs


def load_gt_body(scene_name: str) -> dict[int, dict[int, dict]]:
    """gt[pid][frame] = {global_orient (3,), transl (3,)}"""
    gt_root = RICH_ROOT / "train_body" / scene_name
    gt: dict[int, dict[int, dict]] = {}
    for fd in sorted(gt_root.iterdir()):
        if not fd.is_dir(): continue
        try: fi = int(fd.name)
        except ValueError: continue
        for pkl in sorted(fd.glob("*.pkl")):
            pid = int(pkl.stem)
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            gt.setdefault(pid, {})[fi] = {
                "global_orient": np.asarray(d["global_orient"], np.float32).reshape(3),
                "transl":        np.asarray(d["transl"],        np.float32).reshape(3),
            }
    return gt


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    dR = R1.T @ R2
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def chordal_mean(Rs: list[np.ndarray]) -> np.ndarray:
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _triangulate_dlt(obs: list[tuple[float, float]], pmats: list[np.ndarray]) -> np.ndarray:
    """DLT SVD triangulation. obs: list of (u,v), pmats: list of (3,4). Returns (3,) world."""
    A = []
    for (u, v), P in zip(obs, pmats):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


# MHR70 joint indices for the 3 Procrustes alignment joints.
# Same as _SMPLX_TO_MHR70_ALIGN in placer.py.
_ALIGN = {1: 9, 2: 10, 12: 69}   # smplx_j → mhr70_j  (L-hip, R-hip, neck)
_KP_W, _KP_H = 1440.0, 1053.0    # pred_keypoints_2d native resolution (SAM3D input)


def procrustes_init_gt(
    cam_body: list[dict | None],
    valid_ks: list[int],
    local_ts: dict[int, int],
    gt_exts: list[np.ndarray],   # (3,4) each, world-to-cam extrinsics
    gt_intrs: list[np.ndarray],  # (3,3) each, at _GT_W_ORIG × _GT_H_ORIG
    smplx_model,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """DLT-triangulate 3 joints with GT cameras, then Procrustes → (R_world, pelvis_world).

    Keypoints are assumed to be in _KP_W × _KP_H pixel space (SAM3D input resolution)
    and are scaled to GT intrinsic resolution (_GT_W_ORIG × _GT_H_ORIG) before DLT.
    Returns (R_world (3,3), pelvis_world (3,)) or (None, None) if < 2 joints triangulate.
    """
    sx = _GT_W_ORIG / _KP_W
    sy = _GT_H_ORIG / _KP_H
    Pmats = {k: gt_intrs[k] @ gt_exts[k] for k in valid_ks}

    joint_world: dict[int, np.ndarray] = {}
    for smplx_j, mhr70_j in _ALIGN.items():
        obs, pmats_j = [], []
        for k in valid_ks:
            kp = cam_body[k]["kp2d"][local_ts[k], mhr70_j]  # (2,) [x, y]
            obs.append((float(kp[0]) * sx, float(kp[1]) * sy))
            pmats_j.append(Pmats[k])
        if len(obs) >= 2:
            joint_world[smplx_j] = _triangulate_dlt(obs, pmats_j)

    if len(joint_world) < 2:
        return None, None

    # FK: canonical joints (zero orient, zero transl) for mean body_pose + betas
    mean_betas = np.mean([cam_body[k]["betas"][local_ts[k]]    for k in valid_ks], axis=0)
    mean_bp    = np.mean([cam_body[k]["body_pose"][local_ts[k]] for k in valid_ks], axis=0)
    with torch.no_grad():
        fk = smplx_model(
            betas=torch.tensor(mean_betas[None], dtype=torch.float32, device=device),
            body_pose=torch.tensor(mean_bp[None], dtype=torch.float32, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
            jaw_pose=torch.zeros(1, 3, device=device),
            leye_pose=torch.zeros(1, 3, device=device),
            reye_pose=torch.zeros(1, 3, device=device),
            expression=torch.zeros(1, smplx_model.num_expression_coeffs, device=device),
            left_hand_pose=torch.zeros(1, 45, device=device),
            right_hand_pose=torch.zeros(1, 45, device=device),
        )
    J_can = fk.joints[0].detach().cpu().numpy()  # (127, 3), first 55 are SMPL-X joints

    # Use the same formula as placer.py: R @ J_can[j] + t ≈ J_world[j].
    # This gives t_placer (not SMPLX transl), but:
    #   pelvis_world = R @ J_can[0] + t_placer = transl_smplx + J_can[0]
    # And rendering V_can @ R.T + t_placer is also correct (same algebra).
    # The metric must compare pelvis_world vs GT pelvis, not t_placer vs gt_transl.
    pelvis_can = J_can[0].astype(np.float64)
    vis = sorted(joint_world.keys())
    A = np.stack([joint_world[j] for j in vis])
    B = np.stack([J_can[j]       for j in vis])
    A_mu, B_mu = A.mean(0), B.mean(0)
    H = (B - B_mu).T @ (A - A_mu)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_world = (Vt.T @ np.diag([1.0, 1.0, d]) @ U.T).astype(np.float64)
    t_placer = (A_mu - R_world @ B_mu).astype(np.float64)
    # pelvis_world is what evaluate_on_rich_test.py compares against GT.
    pelvis_world = R_world @ pelvis_can + t_placer
    return R_world, t_placer, pelvis_world


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft 1-IoU. pred: (H,W) float[0,1], target: (H,W) float{0,1}."""
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return 1.0 - inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scene_name: str, stride: int = 10, n_frames: int | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nScene: {scene_name}  device: {device}  render: {RENDER_H}×{RENDER_W}  "
          f"iters: {N_ITER}  stride: {stride}")

    scene_dir = GHOST_OUT / scene_name

    # ── GT cameras ──────────────────────────────────────────────────────
    gt_exts  = load_gt_cameras(scene_name)
    gt_intrs = load_gt_intrinsics(scene_name)
    K_gt = min(len(gt_exts), len(gt_intrs))
    print(f"  GT cameras: {K_gt}")

    # Scale intrinsics from original resolution to RENDER_H×RENDER_W
    sx = RENDER_W / _GT_W_ORIG
    sy = RENDER_H / _GT_H_ORIG
    render_intrs: list[np.ndarray] = []
    for K_mat in gt_intrs[:K_gt]:
        Ks = K_mat.copy()
        Ks[0] *= sx; Ks[1] *= sy
        render_intrs.append(Ks.astype(np.float32))

    # ── SMPL-X model ────────────────────────────────────────────────────
    smplx_model = smplx_lib.create(
        str(SMPLX_PATH / "SMPLX_NEUTRAL.pkl"), model_type="smplx", ext="pkl",
        use_pca=False, flat_hand_mean=True, batch_size=1,
    ).eval().to(device)
    faces = torch.tensor(smplx_model.faces.astype(np.int64), device=device, dtype=torch.long)

    # ── Per-camera body data ─────────────────────────────────────────────
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())
    K_cams = min(len(cam_dirs), K_gt)

    # ── GT body (loaded early for ghost↔GT pid matching) ─────────────────
    gt_body = load_gt_body(scene_name)

    # ── Ghost↔GT pid matching (same logic as fusion_dataset.py) ──────────
    # For each ghost pid, transform smplx_transl (cam frame) to world via GT
    # extrinsics, average across cameras, then pick the ghost pid whose world
    # translation is closest to the GT person's translation.
    all_cand_pids: set[int] = set()
    for cd in cam_dirs[:K_cams]:
        for f in (cd / "body_data").glob("person_*.npz"):
            all_cand_pids.add(int(f.stem.split("_")[1]))

    ghost_world_transl: dict[int, dict[int, np.ndarray]] = {}
    for gpid in sorted(all_cand_pids):
        accum: dict[int, list[np.ndarray]] = {}
        for k, cd in enumerate(cam_dirs[:K_cams]):
            if k >= len(gt_exts): continue
            bf = cd / "body_data" / f"person_{gpid}.npz"
            if not bf.exists(): continue
            d_tmp = np.load(bf, allow_pickle=False)
            if "smplx_transl" not in d_tmp.files: continue
            R_ck = gt_exts[k][:3, :3].astype(np.float64)
            t_ck = gt_exts[k][:3, 3].astype(np.float64)
            tr   = d_tmp["smplx_transl"].astype(np.float64)  # (T_local, 3) cam-k frame
            for local_t, global_t in enumerate(d_tmp["frame_indices"].astype(int)):
                t_world = R_ck.T @ (tr[local_t] - t_ck)
                accum.setdefault(int(global_t), []).append(t_world)
        ghost_world_transl[gpid] = {f: np.mean(ts, axis=0) for f, ts in accum.items()}

    ghost_pid: int | None = None
    best_gt_pid: int | None = None
    if gt_body:
        best_gt_pid = max(gt_body, key=lambda p: len(gt_body[p]))
        gt_transl_map = {f: v["transl"].astype(np.float64) for f, v in gt_body[best_gt_pid].items()}
        best_dist = float("inf")
        for gpid, gframes in ghost_world_transl.items():
            common = set(gframes) & set(gt_transl_map)
            if not common: continue
            mean_dist = float(np.mean([np.linalg.norm(gframes[f] - gt_transl_map[f]) for f in common]))
            if mean_dist < best_dist:
                best_dist = mean_dist
                ghost_pid = gpid
        print(f"  Ghost pid={ghost_pid} → GT pid={best_gt_pid}  (mean dist={best_dist:.3f} m)")
    if ghost_pid is None:
        ghost_pid = max(all_cand_pids, key=lambda p: sum(
            1 for cd in cam_dirs[:K_cams] if (cd / "body_data" / f"person_{p}.npz").exists()))
        print(f"  Ghost pid={ghost_pid} (fallback: most cameras, no GT available)")

    cam_body: list[dict | None] = []
    for cam_dir in cam_dirs[:K_cams]:
        bf = cam_dir / "body_data" / f"person_{ghost_pid}.npz"
        if not bf.exists(): cam_body.append(None); continue
        d = np.load(bf, allow_pickle=False)
        cam_body.append({
            "fi":           d["frame_indices"].astype(int),
            "betas":        d["smplx_betas"],
            "body_pose":    d["smplx_body_pose"],
            "global_orient":d["smplx_global_orient"],
            "kp2d":         d["pred_keypoints_2d"],  # (T_local, 70, 2) MHR70 in _KP_W×_KP_H
        })

    # ── Masks ────────────────────────────────────────────────────────────
    # mask_data.npz keys: mask_{frame:05d}_00; values: (H,W) uint16, person=ghost_pid
    # We resize to RENDER_H×RENDER_W during loading.
    print("  Loading masks...", end=" ", flush=True)
    t_load = time.perf_counter()
    cam_masks: list[dict[int, torch.Tensor]] = []
    for k, cam_dir in enumerate(cam_dirs[:K_cams]):
        masks_k: dict[int, torch.Tensor] = {}
        mask_path = cam_dir / "mask_data.npz"
        if not mask_path.exists(): cam_masks.append({}); continue
        npz = np.load(mask_path)
        for key in npz.files:
            fi = int(key.split("_")[1])
            arr = npz[key]
            binary = torch.tensor((arr == ghost_pid).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(binary, size=(RENDER_H, RENDER_W), mode="nearest").squeeze()
            masks_k[fi] = resized.to(device)
        cam_masks.append(masks_k)
    print(f"{time.perf_counter()-t_load:.1f}s")

    # ── Common frames ────────────────────────────────────────────────────
    valid_cam_bodies = [cb for cb in cam_body if cb is not None]
    if not valid_cam_bodies:
        print("  No body data found."); return
    common_frames = sorted(
        set.intersection(*[set(cb["fi"]) for cb in valid_cam_bodies])
    )
    eval_frames = common_frames[::stride]
    if n_frames is not None:
        eval_frames = eval_frames[:n_frames]
    print(f"  Common frames: {len(common_frames)}, evaluating: {len(eval_frames)}")

    # ── Pre-build PyTorch3D cameras (one per GT camera, fixed) ───────────
    pt3d_cameras: list = []
    for k in range(K_cams):
        R_k  = torch.tensor(gt_exts[k][:3, :3], dtype=torch.float32, device=device).unsqueeze(0)
        t_k  = torch.tensor(gt_exts[k][:3,  3], dtype=torch.float32, device=device).unsqueeze(0)
        K_k  = torch.tensor(render_intrs[k],     dtype=torch.float32, device=device).unsqueeze(0)
        isz  = torch.tensor([[RENDER_H, RENDER_W]], device=device)
        pt3d_cameras.append(cameras_from_opencv_projection(R_k, t_k, K_k, isz))

    # ── Renderer (shared settings, one per camera) ───────────────────────
    blend  = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(
        image_size=(RENDER_H, RENDER_W),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend.sigma,
        faces_per_pixel=50,
        bin_size=0,  # naive rasterization: correct gradients, no bin overflow
    )
    renderers = [
        MeshRenderer(
            rasterizer=MeshRasterizer(cameras=pt3d_cameras[k], raster_settings=raster),
            shader=SoftSilhouetteShader(blend_params=blend),
        )
        for k in range(K_cams)
    ]

    # ── Per-frame optimization ───────────────────────────────────────────
    print(f"\n  {'frame':>5}  {'iters':>5}  {'t/fr(s)':>8}  "
          f"{'IoU →':>8}  {'R°→':>8}  {'t_cm→':>8}")
    print(f"  {'─'*60}")

    results = []

    for frame_idx in eval_frames:
        # Valid cameras for this frame
        valid_ks = [
            k for k in range(K_cams)
            if cam_body[k] is not None
            and frame_idx in {int(x) for x in cam_body[k]["fi"]}
            and frame_idx in cam_masks[k]
        ]
        if len(valid_ks) < 2:
            continue

        # Local indices in per-camera arrays
        local_ts = {
            k: int(np.searchsorted(cam_body[k]["fi"], frame_idx))
            for k in valid_ks
        }

        # Mean betas + body_pose across cameras
        mean_betas = np.mean([cam_body[k]["betas"][local_ts[k]]    for k in valid_ks], axis=0)
        mean_bp    = np.mean([cam_body[k]["body_pose"][local_ts[k]] for k in valid_ks], axis=0)

        # Canonical vertices: zero global_orient, zero transl
        with torch.no_grad():
            vout = smplx_model(
                betas=torch.tensor(mean_betas, dtype=torch.float32, device=device).unsqueeze(0),
                body_pose=torch.tensor(mean_bp,  dtype=torch.float32, device=device).unsqueeze(0),
                global_orient=torch.zeros(1, 3, device=device),
                transl=torch.zeros(1, 3, device=device),
                jaw_pose=torch.zeros(1, 3, device=device),
                leye_pose=torch.zeros(1, 3, device=device),
                reye_pose=torch.zeros(1, 3, device=device),
                expression=torch.zeros(1, smplx_model.num_expression_coeffs, device=device),
                left_hand_pose=torch.zeros(1, 45, device=device),
                right_hand_pose=torch.zeros(1, 45, device=device),
                return_verts=True,
            )
        V_can = vout.vertices[0].detach()  # (V, 3) fixed

        # Init R_world, t_world via DLT+Procrustes with GT cameras
        R_init, t_init, pelvis_init = procrustes_init_gt(
            cam_body, valid_ks, local_ts,
            gt_exts, gt_intrs, smplx_model, device,
        )
        if R_init is None:
            continue
        aa_init = SciR.from_matrix(R_init).as_rotvec()

        # GT reference: pelvis_world = R_gt @ J_can[0] + transl ≈ J_can[0] + transl
        # (same formula placer.py + evaluate_on_rich_test.py use)
        gt_R = gt_pelvis = None
        J_can_0 = smplx_model.get_joints()[0].detach().cpu().numpy() if hasattr(smplx_model, 'get_joints') else None
        # Compute J_can[0] from the canonical FK (already have V_can; reuse vout.joints)
        if best_gt_pid and frame_idx in gt_body[best_gt_pid]:
            g = gt_body[best_gt_pid][frame_idx]
            gt_R = SciR.from_rotvec(g["global_orient"].astype(np.float64)).as_matrix()
            # GT pelvis world = J_can[0] + transl (pelvis is the rotation center → R doesn't move it)
            # SMPLX: J_world[j] = R @ (J_can[j] - J_can[0]) + J_can[0] + transl → at j=0: just J_can[0]+transl
            J_can_0_np = vout.joints[0, 0].detach().cpu().numpy().astype(np.float64)
            gt_pelvis = J_can_0_np + g["transl"].astype(np.float64)

        init_R_err = geodesic_deg(R_init, gt_R)     if gt_R      is not None else None
        init_t_err = np.linalg.norm(pelvis_init - gt_pelvis) * 100 if gt_pelvis is not None else None

        # Target masks for valid cameras
        tgt_masks = [cam_masks[k][frame_idx] for k in valid_ks]

        # Optimizable params
        aa = torch.tensor(aa_init, dtype=torch.float32, device=device, requires_grad=True)
        tw = torch.tensor(t_init,  dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([aa, tw], lr=LR)

        def forward() -> tuple[torch.Tensor, list[float]]:
            R_w = axis_angle_to_matrix(aa.unsqueeze(0))[0]       # (3,3)
            V_w = V_can @ R_w.T + tw.unsqueeze(0)                 # (V,3) world
            total = torch.zeros(1, device=device)
            ious  = []
            for k_idx, (k, tgt) in enumerate(zip(valid_ks, tgt_masks)):
                mesh = Meshes(verts=[V_w], faces=[faces])
                sil  = renderers[k](mesh)[0, :, :, 3]             # (H,W)
                loss = soft_iou_loss(sil, tgt)
                total = total + loss
                ious.append(float(1.0 - loss.detach()))
            return total, ious

        # Initial IoU (no grad)
        with torch.no_grad():
            _, iou_init_list = forward()
        iou_init = float(np.mean(iou_init_list))

        t0     = time.perf_counter()
        n_iter = 0
        prev   = float("inf")
        for it in range(N_ITER):
            opt.zero_grad()
            loss, _ = forward()
            loss.backward()
            opt.step()
            n_iter += 1
            if it > 10 and abs(prev - loss.item()) < 1e-5:
                break
            prev = loss.item()
        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            _, iou_final_list = forward()
        iou_final = float(np.mean(iou_final_list))

        aa_np = aa.detach().cpu().numpy().astype(np.float64)
        tw_np = tw.detach().cpu().numpy().astype(np.float64)
        R_final = SciR.from_rotvec(aa_np).as_matrix()
        final_R_err = geodesic_deg(R_final, gt_R) if gt_R is not None else None
        if gt_pelvis is not None:
            pelvis_final = R_final @ J_can_0_np + tw_np
            final_t_err = float(np.linalg.norm(pelvis_final - gt_pelvis) * 100)
        else:
            final_t_err = None

        results.append(dict(
            frame=frame_idx, iters=n_iter, time_s=elapsed,
            iou_init=iou_init,   iou_final=iou_final,
            init_R=init_R_err,   final_R=final_R_err,
            init_t=init_t_err,   final_t=final_t_err,
        ))

        r_str = (f"{init_R_err:6.1f}°→{final_R_err:5.1f}°"
                 if init_R_err is not None else "     n/a     ")
        t_str = (f"{init_t_err:6.1f}→{final_t_err:5.1f}cm"
                 if init_t_err is not None else "     n/a     ")
        print(f"  {frame_idx:5d}  {n_iter:5d}  {elapsed:8.2f}s  "
              f"{iou_init:.3f}→{iou_final:.3f}  {r_str}  {t_str}")

    # ── Summary ────────────────────────────────────────────────────────
    if not results:
        print("  No results."); return

    print(f"\n  {'─'*60}")
    print(f"  {'Metric':<22} {'mean':>8} {'median':>8} {'p90':>8}")
    print(f"  {'─'*60}")
    for key, label, fmt in [
        ("time_s",   "time/frame (s)",   ".2f"),
        ("iters",    "iters",            ".0f"),
        ("iou_init", "IoU init",         ".3f"),
        ("iou_final","IoU final",        ".3f"),
        ("init_R",   "orient° init",     ".1f"),
        ("final_R",  "orient° final",    ".1f"),
        ("init_t",   "transl cm init",   ".1f"),
        ("final_t",  "transl cm final",  ".1f"),
    ]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"  {label:<22} {np.mean(vals):>8.{fmt[1:]}}  "
                  f"{np.median(vals):>8.{fmt[1:]}}  {np.percentile(vals,90):>8.{fmt[1:]}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", nargs="?", default="BBQ_001_guitar")
    p.add_argument("--stride",   type=int, default=10)
    p.add_argument("--n-frames", type=int, default=None)
    args = p.parse_args()
    run(args.scene, stride=args.stride, n_frames=args.n_frames)
