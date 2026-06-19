"""Debug body placement: print GT vs predicted translation and orientation per frame.

GT values are exactly what the RICH dataset stores (raw SMPL-X ``transl`` and
``global_orient``).  The predicted values are the direct equivalents:

    pred_transl  = pelvis_world - J_can[0](pred_betas)
    pred_orient  = R  (from Procrustes, converted to axis-angle)

Both GT and pred are in the VGGT cam-0 / RICH-world (cam_00) frame.
For scenes where cam_00 is present, these two frames coincide.

Usage:
    pixi run python debug/debug_translation.py \\
        --scene_dir /path/to/ghost_outputs/BBQ_001_guitar \\
        --rich_root /path/to/rich \\
        --smplx_model /path/to/SMPLX_NEUTRAL.pkl \\
        [--split train_body]  # or test_body
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models


# ---------------------------------------------------------------------------
# GT scale from RICH calibration XMLs
# ---------------------------------------------------------------------------

def _scene_to_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def _load_gt_extrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,4) [R|t] world-to-cam matrices (metres), one per XML."""
    calib_dir = rich_root / "scan_calibration" / _scene_to_location(scene_name) / "calibration"
    if not calib_dir.is_dir():
        return None
    exts: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        cam_node = tree.getroot().find("CameraMatrix")
        if cam_node is None:
            continue
        vals = list(map(float, cam_node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return exts if exts else None


def _compute_gt_scale(placer: BodyPlacer, gt_exts: list[np.ndarray], cam_dirs: list[Path]) -> float | None:
    """GT scale = median(baseline_gt / baseline_vggt) over non-reference cameras."""
    m_ref = re.search(r"\d+", cam_dirs[0].name)
    ref_idx = int(m_ref.group()) if m_ref else 0
    if ref_idx >= len(gt_exts):
        return None
    E_ref = gt_exts[ref_idx]
    C_ref = -E_ref[:3, :3].T @ E_ref[:3, 3]

    samples: list[float] = []
    for k in range(1, placer.K):
        m = re.search(r"\d+", cam_dirs[k].name)
        gt_idx = int(m.group()) if m else k
        if gt_idx >= len(gt_exts):
            continue
        E_k = gt_exts[gt_idx]
        C_k = -E_k[:3, :3].T @ E_k[:3, 3]
        baseline_gt = float(np.linalg.norm(C_k - C_ref))
        if baseline_gt < 1e-6:
            continue
        for global_t in np.where(placer.cam_valid[:, k])[0]:
            t_vggt = placer.extrinsics[global_t, k, :3, 3]
            norm_vggt = float(np.linalg.norm(t_vggt))
            if norm_vggt > 1e-6:
                samples.append(baseline_gt / norm_vggt)
    return float(np.median(samples)) if samples else None


# ---------------------------------------------------------------------------
# GT cameras: build patched extrinsics/intrinsics arrays for the placer
# ---------------------------------------------------------------------------

_RICH_ORIG_W = 4112
_RICH_ORIG_H = 3008


def _load_gt_intrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,3) K matrices in full-res pixels, one per XML."""
    calib_dir = rich_root / "scan_calibration" / _scene_to_location(scene_name) / "calibration"
    if not calib_dir.is_dir():
        return None
    ks: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        intr_node = tree.getroot().find("Intrinsics")
        if intr_node is None:
            continue
        vals = list(map(float, intr_node.find("data").text.split()))
        ks.append(np.array(vals, dtype=np.float64).reshape(3, 3))
    return ks if ks else None


def _make_gt_cam_arrays(
    placer:    BodyPlacer,
    gt_exts:   list[np.ndarray],
    gt_intrs:  list[np.ndarray],
    cam_dirs:  list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (extrinsics, intrinsics, cam_valid) arrays compatible with the placer.

    GT cameras are static (same for all T frames) and already in metres.
    Extrinsics are re-expressed relative to cam_dirs[0] so that the reference
    camera is [I|0], matching VGGT's convention.
    Intrinsics are scaled from full-res (4112×3008) to VGGT image resolution
    inferred from the existing placer intrinsics.
    """
    T, K = placer.T, placer.K

    # Infer VGGT image resolution from principal point of first valid camera
    # (cx ≈ W/2, cy ≈ H/2)
    vggt_K0 = placer.intrinsics[0, 0]
    W_vggt  = float(vggt_K0[0, 2]) * 2.0
    H_vggt  = float(vggt_K0[1, 2]) * 2.0
    s_x = W_vggt / _RICH_ORIG_W
    s_y = H_vggt / _RICH_ORIG_H

    # Reference camera XML index
    m_ref   = re.search(r"\d+", cam_dirs[0].name)
    ref_idx = int(m_ref.group()) if m_ref else 0
    E_ref   = gt_exts[ref_idx].astype(np.float64)   # world-to-ref (3,4)
    # Full 4×4 for inversion
    E_ref4  = np.eye(4); E_ref4[:3] = E_ref

    extrinsics_gt = np.zeros((T, K, 3, 4), dtype=np.float32)
    intrinsics_gt = np.zeros((T, K, 3, 3), dtype=np.float32)
    cam_valid_gt  = np.zeros((T, K),       dtype=bool)

    for k, cam_dir in enumerate(cam_dirs):
        m      = re.search(r"\d+", cam_dir.name)
        gt_idx = int(m.group()) if m else k
        if gt_idx >= len(gt_exts) or gt_idx >= len(gt_intrs):
            continue

        E_k  = gt_exts[gt_idx].astype(np.float64)
        E_k4 = np.eye(4); E_k4[:3] = E_k
        # Re-express: cam_k relative to ref cam  (ref → world → cam_k)
        E_k_rel = (E_k4 @ np.linalg.inv(E_ref4))[:3]   # (3,4)

        K_gt = gt_intrs[gt_idx].astype(np.float64).copy()
        K_gt[0, 0] *= s_x; K_gt[0, 2] *= s_x
        K_gt[1, 1] *= s_y; K_gt[1, 2] *= s_y

        extrinsics_gt[:, k] = E_k_rel.astype(np.float32)
        intrinsics_gt[:, k] = K_gt.astype(np.float32)
        cam_valid_gt[:, k]  = True

    return extrinsics_gt, intrinsics_gt, cam_valid_gt


# ---------------------------------------------------------------------------
# GT loaders (read directly from RICH pkl files)
# ---------------------------------------------------------------------------

def _load_gt_raw(
    scene_name: str,
    rich_root: Path,
    split: str,
) -> dict[int, dict[int, dict]]:
    """Return gt[gt_pid][frame_idx] = {'transl': (3,), 'global_orient': (3,), 'betas': (10,)}."""
    gt: dict[int, dict[int, dict]] = {}
    gt_root = rich_root / split / scene_name
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT body root not found: {gt_root}")
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            gt_pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            transl = np.asarray(data["transl"], dtype=np.float32).squeeze()           # (3,)
            orient = np.asarray(data["global_orient"], dtype=np.float32).squeeze()    # (3,) axis-angle
            raw_betas = data.get("betas") if data.get("betas") is not None else data.get("smplx_betas")
            betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)[:10] if raw_betas is not None else np.zeros(10, dtype=np.float32)
            gt.setdefault(gt_pid, {})[frame_idx] = {
                "transl":        transl,
                "global_orient": orient,
                "betas":         betas,
            }
    return gt


# ---------------------------------------------------------------------------
# Predicted betas: mean over frames and cameras per ghost pid
# ---------------------------------------------------------------------------

def _load_pred_betas(cam_dirs: list[Path]) -> dict[int, np.ndarray]:
    accum: dict[int, list[np.ndarray]] = defaultdict(list)
    for cam_dir in cam_dirs:
        for f in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(f.stem.split("_")[1])
            d = np.load(f, allow_pickle=False)
            if "smplx_betas" in d.files:
                accum[pid].append(d["smplx_betas"].mean(axis=0))
    return {pid: np.mean(np.stack(v), axis=0) for pid, v in accum.items()}


# ---------------------------------------------------------------------------
# Pid matching: ghost → GT by nearest mean pelvis_world
# ---------------------------------------------------------------------------

def _gt_pelvis_world(
    gt_raw: dict[int, dict[int, dict]],
    placer: BodyPlacer,
) -> dict[int, dict[int, np.ndarray]]:
    """Compute GT pelvis_world = transl + J_can[0](betas) per gt_pid/frame."""
    zero_pose   = np.zeros((1, 63), dtype=np.float32)
    zero_orient = np.zeros((1, 3),  dtype=np.float32)
    result: dict[int, dict[int, np.ndarray]] = {}
    for gt_pid, frames in gt_raw.items():
        # J_can[0] is the same for all frames (only depends on betas, not pose)
        betas = next(iter(frames.values()))["betas"]
        fk = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)  # (1, 55, 3)
        j0 = fk[0, 0].astype(np.float32)
        result[gt_pid] = {fi: v["transl"] + j0 for fi, v in frames.items()}
    return result


def _match_pids(
    trans_dict: dict[int, dict[int, np.ndarray]],   # ghost_pid → {frame: pelvis_world}
    gt_pelvis:  dict[int, dict[int, np.ndarray]],   # gt_pid → {frame: pelvis_world}
    foreground_pids: set[int],
) -> dict[int, int]:
    """Return {gt_pid: ghost_pid} matched by minimum mean 3D distance."""
    gt_pids = list(gt_pelvis.keys())
    ghost_pids = [p for p in trans_dict if p in foreground_pids]

    if not gt_pids or not ghost_pids:
        return {}

    used_ghost: set[int] = set()
    mapping: dict[int, int] = {}

    for gt_pid in gt_pids:
        gt_frames = gt_pelvis[gt_pid]
        best_ghost, best_dist = None, float("inf")
        for ghost_pid in ghost_pids:
            if ghost_pid in used_ghost:
                continue
            pred_frames = trans_dict[ghost_pid]
            common = set(gt_frames) & set(pred_frames)
            if not common:
                continue
            dist = float(np.mean([
                np.linalg.norm(pred_frames[f] - gt_frames[f]) for f in common
            ]))
            if dist < best_dist:
                best_dist, best_ghost = dist, ghost_pid
        if best_ghost is not None:
            mapping[gt_pid] = best_ghost
            used_ghost.add(best_ghost)

    return mapping  # gt_pid → ghost_pid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",   required=True,  help="Ghost output dir for the scene")
    p.add_argument("--rich_root",   required=True,  help="RICH dataset root")
    p.add_argument("--smplx_model", required=True,  help="Path to SMPLX_NEUTRAL.pkl")
    p.add_argument("--split",       default="train_body",
                   help="RICH body split subfolder (train_body or test_body)")
    p.add_argument("--max_frames",  type=int, default=None,
                   help="Print only this many frames per person (for readability)")
    p.add_argument("--use_gt_scale", action="store_true",
                   help="Replace MapAnything scale with GT scale from RICH calibration XMLs")
    p.add_argument("--use_gt_cams", action="store_true",
                   help="Replace VGGT cameras with GT cameras from RICH calibration XMLs (implies --use_gt_scale)")
    args = p.parse_args()

    scene_dir  = Path(args.scene_dir)
    rich_root  = Path(args.rich_root)
    scene_name = scene_dir.name

    # ── Set up placer ─────────────────────────────────────────────────────────
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer    = BodyPlacer(scene_dir, _smplx_arg)
    cam_dirs  = placer._cam_dirs
    K         = len(cam_dirs)

    # ── Collect ghost pids and determine foreground ───────────────────────────
    pid_cam_count: dict[int, int] = defaultdict(int)
    for cam_dir in cam_dirs:
        for f in (cam_dir / "body_data").glob("person_*.npz"):
            pid = int(f.stem.split("_")[1])
            pid_cam_count[pid] += 1
    all_pids:        set[int] = set(pid_cam_count)
    foreground_pids: set[int] = {p for p, c in pid_cam_count.items() if c >= max(1, K - 1)}

    print(f"Scene : {scene_name}")
    print(f"Cams  : {K}  |  Ghost pids: {sorted(all_pids)}  |  Foreground: {sorted(foreground_pids)}")

    # ── Predicted betas ───────────────────────────────────────────────────────
    pred_betas_by_pid = _load_pred_betas(cam_dirs)

    # ── GT calibration (needed for GT scale and/or GT cams) ──────────────────
    gt_exts  = None
    gt_intrs = None
    if args.use_gt_scale or args.use_gt_cams:
        gt_exts  = _load_gt_extrinsics(scene_name, rich_root)
        gt_intrs = _load_gt_intrinsics(scene_name, rich_root)
        if gt_exts is None or gt_intrs is None:
            print("ERROR: GT calibration XMLs not found")
            return

    # ── Scale ─────────────────────────────────────────────────────────────────
    if args.use_gt_cams:
        # GT cameras are already in metres — no VGGT unit conversion needed
        scale = np.ones(placer.T, dtype=np.float32)
        print("Scale : 1.0 (GT cameras are metric)")
    elif args.use_gt_scale:
        gt_scale_val = _compute_gt_scale(placer, gt_exts, cam_dirs)
        if gt_scale_val is None:
            print("ERROR: GT scale computation returned None")
            return
        scale = np.full(placer.T, gt_scale_val, dtype=np.float32)
        print(f"Scale : GT (from XML baselines)  value={gt_scale_val:.4f} m/VGGT-unit")
    else:
        scale = placer.load_mapanything_scale()
        if scale is not None:
            print(f"Scale : MapAnything  median={float(np.median(scale)):.4f} m/VGGT-unit")
        else:
            print("Scale : MapAnything not found — using triangulation fallback")
            scale = placer.estimate_scale_triangulated(pred_betas_by_pid)

    # ── Patch placer cameras if using GT cams ────────────────────────────────
    orig_extrinsics = orig_intrinsics = orig_cam_valid = None
    if args.use_gt_cams:
        gt_exts_arr, gt_intrs_arr, gt_valid_arr = _make_gt_cam_arrays(
            placer, gt_exts, gt_intrs, cam_dirs
        )
        orig_extrinsics, orig_intrinsics, orig_cam_valid = (
            placer.extrinsics, placer.intrinsics, placer.cam_valid
        )
        placer.extrinsics = gt_exts_arr
        placer.intrinsics = gt_intrs_arr
        placer.cam_valid  = gt_valid_arr
        print("Cams  : GT (from RICH calibration XMLs)")

    # ── Run Procrustes DLT ────────────────────────────────────────────────────
    # fused_pose_by_pid=None → uses raw per-camera SAM3D body_pose
    trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
        scale=scale,
        all_pids=foreground_pids,
        pred_betas_by_pid=pred_betas_by_pid,
        fused_pose_by_pid=None,
    )

    # ── Restore placer cameras ────────────────────────────────────────────────
    if orig_extrinsics is not None:
        placer.extrinsics = orig_extrinsics
        placer.intrinsics = orig_intrinsics
        placer.cam_valid  = orig_cam_valid

    # ── J_can[0] per ghost pid (for converting pelvis_world → transl) ─────────
    zero_pose   = np.zeros((1, 63), dtype=np.float32)
    zero_orient = np.zeros((1, 3),  dtype=np.float32)
    j0_pred: dict[int, np.ndarray] = {}
    for pid in foreground_pids:
        betas = pred_betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))
        fk    = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)
        j0_pred[pid] = fk[0, 0].astype(np.float32)

    # ── Load GT ───────────────────────────────────────────────────────────────
    gt_raw = _load_gt_raw(scene_name, rich_root, args.split)
    if not gt_raw:
        print("ERROR: no GT found — check --rich_root and --split")
        return

    gt_pelvis = _gt_pelvis_world(gt_raw, placer)

    # ── Match ghost pids to GT pids ───────────────────────────────────────────
    mapping = _match_pids(trans_dict, gt_pelvis, foreground_pids)
    if not mapping:
        print("ERROR: could not match any ghost pid to a GT pid")
        return

    print(f"Match : gt_pid → ghost_pid = {mapping}\n")

    # ── Per-frame printout ────────────────────────────────────────────────────
    for gt_pid, ghost_pid in sorted(mapping.items()):
        j0 = j0_pred[ghost_pid]
        pred_frames  = trans_dict.get(ghost_pid, {})
        orient_frames = orient_dict.get(ghost_pid, {})
        gt_frames    = gt_raw.get(gt_pid, {})

        common_frames = sorted(set(pred_frames) & set(gt_frames))
        if args.max_frames is not None:
            common_frames = common_frames[:args.max_frames]

        print(f"{'─'*100}")
        print(f"gt_pid={gt_pid}  ghost_pid={ghost_pid}  common_frames={len(common_frames)}")
        print(
            f"{'frame':>6}  "
            f"{'GT_transl':>32}  "
            f"{'pred_transl':>32}  "
            f"{'transl_err_m':>12}  "
            f"{'GT_orient_aa':>32}  "
            f"{'pred_orient_aa':>32}  "
            f"{'orient_err_deg':>14}"
        )
        print(f"{'─'*100}")

        transl_errs = []
        orient_errs = []

        for frame in common_frames:
            pelvis_world = pred_frames[frame]             # R @ J_can[0] + t
            pred_transl  = pelvis_world - j0              # equivalent SMPL-X transl

            R_pred       = orient_frames.get(frame)
            pred_aa      = SciR.from_matrix(R_pred).as_rotvec() if R_pred is not None else np.full(3, np.nan)

            gt_entry     = gt_frames[frame]
            gt_transl    = gt_entry["transl"]             # raw from pkl
            gt_aa        = gt_entry["global_orient"]      # raw axis-angle from pkl

            # Translation error: compare pelvis_world positions
            # (GT pelvis_world = gt_transl + j0_gt, but we use pelvis_world - j0_pred for pred,
            #  so the comparison transl_err = ||pred_transl - gt_transl|| is the cleanest apples-to-apples)
            transl_err = float(np.linalg.norm(pred_transl - gt_transl))
            transl_errs.append(transl_err)

            orient_err = np.nan
            if R_pred is not None:
                R_gt = SciR.from_rotvec(gt_aa.astype(np.float64)).as_matrix()
                cos  = np.clip((np.trace(R_gt.T @ R_pred) - 1.0) / 2.0, -1.0, 1.0)
                orient_err = float(np.degrees(np.arccos(cos)))
                orient_errs.append(orient_err)

            def _fmt3(v: np.ndarray) -> str:
                return f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]"

            print(
                f"{frame:>6}  "
                f"{_fmt3(gt_transl):>32}  "
                f"{_fmt3(pred_transl):>32}  "
                f"{transl_err:>12.4f}  "
                f"{_fmt3(gt_aa):>32}  "
                f"{_fmt3(pred_aa):>32}  "
                f"{orient_err:>14.2f}"
            )

        if transl_errs:
            te = np.array(transl_errs)
            print(
                f"\n  transl: median={np.median(te):.4f}m  mean={np.mean(te):.4f}m  "
                f"<0.5m={100*(te<0.5).mean():.1f}%  <1.0m={100*(te<1.0).mean():.1f}%"
            )
        if orient_errs:
            oe = np.array(orient_errs)
            print(
                f"  orient: median={np.median(oe):.2f}°  mean={np.mean(oe):.2f}°  "
                f"<30°={100*(oe<30).mean():.1f}%  <60°={100*(oe<60).mean():.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
