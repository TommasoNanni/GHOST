"""Diagnostic: confidence-weighted ViTPose / MHR70 DLT triangulation vs GT 3D joints.

Runs two cam modes (pred-cam with GT scale, GT-cam with XML calibration) for the
specified keypoint modality.

Usage:
    # single scene, all 4 mode combinations
    pixi run python -m scripts.diag_triangulate_joints

    # all train scenes, MHR70 only
    pixi run python -m scripts.diag_triangulate_joints --all --modality mhr70

    # all train scenes, ViTPose only
    pixi run python -m scripts.diag_triangulate_joints --all --modality vitpose
"""
import sys, re, pickle, argparse
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/users/tnanni/ghost")

from mappings.mapper import Mapper as _Mapper
_MAPPER = _Mapper.load("/users/tnanni/ghost/mappings/mhr_smplx_regressor.npy")

# ── defaults for single-scene mode ────────────────────────────────────────────
DEFAULT_SCENE     = "Pavallion_003_plankjack"
GHOST_OUTPUT_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_ROOT         = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
SMPLX_MODEL       = Path("/users/tnanni/ghost/body_models/SMPLX_NEUTRAL.pkl")

# ── joint mappings ─────────────────────────────────────────────────────────────
COCO_JOINTS  = {1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16, 18: 7, 19: 8, 20: 9,  21: 10}
MHR70_JOINTS = {1:  9, 2: 10, 4: 11, 5: 12, 7: 13, 8: 14, 18: 7, 19: 8, 20: 62, 21: 41}
JOINT_NAMES  = {1: "L-hip", 2: "R-hip", 4: "L-knee", 5: "R-knee",
                7: "L-ankle", 8: "R-ankle",
                18: "L-elbow", 19: "R-elbow", 20: "L-wrist", 21: "R-wrist"}

CONF_THRESHOLD = 0.5

# ── scenes / cameras to exclude from evaluation ────────────────────────────────
SKIP_SCENES = {
    "Pavallion_013_plankjack",    # whole scene wrong (bad ghost output)
    "Pavallion_013_phonesiteat",  # wrong cross-view ReID
}
# cameras to skip per scene (camera folder name, e.g. "cam_03")
SKIP_CAMERAS: dict[str, set[str]] = {
    "ParkingLot2_008_pushup2":       {"cam_03"},
    "ParkingLot2_014_takingphotos2": {"cam_01"},
    "Pavallion_003_018_tossball":    {"cam_06"},
}


# ── helpers ────────────────────────────────────────────────────────────────────

def triangulate_dlt(pts, Pmats, weights=None):
    A = np.zeros((2*len(pts), 4))
    for i, ((u, v), P) in enumerate(zip(pts, Pmats)):
        w = float(weights[i]) ** 0.5 if weights is not None else 1.0
        A[2*i]   = w * (u * P[2] - P[0])
        A[2*i+1] = w * (v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)

def orig_to_vggt(kp, oc, W, H):
    x1, y1, x2, y2 = oc
    return x1 + kp[0]*(x2-x1)/W, y1 + kp[1]*(y2-y1)/H

def _scene_to_location(name):
    parts = name.split("_")
    for i, p in enumerate(parts):
        if p.isdigit():
            return "_".join(parts[:i])
    return name

def _row0(arr, n): return np.asarray(arr).reshape(-1, n)[0]


# ── SMPL-X FK (loaded once globally) ─────────────────────────────────────────

_smplx_model = None

def _get_smplx():
    global _smplx_model
    if _smplx_model is None:
        import torch, smplx as smplx_lib
        _smplx_model = smplx_lib.create(str(SMPLX_MODEL), model_type="smplx",
                                        ext=SMPLX_MODEL.suffix.lstrip("."), use_pca=False)
    return _smplx_model

def smplx_fk(betas, body_pose, global_orient, transl, return_verts: bool = False):
    import torch
    model = _get_smplx()
    T_loc = betas.shape[0]
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas, dtype=torch.float32),
            body_pose=torch.tensor(body_pose, dtype=torch.float32),
            global_orient=torch.tensor(global_orient, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
            expression=torch.zeros(T_loc, model.num_expression_coeffs),
            jaw_pose=torch.zeros(T_loc, 3), leye_pose=torch.zeros(T_loc, 3),
            reye_pose=torch.zeros(T_loc, 3), left_hand_pose=torch.zeros(T_loc, 45),
            right_hand_pose=torch.zeros(T_loc, 45),
            return_verts=return_verts,
        )
    joints = out.joints[:, :55].numpy()
    if return_verts:
        return joints, out.vertices.numpy()
    return joints


# ── per-scene runner ──────────────────────────────────────────────────────────

def run_scene(scene_name: str, modality: str, verbose: bool = True):
    """Run triangulation diagnostics for one scene.

    Args:
        scene_name: e.g. "ParkingLot2_008_overfence1"
        modality:   "vitpose" | "mhr70" | "both"
        verbose:    print per-joint tables when True

    Returns:
        dict {(cam_mode, kp_mode): {smplx_j: [errors_cm]}}
        dict {(cam_mode, kp_mode): [(err, frame, joint_name)]}
    """
    if scene_name in SKIP_SCENES:
        return None, None

    scene_dir   = GHOST_OUTPUT_ROOT / scene_name
    gt_body_dir = RICH_ROOT / "train_body" / scene_name

    if not scene_dir.exists():
        print(f"  [skip] {scene_name}: scene_dir not found")
        return None, None

    # ── VGGT cameras ──────────────────────────────────────────────────────────
    cam_npz_path = scene_dir / "vggt_cameras.npz"
    if not cam_npz_path.exists():
        print(f"  [skip] {scene_name}: no vggt_cameras.npz")
        return None, None

    cam_npz        = np.load(cam_npz_path)
    extrinsics     = cam_npz["extrinsics"]
    intrinsics     = cam_npz["intrinsics"]
    original_coords= cam_npz["original_coords"]
    original_size  = cam_npz["original_size"]
    cam_valid      = cam_npz["valid"]
    if not cam_valid.any():
        cam_valid = ~np.isnan(extrinsics[:, :, 0, 0])
    vnames = [n.decode() if isinstance(n, bytes) else n for n in cam_npz["camera_names"]]

    if verbose:
        print(f"Cameras: {vnames}  T={extrinsics.shape[0]}")

    # ── GT calibration ────────────────────────────────────────────────────────
    calib_dir = RICH_ROOT / "scan_calibration" / _scene_to_location(scene_name) / "calibration"
    gt_Ext, gt_K = {}, {}
    if calib_dir.is_dir():
        for xml_f in sorted(calib_dir.glob("*.xml")):
            idx  = int(xml_f.stem)
            root = ET.parse(xml_f).getroot()
            ext_v = list(map(float, root.find("CameraMatrix").find("data").text.split()))
            int_v = list(map(float, root.find("Intrinsics").find("data").text.split()))
            gt_Ext[idx] = np.array(ext_v).reshape(3, 4)
            gt_K[idx]   = np.array(int_v).reshape(3, 3)

    # ── GT scale ──────────────────────────────────────────────────────────────
    cam_dirs = sorted(d for d in scene_dir.iterdir() if d.is_dir() and (d/"body_data").is_dir())
    if not cam_dirs:
        print(f"  [skip] {scene_name}: no cam dirs")
        return None, None

    ref_cam_num = int(re.search(r"\d+", cam_dirs[0].name).group())
    gt_scale = 1.0
    if ref_cam_num in gt_Ext:
        E0_gt = gt_Ext[ref_cam_num]; R0, t0 = E0_gt[:3, :3], E0_gt[:3, 3]
        ratios = []
        for ki, cn in enumerate(vnames):
            if ki == 0: continue
            gidx = int(re.search(r"\d+", cn).group())
            if gidx not in gt_Ext: continue
            Ek = gt_Ext[gidx]
            dist_gt   = np.linalg.norm(-Ek[:3,:3].T@Ek[:3,3] - (-R0.T@t0))
            vmask = cam_valid[:, ki]
            if not vmask.any(): continue
            dist_vggt = np.linalg.norm(np.median(extrinsics[vmask, ki, :3, 3], axis=0))
            if dist_vggt > 1e-6 and dist_gt > 1e-6:
                ratios.append(dist_gt / dist_vggt)
        gt_scale = float(np.median(ratios)) if ratios else 1.0
    else:
        R0, t0 = np.eye(3), np.zeros(3)

    if verbose:
        print(f"GT scale = {gt_scale:.4f}")

    # ── per-camera keypoints ───────────────────────────────────────────────────
    all_pids = sorted({int(p.stem.split("_")[1]) for cd in cam_dirs
                       for p in (cd/"body_data").glob("person_*.npz")})

    # ── Ghost↔GT pid matching (same logic as fusion_dataset.py) ─────────────
    # Pre-load GT transl for all GT persons across all frames.
    gt_pid_transl: dict[int, dict[int, np.ndarray]] = {}
    if gt_body_dir.exists():
        for fd in sorted(gt_body_dir.iterdir()):
            if not fd.is_dir(): continue
            try: fi = int(fd.name)
            except ValueError: continue
            for pkl_p in fd.glob("*.pkl"):
                gtp = int(pkl_p.stem)
                with open(pkl_p, "rb") as fh:
                    d = pickle.load(fh, encoding="latin1")
                tr = np.asarray(d.get("transl", np.zeros(3)), dtype=np.float64).reshape(3)
                gt_pid_transl.setdefault(gtp, {})[fi] = tr

    # Convert each ghost pid's smplx_transl (cam frame) → world via GT extrinsics.
    def _ghost_world_transl(gpid: int) -> dict[int, np.ndarray]:
        accum: dict[int, list] = {}
        for cd in cam_dirs:
            ci = int(re.search(r"\d+", cd.name).group())
            if ci not in gt_Ext: continue
            R_cw = gt_Ext[ci][:3, :3]; t_cw = gt_Ext[ci][:3, 3]
            bp = cd / "body_data" / f"person_{gpid}.npz"
            if not bp.exists(): continue
            d = np.load(bp, allow_pickle=False)
            if "smplx_transl" not in d.files: continue
            fi_arr = d["frame_indices"].astype(int)
            tr_arr = d["smplx_transl"].astype(np.float64)
            for li, gi in enumerate(fi_arr):
                accum.setdefault(int(gi), []).append(R_cw.T @ (tr_arr[li] - t_cw))
        return {f: np.mean(ts, axis=0) for f, ts in accum.items()}

    ghost_transl = {gpid: _ghost_world_transl(gpid) for gpid in all_pids}

    # For each GT person, find the closest ghost person (same pattern as fusion_dataset.py).
    gt_to_ghost: dict[int, int] = {}
    for gtp, gtt in gt_pid_transl.items():
        best_gpid: int | None = None; best_dist = float("inf")
        for gpid in all_pids:
            common_f = set(ghost_transl[gpid]) & set(gtt)
            if not common_f: continue
            d = float(np.mean([np.linalg.norm(ghost_transl[gpid][f] - gtt[f]) for f in common_f]))
            if d < best_dist:
                best_dist = d; best_gpid = gpid
        if best_gpid is not None:
            gt_to_ghost[gtp] = best_gpid
            if verbose:
                print(f"  GT pid={gtp} → Ghost pid={best_gpid}  (mean dist={best_dist:.3f} m)")

    # matched_pairs: (ghost_pid, gt_pid) — one entry per GT person matched
    matched_pairs = [(ghost_pid, gt_pid) for gt_pid, ghost_pid in gt_to_ghost.items()]
    if not matched_pairs:
        matched_pairs = [(all_pids[0], None)] if all_pids else []

    # ── camera indices (person-independent) ───────────────────────────────────
    cam_idxs  = [int(re.search(r"\d+", cd.name).group()) for cd in cam_dirs]
    skip_cams = SKIP_CAMERAS.get(scene_name, set())

    def _pred_proj(local_t, k):
        ext = extrinsics[local_t, k].astype(np.float64).copy()
        ext[:3, 3] *= gt_scale
        return intrinsics[local_t, k].astype(np.float64) @ ext

    def _gt_proj(k, local_t):
        ci = cam_idxs[k]
        if ci not in gt_K: return None
        W_orig  = float(original_size[local_t, k, 0])
        K_c     = gt_K[ci].copy()
        scale   = W_orig / (K_c[0, 2] * 2)
        K_s     = K_c * scale; K_s[2, 2] = 1.0
        return K_s @ gt_Ext[ci]

    # ── GT body frames ─────────────────────────────────────────────────────────
    gt_dirs      = sorted(gt_body_dir.iterdir()) if gt_body_dir.exists() else []
    gt_frame_set = {int(d.name) for d in gt_dirs if d.is_dir()}

    # ── determine which modes to run ──────────────────────────────────────────
    if modality == "both":
        kp_modes = ["vitpose", "mhr70"]
    else:
        kp_modes = [modality]
    MODES = [("pred", km) for km in kp_modes] + [("gt", km) for km in kp_modes]

    errs   = {m: defaultdict(list) for m in MODES}
    confs  = {m: defaultdict(list) for m in MODES}
    worsts = {m: []                for m in MODES}

    # ── per-person loop ────────────────────────────────────────────────────────
    for PID, GT_PID in matched_pairs:
        if GT_PID is None:
            continue

        cam_data: dict[int, dict] = {}
        for k, cd in enumerate(cam_dirs):
            if cd.name in skip_cams: continue
            fi_path = cd / "body_data" / f"person_{PID}.npz"
            if not fi_path.exists(): continue
            d_body  = np.load(fi_path, allow_pickle=False)
            fi      = d_body["frame_indices"].astype(int)
            entry   = {
                "name":    cd.name,
                "cam_idx": cam_idxs[k],
                "local_t": {int(g): int(l) for l, g in enumerate(fi)},
            }
            vp_path = cd / f"vitpose_kps_person_{PID}.npz"
            if vp_path.exists():
                vp = np.load(vp_path)["keypoints"]
                entry["kp_vp"]   = vp[:, :, :2]
                entry["kp_conf"] = vp[:, :,  2]
            if "pred_keypoints_2d" in d_body.files:
                entry["kp_mhr"] = d_body["pred_keypoints_2d"]
            cam_data[k] = entry

        if not cam_data:
            continue

        sets   = [set(v["local_t"].keys()) for v in cam_data.values()]
        common = sorted(sets[0].intersection(*sets[1:]))
        if verbose:
            print(f"\n  PID={PID} GT_PID={GT_PID}  Frames visible in all cams: {len(common)}\n")

        for global_t in common:
            if global_t not in gt_frame_set: continue
            frame_dir = gt_body_dir / f"{global_t:05d}"
            if not frame_dir.exists(): continue
            gt_pkl_path = next((p for p in frame_dir.glob("*.pkl") if int(p.stem) == GT_PID), None)
            if gt_pkl_path is None: continue
            with open(gt_pkl_path, "rb") as f:
                gt_p = pickle.load(f, encoding="latin1")

            gt_b  = np.mean(np.asarray(gt_p.get("betas", np.zeros(10))).reshape(-1,10), 0)[None].astype(np.float32)
            gt_bp = _row0(gt_p.get("body_pose",     np.zeros(63)), 63)[None].astype(np.float32)
            gt_go = _row0(gt_p.get("global_orient", np.zeros(3)),   3)[None].astype(np.float32)
            gt_tr = _row0(gt_p.get("transl",        np.zeros(3)),   3)[None].astype(np.float32)
            J_world, V_world = smplx_fk(gt_b, gt_bp, gt_go, gt_tr, return_verts=True)
            J_world = J_world[0]
            V_world = V_world[0]  # (10475, 3) — posed SMPLX vertices in world frame

            # mapper-corrected GT landmarks in world frame (for mhr70 comparison)
            kp_world = _MAPPER.map(V_world)   # (N_kps, 3)

            local_t_ref = next(bd["local_t"][global_t] for bd in cam_data.values()
                               if global_t in bd["local_t"])
            E0_v   = extrinsics[local_t_ref, 0].copy()

            # FK joints in VGGT frame (for vitpose pred-cam target)
            J_cam0 = J_world @ R0.T + t0
            J_vggt = (J_cam0 - E0_v[:3, 3]*gt_scale) @ E0_v[:3, :3]

            # mapper landmarks in VGGT frame (for mhr70 pred-cam target)
            V_cam0 = V_world @ R0.T + t0
            V_vggt = (V_cam0 - E0_v[:3, 3]*gt_scale) @ E0_v[:3, :3]
            kp_vggt = _MAPPER.map(V_vggt)    # (N_kps, 3)

            for cam_mode, kp_mode in MODES:
                joint_map = COCO_JOINTS if kp_mode == "vitpose" else MHR70_JOINTS
                fe = []

                for smplx_j, src_j in sorted(joint_map.items()):
                    obs, Pmats, ws = [], [], []
                    all_c = []

                    for k, bd in cam_data.items():
                        if global_t not in bd["local_t"]: continue
                        if not cam_valid[local_t_ref, k]: continue
                        if cam_mode == "gt" and bd["cam_idx"] not in gt_Ext: continue
                        lt = bd["local_t"][global_t]

                        if kp_mode == "vitpose":
                            if "kp_vp" not in bd: continue
                            kp   = bd["kp_vp"][lt, src_j]
                            conf = float(bd["kp_conf"][lt, src_j])
                            if conf < CONF_THRESHOLD: continue
                            all_c.append(conf)
                        else:
                            if "kp_mhr" not in bd: continue
                            if src_j >= bd["kp_mhr"].shape[1]: continue
                            kp   = bd["kp_mhr"][lt, src_j]
                            conf = 1.0
                            all_c.append(conf)

                        if cam_mode == "pred":
                            oc  = original_coords[local_t_ref, k]
                            os_ = original_size[local_t_ref, k]
                            u, v = orig_to_vggt(kp, oc, float(os_[0]), float(os_[1]))
                            if 0 <= u < oc[2] and 0 <= v < oc[3]:
                                obs.append((u, v)); Pmats.append(_pred_proj(local_t_ref, k)); ws.append(conf)
                        else:
                            Pg = _gt_proj(k, local_t_ref)
                            if Pg is not None:
                                obs.append((float(kp[0]), float(kp[1]))); Pmats.append(Pg); ws.append(conf)

                    if all_c:
                        confs[(cam_mode, kp_mode)][smplx_j].append(float(np.mean(all_c)))

                    if len(obs) >= 2:
                        if kp_mode == "mhr70":
                            target = kp_vggt[_MAPPER.index(src_j)] if cam_mode == "pred" else kp_world[_MAPPER.index(src_j)]
                        else:
                            target = J_vggt[smplx_j] if cam_mode == "pred" else J_world[smplx_j]
                        try:
                            tri = triangulate_dlt(obs, Pmats, weights=ws if kp_mode=="vitpose" else None)
                            e   = float(np.linalg.norm(tri - target) * 100)
                            errs[(cam_mode, kp_mode)][smplx_j].append(e)
                            fe.append((e, JOINT_NAMES[smplx_j]))
                        except Exception:
                            pass

                if fe:
                    we, wj = max(fe, key=lambda x: x[0])
                    worsts[(cam_mode, kp_mode)].append((we, global_t, wj))

    if not any(errs[m] for m in MODES):
        print(f"  [skip] {scene_name}: no keypoint data")
        return None, None

    # free large per-scene arrays before returning
    del extrinsics, intrinsics, original_coords, original_size, cam_valid
    return errs, worsts, confs


# ── summary printing ──────────────────────────────────────────────────────────

def print_summary(label, errs_dict, worst_list, confs_dict=None):
    all_e = []
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  {'Joint':>10}  {'mean':>8}  {'median':>8}  {'avg conf':>9}  {'n':>5}")
    print(f"  {'-'*55}")
    for smplx_j in sorted(COCO_JOINTS):
        errs_j = errs_dict[smplx_j]
        cfns   = (confs_dict or {}).get(smplx_j, [])
        if not errs_j:
            print(f"  {JOINT_NAMES[smplx_j]:>10}  {'—':>8}  {'—':>8}  {'—':>9}  {0:>5}")
            continue
        all_e.extend(errs_j)
        conf_str = f"{np.mean(cfns):>9.3f}" if cfns else f"{'—':>9}"
        print(f"  {JOINT_NAMES[smplx_j]:>10}  {np.mean(errs_j):>7.1f}cm"
              f"  {np.median(errs_j):>7.1f}cm  {conf_str}  {len(errs_j):>5}")
    if all_e:
        print(f"\n  Overall  mean={np.mean(all_e):.1f}cm  median={np.median(all_e):.1f}cm"
              f"  max={np.max(all_e):.1f}cm  (n={len(all_e)})")
    print(f"\n  Top-10 worst frames:")
    print(f"  {'frame':>6}  {'max err':>8}  {'joint':>12}")
    for e, t, jn in sorted(worst_list, reverse=True)[:10]:
        print(f"  {t:>6}  {e:>7.1f}cm  {jn:>12}")


MODE_LABELS = {
    ("pred", "vitpose"): "PRED cameras + ViTPose (RePoGen-S)",
    ("pred", "mhr70"):   "PRED cameras + MHR70 (SAM3D)",
    ("gt",   "vitpose"): "GT cameras   + ViTPose (RePoGen-S)",
    ("gt",   "mhr70"):   "GT cameras   + MHR70 (SAM3D)",
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Triangulation diagnostic.")
    parser.add_argument("--modality", choices=["vitpose", "mhr70", "both"], default="both",
                        help="Keypoint source (default: both)")
    parser.add_argument("--all", dest="all_scenes", action="store_true",
                        help="Run all scenes in rich_train and print aggregate table")
    args = parser.parse_args()

    if args.all_scenes:
        # ── all-scenes mode ───────────────────────────────────────────────────
        kp_modes  = ["vitpose", "mhr70"] if args.modality == "both" else [args.modality]
        MODES     = [("pred", km) for km in kp_modes] + [("gt", km) for km in kp_modes]

        agg_errs  = {m: defaultdict(list) for m in MODES}
        scene_rows = []

        scenes = sorted(d.name for d in GHOST_OUTPUT_ROOT.iterdir()
                        if d.is_dir() and (d/"vggt_cameras.npz").exists())
        print(f"Found {len(scenes)} scenes. Running with modality={args.modality}\n")

        hdr = f"{'Scene':<45}" + "".join(f"  {MODE_LABELS[m][:22]:>22}" for m in MODES)
        print(hdr)
        print("─" * len(hdr))

        import gc
        for scene_name in scenes:
            result = run_scene(scene_name, args.modality, verbose=False)
            gc.collect()
            if result[0] is None:
                continue
            errs, worsts, confs = result

            row = f"{scene_name:<45}"
            scene_means = {}
            for m in MODES:
                all_e = [e for el in errs[m].values() for e in el]
                mean_e = np.mean(all_e) if all_e else float("nan")
                scene_means[m] = mean_e
                row += f"  {mean_e:>21.1f}cm"
                for j, el in errs[m].items():
                    agg_errs[m][j].extend(el)
            print(row)
            scene_rows.append((scene_name, scene_means))

        # aggregate row
        print("─" * len(hdr))
        agg_row = f"{'AGGREGATE':<45}"
        for m in MODES:
            all_e = [e for el in agg_errs[m].values() for e in el]
            agg_row += f"  {np.mean(all_e):>21.1f}cm" if all_e else f"  {'—':>22}"
        print(agg_row)

    else:
        # ── single-scene mode ─────────────────────────────────────────────────
        print(f"Scene: {DEFAULT_SCENE}\n")
        result = run_scene(DEFAULT_SCENE, args.modality, verbose=True)
        if result[0] is None:
            return
        errs, worsts, confs = result

        kp_modes = ["vitpose", "mhr70"] if args.modality == "both" else [args.modality]
        MODES    = [("pred", km) for km in kp_modes] + [("gt", km) for km in kp_modes]
        for m in MODES:
            print_summary(MODE_LABELS[m], errs[m], worsts[m], confs[m])


if __name__ == "__main__":
    main()
