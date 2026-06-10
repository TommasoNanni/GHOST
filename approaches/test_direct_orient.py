"""Quick test: compare direct orient (R_cw.T @ R_body_cam averaged) vs Procrustes.

Usage:
    pixi run python -m scripts.test_direct_orient [scene_name]
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as SciR

GHOST_OUT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_ROOT  = Path("/capstor/scratch/cscs/tnanni/datasets/rich")


def load_gt_orient(scene_name: str) -> dict[int, dict[int, np.ndarray]]:
    """gt[pid][frame] = R (3,3) world-frame orientation."""
    gt_root = RICH_ROOT / "train_body" / scene_name
    gt: dict[int, dict[int, np.ndarray]] = {}
    for fd in sorted(gt_root.iterdir()):
        if not fd.is_dir(): continue
        try: fi = int(fd.name)
        except ValueError: continue
        for pkl_path in sorted(fd.glob("*.pkl")):
            pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            aa = np.asarray(data["global_orient"], dtype=np.float64).reshape(3)
            gt.setdefault(pid, {})[fi] = SciR.from_rotvec(aa).as_matrix()
    return gt


def chordal_mean_R(Rs: list[np.ndarray]) -> np.ndarray:
    """Average rotation matrices via chordal mean (project mean to SO(3))."""
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    dR = R1.T @ R2
    tr = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def run(scene_name: str) -> None:
    scene_dir = GHOST_OUT / scene_name
    print(f"\nScene: {scene_name}")

    # ── Load VGGT cameras ────────────────────────────────────────────────
    cam_npz = np.load(scene_dir / "vggt_cameras.npz")
    extrinsics  = cam_npz["extrinsics"]   # (T, K, 3, 4)
    cam_valid   = cam_npz["valid"]        # (T, K)
    cam_names   = cam_npz["camera_names"]
    T, K = cam_valid.shape
    print(f"  T={T} K={K}  cameras: {[n.decode() for n in cam_names]}")

    # ── Load GT ─────────────────────────────────────────────────────────
    gt = load_gt_orient(scene_name)
    if not gt:
        print("  No GT found"); return
    gt_pid = list(gt.keys())[0]
    print(f"  GT pid={gt_pid}  frames={len(gt[gt_pid])}")

    # Find ghost pid that has body_data in the most cameras
    ghost_pids: dict[int, int] = {}  # pid → num cameras
    for cam_dir in sorted(d for d in scene_dir.iterdir() if d.is_dir() and (d / "body_data").is_dir()):
        for bf in (cam_dir / "body_data").glob("person_*.npz"):
            pid = int(bf.stem.split("_")[1])
            ghost_pids[pid] = ghost_pids.get(pid, 0) + 1
    ghost_pid = max(ghost_pids, key=ghost_pids.get)
    print(f"  Ghost pid={ghost_pid} (present in {ghost_pids[ghost_pid]} cameras)")

    # ── Load per-camera smplx_global_orient ─────────────────────────────
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())

    # direct_Rs[frame] = list of R_world estimates (one per camera)
    direct_Rs: dict[int, list[np.ndarray]] = {}

    for k, cam_dir in enumerate(cam_dirs):
        bf = cam_dir / "body_data" / f"person_{ghost_pid}.npz"
        if not bf.exists():
            continue
        d = np.load(bf, allow_pickle=False)
        if "smplx_global_orient" not in d.files:
            continue
        fi    = d["frame_indices"].astype(int)
        go_aa = d["smplx_global_orient"]   # (T_local, 3) axis-angle, camera frame

        for local_t, global_t in enumerate(fi):
            vggt_t = global_t  # frame_start=0 for RICH
            if vggt_t < 0 or vggt_t >= T:
                continue
            if not cam_valid[vggt_t, k]:
                continue
            R_cw = extrinsics[vggt_t, k, :3, :3].astype(np.float64)
            R_body_cam = SciR.from_rotvec(go_aa[local_t].astype(np.float64)).as_matrix()
            R_world = R_cw.T @ R_body_cam
            direct_Rs.setdefault(int(global_t), []).append(R_world)

    print(f"  direct_Rs frames collected: {len(direct_Rs)}")

    # ── Load current Procrustes orientation from infer_scene ─────────────
    # Check if there's an existing infer output for this scene
    infer_out = scene_dir / "infer_output.npz"
    has_procrustes = infer_out.exists()
    if has_procrustes:
        inf = np.load(infer_out, allow_pickle=True)
        print(f"  infer_output keys: {list(inf.files)}")

    # ── Compare direct vs GT ─────────────────────────────────────────────
    frames = sorted(
        set(direct_Rs.keys()) & set(gt[gt_pid].keys())
    )
    print(f"  Overlapping frames (direct ∩ GT): {len(frames)}")

    if not frames:
        print("  No overlap — stopping"); return

    direct_errs = []
    per_cam_errs = []
    n_cams_list  = []

    for f in frames:
        R_gt = gt[gt_pid][f]
        Rs   = direct_Rs[f]
        n_cams_list.append(len(Rs))

        # Per-camera errors (before averaging)
        for R in Rs:
            per_cam_errs.append(geodesic_deg(R, R_gt))

        # After averaging
        R_avg = chordal_mean_R(Rs) if len(Rs) > 1 else Rs[0]
        direct_errs.append(geodesic_deg(R_avg, R_gt))

    direct_errs  = np.array(direct_errs)
    per_cam_errs = np.array(per_cam_errs)

    print(f"\n  Per-camera R_world error (before avg):  "
          f"mean={per_cam_errs.mean():.1f}°  "
          f"median={np.median(per_cam_errs):.1f}°  "
          f"p90={np.percentile(per_cam_errs, 90):.1f}°")
    print(f"  After chordal avg ({np.mean(n_cams_list):.1f} cams/frame):  "
          f"mean={direct_errs.mean():.1f}°  "
          f"median={np.median(direct_errs):.1f}°  "
          f"p90={np.percentile(direct_errs, 90):.1f}°")

    # Best-single-camera oracle
    best_cam_errs = []
    for f in frames:
        Rs = direct_Rs[f]
        errs_f = [geodesic_deg(R, gt[gt_pid][f]) for R in Rs]
        best_cam_errs.append(min(errs_f))
    best_cam_errs = np.array(best_cam_errs)
    print(f"  Best-camera oracle (lower bound):  "
          f"mean={best_cam_errs.mean():.1f}°  "
          f"median={np.median(best_cam_errs):.1f}°  "
          f"p90={np.percentile(best_cam_errs, 90):.1f}°")

    # Robust average: one iteration of downweighting cameras far from chordal mean
    robust_errs = []
    for f in frames:
        Rs = direct_Rs[f]
        R_mean = chordal_mean_R(Rs) if len(Rs) > 1 else Rs[0]
        # weight = 1 / (1 + dist_to_mean^2) — soft Huber on SO(3)
        dists = [geodesic_deg(R, R_mean) for R in Rs]
        weights = np.array([1.0 / (1.0 + d**2 / 25.0) for d in dists])  # 25 = 5° scale
        weights /= weights.sum()
        # weighted chordal mean
        M = sum(w * R for w, R in zip(weights, Rs))
        U, _, Vt = np.linalg.svd(M)
        dsgn = np.linalg.det(Vt.T @ U.T)
        R_robust = Vt.T @ np.diag([1.0, 1.0, dsgn]) @ U.T
        robust_errs.append(geodesic_deg(R_robust, gt[gt_pid][f]))
    robust_errs = np.array(robust_errs)
    print(f"  Robust weighted avg:               "
          f"mean={robust_errs.mean():.1f}°  "
          f"median={np.median(robust_errs):.1f}°  "
          f"p90={np.percentile(robust_errs, 90):.1f}°")

    # Now repeat using GT camera extrinsics to isolate VGGT rotation error
    gt_cams = None
    try:
        import sys, importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_on_rich_test",
            str(Path(__file__).parent.parent / "evaluation" / "evaluate_on_rich_test.py"))
        mod = importlib.util.load_from_spec = None  # skip heavy import
    except Exception:
        pass

    import xml.etree.ElementTree as ET
    import re
    location = re.match(r"^([A-Za-z]+[0-9]+)", scene_name)
    location = location.group(1) if location else scene_name
    calib_dir = RICH_ROOT / "scan_calibration" / location / "calibration"
    gt_exts: list[np.ndarray] = []
    if calib_dir.is_dir():
        for xml_path in sorted(calib_dir.glob("*.xml")):
            tree = ET.parse(xml_path)
            cam_node = tree.getroot().find("CameraMatrix")
            if cam_node is None: continue
            vals = list(map(float, cam_node.find("data").text.split()))
            gt_exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))

    if len(gt_exts) >= K - 1:  # allow missing cam_10
        print(f"\n  Repeating with GT cameras ({len(gt_exts)} cameras found):")
        # gt_exts[k] → cam_k extrinsics; pad to K cams (cam_10 may be missing)
        # VGGT cam order matches XML order (000-006); cam_10 is extra
        gt_direct_Rs: dict[int, list[np.ndarray]] = {}
        for k, cam_dir in enumerate(cam_dirs):
            if k >= len(gt_exts): continue  # cam_10 has no GT calib
            if np.all(gt_exts[k] == 0): continue
            R_cw_gt = gt_exts[k][:3, :3]
            bf = cam_dir / "body_data" / f"person_{ghost_pid}.npz"
            if not bf.exists(): continue
            d = np.load(bf, allow_pickle=False)
            if "smplx_global_orient" not in d.files: continue
            fi    = d["frame_indices"].astype(int)
            go_aa = d["smplx_global_orient"]
            for local_t, global_t in enumerate(fi):
                R_body_cam = SciR.from_rotvec(go_aa[local_t].astype(np.float64)).as_matrix()
                R_world = R_cw_gt.T @ R_body_cam
                gt_direct_Rs.setdefault(int(global_t), []).append(R_world)

        gt_frames = sorted(set(gt_direct_Rs.keys()) & set(gt[gt_pid].keys()))
        gt_direct_errs = []
        for f in gt_frames:
            R_gt = gt[gt_pid][f]
            Rs   = gt_direct_Rs[f]
            R_avg = chordal_mean_R(Rs) if len(Rs) > 1 else Rs[0]
            gt_direct_errs.append(geodesic_deg(R_avg, R_gt))
        gt_direct_errs = np.array(gt_direct_errs)
        print(f"    GT cams + direct orient avg:  "
              f"mean={gt_direct_errs.mean():.1f}°  "
              f"median={np.median(gt_direct_errs):.1f}°  "
              f"p90={np.percentile(gt_direct_errs, 90):.1f}°")

    # Show worst 5 frames
    worst_idx = np.argsort(direct_errs)[-5:][::-1]
    print(f"\n  Worst 5 frames (pred cams + direct):")
    for i in worst_idx:
        f = frames[i]
        print(f"    frame {f:5d}: {direct_errs[i]:.1f}°  ({n_cams_list[i]} cams)")


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "ParkingLot2_008_eating1"
    run(scene)
