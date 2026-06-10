"""
Diagnostic: characterise the ~15-16 cm depth error from silhouette fitting.

Does NOT re-run the expensive Powell optimisation.  Uses the raw SAM3D
body_data (transl) as the "prediction" so we can process hundreds of frames
quickly. The four checks are:

  1. Sign & consistency  – signed error (pred_z - GT_z) along the cam0 Z-axis.
  2. Error vs distance   – does the error grow with GT depth (scale issue)
                          or stay roughly constant (fixed offset)?
  3. Surface-vs-pelvis  – for a sample of frames, build the full posed GT mesh
                          and measure the Z gap between the frontmost surface
                          vertex and the GT pelvis (body depth offset).
  4. Convention check   – verify pred_pelvis = transl_pred + J0_pred matches
                          the GT definition; report any residual mean offset.

cam_00 has identity extrinsics ([I|0]), so world-frame Z == cam0-frame Z.

Usage:
    pixi run python debug/diagnose_silhouette_depth.py \
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \
        [--person_id 1] [--split train_body] [--surface_frames 30]
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.stats import linregress

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

# SMPL-X joint indices assigned to the trunk.  Voronoi partition: a vertex is a
# "torso vertex" if its nearest joint (in canonical T-pose) is in this set.
# Excludes hips(1,2), knees(4,5), ankles(7,8), feet(10,11),
#          head(15), shoulders(16,17), elbows(18,19), wrists(20,21), hands.
_TORSO_JOINT_IDS = frozenset([0, 3, 6, 9, 12, 13, 14])
#  0=pelvis  3=spine1  6=spine2  9=spine3  12=neck  13=left_collar  14=right_collar


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pred_body_data(cam_dir: Path, pid: int) -> dict[int, dict] | None:
    path = cam_dir / "body_data" / f"person_{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    req = {"smplx_body_pose", "smplx_global_orient", "smplx_betas",
           "smplx_transl", "frame_indices"}
    if not req.issubset(d.files):
        return None
    frames = d["frame_indices"].astype(int)
    out: dict[int, dict] = {}
    for local_t, global_t in enumerate(frames):
        out[int(global_t)] = {
            "body_pose":     d["smplx_body_pose"][local_t].astype(np.float64),
            "global_orient": d["smplx_global_orient"][local_t].astype(np.float64),
            "betas":         d["smplx_betas"][local_t].astype(np.float64),
            "transl":        d["smplx_transl"][local_t].astype(np.float64),
        }
    return out


def load_gt_frame(rich_root: Path, split: str, scene_name: str,
                  frame_idx: int) -> dict | None:
    gt_dir = rich_root / split / scene_name
    if not gt_dir.is_dir():
        return None
    for frame_dir in gt_dir.iterdir():
        try:
            if int(frame_dir.name) != frame_idx:
                continue
        except ValueError:
            continue
        pkl = next(frame_dir.glob("*.pkl"), None)
        if pkl is None:
            return None
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        raw_betas = d.get("betas") if d.get("betas") is not None else d.get("smplx_betas")
        return {
            "transl":        np.array(d["transl"],        dtype=np.float64).reshape(3),
            "global_orient": np.array(d["global_orient"], dtype=np.float64).reshape(3),
            "body_pose":     np.array(d["body_pose"],     dtype=np.float64).reshape(63),
            "betas":         np.array(raw_betas,          dtype=np.float64).reshape(-1)[:10],
        }
    return None


# ---------------------------------------------------------------------------
# FK helper (zero transl, zero orient → pelvis position = J0(betas))
# ---------------------------------------------------------------------------

def pelvis_from_betas(smplx_model: smplx.SMPLX, betas: np.ndarray) -> np.ndarray:
    """Return J0 (pelvis offset from origin) for given betas, zero pose/transl."""
    with torch.no_grad():
        out = smplx_model(
            betas=torch.tensor(betas[None].astype(np.float32)),
            body_pose=torch.zeros(1, 63),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
        )
    return out.joints[0, 0].numpy().astype(np.float64)


def build_posed_mesh(smplx_model: smplx.SMPLX,
                     betas: np.ndarray,
                     body_pose: np.ndarray,
                     global_orient: np.ndarray,
                     transl: np.ndarray) -> np.ndarray:
    """Return (N_verts, 3) posed vertices in world frame."""
    with torch.no_grad():
        out = smplx_model(
            betas=torch.tensor(betas[None].astype(np.float32)),
            body_pose=torch.tensor(body_pose[None].astype(np.float32)),
            global_orient=torch.tensor(global_orient[None].astype(np.float32)),
            transl=torch.tensor(transl[None].astype(np.float32)),
            return_verts=True,
        )
    return out.vertices[0].numpy().astype(np.float64)


def compute_torso_mask(smplx_model: smplx.SMPLX) -> np.ndarray:
    """Boolean mask (N_verts,) marking torso vertices via canonical-pose Voronoi.

    Each vertex is assigned to its nearest SMPL-X joint (in canonical T-pose,
    mean betas, zero pose/orient).  Vertices assigned to _TORSO_JOINT_IDS are
    returned as True.  Computed once and reused for all frames.
    """
    with torch.no_grad():
        out = smplx_model(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 63),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
            return_verts=True,
        )
    verts  = out.vertices[0].numpy()       # (N_verts, 3)
    joints = out.joints[0, :55].numpy()    # (55, 3)

    # (N_verts, 55) squared distances → nearest joint per vertex
    diff    = verts[:, None, :] - joints[None, :, :]   # (N_verts, 55, 3)
    sq_dist = (diff ** 2).sum(axis=2)                  # (N_verts, 55)
    nearest = sq_dist.argmin(axis=1)                   # (N_verts,)

    mask = np.array([j in _TORSO_JOINT_IDS for j in nearest])
    print(f"Torso mask: {mask.sum()} / {len(mask)} vertices")
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",     required=True)
    p.add_argument("--rich_root",     required=True)
    p.add_argument("--smplx_model",   required=True)
    p.add_argument("--person_id",     type=int, default=1)
    p.add_argument("--split",         default="train_body")
    p.add_argument("--surface_frames", type=int, default=30,
                   help="Number of frames for surface-vs-pelvis check (slow)")
    args = p.parse_args()

    scene_dir  = Path(args.scene_dir)
    rich_root  = Path(args.rich_root)
    scene_name = scene_dir.name

    # Reference camera: cam_00 (VGGT [I|0])
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and re.search(r"cam", d.name, re.I))
    if not cam_dirs:
        print("No camera dirs found"); return
    cam0_dir = cam_dirs[0]
    print(f"Reference cam: {cam0_dir.name}")

    _gender_json = _ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(scene_dir, _smplx_arg)

    smplx_model = smplx.create(
        args.smplx_model, model_type="smplx", ext="pkl",
        use_pca=False, flat_hand_mean=True, batch_size=1,
    )
    smplx_model.eval()

    # Precompute torso vertex mask once (canonical T-pose Voronoi).
    torso_mask = compute_torso_mask(smplx_model)

    pred_body = load_pred_body_data(cam0_dir, args.person_id)
    if pred_body is None:
        print(f"No body data for person {args.person_id}"); return

    # ── Collect per-frame data ────────────────────────────────────────────────
    signed_errors_z: list[float] = []      # pred_z - GT_z  (before correction)
    signed_errors_z_corr: list[float] = [] # pred_z - GT_z  (after correction)
    depth_deltas: list[float] = []         # delta = z_pelvis - z_torso_front
    gt_distances:     list[float] = []      # ||GT_pelvis|| Euclidean
    gt_pelvis_zs:     list[float] = []      # GT pelvis Z
    axis_errors: list[np.ndarray] = []     # (pred - GT) per axis
    j0_diffs: list[np.ndarray] = []

    frame_indices = sorted(pred_body.keys())
    print(f"Total pred frames: {len(frame_indices)}")

    processed = 0
    for fi in frame_indices:
        gt = load_gt_frame(rich_root, args.split, scene_name, fi)
        if gt is None:
            continue

        fd = pred_body[fi]

        # Pelvis positions: transl + J0(betas)
        J0_pred = pelvis_from_betas(smplx_model, fd["betas"])
        J0_gt   = pelvis_from_betas(smplx_model, gt["betas"])

        pelvis_pred = fd["transl"] + J0_pred   # (3,) world = cam0 frame
        pelvis_gt   = gt["transl"] + J0_gt     # (3,)

        # ── Depth-offset correction ──────────────────────────────────────────
        # Build posed predicted mesh (cam0=[I|0], so world frame = cam0 frame)
        verts = build_posed_mesh(smplx_model,
                                 fd["betas"], fd["body_pose"],
                                 fd["global_orient"], fd["transl"])
        z_pelvis      = float(pelvis_pred[2])
        z_torso_front = float(verts[torso_mask, 2].min())  # nearest torso vertex
        delta         = z_pelvis - z_torso_front           # body thickness in Z (>0)
        # Push root backward (away from cam0) by delta along cam0 Z-axis
        pelvis_corr_z = z_pelvis + delta

        depth_deltas.append(delta)
        signed_errors_z_corr.append(pelvis_corr_z - float(pelvis_gt[2]))

        # Z = depth along cam0 optical axis
        signed_errors_z.append(float(pelvis_pred[2] - pelvis_gt[2]))
        gt_distances.append(float(np.linalg.norm(pelvis_gt)))
        gt_pelvis_zs.append(float(pelvis_gt[2]))
        axis_errors.append(pelvis_pred - pelvis_gt)
        j0_diffs.append(J0_pred - J0_gt)

        processed += 1

    print(f"Frames with GT: {processed}\n")
    if processed == 0:
        print("No GT data found — check --split and paths."); return

    se   = np.array(signed_errors_z)        # signed depth errors, before (m)
    se_c = np.array(signed_errors_z_corr)  # signed depth errors, after (m)
    deltas = np.array(depth_deltas)         # per-frame body-thickness deltas (m)
    d_gt = np.array(gt_distances)
    z_gt = np.array(gt_pelvis_zs)
    ax   = np.array(axis_errors)            # (N, 3)
    j0d  = np.array(j0_diffs)              # (N, 3)

    sep = "─" * 62

    # ── Check 1: Sign & consistency ──────────────────────────────────────────
    print(sep)
    print("CHECK 1 — Sign & consistency of depth error (pred_Z − GT_Z)")
    print(sep)
    print(f"  n frames       : {len(se)}")
    print(f"  mean           : {se.mean()*100:+.2f} cm")
    print(f"  std            : {se.std()*100:.2f} cm")
    print(f"  median         : {np.median(se)*100:+.2f} cm")
    print(f"  min / max      : {se.min()*100:+.2f} / {se.max()*100:+.2f} cm")
    pos_frac = (se > 0).mean()
    print(f"  frac > 0 (pred too far)  : {pos_frac:.2%}")
    print(f"  frac < 0 (pred too near) : {1-pos_frac:.2%}")
    print(f"\n  Per-axis mean error (pred − GT):")
    for i, ax_name in enumerate("XYZ"):
        print(f"    {ax_name}: mean={ax[:, i].mean()*100:+.2f} cm  "
              f"std={ax[:, i].std()*100:.2f} cm")
    print()

    # ── Check 2: Constant vs distance-dependent ──────────────────────────────
    print(sep)
    print("CHECK 2 — Error vs GT distance  (constant bias vs scale issue)")
    print(sep)
    # Regress: signed_error = slope * GT_z + intercept
    slope, intercept, r, pval, _ = linregress(z_gt, se)
    print(f"  Linear fit:  error = {slope:.4f} * GT_z + {intercept*100:+.2f} cm")
    print(f"  slope        : {slope:.4f}  ({slope*100:.2f} cm/m  →"
          f"  {slope*100:.1f}% scale bias)")
    print(f"  intercept    : {intercept*100:+.2f} cm")
    print(f"  R²           : {r**2:.4f}   (0=constant, 1=fully proportional)")
    print(f"  p-value      : {pval:.4f}")

    # GT distance deciles
    print(f"\n  Error by GT depth decile:")
    deciles = np.percentile(z_gt, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    bins = np.concatenate([[z_gt.min()-0.01], deciles, [z_gt.max()+0.01]])
    print(f"  {'GT_z range (m)':>20}  {'n':>5}  {'mean err (cm)':>15}  {'std':>8}")
    for i in range(len(bins) - 1):
        mask = (z_gt >= bins[i]) & (z_gt < bins[i+1])
        n = mask.sum()
        if n == 0:
            continue
        m = se[mask].mean() * 100
        s = se[mask].std() * 100
        label = f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
        print(f"  {label:>20}  {n:>5}  {m:>+15.2f}  {s:>8.2f}")
    print()

    # ── Check 3: Surface-vs-pelvis offset ────────────────────────────────────
    print(sep)
    print("CHECK 3 — Surface-vs-pelvis depth offset  (GT posed mesh)")
    print(sep)
    print(f"  Building posed GT mesh for {args.surface_frames} frames …")

    surface_to_pelvis: list[float] = []   # GT_pelvis_z - min_vertex_z
    near_to_pelvis:    list[float] = []   # GT_pelvis_z - min_vertex_z (50px near pelvis)
    frac_frames = max(1, len(frame_indices) // args.surface_frames)
    sample_frames = frame_indices[::frac_frames][: args.surface_frames]

    for fi in sample_frames:
        gt = load_gt_frame(rich_root, args.split, scene_name, fi)
        if gt is None:
            continue

        J0_gt = pelvis_from_betas(smplx_model, gt["betas"])
        pelvis_gt_z = float(gt["transl"][2] + J0_gt[2])

        verts = build_posed_mesh(
            smplx_model,
            gt["betas"], gt["body_pose"],
            gt["global_orient"], gt["transl"],
        )
        min_z = float(verts[:, 2].min())   # frontmost vertex (closest to cam0)

        # Also: frontmost vertex along ray toward pelvis
        # Project GT pelvis to image, take vertices within r_px of that centroid
        # cam0 = [I|0], so image coords: u=x/z*fx+cx, v=y/z*fy+cy
        K = placer.intrinsics[0, 0].astype(np.float64)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        pelvis_cam = gt["transl"] + J0_gt  # same as world (cam0=[I|0])
        px = pelvis_cam[0] / pelvis_cam[2] * fx + cx
        py = pelvis_cam[1] / pelvis_cam[2] * fy + cy

        verts_u = verts[:, 0] / (verts[:, 2] + 1e-9) * fx + cx
        verts_v = verts[:, 1] / (verts[:, 2] + 1e-9) * fy + cy
        r_px = 50.0
        near_pelvis = (np.abs(verts_u - px) < r_px) & (np.abs(verts_v - py) < r_px)
        near_min_z = float(verts[near_pelvis, 2].min()) if near_pelvis.any() else min_z

        gap = pelvis_gt_z - min_z
        gap_near = pelvis_gt_z - near_min_z
        surface_to_pelvis.append(gap)
        near_to_pelvis.append(gap_near)

        if len(surface_to_pelvis) <= 5:
            print(f"    frame {fi:5d}: GT_pelvis_z={pelvis_gt_z:.3f}m  "
                  f"min_vertex_z={min_z:.3f}m  gap={gap*100:.1f}cm  "
                  f"near_gap={gap_near*100:.1f}cm")

    stp  = np.array(surface_to_pelvis)
    ntp  = np.array(near_to_pelvis)
    print(f"\n  Surface→pelvis gap — global min vertex  (pelvis_z - min_vertex_z):")
    print(f"    n frames : {len(stp)}")
    print(f"    mean     : {stp.mean()*100:.2f} cm  (inflated by hands/feet)  ")
    print(f"    median   : {np.median(stp)*100:.2f} cm")
    print(f"\n  Surface→pelvis gap — near-pelvis vertex (50px image radius):")
    print(f"    n frames : {len(ntp)}")
    print(f"    mean     : {ntp.mean()*100:.2f} cm  ← compare to depth error above")
    print(f"    std      : {ntp.std()*100:.2f} cm")
    print(f"    median   : {np.median(ntp)*100:.2f} cm")
    print(f"    min/max  : {ntp.min()*100:.2f} / {ntp.max()*100:.2f} cm")
    print()

    # ── Check 4: Pelvis convention ────────────────────────────────────────────
    print(sep)
    print("CHECK 4 — Pelvis convention:  J0_pred vs J0_gt")
    print(sep)
    print("  Both use:  pelvis = transl + J0,  J0 = smplx_fk(betas, pose=0, orient=0).joints[0]")
    print(f"  J0_diff = J0(betas_pred) - J0(betas_gt)  over {len(j0d)} frames:")
    for i, ax_name in enumerate("XYZ"):
        print(f"    {ax_name}: mean={j0d[:, i].mean()*100:+.3f} cm  "
              f"std={j0d[:, i].std()*100:.3f} cm  "
              f"range=[{j0d[:, i].min()*100:+.2f}, {j0d[:, i].max()*100:+.2f}] cm")
    print(f"  J0 magnitude from GT betas:")
    # Compute a single representative J0 from median betas
    sample_j0 = []
    for fi in frame_indices[:50]:
        gt = load_gt_frame(rich_root, args.split, scene_name, fi)
        if gt is not None:
            sample_j0.append(pelvis_from_betas(smplx_model, gt["betas"]))
    if sample_j0:
        j0_arr = np.array(sample_j0)
        print(f"    mean J0 : X={j0_arr[:,0].mean()*100:+.2f}  "
              f"Y={j0_arr[:,1].mean()*100:+.2f}  "
              f"Z={j0_arr[:,2].mean()*100:+.2f} cm")
    print(sep)
    print()

    # ── Check 5: Deterministic depth-offset correction ────────────────────────
    print(sep)
    print("CHECK 5 — Deterministic depth-offset correction  (torso Voronoi mask)")
    print(sep)
    print("  Per frame:  delta = z_pelvis − z_torso_front  (cam0-frame Z depth)")
    print("              corrected_transl_z += delta  (push body backward along Z)")
    print(f"\n  delta (body thickness along Z) over {len(deltas)} frames:")
    print(f"    mean   : {deltas.mean()*100:.2f} cm")
    print(f"    std    : {deltas.std()*100:.2f} cm")
    print(f"    median : {np.median(deltas)*100:.2f} cm")
    print(f"    min/max: {deltas.min()*100:.2f} / {deltas.max()*100:.2f} cm")
    print(f"\n  Signed depth error (pred_Z − GT_Z):")
    print(f"    before correction:  {se.mean()*100:+.2f} ± {se.std()*100:.2f} cm  "
          f"(median {np.median(se)*100:+.2f})")
    print(f"    after  correction:  {se_c.mean()*100:+.2f} ± {se_c.std()*100:.2f} cm  "
          f"(median {np.median(se_c)*100:+.2f})")
    residual = se_c.mean() * 100
    sign_str = "still too near" if residual < 0 else "over-corrected (now too far)"
    print(f"    residual bias     : {residual:+.2f} cm  ({sign_str})")
    pos_frac_c = (se_c > 0).mean()
    print(f"    frac > 0 after    : {pos_frac_c:.2%}")
    print()
    print("SUMMARY")
    print(f"  Before correction:  mean depth error = {se.mean()*100:+.2f} ± {se.std()*100:.2f} cm")
    print(f"  Pred body thickness (delta):           {deltas.mean()*100:.2f} ± {deltas.std()*100:.2f} cm")
    print(f"  After  correction:  mean depth error = {se_c.mean()*100:+.2f} ± {se_c.std()*100:.2f} cm")


if __name__ == "__main__":
    main()
