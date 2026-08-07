#!/usr/bin/env python
"""Audit EgoHumans ReID by projecting the 3D GT into each camera's image.

Question this answers: do the person tracks ghost produced (and the raw->global
grouping in ``manual_reid.json``) actually correspond to the right real person?

Projection (no distortion model needed — ghost frames are ALREADY undistorted)::

    X_colmap = T_aria01 @ [X_smpl; 1]      # single shared anchor, ALL persons
    x_cam    = R @ X_colmap + t            # images.txt row for that exo cam
    uv       = (s * K_new) @ x_cam         # s = W_ghost / W_calib
    bbox_gt  = min/max of uv over vertices with z > 0

Inputs
    --gt_root      egohumans_gt_full/<activity>:
                     <seq>/processed_data/smpl/<frame:05d>.npy
                        -> {human_name: {vertices, joints, betas, ...}} in the
                           PRIMARY aria's SLAM frame (aria01), not per person
                     <seq>/colmap/workplace/images.txt                 (R, t)
                     <seq>/colmap/workplace/colmap_from_aria_transforms.pkl
    --calib_root   staged data root holding <seq>/exo/<cam>/calibration.json
                   (K_new from fisheye balance=0 undistort, at full res)
    --ghost_root   ghost_outputs/egohumans_new/<activity>:
                     <seq>/<cam>/body_data/person_<pid>.npz  (bbox, frame_indices)
                     <seq>/vggt_cameras_centered.npz         (intrinsics, image size)

Method
    Per frame, GT people and ghost pids are matched by Hungarian assignment on
    IoU. A pid's "identity" is the GT ``human_name`` it matches most often.
    ``manual_reid.json`` groups (global id -> [cXpY, ...]) are then checked: all
    members of one group must resolve to the SAME GT name, or the cross-view
    ReID merged two different people.

Two things that would otherwise masquerade as ReID failures are measured, not
assumed:
    * frame numbering — offsets -1/0/+1 are all scored, the best is reported;
    * world anchor — mean IoU under the aria01 anchor is reported next to the
      per-person anchor, so a wrong-anchor bug is visible as a global collapse.

Example:
    pixi run python debug/audit_egohumans_reid_gt.py \
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/07_tennis \
        --gt_root /iopsstor/scratch/cscs/tnanni/egohumans_gt_full/07_tennis \
        --calib_root /capstor/scratch/cscs/tnanni/datasets/egohumans/07_tennis/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/07_tennis \
        --out eval_explainability/audit_reid_07_tennis.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

OFFSETS = (-1, 0, 1)       # ghost_frame = gt_frame - offset
MIN_VIS_VERTS = 200        # projected verts in front of cam & near image
_TOKEN = re.compile(r"^c(\d+)p(\d+)$")


# --------------------------------------------------------------------------- #
# COLMAP extrinsics (standalone copy of the parsers in render_egohumans_gt.py)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], np.float64)


def parse_static_extrinsics(images_txt: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """cam_name -> (R, t) with X_cam = R @ X_colmap + t.

    Exo cams are static, so the first row for a cam is representative; the
    spread across that cam's rows is checked by the caller via `extrinsic_spread`.
    """
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 10:
                continue
            cam = p[9].split("/")[0]
            q = np.array([float(v) for v in p[1:5]], np.float64)
            t = np.array([float(v) for v in p[5:8]], np.float64)
            out[cam].append((quat_wxyz_to_R(q), t))
    return {c: v[0] for c, v in out.items()}, \
           {c: float(np.std([np.asarray(x[1]) for x in v], axis=0).max()) if len(v) > 1 else 0.0
            for c, v in out.items()}


def load_K(calib_json: Path, w_ghost: int, h_ghost: int) -> np.ndarray:
    """K_new (full-res, balance=0) scaled into the ghost image space.

    NOTE: on the EgoHumans staging this file does NOT describe the frames the
    pipeline ran on (its focal is ~1.36x too short — the frames were undistorted
    by a different code path than the one that wrote it). Kept as a fallback and
    as a diagnostic; the default K source is ``vggt_cameras_centered.npz``.
    """
    d = json.load(open(calib_json))
    K = np.asarray(d["K"], np.float64)
    W, H = int(d["width"]), int(d["height"])
    sx, sy = w_ghost / W, h_ghost / H
    if abs(sx - sy) > 1e-3:
        raise ValueError(f"aspect mismatch: calib {W}x{H} vs ghost {w_ghost}x{h_ghost}")
    S = np.array([[sx, 0, 0], [0, sx, 0], [0, 0, 1]], np.float64)
    return S @ K


def load_vggt_cameras(scene_dir: Path) -> tuple[dict[str, np.ndarray],
                                                 dict[str, tuple[int, int]]]:
    """(cam -> K in ghost pixel space, cam -> (W, H)) from vggt_cameras_centered.npz.

    These are the intrinsics the pipeline itself used, so they are the ones that
    make a GT projection land on the ghost boxes. Arrays are per-frame
    ``(T, C, 3, 3)`` / ``(T, C, 2)``; the per-camera median over valid frames is
    taken. VGGT works at a fixed ``vggt_hw``-style resolution encoded in the
    principal point (cx = W_vggt/2), so K is rescaled to the ghost image size.
    """
    npz = scene_dir / "vggt_cameras_centered.npz"
    if not npz.exists():
        raise FileNotFoundError(f"no vggt_cameras_centered.npz in {scene_dir}")
    z = np.load(npz, allow_pickle=True)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["camera_names"]]
    Kv = np.asarray(z["intrinsics"], np.float64)             # (T, C, 3, 3)
    sizes = np.asarray(z["original_size"]).astype(int)       # (T, C, 2)
    valid = np.asarray(z["valid"]).astype(bool) if "valid" in z.files \
        else np.ones(Kv.shape[:2], bool)

    K_out, wh_out = {}, {}
    for i, n in enumerate(names):
        m = valid[:, i]
        if not m.any():
            continue
        K = np.median(Kv[m, i], axis=0)
        w, h = (int(v) for v in np.median(sizes[m, i], axis=0))
        # VGGT resolution is implicit in the principal point (cx, cy) = (W/2, H/2)
        vw, vh = 2.0 * K[0, 2], 2.0 * K[1, 2]
        K[0] *= w / vw
        K[1] *= h / vh
        K_out[n], wh_out[n] = K, (w, h)
    return K_out, wh_out


# --------------------------------------------------------------------------- #
# GT side
# --------------------------------------------------------------------------- #
def load_gt_frame(npy_path: Path) -> dict[str, np.ndarray]:
    """human_name -> (V,3) vertices in the aria01 SLAM frame."""
    if not npy_path.exists():
        return {}
    arr = np.load(str(npy_path), allow_pickle=True)
    d = arr.item() if arr.dtype == object and arr.shape == () else arr
    if not isinstance(d, dict):
        return {}
    out = {}
    for name, params in d.items():
        if not isinstance(params, dict) or "vertices" not in params:
            continue
        out[str(name)] = np.asarray(params["vertices"], np.float64).reshape(-1, 3)
    return out


def project_box(verts_aria: np.ndarray, T: np.ndarray, R: np.ndarray, t: np.ndarray,
                K: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """aria-frame verts -> ghost-pixel bbox, or None if not usefully visible."""
    Xc = (T[:3, :3] @ verts_aria.T + T[:3, 3:4]).T          # -> colmap world
    Xc = (R @ Xc.T + t.reshape(3, 1)).T                     # -> camera
    front = Xc[:, 2] > 1e-6
    if front.sum() < MIN_VIS_VERTS:
        return None
    uv = (K @ Xc[front].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    near = ((uv[:, 0] > -w) & (uv[:, 0] < 2 * w) &
            (uv[:, 1] > -h) & (uv[:, 1] < 2 * h))
    if near.sum() < MIN_VIS_VERTS:
        return None
    uv = uv[near]
    box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    # must overlap the image at all
    if box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
        return None
    return box


def gt_frame_indices(smpl_dir: Path) -> list[int]:
    idxs = []
    for p in smpl_dir.glob("*.npy"):
        try:
            idxs.append(int(p.stem))
        except ValueError:
            continue
    return sorted(idxs)


# --------------------------------------------------------------------------- #
# ghost side
# --------------------------------------------------------------------------- #
def ghost_tracks(cam_dir: Path) -> dict[int, dict[int, np.ndarray]]:
    """pid -> {frame_index: bbox xyxy} from body_data/person_<pid>.npz."""
    tracks: dict[int, dict[int, np.ndarray]] = {}
    for f in sorted((cam_dir / "body_data").glob("person_*.npz")):
        try:
            pid = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        d = np.load(f, allow_pickle=True)
        if "bbox" not in d.files or "frame_indices" not in d.files:
            continue
        fr = np.asarray(d["frame_indices"]).astype(int)
        bb = np.asarray(d["bbox"], np.float64)
        tracks[pid] = {int(fr[i]): bb[i] for i in range(len(fr))}
    return tracks


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def match_frame(gt_boxes: dict[str, np.ndarray], pred_boxes: dict[int, np.ndarray],
                thr: float) -> list[tuple[str, int, float]]:
    if not gt_boxes or not pred_boxes:
        return []
    names, pids = list(gt_boxes), list(pred_boxes)
    cost = np.zeros((len(names), len(pids)))
    for i, n in enumerate(names):
        for j, p in enumerate(pids):
            cost[i, j] = -iou(gt_boxes[n], pred_boxes[p])
    ri, ci = linear_sum_assignment(cost)
    out = []
    for i, j in zip(ri, ci):
        v = -cost[i, j]
        if v >= thr:
            out.append((names[i], pids[j], v))
    return out


def score(gt_by_frame: dict[int, dict[str, np.ndarray]],
          tracks: dict[int, dict[int, np.ndarray]], off: int, thr: float):
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    iou_sum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    gt_seen: dict[str, int] = defaultdict(int)
    matched, total_iou = 0, 0.0
    for gf, gt_boxes in gt_by_frame.items():
        hf = gf - off
        pred = {pid: t[hf] for pid, t in tracks.items() if hf in t}
        for n in gt_boxes:
            gt_seen[n] += 1
        for name, pid, v in match_frame(gt_boxes, pred, thr):
            counts[pid][name] += 1
            iou_sum[pid][name] += v
            matched += 1
            total_iou += v
    return matched, total_iou, counts, iou_sum, gt_seen


def build_gt_boxes(smpl_dir: Path, sample: list[int], T: np.ndarray,
                   R: np.ndarray, t: np.ndarray, K: np.ndarray, w: int, h: int,
                   per_person_T: dict[str, np.ndarray] | None = None):
    """{frame: {name: bbox}} under one anchor convention."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for gf in sample:
        boxes = {}
        for name, verts in load_gt_frame(smpl_dir / f"{gf:05d}.npy").items():
            Tn = per_person_T.get(name, T) if per_person_T else T
            box = project_box(verts, Tn, R, t, K, w, h)
            if box is not None:
                boxes[name] = box
        if boxes:
            out[gf] = boxes
    return out


def audit_cam(cam: str, smpl_dir: Path, cam_dir: Path, T_anchor: np.ndarray,
              T_all: dict[str, np.ndarray], Rt: tuple[np.ndarray, np.ndarray],
              K: np.ndarray, wh: tuple[int, int], sample: list[int],
              thr: float) -> dict:
    w, h = wh
    R, t = Rt
    tracks = ghost_tracks(cam_dir)
    if not tracks:
        return {"cam": cam, "error": "no ghost tracks"}

    gt_by_frame = build_gt_boxes(smpl_dir, sample, T_anchor, R, t, K, w, h)
    if not gt_by_frame:
        return {"cam": cam, "error": "no GT projected into view"}

    best = None
    for off in OFFSETS:
        matched, total_iou, counts, iou_sum, gt_seen = score(gt_by_frame, tracks, off, thr)
        if best is None or (matched, total_iou) > (best[0], best[1]):
            best = (matched, total_iou, counts, iou_sum, gt_seen, off)
    matched, total_iou, counts, iou_sum, gt_seen, off = best

    # diagnostic: same scoring with the (wrong) per-person anchor
    gt_pp = build_gt_boxes(smpl_dir, sample, T_anchor, R, t, K, w, h, per_person_T=T_all)
    m_pp, i_pp, *_ = score(gt_pp, tracks, off, thr) if gt_pp else (0, 0.0, None, None, None)

    pid_report = {}
    for pid in sorted(tracks):
        hist = counts.get(pid, {})
        if not hist:
            pid_report[str(pid)] = {"track_frames": len(tracks[pid]), "matched": 0,
                                    "identity": None, "purity": 0.0,
                                    "mean_iou": 0.0, "hist": {}}
            continue
        dom = max(hist, key=hist.get)
        tot = sum(hist.values())
        pid_report[str(pid)] = {
            "track_frames": len(tracks[pid]),
            "matched": tot,
            "identity": dom,
            "purity": hist[dom] / tot,
            "mean_iou": iou_sum[pid][dom] / hist[dom],
            "hist": dict(hist),
        }
    return {
        "cam": cam,
        "frame_offset": off,
        "gt_frames_sampled": len(gt_by_frame),
        "gt_people": dict(gt_seen),
        "matched_pairs": matched,
        "mean_iou_all": (total_iou / matched) if matched else 0.0,
        "anchor_check": {"aria01_matched": matched,
                         "aria01_mean_iou": (total_iou / matched) if matched else 0.0,
                         "perperson_matched": m_pp,
                         "perperson_mean_iou": (i_pp / m_pp) if m_pp else 0.0},
        "pids": pid_report,
    }


# --------------------------------------------------------------------------- #
# manual_reid group check
# --------------------------------------------------------------------------- #
def check_groups(groups: dict, cam_reports: dict) -> dict:
    """Each global id must resolve to ONE GT name across all its cXpY members."""
    out = {}
    for gid, tokens in groups.items():
        members = []
        for tok in tokens:
            m = _TOKEN.match(tok)
            if not m:
                members.append({"token": tok, "error": "unparsable"})
                continue
            cam = f"cam{int(m.group(1)):02d}"
            pid = m.group(2)
            rep = cam_reports.get(cam)
            if rep is None or "pids" not in rep:
                members.append({"token": tok, "cam": cam, "pid": pid,
                                "error": "cam not audited"})
                continue
            info = rep["pids"].get(pid)
            if info is None:
                members.append({"token": tok, "cam": cam, "pid": pid,
                                "error": "pid does not exist"})
                continue
            members.append({"token": tok, "cam": cam, "pid": pid,
                            "identity": info["identity"],
                            "purity": info["purity"],
                            "matched": info["matched"],
                            "mean_iou": info["mean_iou"]})
        names = {m.get("identity") for m in members if m.get("identity")}
        out[str(gid)] = {
            "members": members,
            "identities": sorted(names),
            "consistent": len(names) <= 1,
            "unmatched_tokens": [m["token"] for m in members if not m.get("identity")],
        }
    claimed = defaultdict(set)
    for g in out.values():
        for m in g["members"]:
            if m.get("identity"):
                claimed[m["cam"]].add(m["identity"])
    missing = {}
    for cam, rep in cam_reports.items():
        if "gt_people" not in rep:
            continue
        miss = [n for n in rep["gt_people"] if n not in claimed.get(cam, set())]
        if miss:
            missing[cam] = miss
    return {"groups": out, "gt_people_not_in_any_group": missing}


# --------------------------------------------------------------------------- #
def audit_scene(scene: str, ghost_root: Path, gt_root: Path, calib_root: Path | None,
                groups: dict, thr: float, max_frames: int, anchor: str,
                k_source: str) -> dict:
    ghost_scene = ghost_root / scene
    gt_scene = gt_root / scene
    smpl_dir = gt_scene / "processed_data" / "smpl"
    images_txt = gt_scene / "colmap" / "workplace" / "images.txt"
    tf_pkl = gt_scene / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    for p in (smpl_dir, images_txt, tf_pkl):
        if not p.exists():
            return {"scene": scene, "error": f"missing {p}"}

    transforms = {k: np.asarray(v, np.float64)
                  for k, v in pickle.load(open(tf_pkl, "rb")).items()}
    if anchor not in transforms:
        return {"scene": scene, "error": f"anchor {anchor} not in {sorted(transforms)}"}
    T_anchor = transforms[anchor]

    extr, spread = parse_static_extrinsics(images_txt)
    K_vggt, sizes = load_vggt_cameras(ghost_scene)
    frames = gt_frame_indices(smpl_dir)
    if not frames:
        return {"scene": scene, "error": "no GT smpl frames"}
    step = max(1, len(frames) // max_frames)
    sample = frames[::step]

    cam_reports = {}
    for cam in sorted(sizes):
        cam_dir = ghost_scene / cam
        if not (cam_dir / "body_data").is_dir():
            continue
        if cam not in extr:
            cam_reports[cam] = {"cam": cam, "error": "no extrinsics in images.txt"}
            continue
        calib_json = calib_root / scene / "exo" / cam / "calibration.json" \
            if calib_root else None
        try:
            fx_calib = None
            if calib_json is not None and calib_json.exists():
                fx_calib = float(load_K(calib_json, *sizes[cam])[0, 0])
            if k_source == "calib":
                if fx_calib is None:
                    cam_reports[cam] = {"cam": cam, "error": f"no {calib_json}"}
                    continue
                K = load_K(calib_json, *sizes[cam])
            else:
                if cam not in K_vggt:
                    cam_reports[cam] = {
                        "cam": cam,
                        "error": "cam absent from vggt_cameras_centered.npz"}
                    continue
                K = K_vggt[cam]
            rep = audit_cam(cam, smpl_dir, cam_dir, T_anchor, transforms,
                            extr[cam], K, sizes[cam], sample, thr)
            rep["extrinsic_spread_m"] = spread.get(cam, 0.0)
            rep["fx_used"] = float(K[0, 0])
            rep["fx_calib_json"] = fx_calib
            cam_reports[cam] = rep
        except Exception as e:                                # noqa: BLE001
            cam_reports[cam] = {"cam": cam, "error": f"{type(e).__name__}: {e}"}

    res = {"scene": scene, "cams": cam_reports}
    if groups:
        res.update(check_groups(groups, cam_reports))
    return res


def print_scene(res: dict) -> None:
    print(f"\n=== {res['scene']} ===", flush=True)
    if "error" in res:
        print(f"  ERROR {res['error']}")
        return
    for cam, rep in res["cams"].items():
        if "error" in rep:
            print(f"  {cam}: ERROR {rep['error']}")
            continue
        gtp = ", ".join(f"{k}:{v}" for k, v in sorted(rep["gt_people"].items()))
        ac = rep["anchor_check"]
        fxc = rep.get("fx_calib_json")
        fx_note = (f"fx {rep.get('fx_used', 0):.0f}"
                   + (f" (calib.json {fxc:.0f})" if fxc else ""))
        print(f"  {cam}  offset={rep['frame_offset']:+d}  {fx_note}  "
              f"GT frames {rep['gt_frames_sampled']} [{gtp}]  "
              f"matched {rep['matched_pairs']}  mIoU {rep['mean_iou_all']:.2f}  "
              f"(per-person anchor: {ac['perperson_matched']} / "
              f"{ac['perperson_mean_iou']:.2f})")
        print(f"      {'pid':>4} {'trkfr':>6} {'match':>6} {'identity':>9} "
              f"{'purity':>7} {'mIoU':>5}  hist")
        for pid, info in rep["pids"].items():
            hist = ",".join(f"{k}:{v}" for k, v in sorted(info["hist"].items()))
            print(f"      {pid:>4} {info['track_frames']:>6} {info['matched']:>6} "
                  f"{str(info['identity']):>9} {info['purity']:>7.2f} "
                  f"{info['mean_iou']:>5.2f}  {hist}")
    if "groups" in res:
        print("  GROUPS (manual_reid.json)")
        for gid, g in sorted(res["groups"].items(), key=lambda kv: int(kv[0])):
            if not g["consistent"]:
                tag = "MISMATCH"
            elif g["unmatched_tokens"]:
                tag = "PARTIAL"
            else:
                tag = "OK"
            body = "  ".join(
                f"{m['token']}->{m.get('identity') or m.get('error')}"
                f"({m.get('purity', 0):.2f})" for m in g["members"])
            print(f"    [{tag:^8}] g{gid}: {body}")
        for cam, miss in res.get("gt_people_not_in_any_group", {}).items():
            print(f"    MISSING  {cam}: GT {', '.join(miss)} in no group")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True,
                    help="ghost_outputs/egohumans_new/<activity>")
    ap.add_argument("--gt_root", required=True,
                    help="egohumans_gt_full/<activity> (smpl + colmap)")
    ap.add_argument("--calib_root", default=None,
                    help="staged camera_ready/<activity> with exo/<cam>/calibration.json; "
                         "only needed for --k_source calib (else reported as a diagnostic)")
    ap.add_argument("--k_source", choices=("vggt", "calib"), default="vggt",
                    help="vggt = intrinsics from vggt_cameras_centered.npz (what the "
                         "pipeline actually used); calib = staged calibration.json "
                         "(focal ~1.36x too short on EgoHumans — will not land)")
    ap.add_argument("--manual_reid", default="manual_reid.json")
    ap.add_argument("--dataset_key", default="egohumans")
    ap.add_argument("--anchor", default="aria01",
                    help="colmap_from_aria key used as the shared world anchor")
    ap.add_argument("--scene", action="append", default=None,
                    help="repeatable; default = every scene in ghost_root")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max_frames", type=int, default=120,
                    help="GT frames sampled per scene")
    ap.add_argument("--out", default=None, help="write full report as JSON")
    args = ap.parse_args()

    ghost_root = Path(args.ghost_root)
    manual = {}
    mr = Path(args.manual_reid)
    if mr.exists():
        manual = json.load(open(mr)).get(args.dataset_key, {})
    else:
        print(f"WARN no manual_reid at {mr} — group check skipped")

    scenes = args.scene or sorted(d.name for d in ghost_root.iterdir() if d.is_dir())

    report = []
    for scene in scenes:
        groups = manual.get(scene, {}).get("groups", {})
        try:
            res = audit_scene(scene, ghost_root, Path(args.gt_root),
                              Path(args.calib_root) if args.calib_root else None,
                              groups, args.iou, args.max_frames, args.anchor,
                              args.k_source)
        except Exception as e:                                # noqa: BLE001
            res = {"scene": scene, "error": f"{type(e).__name__}: {e}"}
        print_scene(res)
        report.append(res)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out, "w"), indent=1, default=float)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
