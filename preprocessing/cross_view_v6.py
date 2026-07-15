"""
Cross-view person ReID v6 — Hungarian assignment + relative-direction geometry.

Phase 1: Per camera-pair Hungarian assignment on a 4-term similarity matrix:
  (1) betas cosine sim  (2) TransReID/DINOv3 appearance  (3) rotation-aligned kp3d
  (4) relative pelvis direction — pelvis of candidate relative to the anchor person,
      rotated by cam_pair_R (fitted from betas-best anchor pair), compared with
      the corresponding direction in cam_B.

  Rectangular linear_sum_assignment (scipy) is used so no dummy threshold is needed:
  min(|A|,|B|) persons get 1-to-1 matches; the remaining persons in the larger camera
  become singletons naturally. Matches from all pairs are then fed into Union-Find
  (highest-sim first, with camera-uniqueness conflict check) to produce global clusters.

Phase 2: Directional consistency check (legacy, fires only when cam_t_dict populated).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

from data.video_dataset import Scene

# ── Phase 1 parameters ────────────────────────────────────────────────────────
_BETAS_W = 0.50
_APP_W   = 0.50
_KP3D_W  = 1.00
_DIR_W = 1.00

# ── Phase 2 parameters ────────────────────────────────────────────────────────


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _fit_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Pure rotation fit (no scale, no translation):
        min_{R in SO(3)} || dst - R @ src ||²_F
    src, dst: (N, 3) float64.
    Returns R: (3, 3).
    """
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d_sign = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T


def _sector(v: np.ndarray) -> int | None:
    """
    Discretise a 3-D vector to one of 8 horizontal sectors (0 = N … 7 = NW).
    Uses the (x, z) components: z = forward / "north", x = east.
    Returns None if vector is zero.
    """
    x, z = float(v[0]), float(v[2])
    if abs(x) < 1e-6 and abs(z) < 1e-6:
        return None
    return round(math.atan2(x, z) / (math.pi / 4)) % 8


def _sector_dist(a: int, b: int) -> int:
    """Circular distance between two sectors (0-7). Max = 4."""
    return min((a - b) % 8, (b - a) % 8)


# ── Main class ────────────────────────────────────────────────────────────────

class CrossViewReidentifierV6:
    def __init__(
        self,
        reid_ckpt: str | None = None,
        betas_weight: float = _BETAS_W,
        app_weight: float = _APP_W,
        kp3d_weight: float = _KP3D_W,
        dir_weight: float = _DIR_W,
    ) -> None:
        self.reid_ckpt    = reid_ckpt
        self.betas_weight = betas_weight
        self.app_weight   = app_weight
        self.kp3d_weight  = kp3d_weight
        self.dir_weight   = dir_weight
        self._reid = None

    def _get_reid(self):
        if self.reid_ckpt is None:
            return None
        if self._reid is None:
            from preprocessing.transreid_extractor import TransReIDExtractor
            self._reid = TransReIDExtractor(self.reid_ckpt)
        return self._reid

    def match_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path] | None = None,
        intrinsics_map=None,
        dry_run: bool = False,
    ) -> None:
        scene_id  = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent
        if not dry_run and (scene_dir / "cross_view_reid.json").exists():
            logging.info(f"  Scene {scene_id}: cross-view ReID v6 already done, skipping")
            return
        if frames_dirs is None:
            frames_dirs = {}
            for video in scene.videos:
                if hasattr(video, "frames_home") and video.frames_home is not None:
                    frames_dirs[video.video_id] = video.frames_home

        reid = self._get_reid()

        # ── Feature loading ───────────────────────────────────────────────────
        shape_feat:  dict[tuple, np.ndarray]        = {}
        app_feat:    dict[tuple, np.ndarray]        = {}
        kp3d_rel:    dict[tuple, np.ndarray]        = {}   # node → (T, J, 3) root-relative
        kp3d_dict:   dict[tuple, dict[int, np.ndarray]] = {}  # node → {abs_frame: kp(J,3)}
        pose_dict:   dict[tuple, dict[int, np.ndarray]] = {}  # node → {abs_frame: body_pose(63,)}
        cam_t_dict:  dict[tuple, dict[int, np.ndarray]] = {}  # node → {abs_frame: pos(3,)}
        person_pids: dict[str, list[int]]    = {}
        active_vids: list[str]               = []

        for vid_id, vid_dir in video_dirs.items():
            body_dir     = Path(vid_dir) / "body_data"
            summary_path = body_dir / "body_params_summary.json"
            if not summary_path.exists():
                continue
            pids = [int(k) for k in
                    json.loads(summary_path.read_text()).get("persons", {}).keys()]
            if not pids:
                continue

            gallery: dict[int, tuple] = {}
            gpath = body_dir / "appearance_gallery.npz"
            if gpath.exists():
                gd = np.load(str(gpath))
                for k in gd.files:
                    if k.endswith("_conf"):
                        continue
                    feats = gd[k]
                    confs = (gd[f"{k}_conf"] if f"{k}_conf" in gd.files
                             else np.ones(len(feats), np.float32))
                    gallery[int(k)] = (feats, confs)

            fdir       = frames_dirs.get(vid_id)
            jdir       = Path(vid_dir) / "json_data"
            cam_suffix = str(vid_id).split("_")[-1]
            kept: list[int] = []

            for pid in pids:
                npz = body_dir / f"person_{pid}.npz"
                if not npz.exists():
                    continue
                node = (vid_id, pid)

                with np.load(str(npz)) as d:
                    # Per-frame quality mask: discard frames where >30% of 2D kps
                    # fall outside the bbox (SAM3D hallucinates full body from partial crops)
                    good_frames: set[int] = set()
                    n_total = len(d["frame_indices"]) if "frame_indices" in d else 0
                    if ("pred_keypoints_2d" in d and "bbox" in d and "frame_indices" in d
                            and len(d["pred_keypoints_2d"]) > 0):
                        kp2d = d["pred_keypoints_2d"].astype(np.float32)  # (T, J, 2)
                        bb   = d["bbox"].astype(np.float32)                # (T, 4) x1y1x2y2
                        fi_q = d["frame_indices"].astype(int)
                        J = kp2d.shape[1]
                        for i in range(len(fi_q)):
                            x1, y1, x2, y2 = bb[i]
                            kp = kp2d[i]
                            n_out = int(np.sum(
                                (kp[:, 0] < x1) | (kp[:, 0] > x2) |
                                (kp[:, 1] < y1) | (kp[:, 1] > y2)))
                            if n_out / J <= 0.30:
                                good_frames.add(int(fi_q[i]))

                    n_good = len(good_frames)
                    logging.info(f"  [v6-QA] {node}: {n_good}/{n_total} good frames "
                                 f"(≤30% kp2d outside bbox)")
                    if n_good < 15:
                        logging.info(f"  [v6-QA] {node}: DISCARD — only {n_good} good frames")
                        continue

                    # shape: median betas over GOOD frames only (falls back to all if none pass)
                    if "smplx_betas" in d and "frame_indices" in d and len(d["smplx_betas"]) > 0:
                        fi_b = d["frame_indices"].astype(int)
                        betas_all = d["smplx_betas"].astype(np.float32)
                        good_mask = np.array([fi_b[i] in good_frames for i in range(len(fi_b))])
                        betas_use = betas_all[good_mask] if good_mask.any() else betas_all
                        sm = np.median(betas_use, axis=0)
                        nm = np.linalg.norm(sm)
                        if nm > 0:
                            shape_feat[node] = sm / nm
                    # root-relative 3-D keypoints for R fitting + per-frame sim (good frames only)
                    if ("pred_keypoints_3d" in d and "frame_indices" in d
                            and len(d["pred_keypoints_3d"]) > 0):
                        kp = d["pred_keypoints_3d"].astype(np.float32)
                        fi_kp = d["frame_indices"].astype(int)
                        if kp.ndim == 3 and kp.shape[0] > 0:
                            kp3d_rel[node] = kp
                            if len(fi_kp) == len(kp):
                                kp3d_dict[node] = {int(fi_kp[i]): kp[i]
                                                   for i in range(len(kp))
                                                   if int(fi_kp[i]) in good_frames}
                    # smplx_body_pose: per-frame joint angles (63-d axis-angle, camera-independent)
                    if ("smplx_body_pose" in d and "frame_indices" in d
                            and len(d["smplx_body_pose"]) > 0):
                        bp = d["smplx_body_pose"].astype(np.float32)
                        fi_bp = d["frame_indices"].astype(int)
                        if len(fi_bp) == len(bp):
                            pose_dict[node] = {int(fi_bp[i]): bp[i]
                                               for i in range(len(bp))
                                               if int(fi_bp[i]) in good_frames}
                    # pelvis trajectory (good frames only)
                    if ("pred_cam_t" in d and "frame_indices" in d
                            and len(d["pred_cam_t"]) > 0):
                        ct = d["pred_cam_t"].astype(np.float32)
                        fi = d["frame_indices"].astype(int)
                        if len(fi) == len(ct):
                            cam_t_dict[node] = {int(fi[i]): ct[i]
                                                for i in range(len(ct))
                                                if int(fi[i]) in good_frames}
                # appearance: TransReID preferred, DINOv3 gallery-mean fallback
                # If TransReID is available but returns None (occlusion filter), do NOT
                # fall back to gallery — leave appearance absent so geometry handles it.
                av = None
                transreid_attempted = reid is not None and fdir is not None
                if transreid_attempted:
                    try:
                        av = reid.person_feature(fdir, jdir, cam_suffix, pid)
                        if av is None:
                            logging.info(f"  [v6-APP] {node}: TransReID→None (occlusion filter) → no appearance, geometry will resolve")
                    except Exception as e:
                        logging.warning(f"  [v6-APP] {node}: TransReID error ({e}) → DINOv3 fallback")
                        transreid_attempted = False
                if not transreid_attempted and av is None and pid in gallery:
                    f, c = gallery[pid]
                    w = c / (c.sum() + 1e-8)
                    m = f.T @ w
                    n = np.linalg.norm(m)
                    av = (m / n).astype(np.float32) if n > 1e-8 else None
                    if av is not None:
                        logging.info(f"  [v6-APP] {node}: DINOv3 gallery fallback")
                if av is not None:
                    app_feat[node] = av

                if node in shape_feat or node in app_feat:
                    kept.append(pid)

            if kept:
                person_pids[vid_id] = sorted(kept)
                active_vids.append(vid_id)

        if len(active_vids) < 2:
            logging.info(f"Scene {scene_id}: fewer than 2 views with features, skipping")
            return

        nodes = [(v, p) for v in active_vids for p in person_pids[v]]

        # ── Similarity ─────────────────────────────────────────────────────────
        w_b, w_a, w_k, w_d = (self.betas_weight, self.app_weight,
                               self.kp3d_weight, self.dir_weight)
        N_KP_SAMPLES  = 20
        N_DIR_SAMPLES = 20

        # Pre-compute rotation R per camera pair seeded from betas-only best match.
        # R maps cam_A joints → cam_B frame.
        cam_pair_R:    dict[tuple, np.ndarray] = {}
        cam_pair_seed: dict[tuple, tuple]      = {}  # (cam_A,cam_B) → (anchor_ni, anchor_nj)
        for cam_A, cam_B in combinations(active_vids, 2):
            best_s, seed_A, seed_B = -1.0, None, None
            for pA in person_pids[cam_A]:
                sA = shape_feat.get((cam_A, pA))
                if sA is None:
                    continue
                for pB in person_pids[cam_B]:
                    sB = shape_feat.get((cam_B, pB))
                    if sB is None:
                        continue
                    s = float(np.dot(sA, sB).clip(-1, 1))
                    if s > best_s:
                        best_s, seed_A, seed_B = s, (cam_A, pA), (cam_B, pB)
            if seed_A is None:
                continue
            dA, dB = kp3d_dict.get(seed_A), kp3d_dict.get(seed_B)
            if dA is None or dB is None:
                continue
            common = sorted(set(dA) & set(dB))
            if len(common) < 2:
                continue
            kA = np.stack([dA[f] for f in common]).reshape(-1, 3).astype(np.float64)
            kB = np.stack([dB[f] for f in common]).reshape(-1, 3).astype(np.float64)
            cam_pair_R[(cam_A, cam_B)]    = _fit_rotation(kA, kB)
            cam_pair_seed[(cam_A, cam_B)] = (seed_A, seed_B)
            logging.info(f"  [v6-R] fit {cam_A}↔{cam_B} from "
                         f"{seed_A[0]}:P{seed_A[1]}↔{seed_B[0]}:P{seed_B[1]} "
                         f"(betas sim={best_s:.3f}, {len(common)} shared frames)")

        def _kp3d_sim(ni: tuple, nj: tuple) -> float:
            """Cosine similarity of rotation-aligned root-relative 3D joints at shared frames."""
            cam_A, cam_B = ni[0], nj[0]
            if (cam_A, cam_B) in cam_pair_R:
                R = cam_pair_R[(cam_A, cam_B)]
            elif (cam_B, cam_A) in cam_pair_R:
                R = cam_pair_R[(cam_B, cam_A)].T
            else:
                return 0.0
            dA, dB = kp3d_dict.get(ni), kp3d_dict.get(nj)
            if dA is None or dB is None:
                return 0.0
            common = sorted(set(dA) & set(dB))
            if not common:
                return 0.0
            if len(common) > N_KP_SAMPLES:
                idx = np.linspace(0, len(common) - 1, N_KP_SAMPLES, dtype=int)
                common = [common[i] for i in idx]
            sims = []
            for f in common:
                kA_rot = (R @ dA[f].T).T.reshape(-1).astype(np.float64)
                kB_flat = dB[f].reshape(-1).astype(np.float64)
                nA = np.linalg.norm(kA_rot)
                nB = np.linalg.norm(kB_flat)
                if nA < 1e-6 or nB < 1e-6:
                    continue
                sims.append((float(np.dot(kA_rot, kB_flat) / (nA * nB)) + 1) / 2)
            return float(np.mean(sims)) if sims else 0.0

        _POSE_SIGMA = 0.5  # rad per joint; tune based on geodesic logs

        def _aa_to_R(aa: np.ndarray) -> np.ndarray:
            """Axis-angle (3,) → rotation matrix (3,3)."""
            theta = float(np.linalg.norm(aa))
            if theta < 1e-8:
                return np.eye(3)
            ax = aa / theta
            K  = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        def _geodesic(aa_a: np.ndarray, aa_b: np.ndarray) -> float:
            """Geodesic distance (rad) between two axis-angle rotations."""
            R_rel = _aa_to_R(aa_a).T @ _aa_to_R(aa_b)
            cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            return float(np.arccos(cos_a))

        def _pose_sim(ni: tuple, nj: tuple) -> float:
            """Mean per-joint geodesic distance across shared good frames → exp(-dist/sigma).

            Geodesic on SO(3) is the correct metric for axis-angle joint rotations.
            Robust to the direction-vs-magnitude ambiguity of cosine, and unlike full-vector
            L2, averaging per-joint distances is less sensitive to sync offsets (a 1-frame
            offset changes a few joints, not all 21 simultaneously).
            """
            dA, dB = pose_dict.get(ni), pose_dict.get(nj)
            if dA is None or dB is None:
                return 0.0
            common = sorted(set(dA) & set(dB))
            if not common:
                return 0.0
            if len(common) > N_KP_SAMPLES:
                idx = np.linspace(0, len(common) - 1, N_KP_SAMPLES, dtype=int)
                common = [common[i] for i in idx]
            n_joints = dA[common[0]].shape[0] // 3
            # per-joint geodesic distances averaged across sampled frames
            joint_geo = np.zeros(n_joints, dtype=np.float64)
            n_valid   = 0
            for f in common:
                a = dA[f].astype(np.float64).reshape(n_joints, 3)
                b = dB[f].astype(np.float64).reshape(n_joints, 3)
                frame_geo = np.array([_geodesic(a[j], b[j]) for j in range(n_joints)])
                joint_geo += frame_geo
                n_valid   += 1
            if n_valid == 0:
                return 0.0
            joint_geo /= n_valid   # mean geodesic per joint over sampled frames
            mean_geo   = float(joint_geo.mean())
            sim        = float(np.exp(-mean_geo / _POSE_SIGMA))
            # top-5 worst joints for diagnostic logging
            top5_idx  = np.argsort(joint_geo)[::-1][:5]
            top5_str  = "  ".join(f"j{j}={joint_geo[j]:.3f}" for j in top5_idx)
            logging.info(
                f"  [v6-POSE] {ni[0]}:P{ni[1]} ↔ {nj[0]}:P{nj[1]}"
                f"  mean_geo={mean_geo:.3f}rad  sim={sim:.3f}"
                f"  frames={n_valid}  top5: {top5_str}")
            return sim

        def _dir_sim(ni: tuple, nj: tuple,
                     anchor_ni: tuple, anchor_nj: tuple,
                     R: np.ndarray) -> float:
            """Cosine sim of R-rotated relative pelvis direction at shared good frames.

            Direction = (pelvis of candidate) - (pelvis of anchor) in each camera.
            R rotates cam_A frame into cam_B frame (fitted from anchor pair kp3d).
            Anchor pair returns 0.0 (zero vector → skip).
            """
            dNi  = kp3d_dict.get(ni);        dNj  = kp3d_dict.get(nj)
            dAni = kp3d_dict.get(anchor_ni); dAnj = kp3d_dict.get(anchor_nj)
            if any(d is None for d in (dNi, dNj, dAni, dAnj)):
                return 0.0
            common = sorted(set(dNi) & set(dNj) & set(dAni) & set(dAnj))
            if not common:
                return 0.0
            if len(common) > N_DIR_SAMPLES:
                idx    = np.linspace(0, len(common) - 1, N_DIR_SAMPLES, dtype=int)
                common = [common[i] for i in idx]
            sims = []
            for f in common:
                v_A = (dNi[f][0].astype(np.float64)
                       - dAni[f][0].astype(np.float64))
                v_B = (dNj[f][0].astype(np.float64)
                       - dAnj[f][0].astype(np.float64))
                nA = np.linalg.norm(v_A)
                nB = np.linalg.norm(v_B)
                if nA < 1e-4 or nB < 1e-4:
                    continue
                v_A_rot = R @ v_A
                nAr = np.linalg.norm(v_A_rot)
                if nAr < 1e-6:
                    continue
                sims.append((float(np.dot(v_A_rot, v_B) / (nAr * nB)) + 1) / 2)
            return float(np.mean(sims)) if sims else 0.0

        def _sim(ni: tuple, nj: tuple,
                 anchor_ni: tuple | None = None,
                 anchor_nj: tuple | None = None,
                 pair_R: np.ndarray | None = None) -> float:
            num = den = 0.0
            si, sj = shape_feat.get(ni), shape_feat.get(nj)
            if si is not None and sj is not None and si.shape == sj.shape:
                num += w_b * ((float(np.dot(si, sj).clip(-1, 1)) + 1) / 2)
                den += w_b
            ai, aj = app_feat.get(ni), app_feat.get(nj)
            if ai is not None and aj is not None and ai.shape == aj.shape:
                num += w_a * ((float(np.dot(ai, aj).clip(-1, 1)) + 1) / 2)
                den += w_a
            ks = _pose_sim(ni, nj)
            if ks > 0:
                num += w_k * ks
                den += w_k
            if anchor_ni is not None and anchor_nj is not None and pair_R is not None:
                ds = _dir_sim(ni, nj, anchor_ni, anchor_nj, pair_R)
                if ds > 0:
                    num += w_d * ds
                    den += w_d
            return num / den if den > 0 else 0.0

        # ── Phase 1: per-pair Hungarian + Union-Find ───────────────────────────
        from scipy.optimize import linear_sum_assignment

        edges: list[tuple] = []
        for cam_A, cam_B in combinations(active_vids, 2):
            pA_list = person_pids.get(cam_A, [])
            pB_list = person_pids.get(cam_B, [])
            if not pA_list or not pB_list:
                continue
            seed_pair  = cam_pair_seed.get((cam_A, cam_B))
            anchor_ni  = seed_pair[0] if seed_pair else None
            anchor_nj  = seed_pair[1] if seed_pair else None
            pair_R     = cam_pair_R.get((cam_A, cam_B))

            mat = np.zeros((len(pA_list), len(pB_list)), dtype=np.float64)
            debug_pair = dry_run and len(pA_list) > 1 and len(pB_list) > 1
            for i, pA in enumerate(pA_list):
                for j, pB in enumerate(pB_list):
                    ni_, nj_ = (cam_A, pA), (cam_B, pB)
                    mat[i, j] = _sim(ni_, nj_, anchor_ni, anchor_nj, pair_R)
                    if debug_pair:
                        si = shape_feat.get(ni_); sj = shape_feat.get(nj_)
                        ai = app_feat.get(ni_);   aj = app_feat.get(nj_)
                        b_s = ((float(np.dot(si, sj).clip(-1,1))+1)/2
                               if si is not None and sj is not None else float('nan'))
                        a_s = ((float(np.dot(ai, aj).clip(-1,1))+1)/2
                               if ai is not None and aj is not None else float('nan'))
                        p_s = _pose_sim(ni_, nj_)
                        d_s = (_dir_sim(ni_, nj_, anchor_ni, anchor_nj, pair_R)
                               if anchor_ni else float('nan'))
                        logging.info(
                            f"  [v6-DBG] {cam_A}:P{pA} ↔ {cam_B}:P{pB}"
                            f"  total={mat[i,j]:.3f}"
                            f"  betas={b_s:.3f}  app={a_s:.3f}"
                            f"  pose={p_s:.3f}  dir={d_s:.3f}")

            _HUN_THR = 0.70
            row_idx, col_idx = linear_sum_assignment(-mat)
            for r, c in zip(row_idx, col_idx):
                ni = (cam_A, pA_list[r])
                nj = (cam_B, pB_list[c])
                s  = float(mat[r, c])
                if s < _HUN_THR:
                    logging.info(
                        f"  [v6-HUN] DROP {cam_A}:P{pA_list[r]} ↔ {cam_B}:P{pB_list[c]}  sim={s:.3f} < {_HUN_THR}")
                    continue
                logging.info(
                    f"  [v6-HUN] {cam_A}:P{pA_list[r]} ↔ {cam_B}:P{pB_list[c]}  sim={s:.3f}")
                edges.append((s, ni, nj))

        parent = {n: n for n in nodes}

        def _find(x: tuple) -> tuple:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _cams(root: tuple) -> set:
            return {n[0] for n in nodes if _find(n) == root}

        n_cams = len(active_vids)
        n_p1_merged = 0
        edges.sort(key=lambda e: -e[0])
        for s, ni, nj in edges:
            ri, rj = _find(ni), _find(nj)
            if ri == rj:
                continue
            ci, cj = _cams(ri), _cams(rj)
            if ci & cj or len(ci | cj) > n_cams:
                logging.info(
                    f"  [v6-P1] REJECT {ni[0]}:P{ni[1]} ↔ {nj[0]}:P{nj[1]}  sim={s:.3f}"
                    f"  conflict={ci & cj or 'overflow'}")
                continue
            parent[rj] = ri
            n_p1_merged += 1
            logging.info(
                f"  [v6-P1] merge {ni[0]}:P{ni[1]} ↔ {nj[0]}:P{nj[1]}  sim={s:.3f}")

        # Build mutable cluster sets
        root_to_idx: dict[tuple, int] = {}
        clusters: list[set] = []
        for n in nodes:
            r = _find(n)
            if r not in root_to_idx:
                root_to_idx[r] = len(clusters)
                clusters.append(set())
            clusters[root_to_idx[r]].add(n)

        # ── Phase 2: per-sample directional voting ────────────────────────────
        if kp3d_rel and cam_t_dict:
            clusters = self._geometry_phase(
                clusters, kp3d_rel, cam_t_dict, _sim, active_vids)

        # ── Global ID assignment ───────────────────────────────────────────────
        global_remap = {v: {} for v in active_vids}
        gid     = 1
        pending = []
        for members in clusters:
            cam_set = {v for v, _ in members}
            if len(cam_set) >= 2:
                for v, p in members:
                    if p != gid:
                        global_remap[v][p] = gid
                gid += 1
            else:
                pending.append((min(p for _, p in members), members))
        for _, members in sorted(pending):
            for v, p in members:
                if p != gid:
                    global_remap[v][p] = gid
            gid += 1

        n_comps  = len(clusters)
        n_remaps = sum(len(m) for m in global_remap.values())
        logging.info(
            f"Scene {scene_id}: v6 → {n_comps} global person(s), "
            f"{n_p1_merged} P1 merge(s), {n_remaps} remap(s) "
            f"across {len(active_vids)} view(s)")

        if dry_run:
            print(f"\n[DRY RUN v6] Scene {scene_id}: {n_comps} global persons")
            for cl in sorted(clusters, key=lambda c: -len({v for v, _ in c})):
                cams = sorted({v for v, _ in cl})
                tag  = f"{len(cams)}-cam" if len(cams) >= 2 else "singleton"
                mstr = " ".join(
                    f"{v.split('_')[-1]}:P{p}" for v, p in sorted(cl))
                print(f"  [{tag}] {mstr}")
            return

        # ── Apply remaps ───────────────────────────────────────────────────────
        from preprocessing.cross_view_reid_v2 import CrossVideoReidentifierV2
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir  = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"
            tmp: list[tuple[Path, Path]] = []
            for old, new in remap.items():
                src_path = body_dir / f"person_{old}.npz"
                if src_path.exists():
                    tmp_path = body_dir / f"person_{old}.v6tmp.npz"
                    src_path.rename(tmp_path)
                    tmp.append((tmp_path, body_dir / f"person_{new}.npz"))
            for tmp_path, dst_path in tmp:
                if dst_path.exists():
                    tmp_path.unlink()
                else:
                    tmp_path.rename(dst_path)
            CrossVideoReidentifierV2.apply_reid_remap(vid_dir, remap)

        (scene_dir / "cross_view_reid.json").write_text(
            json.dumps({"status": "done", "n_global": n_comps}, indent=2))

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _geometry_phase(
        self,
        clusters: list[set],
        kp3d_rel: dict[tuple, np.ndarray],
        cam_t_dict: dict[tuple, dict[int, np.ndarray]],
        sim_fn: Callable,
        active_vids: list[str],
    ) -> list[set]:
        """
        Directional consistency with temporally-aligned per-frame voting.

        cam_t_dict[node] = {abs_frame_idx: pos(3,)} built from json frame indices
        and pred_cam_t trajectory. Direction vectors are computed only at frames
        where ALL four nodes in a comparison are simultaneously detected, so
        partial/gappy tracks don't produce spurious direction samples.

        Structure (same eviction/reassignment as before):
          For each camera pair (A, B) with ≥2 Phase-1 matched pairs:
            1. Fit R from highest-sim pair's kp3d.
            2. For every ordered pair (m, n) of matched pairs:
                 frames = detected(A_m) ∩ detected(A_n) ∩ detected(B_m) ∩ detected(B_n)
                 Sample N_SAMPLES from those frames.
                 v_A[f] = pos_A_n[f] - pos_A_m[f]
                 v_B[f] = pos_B_n[f] - pos_B_m[f]
                 majority vote: mismatch if >50% samples have sector_dist > 1.
            3. Evict most-inconsistent node (cascade).
            4. Geo-reassign evicted nodes; never reassign to origin cluster.
        """
        N_SAMPLES = 20

        # ── Build corr per camera pair from Phase-1 clusters ─────────────────
        corr: dict[tuple, list] = defaultdict(list)
        for cl_idx, cl in enumerate(clusters):
            cams = sorted({v for v, _ in cl})
            for cam_A, cam_B in combinations(cams, 2):
                nA_list = [(v, p) for v, p in cl if v == cam_A]
                nB_list = [(v, p) for v, p in cl if v == cam_B]
                if len(nA_list) != 1 or len(nB_list) != 1:
                    continue
                nA, nB = nA_list[0], nB_list[0]
                corr[(cam_A, cam_B)].append(
                    (sim_fn(nA, nB), nA[1], nB[1], cl_idx))
        for key in corr:
            corr[key].sort(key=lambda x: -x[0])

        # ── Temporally-aligned sample helper ─────────────────────────────────
        def _shared_samples(*nodes: tuple) -> list[int]:
            """Sorted absolute frame indices detected in ALL nodes, sampled to N_SAMPLES."""
            dicts = [cam_t_dict.get(n) for n in nodes]
            if any(d is None or len(d) == 0 for d in dicts):
                return []
            common = set(dicts[0].keys())
            for d in dicts[1:]:
                common &= d.keys()
            if not common:
                return []
            common_sorted = sorted(common)
            if len(common_sorted) <= N_SAMPLES:
                return common_sorted
            idx = np.linspace(0, len(common_sorted) - 1, N_SAMPLES, dtype=int)
            return [common_sorted[i] for i in idx]

        # ── Per-frame direction check: majority vote → True = mismatch ────────
        def _is_mismatch(nA_m, nA_n, nB_m, nB_n, R) -> bool:
            frames = _shared_samples(nA_m, nA_n, nB_m, nB_n)
            if not frames:
                return False
            dA_m = cam_t_dict[nA_m]; dA_n = cam_t_dict[nA_n]
            dB_m = cam_t_dict[nB_m]; dB_n = cam_t_dict[nB_n]
            votes_bad = votes_ok = 0
            for f in frames:
                v_A = dA_n[f].astype(np.float64) - dA_m[f].astype(np.float64)
                v_B = dB_n[f].astype(np.float64) - dB_m[f].astype(np.float64)
                if np.linalg.norm(v_A) < 1e-6 or np.linalg.norm(v_B) < 1e-6:
                    continue
                se = _sector(R @ v_A)
                sa = _sector(v_B)
                if se is None or sa is None:
                    continue
                if _sector_dist(se, sa) > 1:
                    votes_bad += 1
                else:
                    votes_ok += 1
            return votes_bad > votes_ok

        # ── Fit R, collect inconsistencies ────────────────────────────────────
        inconsistencies: list[tuple[tuple, tuple]] = []
        node_min_sim:    dict[tuple, float]        = {}
        fitted_R:        dict[tuple, np.ndarray]   = {}

        for (cam_A, cam_B), cands in corr.items():
            if len(cands) < 2:
                continue
            seed_s, seed_pA, seed_pB, _ = cands[0]
            kA = kp3d_rel.get((cam_A, seed_pA))
            kB = kp3d_rel.get((cam_B, seed_pB))
            if kA is None or kB is None:
                continue
            T = min(len(kA), len(kB))
            if T < 2:
                continue
            R = _fit_rotation(
                kA[:T].reshape(-1, 3).astype(np.float64),
                kB[:T].reshape(-1, 3).astype(np.float64))
            fitted_R[(cam_A, cam_B)] = R

            pairs_data = []
            for s, pA, pB, ci in cands:
                nA, nB = (cam_A, pA), (cam_B, pB)
                if cam_t_dict.get(nA) is None or cam_t_dict.get(nB) is None:
                    continue
                pairs_data.append((pA, pB, ci, s))
                for node in (nA, nB):
                    node_min_sim[node] = min(node_min_sim.get(node, 1.0), s)

            if len(pairs_data) < 2:
                continue

            logging.info(f"  [v6-GEO] checking {cam_A}\u2194{cam_B}: "
                         f"{len(pairs_data)} pairs, seed sim={cands[0][0]:.3f}")

            for i, (pA_m, pB_m, ci_m, s_m) in enumerate(pairs_data):
                for j, (pA_n, pB_n, ci_n, s_n) in enumerate(pairs_data):
                    if i == j:
                        continue
                    nA_m, nA_n = (cam_A, pA_m), (cam_A, pA_n)
                    nB_m, nB_n = (cam_B, pB_m), (cam_B, pB_n)
                    if _is_mismatch(nA_m, nA_n, nB_m, nB_n, R):
                        inconsistencies.append(((cam_A, pA_m), (cam_B, pB_m)))
                        logging.info(
                            f"  [v6-P2] dir-mismatch {cam_A}:P{pA_m}\u2194{cam_B}:P{pB_m} "
                            f"(sim={s_m:.3f}) vs {cam_A}:P{pA_n}\u2194{cam_B}:P{pB_n} "
                            f"(sim={s_n:.3f})")

        if not inconsistencies:
            return clusters

        # ── Eviction ─────────────────────────────────────────────────────────
        cl_sets:      list[set]        = [set(cl) for cl in clusters]
        node_to_cl:   dict[tuple, int] = {n: i for i, cl in enumerate(cl_sets) for n in cl}
        to_evict:     set[tuple]       = set()
        evicted_from: dict[tuple, int] = {}

        while True:
            fail_count: dict[tuple, int] = defaultdict(int)
            for node_mA, node_mB in inconsistencies:
                if node_mA in to_evict or node_mB in to_evict:
                    continue
                fail_count[node_mA] += 1
                fail_count[node_mB] += 1
            if not fail_count:
                break
            worst = max(fail_count,
                        key=lambda n: (fail_count[n], -node_min_sim.get(n, 1.0)))
            if fail_count[worst] == 0:
                break
            to_evict.add(worst)

        for node in to_evict:
            ci = node_to_cl.get(node)
            if ci is not None:
                evicted_from[node] = ci
                cl_sets[ci].discard(node)
            logging.info(f"  [v6-P2] evict {node[0]}:P{node[1]}")

        if not to_evict:
            return [cl for cl in cl_sets if cl]

        # ── Geometric reassignment ────────────────────────────────────────────
        def _geo_score(ev_node: tuple, cl_idx: int) -> int:
            cam_E = ev_node[0]
            dE = cam_t_dict.get(ev_node)
            if dE is None:
                return 0
            cl = cl_sets[cl_idx]
            score = 0
            for (cam_A, cam_B), R in fitted_R.items():
                if cam_A == cam_E:
                    cam_O, R_eff = cam_B, R.T
                elif cam_B == cam_E:
                    cam_O, R_eff = cam_A, R
                else:
                    continue
                j_nodes = [(cam_O, p) for v, p in cl
                           if v == cam_O and (cam_O, p) not in to_evict]
                for j_node in j_nodes:
                    dJ = cam_t_dict.get(j_node)
                    if dJ is None:
                        continue
                    for other_idx, other_cl in enumerate(cl_sets):
                        if other_idx == cl_idx or not other_cl:
                            continue
                        k_nodes = [(cam_O, p) for v, p in other_cl
                                   if v == cam_O and (cam_O, p) not in to_evict]
                        for k_node in k_nodes:
                            dK = cam_t_dict.get(k_node)
                            if dK is None:
                                continue
                            refs = [(cam_E, p) for v, p in other_cl if v == cam_E]
                            if not refs:
                                refs = [n for n in to_evict
                                        if n[0] == cam_E and n != ev_node
                                        and evicted_from.get(n) == other_idx]
                            for ref in refs:
                                dR = cam_t_dict.get(ref)
                                if dR is None:
                                    continue
                                frames = _shared_samples(ev_node, ref, j_node, k_node)
                                for f in frames:
                                    v_O = (dK[f].astype(np.float64)
                                           - dJ[f].astype(np.float64))
                                    if np.linalg.norm(v_O) < 1e-6:
                                        continue
                                    v_E_exp = R_eff @ v_O
                                    v_E_act = (dR[f].astype(np.float64)
                                               - dE[f].astype(np.float64))
                                    if np.linalg.norm(v_E_act) < 1e-6:
                                        continue
                                    se = _sector(v_E_exp)
                                    sa = _sector(v_E_act)
                                    if se is None or sa is None:
                                        continue
                                    score += 1 if _sector_dist(se, sa) <= 1 else -1
            return score

        def _best_geo_score(node: tuple) -> int:
            cam_e = node[0]
            fE = cam_t_dict.get(node, {})
            origin_ci = evicted_from.get(node, -1)
            best = 0
            for ci, cl in enumerate(cl_sets):
                if not cl or ci == origin_ci:
                    continue
                conflict = any(
                    v == cam_e and (v, p) not in to_evict and
                    bool(set(fE) & set(cam_t_dict.get((v, p), {})))
                    for v, p in cl
                )
                if conflict:
                    continue
                if len({v for v, _ in cl}) >= len(active_vids):
                    continue
                if len({v for v, _ in cl}) < 2:  # singleton cluster: no geo structure
                    continue
                best = max(best, _geo_score(node, ci))
            return best

        for node in sorted(to_evict, key=_best_geo_score, reverse=True):
            cam_e = node[0]
            fE = cam_t_dict.get(node, {})
            origin_ci = evicted_from.get(node, -1)
            best_score  = 0
            best_cl_idx = -1
            for ci, cl in enumerate(cl_sets):
                if not cl or ci == origin_ci:
                    continue
                conflict = any(
                    v == cam_e and (v, p) not in to_evict and
                    bool(set(fE) & set(cam_t_dict.get((v, p), {})))
                    for v, p in cl
                )
                if conflict:
                    continue
                if len({v for v, _ in cl}) >= len(active_vids):
                    continue
                if len({v for v, _ in cl}) < 2:  # singleton cluster: no geo structure
                    continue
                s = _geo_score(node, ci)
                if s > best_score:
                    best_score  = s
                    best_cl_idx = ci
            if best_cl_idx >= 0:
                cl_sets[best_cl_idx].add(node)
                logging.info(f"  [v6-P2] geo-reassign {cam_e}:P{node[1]} "
                             f"\u2192 cluster {best_cl_idx}  score={best_score}")
            else:
                cl_sets.append({node})
                logging.info(f"  [v6-P2] singleton {cam_e}:P{node[1]} "
                             f"(no positive-score cluster)")

        return [cl for cl in cl_sets if cl]
