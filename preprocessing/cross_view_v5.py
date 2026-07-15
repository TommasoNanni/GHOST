"""
Cross-view person ReID v5.

Delta estimation: identical to v2/v3/v4 (weighted xcorr + ALS per candidate + consensus clocks).
Assignment at fixed delta: two phases.

  Phase 1 — Dynamic people (unchanged from v4 Tier-1).
            Split persons by motion (pose_std > STD_THRESHOLD); if nobody exceeds it,
            fall back to top-TIER1_FALLBACK_N by std. Match high-motion persons via
            canonical joint similarity (camera-invariant, no geometry). Accept ≥ CANON_THR.
            Matching is pairwise (per camera pair) and unions accepted pairs.

  Phase 2 — Static people (new). After ALL pairwise Phase-1 matching is done, every person
            still in a singleton union-find component is clustered *globally* across all
            cameras at once via single-linkage on a betas+appearance feature:

              edge (i, j) is accepted only if
                  sim(i, j) ≥ STATIC_SIM_FLOOR                  (absolute backstop)
                  ∧ i and j are mutual best matches             (relative evidence)
                  ∧ sim(i, j) beats each runner-up by RATIO_THR (discriminability / SDS)
              greedily, strongest edge first, subject to
                  source constraint: one person per camera per cluster
                  size   constraint: cluster size ≤ N cameras

            betas are view-invariant and weighted above appearance. There is NO geometric
            fallback — when the three gates leave a person ambiguous, it stays isolated.

Phase-2 clusters never share a camera (source constraint), so they introduce no new
same-camera conflicts; conflict resolution and global-ID assignment are unchanged from v4.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as _Rot

from data.video_dataset import Scene

# ── Constants (delta estimation — identical to v2/v3/v4) ──────────────────────
_MIN_OVERLAP              = 30
_MIN_DELTA_OVERLAP_FRAMES = 100
_OVERLAP_PENALTY_K        = 30.0
_OVERLAP_REF_FRAMES       = 500
_DELTA_PRIOR_WEIGHT       = 0.0
_DELTA_PRIOR_SCALE        = 30.0

# ── Constants (Phase 1 — dynamic, identical to v4 Tier-1) ─────────────────────
_STD_THRESHOLD    = 0.30   # m — max per-joint std over time; above = high-motion
_TIER1_FALLBACK_N = 2      # take top-N per side if nobody exceeds _STD_THRESHOLD
_CANON_THR        = 0.75   # acceptance: canonical sim must exceed this

# ── Constants (Phase 2 — static graph clustering) ─────────────────────────────
_STATIC_SIM_FLOOR = 0.60   # absolute similarity backstop for a static merge
_STATIC_RATIO_THR = 1.05   # best match must beat runner-up by this ratio (SDS gate)
_STATIC_BETAS_W   = 0.60   # betas (view-invariant) weighted above appearance
_STATIC_APP_W     = 0.40


class CrossViewReidentifierV5:
    """Cross-view ReID: dynamic pairwise matching (Phase 1) + global static clustering (Phase 2).

    Phase 1 is v4's Tier-1 canonical-pose matcher verbatim. Phase 2 replaces v4's
    geometric Tier-2 with a PME-style single-linkage clustering over the remaining
    singletons, using view-invariant betas + appearance and conservative-by-default
    merge gates (mutual best + discriminability ratio). No geometric fallback.
    """

    _THRESHOLD         = 0.50
    _APPEARANCE_WEIGHT = 0.30
    _SHAPE_WEIGHT      = 0.30
    _POSE_WEIGHT       = 0.40
    _MATCH_RMSE_THR    = 0.60

    _DROID_WEIGHTS: str | None = None
    _DROID_ROOT: str = "/users/tnanni/ghost/DROID-SLAM"
    _SLAM_CAMS: set[str] | None = None
    _DEFAULT_INTRINSICS: tuple[float, float, float, float] = (1000.0, 1000.0, 512.0, 384.0)

    def __init__(
        self,
        droid_weights: str | None = None,
        droid_root: str | None = None,
        slam_cams: set[str] | list[str] | None = None,
        default_intrinsics: tuple[float, float, float, float] | None = None,
        threshold: float = _THRESHOLD,
        appearance_weight: float = _APPEARANCE_WEIGHT,
        shape_weight: float = _SHAPE_WEIGHT,
        pose_weight: float = _POSE_WEIGHT,
        match_rmse_thr: float = _MATCH_RMSE_THR,
        static_sim_floor: float = _STATIC_SIM_FLOOR,
        static_ratio_thr: float = _STATIC_RATIO_THR,
        static_betas_weight: float = _STATIC_BETAS_W,
        static_app_weight: float = _STATIC_APP_W,
        reid_ckpt: str | None = None,
    ) -> None:
        self.droid_weights      = droid_weights if droid_weights is not None else self._DROID_WEIGHTS
        self.droid_root         = droid_root    if droid_root    is not None else self._DROID_ROOT
        self.slam_cams: set[str] | None = set(slam_cams) if slam_cams is not None else self._SLAM_CAMS
        self.default_intrinsics = default_intrinsics if default_intrinsics is not None else self._DEFAULT_INTRINSICS
        self.threshold         = threshold
        self.appearance_weight = appearance_weight
        self.shape_weight      = shape_weight
        self.pose_weight       = pose_weight
        self.match_rmse_thr    = match_rmse_thr
        self.static_sim_floor    = static_sim_floor
        self.static_ratio_thr    = static_ratio_thr
        self.static_betas_weight = static_betas_weight
        self.static_app_weight   = static_app_weight
        self.reid_ckpt           = reid_ckpt
        self._reid = None        # TransReIDExtractor, lazy-built on first use

    def _get_reid(self):
        """Lazy TransReID extractor (cross-view appearance). None if no ckpt configured."""
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
        intrinsics_map: dict[str, np.ndarray] | None = None,
        dry_run: bool = False,
    ) -> None:
        scene_id  = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent
        if not dry_run and (scene_dir / "cross_view_reid.json").exists():
            logging.info(f"  Scene {scene_id}: cross-view ReID v5 already done, skipping")
            return
        if frames_dirs is None:
            frames_dirs = {}
            for video in scene.videos:
                if hasattr(video, "frames_home") and video.frames_home is not None:
                    frames_dirs[video.video_id] = video.frames_home
        self._core(
            video_dirs=video_dirs,
            scene_id=scene_id,
            scene_dir=scene_dir,
            frames_dirs=frames_dirs,
            intrinsics_map=intrinsics_map or {},
            dry_run=dry_run,
        )

    # ── Core ──────────────────────────────────────────────────────────────────

    def _core(
        self,
        video_dirs: dict[str, Path],
        scene_id: str,
        scene_dir: Path,
        frames_dirs: dict[str, Path],
        intrinsics_map: dict[str, np.ndarray],
        dry_run: bool,
    ) -> None:
        video_ids = list(video_dirs.keys())

        # ── Data loading (identical to v2/v3/v4) ──────────────────────────────
        person_descs  : dict[str, dict[int, tuple]] = {}
        person_pids   : dict[str, list[int]]        = {}
        person_kpts3d : dict[str, dict[int, tuple]] = {}
        person_cam_t  : dict[str, dict[int, tuple]] = {}  # vid → pid → (frames, cam_t)

        for vid_id, vid_dir in video_dirs.items():
            body_dir     = Path(vid_dir) / "body_data"
            gallery_path = body_dir / "appearance_gallery.npz"
            summary_path = body_dir / "body_params_summary.json"
            if not summary_path.exists():
                logging.warning(f"{scene_id}/{vid_id}: no body_params_summary.json, skipping")
                continue
            with open(summary_path) as _f:
                summary = json.load(_f)
            pids = [int(k) for k in summary.get("persons", {}).keys()]
            if not pids:
                continue

            app_gallery: dict[int, tuple] = {}
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                for k in gdata.files:
                    if k.endswith("_conf"):
                        continue
                    conf_key = f"{k}_conf"
                    feats = gdata[k]
                    confs = gdata[conf_key] if conf_key in gdata.files else np.ones(len(feats), dtype=np.float32)
                    app_gallery[int(k)] = (feats, confs)

            descs: dict[int, tuple] = {}
            for pid in pids:
                npz_path = body_dir / f"person_{pid}.npz"
                if not npz_path.exists():
                    continue
                shape_feat: np.ndarray | None = None
                pose_feat:  np.ndarray | None = None
                with np.load(str(npz_path)) as pdata:
                    if "smplx_betas" in pdata:
                        bv = pdata["smplx_betas"]
                        if len(bv) > 0:
                            conf_kps = pdata.get("pred_joint_confidence")
                            if conf_kps is not None and len(conf_kps) == len(bv):
                                fc = np.mean(conf_kps, axis=-1).astype(np.float32)
                                tc = fc.sum()
                                sm = (fc[:, None] * bv).sum(0) / tc if tc > 0 else np.median(bv, 0).astype(np.float32)
                            else:
                                sm = np.median(bv, axis=0).astype(np.float32)
                            nm = np.linalg.norm(sm)
                            shape_feat = sm / nm if nm > 0 else sm
                    if "pred_keypoints_3d" in pdata and "frame_indices" in pdata:
                        person_kpts3d.setdefault(vid_id, {})[pid] = (
                            pdata["frame_indices"].copy(),
                            pdata["pred_keypoints_3d"].copy(),
                        )
                    for _cam_t_key in ("pred_cam_t", "smplx_transl"):
                        if _cam_t_key in pdata and "frame_indices" in pdata:
                            person_cam_t.setdefault(vid_id, {})[pid] = (
                                pdata["frame_indices"].copy(),
                                pdata[_cam_t_key].copy(),
                            )
                            break
                    if "pred_keypoints_3d" in pdata and "smplx_global_orient" in pdata:
                        kps     = pdata["pred_keypoints_3d"].astype(np.float32)
                        gorient = pdata["smplx_global_orient"].astype(np.float32)
                        N = len(kps)
                        if N > 0:
                            kps_rel = kps - kps[:, 0:1, :]
                            rm_     = _Rot.from_rotvec(gorient).inv().as_matrix()
                            kc      = np.einsum("nij,nkj->nki", rm_, kps_rel).astype(np.float32)
                            pv      = kc.reshape(N, -1)
                            npn     = np.linalg.norm(pv, axis=1, keepdims=True)
                            pv      = np.where(npn > 1e-6, pv / npn, pv)
                            pose_feat = pv
                app_feat = app_gallery.get(pid)
                if app_feat is None and shape_feat is None and pose_feat is None:
                    continue
                descs[pid] = (app_feat, shape_feat, pose_feat)
            if descs:
                person_descs[vid_id] = descs
                person_pids[vid_id]  = sorted(descs.keys())

        active_vids = [v for v in video_ids if v in person_descs]
        if len(active_vids) < 2:
            logging.info(f"Scene {scene_id}: fewer than 2 videos with descriptors, skipping")
            return

        # ── SLAM: correct moving cameras to world frame ────────────────────────
        import sys as _sys

        def _quat_to_rot(q: np.ndarray) -> np.ndarray:
            qx, qy, qz, qw = q
            return np.array([
                [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
            ], dtype=np.float64)

        def _slam_image_stream(frames_dir: Path, fx, fy, cx, cy):
            import cv2, torch as _torch
            img_files = sorted(p for p in frames_dir.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            for t, imfile in enumerate(img_files):
                image = cv2.imread(str(imfile))
                h0, w0 = image.shape[:2]
                scale = np.sqrt((384 * 512) / (h0 * w0))
                h1, w1 = int(h0 * scale), int(w0 * scale)
                image = cv2.resize(image, (w1, h1))
                image = image[:h1 - h1 % 8, :w1 - w1 % 8]
                image = _torch.as_tensor(image).permute(2, 0, 1)
                intr  = _torch.tensor([fx * w1/w0, fy * h1/h0, cx * w1/w0, cy * h1/h0])
                yield t, image[None], intr

        def _run_droid_slam(frames_dir: Path, fx, fy, cx, cy, weights: str, droid_root: str):
            import argparse, cv2, torch as _torch
            if droid_root not in _sys.path:
                _sys.path.insert(0, str(droid_root))
                _sys.path.insert(0, str(Path(droid_root) / "droid_slam"))
            from droid import Droid
            try:
                _torch.multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
            img_files = sorted(p for p in frames_dir.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            if not img_files:
                logging.warning(f"DROID-SLAM: no images found in {frames_dir}, skipping")
                return None
            first  = cv2.imread(str(img_files[0]))
            H_orig, W_orig = first.shape[:2]
            stream = list(_slam_image_stream(frames_dir, fx, fy, cx, cy))
            H_slam = stream[0][1].shape[2]
            W_slam = stream[0][1].shape[3]
            args = argparse.Namespace(
                weights=weights, image_size=[H_slam, W_slam], buffer=512,
                stereo=False, disable_vis=True, beta=0.3, filter_thresh=2.4,
                warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
                frontend_window=25, frontend_radius=2, frontend_nms=1,
                backend_thresh=22.0, backend_radius=2, backend_nms=3, upsample=False,
            )
            droid = Droid(args)
            for t, image, intrinsics in stream:
                droid.track(t, image, intrinsics=intrinsics)
            try:
                poses = droid.terminate(iter(stream))
            except (ValueError, RuntimeError) as e:
                logging.warning(f"DROID-SLAM backend failed ({e}); skipping.")
                del droid
                import torch as _t; _t.cuda.empty_cache()
                return None
            n_kf   = droid.video.counter.value
            disps  = droid.video.disps[:n_kf].cpu().numpy()
            tstamp = droid.video.tstamp[:n_kf].cpu().numpy().astype(int)
            del droid
            import torch as _t; _t.cuda.empty_cache()
            return poses, disps, tstamp, (H_orig, W_orig)

        def _estimate_slam_scale(vid_id, fx, fy, cx, cy, disps, tstamp,
                                  frame_offset, orig_hw) -> float:
            H_orig, W_orig = orig_hw
            H_disp, W_disp = disps.shape[1], disps.shape[2]
            sx, sy = W_disp / W_orig, H_disp / H_orig
            lambdas = []
            for pid, (frames, cam_t) in person_cam_t.get(vid_id, {}).items():
                frame_to_idx = {int(f): i for i, f in enumerate(frames)}
                for kf_i, t in enumerate(tstamp):
                    arr_idx = frame_to_idx.get(int(t) + frame_offset)
                    if arr_idx is None:
                        continue
                    x, y, z = cam_t[arr_idx]
                    if z < 0.5:
                        continue
                    px = int(round((fx * x / z + cx) * sx))
                    py = int(round((fy * y / z + cy) * sy))
                    if not (0 <= px < W_disp and 0 <= py < H_disp):
                        continue
                    d_inv = float(disps[kf_i, py, px])
                    if d_inv < 1e-6:
                        continue
                    lambdas.append(z * d_inv)
            if not lambdas:
                logging.warning(f"{vid_id}: no valid λ estimates, defaulting to 1.0")
                return 1.0
            lam = float(np.median(lambdas))
            logging.info(f"{vid_id}: SLAM scale λ={lam:.4f} ({len(lambdas)} samples)")
            return lam

        _slam_corrected: set[str] = set()
        if self.droid_weights and frames_dirs:
            _slam_targets = (
                [v for v in active_vids if v in self.slam_cams]
                if self.slam_cams is not None
                else active_vids
            )
            logging.info(
                f"Scene {scene_id}: running DROID-SLAM on "
                + (str(self.slam_cams) if self.slam_cams is not None else "all cameras")
            )
            fx0, fy0, cx0, cy0 = self.default_intrinsics
            for vid_id in _slam_targets:
                fdir = frames_dirs.get(vid_id)
                if fdir is None or not Path(fdir).is_dir():
                    logging.warning(f"  {vid_id}: no frames dir for SLAM, skipping")
                    continue
                K  = intrinsics_map.get(vid_id)
                fx = float(K[0, 0]) if K is not None else fx0
                fy = float(K[1, 1]) if K is not None else fy0
                cx = float(K[0, 2]) if K is not None else cx0
                cy = float(K[1, 2]) if K is not None else cy0
                result = _run_droid_slam(Path(fdir), fx, fy, cx, cy,
                                         self.droid_weights, self.droid_root)
                if result is None:
                    logging.warning(f"  {vid_id}: SLAM returned no poses, skipping")
                    continue
                poses, disps, tstamp, orig_hw = result
                traj_std = float(np.std(poses[:, :3]))
                if traj_std < 0.05:
                    logging.warning(
                        f"  {vid_id}: negligible SLAM motion (std={traj_std:.4f} m), treating as static"
                    )
                    continue
                kpts_data = person_kpts3d.get(vid_id, {})
                if not kpts_data:
                    continue
                frame_offset = int(min(frames.min() for frames, _ in kpts_data.values()))
                lam = _estimate_slam_scale(vid_id, fx, fy, cx, cy, disps, tstamp,
                                           frame_offset, orig_hw)
                cam_t_data = person_cam_t.get(vid_id, {})
                for pid, (frames, kpts) in list(kpts_data.items()):
                    cam_t = cam_t_data.get(pid, (None, None))[1]
                    corrected = np.full_like(kpts, np.nan)
                    for i, frame_idx in enumerate(frames):
                        t_idx = int(frame_idx) - frame_offset
                        if t_idx < 0 or t_idx >= len(poses):
                            continue
                        pose = poses[t_idx]
                        if not np.isfinite(pose).all():
                            continue
                        R = _quat_to_rot(pose[3:])
                        root_cam = cam_t[i] if cam_t is not None else np.zeros(3)
                        corrected[i] = (R @ kpts[i].T).T + R @ root_cam + lam * pose[:3]
                    person_kpts3d[vid_id][pid] = (frames, corrected)
                logging.info(f"  {vid_id}: world-frame correction applied (λ={lam:.4f})")
                _slam_corrected.add(vid_id)
        else:
            logging.info(
                f"Scene {scene_id}: no droid_weights — assuming static cameras"
                if not self.droid_weights
                else f"Scene {scene_id}: droid_weights set but no frames_dirs — SLAM skipped"
            )

        # ── Absolute-position correction for non-SLAM cameras ─────────────────
        # pred_keypoints_3d is root-relative; absolute pos = kpts + pred_cam_t
        for vid_id in active_vids:
            if vid_id in _slam_corrected:
                continue
            kpts_data  = person_kpts3d.get(vid_id, {})
            cam_t_data = person_cam_t.get(vid_id, {})
            for pid, (frames, kpts) in list(kpts_data.items()):
                cam_t_entry = cam_t_data.get(pid)
                if cam_t_entry is None:
                    continue
                _, cam_t = cam_t_entry
                person_kpts3d[vid_id][pid] = (frames, kpts + cam_t[:, None, :])

        # ── Union-Find with edge tracking ──────────────────────────────────────
        parent:  dict[tuple, tuple] = {}
        rank_uf: dict[tuple, int]   = {}
        edges:   list[tuple]        = []

        def _find(x: tuple) -> tuple:
            if parent.setdefault(x, x) != x:
                parent[x] = _find(parent[x])
            return parent[x]

        def _union(x: tuple, y: tuple) -> None:
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return
            if rank_uf.get(rx, 0) < rank_uf.get(ry, 0):
                rx, ry = ry, rx
            parent[ry] = rx
            if rank_uf.get(rx, 0) == rank_uf.get(ry, 0):
                rank_uf[rx] = rank_uf.get(rx, 0) + 1

        for _v in active_vids:
            for _p in person_pids[_v]:
                _find((_v, _p))

        # ── Single-image mode detection ────────────────────────────────────────
        max_track_len = max(
            (len(kd[0]) for vid in active_vids for kd in person_kpts3d.get(vid, {}).values()),
            default=0,
        )
        _single_image_mode = max_track_len == 1

        # ── Appearance helpers (identical to v2/v3/v4) ─────────────────────────

        def _chamfer_sim(fa, ca, fb, cb) -> float:
            S = fa @ fb.T
            return 0.5 * (float(S.max(axis=1).mean()) + float(S.max(axis=0).mean()))

        def _xcorr_sim(fa: np.ndarray, fb: np.ndarray) -> tuple[float, int]:
            N, M = len(fa), len(fb)
            fa = fa - fa.mean(axis=0); fb = fb - fb.mean(axis=0)
            na = np.linalg.norm(fa, axis=1, keepdims=True)
            nb = np.linalg.norm(fb, axis=1, keepdims=True)
            fa = np.where(na > 1e-6, fa / na, fa)
            fb = np.where(nb > 1e-6, fb / nb, fb)
            S = fa @ fb.T
            best_score, best_off = -1.0, 0
            for tau in range(-(M - 1), N):
                diag = np.diagonal(S, offset=-tau)
                if len(diag) < _MIN_OVERLAP:
                    continue
                score = float(diag.mean())
                if score > best_score:
                    best_score, best_off = score, tau
            return float(max(0.0, best_score)), best_off

        def _weighted_sim_mat(pids_a, pids_b, descs_a, descs_b) -> np.ndarray:
            Na, Nb = len(pids_a), len(pids_b)
            sim_mat = np.zeros((Na, Nb), dtype=np.float32)
            wgt_mat = np.zeros((Na, Nb), dtype=np.float32)
            w_app, w_shape, w_pose = self.appearance_weight, self.shape_weight, self.pose_weight
            for i, pa in enumerate(pids_a):
                app_a = descs_a[pa][0]
                if app_a is None:
                    continue
                for j, pb in enumerate(pids_b):
                    app_b = descs_b[pb][0]
                    if app_b is None:
                        continue
                    s = _chamfer_sim(app_a[0], app_a[1], app_b[0], app_b[1])
                    sim_mat[i, j] += w_app * s; wgt_mat[i, j] += w_app
            sa_list = [descs_a[p][1] for p in pids_a]
            sb_list = [descs_b[p][1] for p in pids_b]
            mka = np.array([f is not None for f in sa_list], dtype=np.float32)
            mkb = np.array([f is not None for f in sb_list], dtype=np.float32)
            if mka.any() and mkb.any():
                dim  = next(f for f in sa_list if f is not None).shape[0]
                zero = np.zeros(dim, dtype=np.float32)
                mat_a = np.stack([f if f is not None else zero for f in sa_list])
                mat_b = np.stack([f if f is not None else zero for f in sb_list])
                ssim  = mat_a @ mat_b.T
                sw    = np.outer(mka, mkb) * w_shape
                sim_mat += sw * ssim; wgt_mat += sw
            for i, pa in enumerate(pids_a):
                pose_a = descs_a[pa][2]
                if pose_a is None:
                    continue
                for j, pb in enumerate(pids_b):
                    pose_b = descs_b[pb][2]
                    if pose_b is None:
                        continue
                    s, _ = _xcorr_sim(pose_a, pose_b)
                    sim_mat[i, j] += w_pose * s; wgt_mat[i, j] += w_pose
            return np.where(wgt_mat > 0, sim_mat / wgt_mat, 0.0)

        # ── Geometry helpers (identical to v2/v3/v4) ───────────────────────────

        def _overlap_factor(n_frames: int) -> float:
            if n_frames <= 0:
                return float("inf")
            raw = 1.0 + _OVERLAP_PENALTY_K / np.sqrt(float(n_frames))
            ref = 1.0 + _OVERLAP_PENALTY_K / np.sqrt(float(_OVERLAP_REF_FRAMES))
            return max(raw / ref, 1.0)

        def _delta_prior_cost(delta: int) -> float:
            return _DELTA_PRIOR_WEIGHT * abs(int(delta)) / _DELTA_PRIOR_SCALE

        def _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta):
            """Returns (cam_b_kpts, cam_a_kpts) both [T*J, 3] at co-visible frames."""
            kb = person_kpts3d.get(vid_b, {}); ka = person_kpts3d.get(vid_a, {})
            if pid_b not in kb or pid_a not in ka:
                return None, None
            frames_b, kpts_b = kb[pid_b]; frames_a, kpts_a = ka[pid_a]
            fb2i = {int(f): i for i, f in enumerate(frames_b)}
            fa2i = {int(f): i for i, f in enumerate(frames_a)}
            common = sorted(f for f in fb2i if f + delta in fa2i)
            if len(common) < _MIN_OVERLAP:
                return None, None
            ib = [fb2i[f] for f in common]; ia = [fa2i[f + delta] for f in common]
            return kpts_b[ib].reshape(-1, 3), kpts_a[ia].reshape(-1, 3)

        def _track_std(vid: str, pid: int) -> float:
            kd = person_kpts3d.get(vid, {})
            if pid not in kd:
                return 0.0
            _, kpts = kd[pid]
            valid = kpts[np.isfinite(kpts).all(axis=(1, 2))]
            return float(valid.std(axis=0).max()) if len(valid) > 0 else 0.0

        def _get_top_k_peaks(scores: dict, k: int = 10) -> list:
            if not scores:
                return [0]
            deltas = sorted(scores); vals = [scores[d] for d in deltas]
            peaks = []
            for i, (d, v) in enumerate(zip(deltas, vals)):
                left  = vals[i - 1] if i > 0 else -1.0
                right = vals[i + 1] if i < len(vals) - 1 else -1.0
                if v >= left and v >= right and v > 0:
                    peaks.append((v, d))
            peaks.sort(reverse=True)
            if not peaks:
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                return [d for d, _ in ranked[:k]]
            return [d for _, d in peaks[:k]]

        def _constrained_fit(assignment, vid_b, vid_a, delta, max_iter=10):
            pairs = []
            for pb, pa in assignment:
                src, dst = _get_aligned_flat(pb, vid_b, pa, vid_a, delta)
                if src is None:
                    continue
                valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
                if valid.sum() < _MIN_OVERLAP:
                    continue
                pairs.append((pb, src[valid], dst[valid]))
            if not pairs:
                return None
            R = np.eye(3, dtype=np.float64); t = np.zeros(3, dtype=np.float64)
            lambdas = {pb: 1.0 for pb, _, _ in pairs}
            for _ in range(max_iter):
                all_s = np.concatenate([src for _, src, _ in pairs])
                all_d = np.concatenate([dst / lambdas[pb] for pb, _, dst in pairs])
                mu_s, mu_d = all_s.mean(0), all_d.mean(0)
                H = (all_s - mu_s).T @ (all_d - mu_d)
                U, _, Vt = np.linalg.svd(H)
                d_sign = np.linalg.det(Vt.T @ U.T)
                R = Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T
                t = mu_d - R @ mu_s
                for pb, src, dst in pairs:
                    rsrc = (R @ src.T).T + t
                    num = float(np.sum(rsrc * dst)); den = float(np.sum(rsrc * rsrc))
                    lambdas[pb] = max(0.1, min(10.0, num / den)) if den > 1e-10 else 1.0
            rmses = []
            for pb, src, dst in pairs:
                pred = lambdas[pb] * ((R @ src.T).T + t)
                rmses.append(float(np.sqrt(((pred - dst) ** 2).sum(1).mean())))
            return R, t, lambdas, float(np.mean(rmses))

        def _pair_rmse_with_Rt(R, t, pid_b, vid_b, pid_a, vid_a, delta):
            """Used only during delta-candidate scoring (not for assignment)."""
            src, dst = _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta)
            if src is None:
                return float("inf"), 1.0
            valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() < _MIN_OVERLAP:
                return float("inf"), 1.0
            src, dst = src[valid], dst[valid]
            rsrc = (R @ src.T).T + t
            den  = float(np.sum(rsrc * rsrc))
            lam  = max(0.1, min(10.0, float(np.sum(rsrc * dst)) / den)) if den > 1e-10 else 1.0
            pred = lam * rsrc
            r    = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
            return (r if np.isfinite(r) else float("inf")), lam

        # ── Delta estimation (identical to v2/v3/v4) ───────────────────────────

        def _pair_delta_candidates(vid_a: str, vid_b: str) -> dict:
            pids_a = person_pids[vid_a]; pids_b = person_pids[vid_b]
            sim_mat = _weighted_sim_mat(pids_a, pids_b, person_descs[vid_a], person_descs[vid_b])

            delta_scores: dict[int, float] = defaultdict(float)
            kd_a = person_kpts3d.get(vid_a, {}); kd_b = person_kpts3d.get(vid_b, {})
            for i, pa in enumerate(pids_a):
                pose_a = person_descs[vid_a][pa][2]
                if pose_a is None or pa not in kd_a:
                    continue
                for j, pb in enumerate(pids_b):
                    pose_b = person_descs[vid_b][pb][2]
                    if pose_b is None or pb not in kd_b:
                        continue
                    w = (max(_track_std(vid_a, pa), _track_std(vid_b, pb))
                         * max(float(sim_mat[i, j]), 0.05))
                    if w < 1e-6:
                        continue
                    offset = int(kd_a[pa][0][0]) - int(kd_b[pb][0][0])
                    fva = pose_a - pose_a.mean(axis=0); fvb = pose_b - pose_b.mean(axis=0)
                    na_ = np.linalg.norm(fva, axis=1, keepdims=True)
                    nb_ = np.linalg.norm(fvb, axis=1, keepdims=True)
                    fva = np.where(na_ > 1e-6, fva / na_, fva)
                    fvb = np.where(nb_ > 1e-6, fvb / nb_, fvb)
                    S = fva @ fvb.T
                    T_a_len, T_b_len = S.shape
                    for tau in range(-(T_b_len - 1), T_a_len):
                        diag = np.diagonal(S, offset=-tau)
                        if len(diag) < _MIN_OVERLAP:
                            continue
                        score = float(diag.mean())
                        if score > 0:
                            delta_scores[offset + tau] += w * score

            delta_candidates = _get_top_k_peaks(delta_scores, k=10)
            frames_a_all: set[int] = set()
            for _pa2 in kd_a:
                frames_a_all.update(int(_f) for _f in kd_a[_pa2][0])
            frames_b_all: set[int] = set()
            for _pb2 in kd_b:
                frames_b_all.update(int(_f) for _f in kd_b[_pb2][0])
            video_overlap = {
                dk: sum(1 for _f in frames_b_all if _f + dk in frames_a_all)
                for dk in delta_candidates
            }
            filtered = [dk for dk in delta_candidates if video_overlap[dk] >= _MIN_DELTA_OVERLAP_FRAMES]
            if filtered:
                delta_candidates = filtered
            else:
                logging.warning(f"  [{vid_a}↔{vid_b}] all δ candidates < {_MIN_DELTA_OVERLAP_FRAMES} overlap")

            logging.info(f"  [{vid_a}↔{vid_b}] δ candidates: {delta_candidates}")

            best_sim_i, best_sim_j = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
            best_seed = [(pids_b[best_sim_j], pids_a[best_sim_i])]
            pids_a_s  = sorted(pids_a); pids_b_s = sorted(pids_b)

            cands: list[dict] = []
            for dk in delta_candidates:
                res = _constrained_fit(best_seed, vid_b, vid_a, dk)
                if res is None:
                    continue
                R_k, t_k = res[0], res[1]
                rm = np.full((len(pids_b_s), len(pids_a_s)), 1e6)
                for ii, pb in enumerate(pids_b_s):
                    for jj, pa in enumerate(pids_a_s):
                        r, _ = _pair_rmse_with_Rt(R_k, t_k, pb, vid_b, pa, vid_a, dk)
                        rm[ii, jj] = r if np.isfinite(r) else 1e6
                ri, ci = linear_sum_assignment(rm)
                res2 = _constrained_fit(
                    [(pids_b_s[ri[k2]], pids_a_s[ci[k2]]) for k2 in range(len(ri))],
                    vid_b, vid_a, dk,
                )
                R_k, t_k, _, rmse_k = res2 if res2 is not None else (*res[:2], {}, res[3])
                n_k = video_overlap.get(dk, 0)
                score_k = rmse_k * _overlap_factor(n_k) + _delta_prior_cost(dk)
                logging.info(f"    [{vid_a}↔{vid_b}] δ={dk}  raw_rmse={rmse_k:.3f}  overlap={n_k}  score={score_k:.3f}")
                cands.append({"delta": int(dk), "score": float(score_k), "rmse": float(rmse_k), "R": R_k, "t": t_k})
            cands.sort(key=lambda c: c["score"])
            return {
                "pids_a": pids_a, "pids_b": pids_b,
                "pids_a_s": pids_a_s, "pids_b_s": pids_b_s,
                "sim_mat": sim_mat, "best_seed": best_seed,
                "frames_a_all": frames_a_all, "frames_b_all": frames_b_all,
                "cands": cands,
            }

        def _solve_consensus_offsets(pair_data: dict) -> dict:
            scored = sorted(
                (pd["cands"][0]["score"], key, pd["cands"][0]["delta"])
                for key, pd in pair_data.items() if pd["cands"]
            )
            if not scored:
                return {key: 0 for key in pair_data}
            clock: dict[str, float] = {v: 0.0 for v in active_vids}
            comp:  dict[str, str]   = {v: v for v in active_vids}

            def _root(v: str) -> str:
                while comp[v] != v:
                    comp[v] = comp[comp[v]]; v = comp[v]
                return v

            for s, (va, vb), d in scored:
                ra, rb = _root(va), _root(vb)
                if ra == rb:
                    continue
                shift = clock[va] - d - clock[vb]
                for v in active_vids:
                    if _root(v) == rb:
                        clock[v] += shift
                comp[rb] = ra
                logging.info(f"    consensus tree += {va}↔{vb}  δ={d}  score={s:.3f}")

            logging.info(
                f"Scene {scene_id}: consensus clocks "
                + "  ".join(f"{v}={clock[v]:+.1f}" for v in active_vids)
            )
            return {key: int(round(clock[key[0]] - clock[key[1]])) for key in pair_data}

        # ── Phase 1 helper: canonical similarity at fixed δ ────────────────────

        def _canon_sim_at_delta(
            pose_a: np.ndarray, frames_a: np.ndarray,
            pose_b: np.ndarray, frames_b: np.ndarray,
            delta: int,
        ) -> float:
            """Canonical joint cosine similarity between two persons at a fixed δ."""
            fa_dc = pose_a - pose_a.mean(axis=0)
            fb_dc = pose_b - pose_b.mean(axis=0)
            na_ = np.linalg.norm(fa_dc, axis=1, keepdims=True)
            nb_ = np.linalg.norm(fb_dc, axis=1, keepdims=True)
            fa_n = np.where(na_ > 1e-6, fa_dc / na_, fa_dc)
            fb_n = np.where(nb_ > 1e-6, fb_dc / nb_, fb_dc)
            fb_idx = {int(f): i for i, f in enumerate(frames_b)}
            rows = [
                (i, fb_idx[int(frames_a[i]) - delta])
                for i in range(len(frames_a))
                if int(frames_a[i]) - delta in fb_idx
            ]
            if len(rows) < _MIN_OVERLAP:
                return 0.0
            ia, ib = zip(*rows)
            sims = np.sum(fa_n[list(ia)] * fb_n[list(ib)], axis=1)
            return max(0.0, float(sims.mean()))

        # ── Phase 1: dynamic (high-motion) matching — v4 Tier-1 verbatim ───────

        def _dynamic_assign(
            vid_a: str, vid_b: str, pdata: dict, delta_bar: int,
            single_image: bool = False,
        ) -> list[tuple[int, int, float]]:
            """
            Pairwise dynamic matching at fixed delta_bar (unchanged from v4 Tier-1).

            High-motion persons (pose_std > STD_THRESHOLD, or top-TIER1_FALLBACK_N if none)
            are matched via canonical joint similarity → Hungarian → accept ≥ CANON_THR.
            Unmatched persons are simply left as singletons for the Phase-2 static clustering.
            (No rescue anchor: it existed in v4 only to seed Tier-2 geometry, which is gone.)
            In single_image mode all persons are treated as dynamic and pose cosine sim is
            computed directly (DC removal would zero a single frame).
            """
            pids_a_s = pdata["pids_a_s"]
            pids_b_s = pdata["pids_b_s"]
            kd_a     = person_kpts3d.get(vid_a, {})
            kd_b     = person_kpts3d.get(vid_b, {})

            # ── Motion split (skipped in single-image mode) ────────────────────
            if single_image:
                high_a = list(pids_a_s)
                high_b = list(pids_b_s)
                logging.info(
                    f"  [{vid_a}↔{vid_b}] single-image: all {len(high_a)}a/{len(high_b)}b → dynamic"
                )
            else:
                stds_a = {pa: _track_std(vid_a, pa) for pa in pids_a_s}
                stds_b = {pb: _track_std(vid_b, pb) for pb in pids_b_s}

                high_a = [pa for pa in pids_a_s if stds_a[pa] > _STD_THRESHOLD]
                high_b = [pb for pb in pids_b_s if stds_b[pb] > _STD_THRESHOLD]

                if not high_a:
                    high_a = sorted(pids_a_s, key=lambda p: -stds_a[p])[:_TIER1_FALLBACK_N]
                    logging.info(f"  [{vid_a}↔{vid_b}] dynamic fallback (cam_a): top-{_TIER1_FALLBACK_N} by std")
                if not high_b:
                    high_b = sorted(pids_b_s, key=lambda p: -stds_b[p])[:_TIER1_FALLBACK_N]
                    logging.info(f"  [{vid_a}↔{vid_b}] dynamic fallback (cam_b): top-{_TIER1_FALLBACK_N} by std")

                logging.info(
                    f"  [{vid_a}↔{vid_b}] dynamic candidates: {len(high_a)}a/{len(high_b)}b"
                )

            # ── Canonical similarity → Hungarian ───────────────────────────────
            Na_h, Nb_h = len(high_a), len(high_b)
            canon_sim = np.zeros((Na_h, Nb_h), dtype=np.float32)
            for i, pa in enumerate(high_a):
                pose_a = person_descs[vid_a][pa][2]
                if pose_a is None:
                    continue
                for j, pb in enumerate(high_b):
                    pose_b = person_descs[vid_b][pb][2]
                    if pose_b is None:
                        continue
                    if single_image:
                        # DC removal zeros a single frame — use raw normalised pose directly
                        canon_sim[i, j] = max(0.0, float(np.dot(pose_a[0], pose_b[0])))
                    else:
                        if pa not in kd_a or pb not in kd_b:
                            continue
                        canon_sim[i, j] = _canon_sim_at_delta(
                            pose_a, kd_a[pa][0], pose_b, kd_b[pb][0], delta_bar
                        )

            ri, ci = linear_sum_assignment(1.0 - canon_sim)
            dyn_pairs: list[tuple[int, int]] = []
            for r, c in zip(ri, ci):
                pa, pb = high_a[r], high_b[c]
                if canon_sim[r, c] >= _CANON_THR:
                    dyn_pairs.append((pa, pb))
                    logging.info(
                        f"  dynamic accept {vid_a}:P{pa} ↔ {vid_b}:P{pb}  "
                        f"canon_sim={canon_sim[r,c]:.3f}"
                    )
                else:
                    logging.info(
                        f"  dynamic REJECT {vid_a}:P{pa} ↔ {vid_b}:P{pb}  "
                        f"canon_sim={canon_sim[r,c]:.3f}"
                    )

            return [
                (pa, pb, float(canon_sim[high_a.index(pa), high_b.index(pb)]))
                for pa, pb in dyn_pairs
            ]

        # ── Phase 2: global static clustering on singletons ────────────────────

        def _static_cluster() -> None:
            """
            Single-linkage graph clustering over every person still in a singleton
            UF component, using a view-invariant betas + appearance feature.

            An edge (i, j) is merged, strongest-first, only if all of:
              - sim(i, j) ≥ static_sim_floor                       (absolute backstop)
              - i and j are mutual best matches across cameras     (relative evidence)
              - sim(i, j) beats each node's runner-up by ratio_thr (discriminability)
              - the two clusters share no camera                   (source constraint)
              - the merged cluster spans ≤ N cameras               (size constraint)
            Ambiguous persons are left isolated; there is no geometric fallback.
            """
            comps = _get_components()
            singletons = [m[0] for m in comps.values() if len(m) == 1]
            if len(singletons) < 2:
                logging.info(f"  [static] {len(singletons)} singleton(s) — clustering skipped")
                return

            # ── Per-node feature: appearance + L2 betas ────────────────────────
            # Appearance = TransReID view-invariant descriptor when a ckpt is
            # configured and frames are available; otherwise fall back to the
            # DINOv3 conf-weighted gallery mean. Betas always from body params.
            reid = self._get_reid()
            feats: dict[tuple, tuple] = {}
            for node in singletons:
                vid, pid = node
                desc = person_descs.get(vid, {}).get(pid)
                if desc is None:
                    continue
                app_raw, shape_feat, _ = desc
                app_vec = None
                fdir = frames_dirs.get(vid)
                if reid is not None and fdir is not None:
                    jdir = Path(video_dirs[vid]) / "json_data"
                    cam_suffix = str(vid).split("_")[-1]
                    try:
                        app_vec = reid.person_feature(fdir, jdir, cam_suffix, pid)
                    except Exception as e:
                        logging.warning(f"  [static] TransReID failed {vid}:P{pid} ({e}); falling back to DINOv3")
                        app_vec = None
                if app_vec is None and app_raw is not None:   # DINOv3 fallback
                    f, c = app_raw
                    w = c / (c.sum() + 1e-8)
                    m = f.T @ w
                    n = float(np.linalg.norm(m))
                    if n > 1e-8:
                        app_vec = (m / n).astype(np.float32)
                if app_vec is not None or shape_feat is not None:
                    feats[node] = (app_vec, shape_feat)
            nodes = [n for n in singletons if n in feats]
            if len(nodes) < 2:
                logging.info("  [static] <2 nodes with features — clustering skipped")
                return

            w_b, w_a = self.static_betas_weight, self.static_app_weight

            def _sim(ni: tuple, nj: tuple) -> float:
                ai, si = feats[ni]; aj, sj = feats[nj]
                num = 0.0; den = 0.0
                if si is not None and sj is not None:
                    num += w_b * ((float(np.dot(si, sj).clip(-1.0, 1.0)) + 1.0) / 2.0); den += w_b
                if ai is not None and aj is not None:
                    num += w_a * ((float(np.dot(ai, aj).clip(-1.0, 1.0)) + 1.0) / 2.0); den += w_a
                return num / den if den > 0 else 0.0

            # ── Cross-camera candidate similarities per node ───────────────────
            cand: dict[tuple, list] = {n: [] for n in nodes}
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    ni, nj = nodes[a], nodes[b]
                    if ni[0] == nj[0]:
                        continue
                    s = _sim(ni, nj)
                    cand[ni].append((s, nj)); cand[nj].append((s, ni))
            best:   dict[tuple, tuple] = {}
            second: dict[tuple, float] = {}
            for n in nodes:
                lst = sorted(cand[n], key=lambda x: -x[0])
                best[n]   = lst[0] if lst else (0.0, None)
                second[n] = lst[1][0] if len(lst) > 1 else 0.0

            # ── Eligible edges: mutual-best ∧ floor ∧ ratio ────────────────────
            eligible: list[tuple] = []
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    ni, nj = nodes[a], nodes[b]
                    if ni[0] == nj[0]:
                        continue
                    s = _sim(ni, nj)
                    if s < self.static_sim_floor:
                        continue
                    if best[ni][1] != nj or best[nj][1] != ni:
                        continue  # not mutual best
                    ratio_i = s / (second[ni] + 1e-8)
                    ratio_j = s / (second[nj] + 1e-8)
                    if ratio_i < self.static_ratio_thr or ratio_j < self.static_ratio_thr:
                        continue  # ambiguous — leave isolated
                    eligible.append((s, ni, nj))
            eligible.sort(key=lambda x: -x[0])

            # ── Greedy single-linkage merge with source + size constraints ─────
            n_cams = len(active_vids)

            def _root_cams(root: tuple) -> set:
                return {n[0] for n in nodes if _find(n) == root}

            n_merged = 0
            for s, ni, nj in eligible:
                if _find(ni) == _find(nj):
                    continue
                cams_i = _root_cams(_find(ni))
                cams_j = _root_cams(_find(nj))
                if cams_i & cams_j:
                    continue  # source constraint: one person per camera per cluster
                if len(cams_i | cams_j) > n_cams:
                    continue  # size constraint
                edges.append((float(s), ni, nj))
                _union(ni, nj)
                n_merged += 1
                logging.info(
                    f"  [static] merge {ni[0]}:P{ni[1]} ↔ {nj[0]}:P{nj[1]}  sim={s:.3f}"
                )
            logging.info(
                f"  [static] clustered {len(nodes)} singleton(s) → {n_merged} merge(s)"
            )

        # ── All-pairs: delta candidates → consensus → Phase 1 → Phase 2 ───────
        camera_pair_offsets: dict[str, int] = {}

        if _single_image_mode:
            logging.info(
                f"Scene {scene_id}: single-image mode (max_track_len={max_track_len}) — Phase 1 skipped, all → Phase 2 clustering"
            )

        if not _single_image_mode:
            pair_data: dict[tuple, dict] = {}
            for ii, vid_a in enumerate(active_vids):
                for vid_b in active_vids[ii + 1:]:
                    pair_data[(vid_a, vid_b)] = _pair_delta_candidates(vid_a, vid_b)

            consensus = _solve_consensus_offsets(pair_data)

            # Phase 1 — pairwise dynamic matching
            for (vid_a, vid_b), pdata in pair_data.items():
                delta_bar = consensus[(vid_a, vid_b)]
                matches   = _dynamic_assign(vid_a, vid_b, pdata, delta_bar)
                camera_pair_offsets[f"{vid_a}→{vid_b}"] = delta_bar
                for pa, pb, conf in matches:
                    edges.append((conf, (vid_a, pa), (vid_b, pb)))
                    _union((vid_a, pa), (vid_b, pb))

        # ── Conflict resolution (identical to v2/v3/v4) ────────────────────────
        def _get_components() -> dict[tuple, list[tuple]]:
            comps: dict[tuple, list[tuple]] = {}
            for _v in active_vids:
                for _p in person_pids[_v]:
                    _node = (_v, _p)
                    comps.setdefault(_find(_node), []).append(_node)
            return comps

        # Phase 2 — global static clustering on the remaining singletons
        # In single-image mode all persons are singletons (Phase 1 skipped) so this handles everything
        _static_cluster()

        def _find_path_min_edge(src: tuple, dst: tuple) -> int:
            conflict_root = _find(src)
            adj: dict[tuple, list] = {}
            for ei, (s, na, nb) in enumerate(edges):
                if _find(na) != conflict_root:
                    continue
                adj.setdefault(na, []).append((nb, ei, s))
                adj.setdefault(nb, []).append((na, ei, s))
            prev: dict[tuple, tuple | None] = {src: None}
            queue: deque = deque([src])
            while queue:
                cur = queue.popleft()
                if cur == dst:
                    break
                for nbr, ei, _ in adj.get(cur, []):
                    if nbr not in prev:
                        prev[nbr] = (cur, ei)
                        queue.append(nbr)
            else:
                return -1
            path_ei: list[int] = []; cur = dst
            while prev[cur] is not None:
                par, ei = prev[cur]; path_ei.append(ei); cur = par
            adj_nodes = {n: {x[0] for x in nbrs} for n, nbrs in adj.items()}
            def _tri(ei: int) -> int:
                _, na, nb = edges[ei]
                return len(adj_nodes.get(na, set()) & adj_nodes.get(nb, set()))
            return min(path_ei, key=lambda ei: (_tri(ei), edges[ei][0]))

        def _resolve_conflicts() -> None:
            for _ in range(len(edges) + 1):
                comps = _get_components(); conflict_found = False
                for members in comps.values():
                    v2n: dict[str, list] = {}
                    for node in members:
                        v2n.setdefault(node[0], []).append(node)
                    for vid_id, nodes in v2n.items():
                        if len(nodes) < 2:
                            continue
                        conflict_found = True
                        worst_idx = _find_path_min_edge(nodes[0], nodes[1])
                        if worst_idx >= 0:
                            rs, rna, rnb = edges[worst_idx]
                            logging.info(
                                f"Scene {scene_id}: removed conflict "
                                f"{rna[0]}/P{rna[1]} ↔ {rnb[0]}/P{rnb[1]} (sim={rs:.3f})"
                            )
                            edges.pop(worst_idx)
                            parent.clear(); rank_uf.clear()
                            for _v in active_vids:
                                for _p in person_pids[_v]:
                                    _find((_v, _p))
                            for _s, _na, _nb in edges:
                                _union(_na, _nb)
                        break
                if not conflict_found:
                    break

        _resolve_conflicts()

        # ── Global ID assignment (identical to v2/v3/v4) ───────────────────────
        comps = _get_components()
        global_remap: dict[str, dict[int, int]] = {v: {} for v in active_vids}
        pending_single: list[tuple[int, list[tuple]]] = []
        global_counter = 1
        used_global_ids: set[int] = set()
        for members in comps.values():
            if len({v for v, _ in members}) >= 2:
                gid = global_counter; global_counter += 1; used_global_ids.add(gid)
                for vid_id, pid in members:
                    if pid != gid:
                        global_remap[vid_id][pid] = gid
            else:
                pending_single.append((min(pid for _, pid in members), list(members)))
        nni = global_counter
        for _, members in sorted(pending_single):
            gid = nni; nni += 1; used_global_ids.add(gid)
            for vid_id, pid in members:
                if pid != gid:
                    global_remap[vid_id][pid] = gid

        n_comps  = len(comps)
        n_remaps = sum(len(m) for m in global_remap.values())
        logging.info(
            f"Scene {scene_id}: v5 → {n_comps} global person(s), "
            f"{n_remaps} remap(s) across {len(active_vids)} view(s)"
        )

        if dry_run:
            print(
                f"\n[DRY RUN v5] Scene {scene_id}: {n_comps} global persons, "
                f"camera pair offsets: {camera_pair_offsets}"
            )
            for vid_id, remap in global_remap.items():
                if remap:
                    print(f"  {vid_id}: would remap {remap}")
                else:
                    print(f"  {vid_id}: no remaps needed")
            return

        # ── Apply remaps (identical to v3/v4) ──────────────────────────────────
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir  = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"
            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.v5tmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    logging.warning(f"{vid_id}: v5 remap — {dst.name} already exists, discarding")
                    tmp.unlink()
                else:
                    tmp.rename(dst)
            from preprocessing.cross_view_reid_v2 import CrossVideoReidentifierV2
            CrossVideoReidentifierV2.apply_reid_remap(vid_dir, remap)

        (scene_dir / "cross_view_reid.json").write_text(
            json.dumps({"status": "done", "n_global": n_comps}, indent=2)
        )
