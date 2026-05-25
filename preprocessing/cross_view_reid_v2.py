"""Cross-view person re-identification v2 — combined appearance + SLAM + RANSAC.

Full algorithm, matching notebooks/reid_geometric.ipynb + combined_reid.ipynb:

  0. DROID-SLAM runs on every camera (static or moving) to estimate per-frame
     camera poses.  A metric scale λ is computed from pred_cam_t disparities.
     All 3-D keypoints are then projected to a common world frame.

  1. Appearance Hungarian gives initial assignment (appearance + shape + pose
     xcorr) and provides seeds for the geometric RANSAC step.

  2. RANSAC with delta search: for each appearance-seeded anchor candidate
     (high sim ≥ HIGH_APP_THR), integer frame offsets δ in
     [xcorr_estimate − delta_search, xcorr_estimate + delta_search] are tried.
     The 12-DOF affine is fitted for each (anchor, δ) and inliers are counted.
     Prefers dynamic anchors (joint_std ≥ min_anchor_std).  Falls back to all
     pairs as RANSAC seeds when no high-confidence appearance pair exists.

  3. Joint refinement: iteratively re-fits the affine from the inlier set and
     searches a ±delta_search neighbourhood around the current best δ, repeating
     until convergence.

  4. Hungarian on the RMSE matrix under the refined affine + δ to produce the
     final assignment.

  5. Pairs with RMSE < match_rmse_thr are merged.  Same-camera conflict guard
     is enforced via Union-Find.

  6. Per-camera-pair frame offsets (best δ) are saved to cross_view_reid.json
     for downstream temporal synchronisation.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.video_dataset import Scene


class CrossVideoReidentifierV2:
    """Assign consistent global person IDs across camera views — v2 algorithm.

    Parameters
    ----------
    threshold : float
        Fallback similarity threshold when RANSAC fails for a camera pair.
    appearance_weight, shape_weight, pose_weight : float
        Weights for the hybrid appearance descriptor.
    high_app_thr : float
        Pairs with sim >= high_app_thr seed the RANSAC anchor search.
    low_app_thr : float
        When no affine is available, pairs below this threshold are rejected.
    match_rmse_thr : float
        Maximum world-frame RMSE (m) to accept a final matched pair.
    min_anchor_std : float
        Minimum joint temporal std (m) for a track to count as dynamic.
        Static-only scenes fall back to best-RMSE anchor.
    ransac_anchor_thr : float
        Maximum RMSE (m) for the RANSAC anchor fit itself to be accepted.
    inlier_thr : float
        RMSE threshold (m) for counting RANSAC inliers.
    delta_search : int
        Neighbourhood half-width for δ refinement in joint_refine.
    droid_weights : str | None
        Path to droid.pth.  If None SLAM is skipped (assumes static cameras).
    droid_root : str
        Path to the DROID-SLAM repo root (for sys.path injection).
    default_intrinsics : tuple[float, float, float, float]
        Fallback (fx, fy, cx, cy) when per-camera calibration is not available.
    """

    _THRESHOLD: float = 0.4
    _APPEARANCE_WEIGHT: float = 0.5
    _SHAPE_WEIGHT: float = 0.2
    _POSE_WEIGHT: float = 0.3
    _HIGH_APP_THR: float = 0.60
    _LOW_APP_THR: float = 0.35
    _MATCH_RMSE_THR: float = 0.25
    _MIN_ANCHOR_STD: float = 0.05
    _RANSAC_ANCHOR_THR: float = 0.20
    _INLIER_THR: float = 0.25
    _DELTA_SEARCH: int = 2
    _DROID_WEIGHTS: str | None = None
    _DROID_ROOT: str = "/users/tnanni/ghost/DROID-SLAM"
    _DEFAULT_INTRINSICS: tuple[float, float, float, float] = (1200.0, 1200.0, 720.0, 526.0)
    _SLAM_CAMS: set[str] | None = None  # None = run SLAM on all cameras

    def __init__(
        self,
        threshold: float | None = None,
        appearance_weight: float | None = None,
        shape_weight: float | None = None,
        pose_weight: float | None = None,
        high_app_thr: float | None = None,
        low_app_thr: float | None = None,
        match_rmse_thr: float | None = None,
        min_anchor_std: float | None = None,
        ransac_anchor_thr: float | None = None,
        inlier_thr: float | None = None,
        delta_search: int | None = None,
        droid_weights: str | None = None,
        droid_root: str | None = None,
        default_intrinsics: tuple[float, float, float, float] | None = None,
        slam_cams: set[str] | list[str] | None = None,
    ):
        self.threshold = threshold if threshold is not None else self._THRESHOLD
        self.appearance_weight = (
            appearance_weight if appearance_weight is not None else self._APPEARANCE_WEIGHT
        )
        self.shape_weight = shape_weight if shape_weight is not None else self._SHAPE_WEIGHT
        self.pose_weight = pose_weight if pose_weight is not None else self._POSE_WEIGHT
        self.high_app_thr = high_app_thr if high_app_thr is not None else self._HIGH_APP_THR
        self.low_app_thr = low_app_thr if low_app_thr is not None else self._LOW_APP_THR
        self.match_rmse_thr = (
            match_rmse_thr if match_rmse_thr is not None else self._MATCH_RMSE_THR
        )
        self.min_anchor_std = (
            min_anchor_std if min_anchor_std is not None else self._MIN_ANCHOR_STD
        )
        self.ransac_anchor_thr = (
            ransac_anchor_thr if ransac_anchor_thr is not None else self._RANSAC_ANCHOR_THR
        )
        self.inlier_thr = inlier_thr if inlier_thr is not None else self._INLIER_THR
        self.delta_search = delta_search if delta_search is not None else self._DELTA_SEARCH
        self.droid_weights = droid_weights if droid_weights is not None else self._DROID_WEIGHTS
        self.droid_root = droid_root if droid_root is not None else self._DROID_ROOT
        self.default_intrinsics = (
            default_intrinsics if default_intrinsics is not None else self._DEFAULT_INTRINSICS
        )
        self.slam_cams: set[str] | None = (
            set(slam_cams) if slam_cams is not None else self._SLAM_CAMS
        )

    def match_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path] | None = None,
        intrinsics_map: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Assign consistent global person IDs across all camera views in a scene.

        Parameters
        ----------
        frames_dirs : dict[str, Path], optional
            Per-camera directory of extracted JPEG frames, used to run DROID-SLAM.
            Derived from scene.videos.frames_home when not provided.
        intrinsics_map : dict[str, np.ndarray], optional
            Per-camera 3×3 intrinsic matrix K.  Falls back to default_intrinsics
            when not provided.
        """
        scene_id = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent

        if (scene_dir / "cross_view_reid.json").exists():
            logging.info(f"  Scene {scene_id}: cross-view ReID v2 already done, skipping")
            return

        # Derive frames_dirs from scene.videos when not provided.
        if frames_dirs is None:
            frames_dirs = {}
            for video in scene.videos:
                if video.video_id in video_dirs and getattr(video, "frames_home", None):
                    frames_dirs[video.video_id] = video.frames_home

        return self._cross_view_reid_core(
            video_dirs=video_dirs,
            scene_id=scene_id,
            scene_dir=scene_dir,
            cross_view_reid_threshold=self.threshold,
            appearance_weight=self.appearance_weight,
            shape_weight=self.shape_weight,
            pose_weight=self.pose_weight,
            high_app_thr=self.high_app_thr,
            low_app_thr=self.low_app_thr,
            match_rmse_thr=self.match_rmse_thr,
            min_anchor_std=self.min_anchor_std,
            ransac_anchor_thr=self.ransac_anchor_thr,
            inlier_thr=self.inlier_thr,
            delta_search=self.delta_search,
            droid_weights=self.droid_weights,
            droid_root=self.droid_root,
            default_intrinsics=self.default_intrinsics,
            frames_dirs=frames_dirs,
            intrinsics_map=intrinsics_map or {},
            slam_cams=self.slam_cams,
        )

    # ── apply_reid_remap (identical to v1) ────────────────────────────────────

    @staticmethod
    def apply_reid_remap(
        video_dir: Path,
        id_remap: dict[int, int],
        gpu_label: str = "",
    ) -> None:
        """Rewrite mask_data.npz and json_data/*.json with re-identified IDs."""
        if not id_remap or all(k == v for k, v in id_remap.items()):
            return

        npz_path = video_dir / "mask_data.npz"
        json_dir = video_dir / "json_data"

        if npz_path.exists():
            tmp_path = npz_path.with_suffix(".tmp.npz")
            with (
                zipfile.ZipFile(str(npz_path), "r") as zf_in,
                zipfile.ZipFile(
                    str(tmp_path), "w",
                    compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                ) as zf_out,
            ):
                for name in sorted(zf_in.namelist()):
                    with zf_in.open(name) as f:
                        mask_img = np.load(io.BytesIO(f.read()))
                    total_pixels = mask_img.shape[0] * mask_img.shape[1]
                    new_mask = np.zeros_like(mask_img)
                    for old_id, new_id in id_remap.items():
                        old_region = mask_img == old_id
                        if int(old_region.sum()) > 0.80 * total_pixels:
                            logging.warning(
                                f"{gpu_label}Skipping remap {old_id}→{new_id} "
                                f"in {name}: mask covers >80% of frame"
                            )
                            continue
                        new_mask[old_region] = new_id
                    for uid in set(np.unique(mask_img)) - {0} - set(id_remap.keys()):
                        new_mask[mask_img == uid] = uid
                    buf = io.BytesIO()
                    np.save(buf, new_mask)
                    zf_out.writestr(name, buf.getvalue())
            tmp_path.replace(npz_path)

        for json_path in sorted(json_dir.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            if "labels" in data:
                new_labels = {}
                sorted_items = sorted(
                    data["labels"].items(),
                    key=lambda kv: 0 if id_remap.get(int(kv[0]), int(kv[0])) == int(kv[0]) else 1,
                )
                for str_id, info in sorted_items:
                    old_id = int(str_id)
                    new_id = id_remap.get(old_id, old_id)
                    info["instance_id"] = new_id
                    if str(new_id) in new_labels:
                        logging.warning(
                            f"{gpu_label}Re-ID collision for id {new_id} in "
                            f"{json_path.name}: keeping original id {old_id}"
                        )
                        info["instance_id"] = old_id
                        new_labels[str(old_id)] = info
                    else:
                        new_labels[str(new_id)] = info
                data["labels"] = new_labels
            with open(json_path, "w") as f:
                json.dump(data, f)

        logging.info(
            f"{gpu_label}Re-ID segmentation remap applied in {video_dir.name}: "
            f"{id_remap}"
        )

    # ── Core ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cross_view_reid_core(
        video_dirs: dict[str, Path],
        scene_id: str,
        scene_dir: Path,
        cross_view_reid_threshold: float,
        appearance_weight: float,
        shape_weight: float,
        pose_weight: float,
        high_app_thr: float,
        low_app_thr: float,
        match_rmse_thr: float,
        min_anchor_std: float,
        ransac_anchor_thr: float,
        inlier_thr: float,
        delta_search: int,
        droid_weights: str | None,
        droid_root: str,
        default_intrinsics: tuple[float, float, float, float],
        frames_dirs: dict[str, Path],
        intrinsics_map: dict[str, np.ndarray],
        slam_cams: set[str] | None,
    ) -> None:

        MIN_OVERLAP = 30
        MIN_A_SINGULAR_VALUE = 0.3

        video_ids = list(video_dirs.keys())

        # ── Per-person descriptors and raw keypoints ───────────────────────────
        person_descs: dict[str, dict[int, tuple]] = {}
        person_pids: dict[str, list[int]] = {}
        # kpts3d[vid][pid] = (frames_array, kpts_T×70×3)  — corrected to world frame after SLAM
        person_kpts3d: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        # cam_t[vid][pid] = (frames_array, pred_cam_t_T×3)  — needed for SLAM scale
        person_cam_t: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

        for vid_id, vid_dir in video_dirs.items():
            body_dir = Path(vid_dir) / "body_data"
            gallery_path = body_dir / "appearance_gallery.npz"
            summary_path = body_dir / "body_params_summary.json"

            if not summary_path.exists():
                logging.warning(
                    f"{scene_id}/{vid_id}: no body_params_summary.json, "
                    f"skipping cross-view re-ID for this video"
                )
                continue

            with open(summary_path) as _f:
                summary = json.load(_f)
            pids = [int(k) for k in summary.get("persons", {}).keys()]
            if not pids:
                continue

            app_gallery: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                for k in gdata.files:
                    if k.endswith("_conf"):
                        continue
                    pid_key = int(k)
                    conf_key = f"{k}_conf"
                    feats = gdata[k]
                    confs = (
                        gdata[conf_key]
                        if conf_key in gdata.files
                        else np.ones(len(feats), dtype=np.float32)
                    )
                    app_gallery[pid_key] = (feats, confs)

            descs: dict[int, tuple] = {}
            for pid in pids:
                npz_path = body_dir / f"person_{pid}.npz"
                if not npz_path.exists():
                    continue

                shape_feat: np.ndarray | None = None
                pose_feat: np.ndarray | None = None

                with np.load(str(npz_path)) as pdata:
                    # Shape (betas)
                    if "smplx_betas" in pdata:
                        bv = pdata["smplx_betas"]
                        if len(bv) > 0:
                            conf = pdata.get("pred_joint_confidence")
                            if conf is not None and len(conf) == len(bv):
                                fc = np.mean(conf, axis=-1).astype(np.float32)
                                tc = fc.sum()
                                sm = (fc[:, None] * bv).sum(0) / tc if tc > 0 else np.median(bv, 0).astype(np.float32)
                            else:
                                sm = np.median(bv, axis=0).astype(np.float32)
                            nm = np.linalg.norm(sm)
                            shape_feat = sm / nm if nm > 0 else sm

                    # Camera-frame root translation (needed for SLAM scale)
                    for key in ("pred_cam_t", "smplx_transl"):
                        if key in pdata and "frame_indices" in pdata:
                            person_cam_t.setdefault(vid_id, {})[pid] = (
                                pdata["frame_indices"].copy(),
                                pdata[key].copy(),
                            )
                            break

                    # Raw 3-D keypoints (camera space; corrected to world after SLAM)
                    if "pred_keypoints_3d" in pdata and "frame_indices" in pdata:
                        person_kpts3d.setdefault(vid_id, {})[pid] = (
                            pdata["frame_indices"].copy(),
                            pdata["pred_keypoints_3d"].copy(),
                        )

                    # Pose feature (canonical, root-relative, orient-free)
                    if "pred_keypoints_3d" in pdata and "smplx_global_orient" in pdata:
                        from scipy.spatial.transform import Rotation as _Rot
                        kps = pdata["pred_keypoints_3d"].astype(np.float32)
                        gorient = pdata["smplx_global_orient"].astype(np.float32)
                        conf_kps = pdata.get("pred_joint_confidence")
                        N = len(kps)
                        if N > 0:
                            kps_rel = kps - kps[:, 0:1, :]
                            rm = _Rot.from_rotvec(gorient).inv().as_matrix()
                            kc = np.einsum("nij,nkj->nki", rm, kps_rel).astype(np.float32)
                            pv = kc.reshape(N, -1)
                            np_norms = np.linalg.norm(pv, axis=1, keepdims=True)
                            pv = np.where(np_norms > 1e-6, pv / np_norms, pv)
                            pose_feat = pv

                app_feat = app_gallery.get(pid)
                if app_feat is None and shape_feat is None and pose_feat is None:
                    continue
                descs[pid] = (app_feat, shape_feat, pose_feat)

            if descs:
                person_descs[vid_id] = descs
                person_pids[vid_id] = sorted(descs.keys())

        active_vids = [v for v in video_ids if v in person_descs]

        if len(active_vids) < 2:
            logging.info(f"Scene {scene_id}: fewer than 2 videos with descriptors, skipping")
            return set()

        # ── SLAM: correct all cameras to world frame ───────────────────────────
        # Runs on every camera regardless of whether it is static or moving,
        # because we do not know a priori which cameras have ego-motion.

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
                intr = _torch.tensor([fx * w1/w0, fy * h1/h0, cx * w1/w0, cy * h1/h0])
                yield t, image[None], intr

        def _run_droid_slam(frames_dir: Path, fx, fy, cx, cy, weights: str):
            import argparse, cv2, torch as _torch
            if droid_root not in sys.path:
                sys.path.insert(0, str(droid_root))
                sys.path.insert(0, str(Path(droid_root) / "droid_slam"))
            from droid import Droid

            try:
                _torch.multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            img_files = sorted(p for p in frames_dir.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            if not img_files:
                return None
            first = cv2.imread(str(img_files[0]))
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
                poses = droid.terminate(iter(stream))   # (N_frames, 7): [tx,ty,tz, qx,qy,qz,qw]
            except (ValueError, RuntimeError) as e:
                # Too few keyframes (e.g. static camera) — backend factor graph is empty.
                logging.warning(f"DROID-SLAM backend failed ({e}); camera may be static — skipping.")
                del droid
                import torch as _t; _t.cuda.empty_cache()
                return None

            n_kf   = droid.video.counter.value
            disps  = droid.video.disps[:n_kf].cpu().numpy()
            tstamp = droid.video.tstamp[:n_kf].cpu().numpy().astype(int)
            del droid
            import torch as _t; _t.cuda.empty_cache()

            return poses, disps, tstamp, (H_orig, W_orig), (H_slam, W_slam)

        def _estimate_slam_scale(vid_id, fx, fy, cx, cy, disps, tstamp,
                                  frame_offset, orig_hw) -> float:
            H_orig, W_orig = orig_hw
            H_disp, W_disp = disps.shape[1], disps.shape[2]
            sx = W_disp / W_orig
            sy = H_disp / H_orig
            lambdas = []
            cam_t_data = person_cam_t.get(vid_id, {})
            for pid, (frames, cam_t) in cam_t_data.items():
                frame_to_idx = {int(f): i for i, f in enumerate(frames)}
                for kf_i, t in enumerate(tstamp):
                    global_frame = int(t) + frame_offset
                    arr_idx = frame_to_idx.get(global_frame)
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

        _slam_targets: list[str] = []
        _slam_corrected: set[str] = set()
        if droid_weights and frames_dirs:
            _slam_targets = (
                [v for v in active_vids if v in slam_cams]
                if slam_cams is not None
                else active_vids
            )
            logging.info(
                f"Scene {scene_id}: running DROID-SLAM on "
                + (str(slam_cams) if slam_cams is not None else "all cameras")
            )
            for vid_id in _slam_targets:
                fdir = frames_dirs.get(vid_id)
                if fdir is None or not Path(fdir).is_dir():
                    logging.warning(f"  {vid_id}: no frames dir for SLAM, skipping")
                    continue
                K = intrinsics_map.get(vid_id)
                fx = float(K[0, 0]) if K is not None else default_intrinsics[0]
                fy = float(K[1, 1]) if K is not None else default_intrinsics[1]
                cx = float(K[0, 2]) if K is not None else default_intrinsics[2]
                cy = float(K[1, 2]) if K is not None else default_intrinsics[3]

                result = _run_droid_slam(Path(fdir), fx, fy, cx, cy, droid_weights)
                if result is None:
                    logging.warning(f"  {vid_id}: SLAM returned no poses, skipping correction")
                    continue
                poses, disps, tstamp, orig_hw, _ = result

                kpts_data = person_kpts3d.get(vid_id, {})
                if not kpts_data:
                    continue
                frame_offset = int(min(frames.min() for frames, _ in kpts_data.values()))
                lam = _estimate_slam_scale(vid_id, fx, fy, cx, cy, disps, tstamp,
                                           frame_offset, orig_hw)

                # Project all keypoints to world frame:
                #   world_j = R @ kpts_cam[j] + R @ root_cam + λ * t_slam
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
                        R = _quat_to_rot(pose[3:])   # cam-to-world rotation
                        root_cam = cam_t[i] if cam_t is not None else np.zeros(3)
                        # world = R @ (root + joint_offsets) + λ*t
                        corrected[i] = (R @ kpts[i].T).T + R @ root_cam + lam * pose[:3]
                    person_kpts3d[vid_id][pid] = (frames, corrected)

                logging.info(f"  {vid_id}: world-frame correction applied (λ={lam:.4f})")
                _slam_corrected.add(vid_id)
        else:
            if droid_weights:
                logging.warning(
                    f"Scene {scene_id}: droid_weights set but no frames_dirs — SLAM skipped"
                )
            else:
                logging.info(
                    f"Scene {scene_id}: no droid_weights — assuming static cameras"
                )

        # ── Absolute-position correction for cameras without SLAM ──────────────
        # pred_keypoints_3d is root-relative; absolute camera-frame position =
        # pred_cam_t + kpts.  For SLAM-corrected cameras this is already done
        # inside the SLAM block.  For static cameras we apply R=I, t=0:
        #   abs_j = kpts[j] + pred_cam_t
        slam_corrected = _slam_corrected
        for vid_id in active_vids:
            if vid_id in slam_corrected:
                continue
            kpts_data = person_kpts3d.get(vid_id, {})
            cam_t_data = person_cam_t.get(vid_id, {})
            for pid, (frames, kpts) in list(kpts_data.items()):
                cam_t_entry = cam_t_data.get(pid)
                if cam_t_entry is None:
                    continue
                _, cam_t = cam_t_entry
                corrected = kpts + cam_t[:, None, :]  # broadcast over 70 joints
                person_kpts3d[vid_id][pid] = (frames, corrected)
            if kpts_data:
                logging.info(f"  {vid_id}: absolute-position correction applied (static camera)")

        # ── Union-Find with edge tracking ──────────────────────────────────────
        parent: dict[tuple, tuple] = {}
        rank_uf: dict[tuple, int] = {}
        edges: list[tuple[float, tuple, tuple]] = []

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

        for _vid in active_vids:
            for _pid in person_pids[_vid]:
                _find((_vid, _pid))

        # ── Appearance descriptor helpers ──────────────────────────────────────

        def _xcorr_sim(fa: np.ndarray, fb: np.ndarray) -> tuple[float, int]:
            N, M = len(fa), len(fb)
            fa = fa - fa.mean(axis=0)
            fb = fb - fb.mean(axis=0)
            na = np.linalg.norm(fa, axis=1, keepdims=True)
            nb = np.linalg.norm(fb, axis=1, keepdims=True)
            fa = np.where(na > 1e-6, fa / na, fa)
            fb = np.where(nb > 1e-6, fb / nb, fb)
            S = fa @ fb.T
            best_score, best_off = -1.0, 0
            for tau in range(-(M - 1), N):
                diag = np.diagonal(S, offset=-tau)
                if len(diag) < MIN_OVERLAP:
                    continue
                score = float(diag.mean())
                if score > best_score:
                    best_score, best_off = score, tau
            return float(max(0.0, best_score)), best_off

        def _chamfer_sim(fa, ca, fb, cb) -> float:
            S = fa @ fb.T
            return 0.5 * (float(S.max(axis=1).mean()) + float(S.max(axis=0).mean()))

        def _weighted_sim_mat(
            pids_a, pids_b, descs_a, descs_b, w_app, w_shape, w_pose, vid_a="", vid_b=""
        ) -> tuple[np.ndarray, np.ndarray]:
            Na, Nb = len(pids_a), len(pids_b)
            sim_mat = np.zeros((Na, Nb), dtype=np.float32)
            wgt_mat = np.zeros((Na, Nb), dtype=np.float32)
            off_mat = np.zeros((Na, Nb), dtype=np.int32)

            for i, pa in enumerate(pids_a):
                app_a = descs_a[pa][0]
                if app_a is None:
                    continue
                feats_a, confs_a = app_a
                for j, pb in enumerate(pids_b):
                    app_b = descs_b[pb][0]
                    if app_b is None:
                        continue
                    feats_b, confs_b = app_b
                    s = _chamfer_sim(feats_a, confs_a, feats_b, confs_b)
                    sim_mat[i, j] += w_app * s
                    wgt_mat[i, j] += w_app

            sa_list = [descs_a[p][1] for p in pids_a]
            sb_list = [descs_b[p][1] for p in pids_b]
            mka = np.array([f is not None for f in sa_list], dtype=np.float32)
            mkb = np.array([f is not None for f in sb_list], dtype=np.float32)
            if mka.any() and mkb.any():
                dim = next(f for f in sa_list if f is not None).shape[0]
                zero = np.zeros(dim, dtype=np.float32)
                mat_a = np.stack([f if f is not None else zero for f in sa_list])
                mat_b = np.stack([f if f is not None else zero for f in sb_list])
                ssim = mat_a @ mat_b.T
                sw = np.outer(mka, mkb) * w_shape
                sim_mat += sw * ssim
                wgt_mat += sw

            for i, pa in enumerate(pids_a):
                pose_a = descs_a[pa][2]
                if pose_a is None:
                    continue
                fva = pose_a
                for j, pb in enumerate(pids_b):
                    pose_b = descs_b[pb][2]
                    if pose_b is None:
                        continue
                    fvb = pose_b
                    s, off = _xcorr_sim(fva, fvb)
                    off_mat[i, j] = off
                    logging.info(
                        f"  pose xcorr {vid_a}:P{pa} vs {vid_b}:P{pb}"
                        f"  sim={s:.3f}  offset={off:+d}"
                    )
                    sim_mat[i, j] += w_pose * s
                    wgt_mat[i, j] += w_pose

            return np.where(wgt_mat > 0, sim_mat / wgt_mat, 0.0), off_mat

        # ── Geometric helpers ──────────────────────────────────────────────────

        def _affine_fit(src: np.ndarray, dst: np.ndarray):
            X = np.concatenate([src, np.ones((len(src), 1), dtype=src.dtype)], axis=1)
            valid = np.isfinite(X).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() < 12:
                return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            M, _, _, _ = np.linalg.lstsq(X[valid], dst[valid], rcond=None)
            return M[:3].T, M[3]

        def _apply_T(A, t_vec, kpts):
            shape = kpts.shape
            return (A @ kpts.reshape(-1, 3).T).T.reshape(shape) + t_vec

        def _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta):
            """Common-frame keypoints, flat (N*70, 3), or (None, None)."""
            kb = person_kpts3d.get(vid_b, {})
            ka = person_kpts3d.get(vid_a, {})
            if pid_b not in kb or pid_a not in ka:
                return None, None
            frames_b, kpts_b = kb[pid_b]
            frames_a, kpts_a = ka[pid_a]
            fb2i = {int(f): i for i, f in enumerate(frames_b)}
            fa2i = {int(f): i for i, f in enumerate(frames_a)}
            common = sorted(f for f in fb2i if f + delta in fa2i)
            if len(common) < MIN_OVERLAP:
                return None, None
            ib = [fb2i[f] for f in common]
            ia = [fa2i[f + delta] for f in common]
            return kpts_b[ib].reshape(-1, 3), kpts_a[ia].reshape(-1, 3)

        def _direct_rmse(A, t_vec, pid_b, vid_b, pid_a, vid_a, delta) -> float:
            src, dst = _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta)
            if src is None:
                return float("inf")
            pred = (A @ src.T).T + t_vec
            valid = np.isfinite(pred).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() == 0:
                return float("inf")
            r = float(np.sqrt(((pred[valid] - dst[valid]) ** 2).sum(1).mean()))
            return r if np.isfinite(r) else float("inf")

        def _fit_pair_delta(pid_b, vid_b, pid_a, vid_a, delta):
            """Fit affine for one anchor pair + delta. Returns (A, t, rmse) or None."""
            src, dst = _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta)
            if src is None:
                return None
            A, t = _affine_fit(src, dst)
            if np.linalg.svd(A, compute_uv=False).min() < MIN_A_SINGULAR_VALUE:
                return None
            pred = (A @ src.T).T + t
            valid = np.isfinite(pred).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() == 0:
                return None
            rmse = float(np.sqrt(((pred[valid] - dst[valid]) ** 2).sum(1).mean()))
            return (A, t, rmse) if np.isfinite(rmse) else None

        def _fit_two_pairs_delta(pb1, vid_b, pa1, vid_a, pb2, pa2, delta):
            """Fit affine jointly from two anchor pairs. Returns (A, t, rmse) or None."""
            src1, dst1 = _get_aligned_flat(pb1, vid_b, pa1, vid_a, delta)
            src2, dst2 = _get_aligned_flat(pb2, vid_b, pa2, vid_a, delta)
            if src1 is None or src2 is None:
                return None
            src = np.concatenate([src1, src2])
            dst = np.concatenate([dst1, dst2])
            A, t = _affine_fit(src, dst)
            if np.linalg.svd(A, compute_uv=False).min() < MIN_A_SINGULAR_VALUE:
                return None
            pred = (A @ src.T).T + t
            valid = np.isfinite(pred).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() == 0:
                return None
            rmse = float(np.sqrt(((pred[valid] - dst[valid]) ** 2).sum(1).mean()))
            return (A, t, rmse) if np.isfinite(rmse) else None

        def _get_inlier_pairs(A, t_vec, delta, vid_b, vid_a, exclude_pids_b):
            """For each pid_b (except anchors), find best pid_a by RMSE."""
            inliers = []
            for pid_b in person_pids.get(vid_b, []):
                if pid_b in exclude_pids_b:
                    continue
                # Skip static persons — same guard as anchor selection and joint refine.
                if (_track_std(vid_b, pid_b) < min_anchor_std
                        and all(_track_std(vid_a, pa) < min_anchor_std
                                for pa in person_pids.get(vid_a, []))):
                    continue
                best_rmse, best_pa = float("inf"), None
                for pid_a in person_pids.get(vid_a, []):
                    r = _direct_rmse(A, t_vec, pid_b, vid_b, pid_a, vid_a, delta)
                    if r < best_rmse:
                        best_rmse, best_pa = r, pid_a
                if best_pa is not None and best_rmse < inlier_thr:
                    inliers.append((pid_b, best_pa, best_rmse))
            return inliers

        def _track_std(vid: str, pid: int) -> float:
            kd = person_kpts3d.get(vid, {})
            if pid not in kd:
                return 0.0
            _, kpts = kd[pid]
            valid = kpts[np.isfinite(kpts).all(axis=(1, 2))]
            return float(valid.std(axis=0).mean()) if len(valid) > 0 else 0.0

        def _joint_refine(assignment, vid_b, vid_a, delta_init):
            """Iteratively refit affine + search delta neighbourhood."""
            delta = delta_init
            A, t_vec = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            prev_rmse = float("inf")

            # Prefer dynamic pairs for fitting; fall back to all pairs if all static.
            fit_pairs = [
                (pb, pa) for pb, pa in assignment
                if (_track_std(vid_b, pb) >= min_anchor_std
                    or _track_std(vid_a, pa) >= min_anchor_std)
            ]
            if not fit_pairs:
                fit_pairs = list(assignment)

            best_d_rmse = float("inf")
            for _ in range(20):
                # Refit affine from all pairs at current delta.
                all_src, all_dst = [], []
                for pb, pa in fit_pairs:
                    src, dst = _get_aligned_flat(pb, vid_b, pa, vid_a, delta)
                    if src is not None:
                        all_src.append(src)
                        all_dst.append(dst)
                if not all_src:
                    break
                A, t_vec = _affine_fit(
                    np.concatenate(all_src), np.concatenate(all_dst)
                )

                # Search delta in ±delta_search neighbourhood.
                best_d, best_d_rmse = delta, float("inf")
                for d in range(delta - delta_search, delta + delta_search + 1):
                    sl_pred, sl_dst = [], []
                    for pb, pa in fit_pairs:
                        src, dst = _get_aligned_flat(pb, vid_b, pa, vid_a, d)
                        if src is None:
                            continue
                        sl_pred.append((_apply_T(A, t_vec, src)))
                        sl_dst.append(dst)
                    if not sl_pred:
                        continue
                    rmse = float(np.sqrt(
                        ((np.concatenate(sl_pred) - np.concatenate(sl_dst)) ** 2)
                        .sum(1).mean()
                    ))
                    if np.isfinite(rmse) and rmse < best_d_rmse:
                        best_d_rmse, best_d = rmse, d

                if abs(prev_rmse - best_d_rmse) < 1e-5:
                    break
                prev_rmse, delta = best_d_rmse, best_d

            return A, t_vec, delta, best_d_rmse

        # ── Combined matching: appearance seed → RANSAC+δ → joint refine → Hungarian

        def _match_with_combined_approach(
            vid_a: str, vid_b: str
        ) -> tuple[list[tuple[int, int, float]], int]:
            """Returns (accepted_pairs, best_delta)."""
            pids_a = person_pids[vid_a]
            pids_b = person_pids[vid_b]

            # Step 1: appearance sim matrix + xcorr offset estimation.
            sim_mat, off_mat = _weighted_sim_mat(
                pids_a, pids_b,
                person_descs[vid_a], person_descs[vid_b],
                appearance_weight, shape_weight, pose_weight,
                vid_a=vid_a, vid_b=vid_b,
            )
            row_ind, col_ind = linear_sum_assignment(1.0 - sim_mat)
            app_assignment = [
                (pids_a[r], pids_b[c], float(sim_mat[r, c]), int(off_mat[r, c]))
                for r, c in zip(row_ind, col_ind)
            ]

            # Estimate camera frame offset by accumulating xcorr score curves
            # from all high-confidence dynamic pairs in camera-δ space.
            # Each pair contributes its full curve; the true δ is reinforced
            # across pairs while spurious individual peaks average out.
            from collections import defaultdict as _dd
            delta_scores: dict[int, float] = _dd(float)
            n_contrib = 0
            for pa, pb, sim, _ in app_assignment:
                if sim < high_app_thr:
                    continue
                pose_a = person_descs[vid_a][pa][2]
                pose_b = person_descs[vid_b][pb][2]
                if pose_a is None or pose_b is None:
                    continue
                std_a = _track_std(vid_a, pa)
                std_b = _track_std(vid_b, pb)
                if std_a < min_anchor_std and std_b < min_anchor_std:
                    continue
                kd_a = person_kpts3d.get(vid_a, {})
                kd_b = person_kpts3d.get(vid_b, {})
                if pa not in kd_a or pb not in kd_b:
                    continue
                offset = int(kd_a[pa][0][0]) - int(kd_b[pb][0][0])
                w = max(std_a, std_b) * float(sim)
                fva = pose_a - pose_a.mean(axis=0)
                fvb = pose_b - pose_b.mean(axis=0)
                na = np.linalg.norm(fva, axis=1, keepdims=True)
                nb = np.linalg.norm(fvb, axis=1, keepdims=True)
                fva = np.where(na > 1e-6, fva / na, fva)
                fvb = np.where(nb > 1e-6, fvb / nb, fvb)
                S = fva @ fvb.T
                T_a, T_b = S.shape
                for tau in range(-(T_b - 1), T_a):
                    diag = np.diagonal(S, offset=-tau)
                    if len(diag) < MIN_OVERLAP:
                        continue
                    score = float(diag.mean())
                    if score > 0:
                        delta_scores[offset + tau] += w * score
                n_contrib += 1
            if delta_scores:
                xcorr_delta = max(delta_scores, key=lambda d: delta_scores[d])
            else:
                xcorr_delta = 0
            logging.info(
                f"  [{vid_a}↔{vid_b}] xcorr frame offset estimate: {xcorr_delta:+d}"
                f"  (from {n_contrib} pairs)"
            )

            # Step 2: RANSAC with full delta search.
            # Use high-confidence appearance pairs as anchor candidates.
            # Fall back to all pairs when no high-confidence pair exists.
            high_conf = [(pa, pb) for pa, pb, sim, _ in app_assignment if sim >= high_app_thr]
            anchor_candidates = high_conf if high_conf else [
                (pa, pb) for pa, pb, _, _ in app_assignment
            ]

            _empty = dict(A=None, t=None, delta=xcorr_delta, rmse=float("inf"),
                          inliers=-1, pa=None, pb=None, is_dyn=False,
                          init_asgn=[], pbs=set())
            best_dyn = {**_empty}
            best_any = {**_empty}

            def _better(cand, ref):
                if cand["is_dyn"] and not ref["is_dyn"]:
                    return True
                if cand["is_dyn"] == ref["is_dyn"]:
                    return (cand["inliers"] > ref["inliers"] or
                            (cand["inliers"] == ref["inliers"]
                             and cand["rmse"] < ref["rmse"]))
                return False

            d_lo = xcorr_delta - delta_search
            d_hi = xcorr_delta + delta_search

            # Step 2a: 1-pair RANSAC — baseline single-anchor search.
            for pa, pb in anchor_candidates:
                std_a = _track_std(vid_a, pa)
                std_b = _track_std(vid_b, pb)
                is_dyn = std_a >= min_anchor_std or std_b >= min_anchor_std

                best_for = {**_empty}
                for delta in range(d_lo, d_hi + 1):
                    res = _fit_pair_delta(pb, vid_b, pa, vid_a, delta)
                    if res is None:
                        continue
                    A_c, t_c, rmse_c = res
                    n_in = len(_get_inlier_pairs(A_c, t_c, delta, vid_b, vid_a, {pb}))
                    if (n_in > best_for["inliers"] or
                            (n_in == best_for["inliers"] and rmse_c < best_for["rmse"])):
                        best_for = dict(A=A_c, t=t_c, delta=delta, rmse=rmse_c,
                                        inliers=n_in, pa=pa, pb=pb, is_dyn=is_dyn,
                                        init_asgn=[(pb, pa)], pbs={pb})

                if best_for["A"] is None:
                    continue
                if _better(best_for, best_any):
                    best_any = best_for
                if is_dyn and _better(best_for, best_dyn):
                    best_dyn = best_for

            best_1 = best_dyn if best_dyn["A"] is not None else best_any

            # Step 2b: 2-pair RANSAC — fit affine jointly from pairs of anchor candidates.
            # Gives better depth coverage and more robust Z estimation.
            from itertools import combinations as _comb
            best_2_dyn = {**_empty}
            best_2_any = {**_empty}

            for (pa1, pb1), (pa2, pb2) in _comb(anchor_candidates, 2):
                is_dyn = (
                    _track_std(vid_a, pa1) >= min_anchor_std or
                    _track_std(vid_b, pb1) >= min_anchor_std or
                    _track_std(vid_a, pa2) >= min_anchor_std or
                    _track_std(vid_b, pb2) >= min_anchor_std
                )
                best_for = {**_empty}
                for delta in range(d_lo, d_hi + 1):
                    res = _fit_two_pairs_delta(pb1, vid_b, pa1, vid_a, pb2, pa2, delta)
                    if res is None:
                        continue
                    A_c, t_c, rmse_c = res
                    n_in = len(_get_inlier_pairs(
                        A_c, t_c, delta, vid_b, vid_a, {pb1, pb2}))
                    if (n_in > best_for["inliers"] or
                            (n_in == best_for["inliers"] and rmse_c < best_for["rmse"])):
                        best_for = dict(A=A_c, t=t_c, delta=delta, rmse=rmse_c,
                                        inliers=n_in, pa=pa1, pb=pb1, is_dyn=is_dyn,
                                        init_asgn=[(pb1, pa1), (pb2, pa2)],
                                        pbs={pb1, pb2})

                if best_for["A"] is None:
                    continue
                if _better(best_for, best_2_any):
                    best_2_any = best_for
                if is_dyn and _better(best_for, best_2_dyn):
                    best_2_dyn = best_for

            best_2 = best_2_dyn if best_2_dyn["A"] is not None else best_2_any

            # Prefer 2-pair when it is geometrically valid (meets the RMSE threshold)
            # and is at least as dynamic as the 1-pair winner.  We do NOT compare RMSE
            # across the two types because a 2-pair joint fit is constrained by two
            # independent trajectories and will naturally have a higher combined RMSE
            # than a 1-pair fit that overfits to a single person's data.
            if (best_2["A"] is not None
                    and best_2["rmse"] <= ransac_anchor_thr
                    and (best_2["is_dyn"] or not best_1["is_dyn"])):
                best = best_2
            else:
                best = best_1

            if best["A"] is None or best["rmse"] > ransac_anchor_thr:
                # RANSAC failed: fall back to pure appearance threshold.
                logging.info(
                    f"  [{vid_a}↔{vid_b}] RANSAC failed (best RMSE={best['rmse']:.3f}) "
                    f"— appearance-only fallback"
                )
                accepted = [
                    (pa, pb, sim) for pa, pb, sim, _ in app_assignment
                    if sim >= cross_view_reid_threshold
                ]
                return accepted, xcorr_delta

            if len(best["init_asgn"]) == 2:
                (pb1, pa1), (pb2, pa2) = best["init_asgn"]
                logging.info(
                    f"  [{vid_a}↔{vid_b}] 2-pair anchor: "
                    f"P{pb1}+P{pb2}→P{pa1}+P{pa2}  δ={best['delta']}  "
                    f"RMSE={best['rmse']:.3f}  inliers={best['inliers']}"
                )
            else:
                logging.info(
                    f"  [{vid_a}↔{vid_b}] RANSAC anchor: "
                    f"P{best['pb']}→P{best['pa']}  δ={best['delta']}  "
                    f"RMSE={best['rmse']:.3f}  inliers={best['inliers']}"
                )

            # Step 3: joint refinement.
            inlier_pairs = _get_inlier_pairs(
                best["A"], best["t"], best["delta"], vid_b, vid_a, best["pbs"]
            )
            init_asgn = best["init_asgn"] + [(pb, pa) for pb, pa, _ in inlier_pairs]
            A, t_vec, delta, ref_rmse = _joint_refine(init_asgn, vid_b, vid_a, best["delta"])
            logging.info(
                f"  [{vid_a}↔{vid_b}] after refinement: δ={delta}  RMSE={ref_rmse:.3f}"
            )

            # Step 4: Hungarian on full RMSE matrix under refined affine + delta.
            pids_b_s = sorted(person_pids.get(vid_b, []))
            pids_a_s = sorted(person_pids.get(vid_a, []))
            rmse_mat = np.full((len(pids_b_s), len(pids_a_s)), 1e6, dtype=np.float64)
            for i, pb in enumerate(pids_b_s):
                for j, pa in enumerate(pids_a_s):
                    r = _direct_rmse(A, t_vec, pb, vid_b, pa, vid_a, delta)
                    rmse_mat[i, j] = r if np.isfinite(r) else 1e6
                    n_fr = 0
                    if pb in (person_kpts3d.get(vid_b) or {}):
                        src, dst = _get_aligned_flat(pb, vid_b, pa, vid_a, delta)
                        if src is not None:
                            n_fr = len(src) // 70
                    logging.info(
                        f"    {vid_b}:P{pb} → {vid_a}:P{pa}  frames={n_fr}  RMSE={r:.3f}"
                    )

            row_ind, col_ind = linear_sum_assignment(rmse_mat)

            # Step 5: accept pairs below threshold.
            accepted: list[tuple[int, int, float]] = []
            for r, c in zip(row_ind, col_ind):
                pb, pa = pids_b_s[r], pids_a_s[c]
                rmse_val = rmse_mat[r, c]
                if rmse_val < match_rmse_thr:
                    logging.info(
                        f"  {vid_a}:P{pa} ↔ {vid_b}:P{pb}  "
                        f"RMSE={rmse_val:.3f} → accepted"
                    )
                    accepted.append((pa, pb, float(1.0 - rmse_val / match_rmse_thr)))
                else:
                    logging.info(
                        f"  {vid_a}:P{pa} ↔ {vid_b}:P{pb}  "
                        f"RMSE={rmse_val:.3f} → rejected"
                    )

            return accepted, delta

        # ── All-pairs matching ─────────────────────────────────────────────────
        camera_pair_offsets: dict[str, int] = {}

        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1:]:
                matches, cam_delta = _match_with_combined_approach(vid_a, vid_b)
                camera_pair_offsets[f"{vid_a}→{vid_b}"] = cam_delta
                for pa, pb, sim in matches:
                    edges.append((sim, (vid_a, pa), (vid_b, pb)))
                    _union((vid_a, pa), (vid_b, pb))

        # ── Conflict resolution (identical to v1) ──────────────────────────────
        def _get_components() -> dict[tuple, list[tuple]]:
            comps: dict[tuple, list[tuple]] = {}
            for _vid in active_vids:
                for _pid in person_pids[_vid]:
                    _node = (_vid, _pid)
                    comps.setdefault(_find(_node), []).append(_node)
            return comps

        def _find_path_min_edge(src: tuple, dst: tuple) -> int:
            conflict_root = _find(src)
            adj: dict[tuple, list] = {}
            for ei, (s, na, nb) in enumerate(edges):
                if _find(na) != conflict_root:
                    continue
                adj.setdefault(na, []).append((nb, ei, s))
                adj.setdefault(nb, []).append((na, ei, s))
            from collections import deque
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
            path_ei = []
            cur = dst
            while prev[cur] is not None:
                par, ei = prev[cur]
                path_ei.append(ei)
                cur = par
            return min(path_ei, key=lambda ei: edges[ei][0])

        def _resolve_conflicts() -> None:
            for _ in range(len(edges) + 1):
                comps = _get_components()
                conflict_found = False
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
                                f"Scene {scene_id}: removed conflicting edge "
                                f"{rna[0]}/P{rna[1]} ↔ {rnb[0]}/P{rnb[1]} "
                                f"(sim={rs:.3f}) — same-video duplicate in {vid_id}"
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

        # ── Consolidation pass (identical to v1) ───────────────────────────────
        consolidation_threshold = max(0.15, cross_view_reid_threshold - 0.15)
        comps = _get_components()
        multi_view_comps = {
            root: members
            for root, members in comps.items()
            if len({v for v, _ in members}) >= 2
        }
        isolated = [
            (vid, pid)
            for root, members in comps.items()
            if root not in multi_view_comps
            for vid, pid in members
        ]
        comp_centroids: dict[tuple, tuple] = {}
        for root, members in multi_view_comps.items():
            app_vecs = [
                person_descs[vid][pid][0]
                for vid, pid in members
                if vid in person_descs and pid in person_descs.get(vid, {})
                and person_descs[vid][pid][0] is not None
            ]
            shape_vecs = [
                person_descs[vid][pid][1]
                for vid, pid in members
                if vid in person_descs and pid in person_descs.get(vid, {})
                and person_descs[vid][pid][1] is not None
            ]
            app_c: np.ndarray | None = None
            shape_c: np.ndarray | None = None
            if app_vecs:
                af = np.concatenate([f for f, _ in app_vecs])
                ac = np.concatenate([c for _, c in app_vecs])
                tw = ac.sum()
                m = (ac[:, None] * af).sum(0) / tw if tw > 0 else af.mean(0)
                n = np.linalg.norm(m)
                app_c = (m / n if n > 0 else m).astype(np.float32)
            if shape_vecs:
                m = np.mean(np.stack(shape_vecs), 0).astype(np.float32)
                n = np.linalg.norm(m)
                shape_c = m / n if n > 0 else m
            if app_c is not None or shape_c is not None:
                comp_centroids[root] = (app_c, shape_c)

        def _sim_to_centroid(feat, centroid, w_app, w_shape) -> float:
            sim, weight = 0.0, 0.0
            if feat[0] is not None and centroid[0] is not None:
                feats, confs = feat[0]
                tw = confs.sum()
                mf = (confs[:, None] * feats).sum(0) / tw if tw > 0 else feats.mean(0)
                sim += w_app * float(np.dot(mf, centroid[0]))
                weight += w_app
            if feat[1] is not None and centroid[1] is not None:
                sim += w_shape * float(np.dot(feat[1], centroid[1]))
                weight += w_shape
            return sim / weight if weight > 0 else 0.0

        consolidation_edges_added = False
        for vid, pid in isolated:
            if vid not in person_descs or pid not in person_descs.get(vid, {}):
                continue
            feat = person_descs[vid][pid]
            best_root, best_sim = None, -1.0
            for root, centroid in comp_centroids.items():
                if any(v == vid for v, _ in multi_view_comps[root]):
                    continue
                s = _sim_to_centroid(feat, centroid, appearance_weight, shape_weight)
                if s > best_sim:
                    best_sim, best_root = s, root
            if best_root is not None and best_sim >= consolidation_threshold:
                cm = multi_view_comps[best_root][0]
                edges.append((best_sim, (vid, pid), cm))
                _union((vid, pid), cm)
                consolidation_edges_added = True
                logging.info(
                    f"Scene {scene_id}: consolidation linked "
                    f"{vid}/P{pid} → component {best_root} (sim={best_sim:.3f})"
                )

        if consolidation_edges_added:
            for _ in range(len(edges) + 1):
                comps = _get_components()
                conflict_found = False
                for members in comps.values():
                    v2n: dict[str, list] = {}
                    for node in members:
                        v2n.setdefault(node[0], []).append(node)
                    for vid_id, nodes in v2n.items():
                        if len(nodes) < 2:
                            continue
                        conflict_found = True
                        cr = _find(nodes[0])
                        ws, wi = float("inf"), -1
                        for ei, (s, na, nb) in enumerate(edges):
                            if _find(na) == cr and s < ws:
                                ws, wi = s, ei
                        if wi >= 0:
                            edges.pop(wi)
                            parent.clear(); rank_uf.clear()
                            for _v in active_vids:
                                for _p in person_pids[_v]:
                                    _find((_v, _p))
                            for _s, _na, _nb in edges:
                                _union(_na, _nb)
                        break
                if not conflict_found:
                    break

        # ── Global ID assignment ────────────────────────────────────────────────
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

        logging.info(
            f"Scene {scene_id}: v2 → {len(comps)} global person(s), "
            f"{sum(len(m) for m in global_remap.values())} remap(s) "
            f"across {len(active_vids)} view(s)"
        )

        # ── Apply remaps (identical to v1) ─────────────────────────────────────
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"

            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.xviewtmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    logging.warning(
                        f"{vid_id}: cross-view remap — {dst.name} already exists, discarding"
                    )
                    tmp.unlink()
                else:
                    tmp.rename(dst)

            summary_path = body_dir / "body_params_summary.json"
            if summary_path.exists():
                with open(summary_path) as _f:
                    summary = json.load(_f)
                new_persons: dict[str, object] = {}
                for str_id, info in summary.get("persons", {}).items():
                    new_persons[str(remap.get(int(str_id), int(str_id)))] = info
                summary["persons"] = new_persons
                with open(summary_path, "w") as _f:
                    json.dump(summary, _f, indent=2)

            gallery_path = body_dir / "appearance_gallery.npz"
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                new_gallery = {}
                for k in gdata.files:
                    if k.endswith("_conf"):
                        old_pid = int(k[:-5])
                        new_gallery[f"{remap.get(old_pid, old_pid)}_conf"] = gdata[k]
                    else:
                        old_pid = int(k)
                        new_gallery[str(remap.get(old_pid, old_pid))] = gdata[k]
                np.savez(str(gallery_path), **new_gallery)

            with open(body_dir / "cross_view_id_mapping.json", "w") as _f:
                json.dump({str(k): v for k, v in remap.items()}, _f, indent=2)

            CrossVideoReidentifierV2.apply_reid_remap(vid_dir, remap)
            print(f"  {vid_id}: cross-view re-ID v2 — {len(remap)} remap(s): {remap}")

        # ── Scene-level summary ────────────────────────────────────────────────
        reid_summary = {
            "version": 2,
            "remaps": {
                vid_id: {str(k): v for k, v in remap.items()}
                for vid_id, remap in global_remap.items()
            },
            "camera_pair_offsets": camera_pair_offsets,
        }
        with open(scene_dir / "cross_view_reid.json", "w") as _f:
            json.dump(reid_summary, _f, indent=2)
