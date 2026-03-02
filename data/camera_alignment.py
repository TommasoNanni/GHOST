"""Camera alignment utilities.

This module provides a compact `CameraAlignment` class that estimates
relative camera poses from per-view body outputs and offers simple I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np



class CameraAlignment:
    """Estimate and persist relative camera poses using Kabsch."""

    @staticmethod
    def _kabsch(pts_a: np.ndarray, pts_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if pts_a.shape != pts_b.shape or pts_a.ndim != 2 or pts_a.shape[1] != 3:
            raise ValueError("pts_a and pts_b must both be (N, 3)")
        if len(pts_a) < 3:
            raise ValueError("Need at least 3 point pairs for rotation")

        mu_a = pts_a.mean(axis=0)
        mu_b = pts_b.mean(axis=0)
        A_c = pts_a - mu_a
        B_c = pts_b - mu_b
        H = A_c.T @ B_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = mu_b - R @ mu_a
        return R, t

    @staticmethod
    def _load_person_npz(body_dir: Path, person_id: int) -> dict[str, np.ndarray] | None:
        path = body_dir / f"person_{person_id}.npz"
        if not path.exists():
            return None
        with np.load(str(path)) as f:
            return dict(f)

    @staticmethod
    def _absolute_joints(data: dict[str, np.ndarray], row: int) -> np.ndarray | None:
        cam_t = data.get("pred_cam_t")
        kpts = data.get("pred_keypoints_3d")
        if cam_t is None:
            return None
        root = cam_t[row]
        if kpts is not None:
            joints = kpts[row]
            abs_joints = joints + root[None, :]
            valid = np.any(abs_joints != 0, axis=-1)
            if valid.sum() >= 1:
                return abs_joints[valid]
        return root[None, :]

    def estimate(
        self,
        video_dirs: dict[str, Path],
        min_correspondences: int = 30,
    ) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        video_ids = list(video_dirs.keys())
        video_persons: dict[str, dict[int, dict[str, np.ndarray]]] = {}

        for vid_id, vid_dir in video_dirs.items():
            body_dir = Path(vid_dir) / "body_data"
            summary_path = body_dir / "body_params_summary.json"
            if not summary_path.exists():
                logging.warning(f"Camera alignment: {vid_id}: missing summary, skipping")
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            persons: dict[int, dict[str, np.ndarray]] = {}
            for k in summary.get("persons", {}):
                pid = int(k)
                data = self._load_person_npz(body_dir, pid)
                if data is not None and "frame_indices" in data:
                    persons[pid] = data
            if persons:
                video_persons[vid_id] = persons

        active_vids = [v for v in video_ids if v in video_persons]
        results: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1 :]:
                persons_a = video_persons[vid_a]
                persons_b = video_persons[vid_b]
                shared_pids = sorted(set(persons_a) & set(persons_b))
                if not shared_pids:
                    logging.info(f"Camera alignment: {vid_a} ↔ {vid_b}: no shared persons")
                    continue

                pts_a_parts: list[np.ndarray] = []
                pts_b_parts: list[np.ndarray] = []

                for pid in shared_pids:
                    data_a = persons_a[pid]
                    data_b = persons_b[pid]
                    frames_a = {int(fi): idx for idx, fi in enumerate(data_a["frame_indices"])}
                    frames_b = {int(fi): idx for idx, fi in enumerate(data_b["frame_indices"])}
                    common_frames = sorted(set(frames_a) & set(frames_b))
                    if not common_frames:
                        continue
                    for fi in common_frames:
                        row_a = frames_a[fi]
                        row_b = frames_b[fi]
                        joints_a = self._absolute_joints(data_a, row_a)
                        joints_b = self._absolute_joints(data_b, row_b)
                        if joints_a is None or joints_b is None:
                            continue
                        n = min(len(joints_a), len(joints_b))
                        if n < 1:
                            continue
                        pts_a_parts.append(joints_a[:n])
                        pts_b_parts.append(joints_b[:n])

                if not pts_a_parts:
                    logging.warning(f"Camera alignment: {vid_a} ↔ {vid_b}: no common frames")
                    continue

                pts_a = np.concatenate(pts_a_parts, axis=0).astype(np.float64)
                pts_b = np.concatenate(pts_b_parts, axis=0).astype(np.float64)
                n_pairs = len(pts_a)
                if n_pairs < min_correspondences:
                    logging.warning(
                        f"Camera alignment: {vid_a} ↔ {vid_b}: only {n_pairs} correspondences (< {min_correspondences}), skipping"
                    )
                    continue

                try:
                    R, t = self._kabsch(pts_a, pts_b)
                except ValueError as e:
                    logging.warning(f"Camera alignment: {vid_a} ↔ {vid_b}: Kabsch failed: {e}")
                    continue

                residuals = pts_b - (pts_a @ R.T + t[None, :])
                rmse = float(np.sqrt((residuals ** 2).sum(axis=-1).mean()))
                logging.info(
                    f"Camera alignment: {vid_a} ↔ {vid_b}: {n_pairs} correspondences, RMSE = {rmse:.4f} m, |t| = {np.linalg.norm(t):.3f} m"
                )
                results[(vid_a, vid_b)] = (R, t)

        return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

    @staticmethod
    def save(alignment: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for (vid_a, vid_b), (R, t) in alignment.items():
            prefix = f"{vid_a}__to__{vid_b}"
            arrays[f"{prefix}__R"] = R
            arrays[f"{prefix}__t"] = t
        path = output_dir / "camera_alignment.npz"
        np.savez(str(path), **arrays)
        logging.info(f"Camera alignment: saved {len(alignment)} pair(s) to {path}")
        return path


    @staticmethod
    def load(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        results: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        with np.load(str(path)) as f:
            prefixes: set[str] = set()
            for key in f.files:
                if key.endswith("__R"):
                    prefixes.add(key[:-3])
            for prefix in sorted(prefixes):
                parts = prefix.split("__to__")
                if len(parts) != 2:
                    continue
                vid_a, vid_b = parts
                R = f[f"{prefix}__R"]
                t = f[f"{prefix}__t"]
                results[(vid_a, vid_b)] = (R, t)
        return results


    @staticmethod
    def relative_pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def camera_center_in_A(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return -(R.T @ t)


# Compatibility wrappers (small convenience functions)

def estimate_relative_camera_poses(
    video_dirs: dict[str, Path], min_correspondences: int = 30
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    return CameraAlignment().estimate(video_dirs, min_correspondences)


def save_camera_alignment(
    alignment: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], output_dir: Path
) -> Path:
    return CameraAlignment.save(alignment, output_dir)


def load_camera_alignment(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    return CameraAlignment.load(path)


def relative_pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return CameraAlignment.relative_pose_to_matrix(R, t)


def camera_center_in_A(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return CameraAlignment.camera_center_in_A(R, t)
