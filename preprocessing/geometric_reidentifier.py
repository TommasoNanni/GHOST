"""Geometric post-ReID using 3D world-space proximity.

After temporal sync and camera alignment, background persons that are
spatially close at the same physical time across cameras are merged into
the same global ID.  This is a second-pass correction on top of the
appearance-based cross-view ReID.

Prerequisites (all produced by earlier pipeline stages):
    - ``cross_view_reid.json``   — foreground set + remaps
    - ``temporal_offsets.json``  — per-camera global start frame
    - ``camera_alignment.npz``   — pairwise (R, t) transforms
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from data.video_dataset import Scene
from preprocessing.parameters_extraction import ParametersExtractor


class GeometricReidentifier:
    """Correct background-person ReID using 3D world-space proximity.

    For each aligned camera pair, background persons whose root positions
    (``pred_cam_t``) are within ``distance_threshold`` metres in world space
    at overlapping times are merged into the same global ID.
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        min_overlap_frames: int = 10,
    ):
        """
        Parameters
        ----------
        distance_threshold : float
            Maximum median distance (metres, world space) for two background
            persons to be considered the same individual.
        min_overlap_frames : int
            Minimum temporally-overlapping frames required for a proximity
            check to be attempted.
        """
        self.distance_threshold = distance_threshold
        self.min_overlap_frames = min_overlap_frames

    def reidentify(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
    ) -> None:
        """Run geometric post-ReID and apply remaps in-place.

        Loads prerequisites from the scene-level output directory, finds
        background persons that are spatially close, and renames their
        ``body_data/`` files + updates masks/JSON to use a consistent global ID.
        """
        scene_id  = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent

        reid_path      = scene_dir / "cross_view_reid.json"
        offsets_path   = scene_dir / "temporal_offsets.json"
        alignment_path = scene_dir / "camera_alignment.npz"

        if not reid_path.exists():
            logging.warning(f"Geometric ReID [{scene_id}]: cross_view_reid.json missing — skipping")
            return
        if not alignment_path.exists():
            logging.warning(f"Geometric ReID [{scene_id}]: camera_alignment.npz missing — skipping")
            return

        # ── Load prerequisites ────────────────────────────────────────────────
        with open(reid_path) as f:
            reid_data = json.load(f)
        foreground: dict[str, set[int]] = {
            vid: {int(p) for p in pids}
            for vid, pids in reid_data.get("foreground", {}).items()
        }

        offsets: dict[str, int] = {}
        if offsets_path.exists():
            with open(offsets_path) as f:
                offsets = {k: int(v) for k, v in json.load(f).items()}
        else:
            logging.warning(
                f"Geometric ReID [{scene_id}]: temporal_offsets.json missing "
                f"— assuming all cameras are synchronised"
            )

        # Load pairwise (R, t): X_b = R @ X_a + t
        alignment: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        with np.load(str(alignment_path)) as f:
            prefixes: set[str] = {k[:-3] for k in f.files if k.endswith("__R")}
            for prefix in sorted(prefixes):
                parts = prefix.split("__to__")
                if len(parts) != 2:
                    continue
                vid_a, vid_b = parts
                alignment[(vid_a, vid_b)] = (f[f"{prefix}__R"].copy(), f[f"{prefix}__t"].copy())

        # ── Load background persons ───────────────────────────────────────────
        # background_data[vid_id][pid] = {"pred_cam_t": (T,3), "frame_indices": (T,)}
        background_data: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for vid_id, vid_dir in video_dirs.items():
            body_dir = Path(vid_dir) / "body_data"
            fg_pids  = foreground.get(vid_id, set())
            persons: dict[int, dict[str, np.ndarray]] = {}
            for npz_path in sorted(body_dir.glob("person_*.npz")):
                pid = int(npz_path.stem.split("_")[1])
                if pid in fg_pids:
                    continue
                with np.load(str(npz_path)) as d:
                    if "pred_cam_t" not in d or "frame_indices" not in d:
                        continue
                    persons[pid] = {
                        "pred_cam_t":    d["pred_cam_t"].copy(),
                        "frame_indices": d["frame_indices"].copy(),
                    }
            if persons:
                background_data[vid_id] = persons

        # ── Union-Find ────────────────────────────────────────────────────────
        parent: dict[tuple[str, int], tuple[str, int]] = {}

        def _find(x: tuple[str, int]) -> tuple[str, int]:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def _union(x: tuple[str, int], y: tuple[str, int]) -> None:
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return
            # smaller global pid becomes the root (consistent with ReID convention)
            if rx[1] > ry[1]:
                rx, ry = ry, rx
            parent[ry] = rx

        # ── Proximity check for each aligned camera pair ──────────────────────
        for (vid_a, vid_b), (R, t) in alignment.items():
            bg_a = background_data.get(vid_a, {})
            bg_b = background_data.get(vid_b, {})
            if not bg_a or not bg_b:
                continue

            # Frame fi_a in cam_a is at the same physical time as
            # frame (fi_a + delta) in cam_b.
            delta = offsets.get(vid_a, 0) - offsets.get(vid_b, 0)

            for pid_a, data_a in bg_a.items():
                frames_a = {int(fi): idx for idx, fi in enumerate(data_a["frame_indices"])}
                for pid_b, data_b in bg_b.items():
                    frames_b = {int(fi): idx for idx, fi in enumerate(data_b["frame_indices"])}

                    common = [
                        (frames_a[fi], frames_b[fi + delta])
                        for fi in frames_a
                        if (fi + delta) in frames_b
                    ]
                    if len(common) < self.min_overlap_frames:
                        continue

                    rows_a = [r for r, _ in common]
                    rows_b = [r for _, r in common]
                    pos_a = data_a["pred_cam_t"][rows_a]  # (N, 3)
                    pos_b = data_b["pred_cam_t"][rows_b]  # (N, 3)

                    # Transform cam_a positions into cam_b space: X_b = R @ X_a + t
                    pos_a_in_b  = pos_a @ R.T + t[None, :]  # (N, 3)
                    median_dist = float(np.median(np.linalg.norm(pos_a_in_b - pos_b, axis=-1)))

                    if median_dist < self.distance_threshold:
                        logging.info(
                            f"Geometric ReID [{scene_id}]: "
                            f"{vid_a}/P{pid_a} ↔ {vid_b}/P{pid_b} — "
                            f"median dist={median_dist:.3f} m over {len(common)} frames → merging"
                        )
                        _union((vid_a, pid_a), (vid_b, pid_b))

        # ── Collect merge groups ──────────────────────────────────────────────
        all_nodes: set[tuple[str, int]] = set(parent.keys()) | set(parent.values())
        groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for node in all_nodes:
            root = _find(node)
            groups.setdefault(root, []).append(node)

        if not groups:
            logging.info(f"Geometric ReID [{scene_id}]: no background merges found")
            return

        # ── Build per-camera remaps ───────────────────────────────────────────
        per_cam_remap: dict[str, dict[int, int]] = {}
        for root, members in groups.items():
            _, new_pid = root
            for vid_id, old_pid in members:
                if old_pid == new_pid:
                    continue
                per_cam_remap.setdefault(vid_id, {})[old_pid] = new_pid

        # ── Apply remaps ──────────────────────────────────────────────────────
        for vid_id, remap in per_cam_remap.items():
            if not remap:
                continue
            vid_dir  = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"

            # Skip entries where the target ID already exists in this camera —
            # that means a different physical person already holds that global ID
            # and overwriting would create a collision.
            safe_remap: dict[int, int] = {}
            for old_id, new_id in remap.items():
                if (body_dir / f"person_{new_id}.npz").exists():
                    logging.info(
                        f"Geometric ReID [{scene_id}]: {vid_id} — "
                        f"skipping {old_id}→{new_id}: person_{new_id}.npz already exists"
                    )
                else:
                    safe_remap[old_id] = new_id
            if not safe_remap:
                continue
            remap = safe_remap

            # Rename via tmp to avoid clobbering when IDs swap.
            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.geotmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    logging.warning(
                        f"{vid_id}: geometric ReID — {dst.name} already exists, "
                        f"discarding duplicate from {tmp.name}"
                    )
                    tmp.unlink()
                else:
                    tmp.rename(dst)

            summary_path = body_dir / "body_params_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                new_persons: dict[str, object] = {}
                for str_id, info in summary.get("persons", {}).items():
                    new_persons[str(remap.get(int(str_id), int(str_id)))] = info
                summary["persons"] = new_persons
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

            gallery_path = body_dir / "appearance_gallery.npz"
            if gallery_path.exists():
                gdata = np.load(str(gallery_path))
                new_gallery: dict[str, np.ndarray] = {}
                for k in gdata.files:
                    if k.endswith("_conf"):
                        old_pid = int(k[:-5])
                        new_key = f"{remap.get(old_pid, old_pid)}_conf"
                    else:
                        old_pid = int(k)
                        new_key = str(remap.get(old_pid, old_pid))
                    new_gallery[new_key] = gdata[k]
                np.savez(str(gallery_path), **new_gallery)

            ParametersExtractor._apply_reid_remap(vid_dir, remap)
            logging.info(f"  {vid_id}: geometric ReID — {len(remap)} remap(s): {remap}")
