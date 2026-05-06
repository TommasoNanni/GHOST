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
from preprocessing.cross_view_reidentifier import CrossVideoReidentifier


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
        second_pass_threshold: float = 1.5,
    ):
        """
        Parameters
        ----------
        distance_threshold : float
            Maximum median distance (metres, world space) for two background
            persons to be considered the same individual (first pass).
        min_overlap_frames : int
            Minimum temporally-overlapping frames required for a proximity
            check to be attempted.
        second_pass_threshold : float
            Distance threshold for the second proximity pass, which runs only
            on displaced nodes and singletons (persons never matched in the
            first pass).  A larger value is appropriate here because these
            persons often have noisier 3-D estimates.
        """
        self.distance_threshold = distance_threshold
        self.min_overlap_frames = min_overlap_frames
        self.second_pass_threshold = second_pass_threshold

    @staticmethod
    def _merge_npz(src: Path, dst: Path) -> None:
        """Merge per-frame NPZ data from src into dst, keeping dst's file.

        All fields are concatenated along axis 0, sorted by frame_indices, and
        deduplicated — for duplicate frame indices the detection with higher mean
        joint confidence is kept.  src is deleted after the merge.
        """
        a = dict(np.load(str(dst), allow_pickle=False))
        b = dict(np.load(str(src), allow_pickle=False))

        merged: dict[str, np.ndarray] = {}
        for k in set(a) | set(b):
            if k not in a:
                merged[k] = b[k]
            elif k not in b:
                merged[k] = a[k]
            else:
                merged[k] = np.concatenate([a[k], b[k]], axis=0)

        # Sort by frame_indices
        fi = merged["frame_indices"]
        order = np.argsort(fi, kind="stable")
        for k in merged:
            if merged[k].ndim >= 1 and merged[k].shape[0] == len(fi):
                merged[k] = merged[k][order]

        # Deduplicate: keep the detection with higher mean joint confidence
        fi_sorted = merged["frame_indices"]
        conf = merged.get("pred_joint_confidence")
        keep = np.ones(len(fi_sorted), dtype=bool)
        if conf is not None:
            mean_conf = conf.mean(axis=-1)
            seen: dict[int, int] = {}
            for i, fidx in enumerate(fi_sorted.tolist()):
                fidx = int(fidx)
                if fidx in seen:
                    prev = seen[fidx]
                    if mean_conf[i] > mean_conf[prev]:
                        keep[prev] = False
                        seen[fidx] = i
                    else:
                        keep[i] = False
                else:
                    seen[fidx] = i

        for k in merged:
            if merged[k].ndim >= 1 and merged[k].shape[0] == len(fi_sorted):
                merged[k] = merged[k][keep]

        np.savez(str(dst), **merged)
        src.unlink()
        logging.info(
            f"  merged {src.name} → {dst.name} "
            f"({len(a['frame_indices'])} + {len(b['frame_indices'])} → {int(keep.sum())} frames)"
        )

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

        from preprocessing.camera_alignment import CameraAlignment
        alignment = CameraAlignment.load(alignment_path)

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
                    cam_t = d["pred_cam_t"].copy()
                    persons[pid] = {
                        "pred_cam_t":    cam_t,
                        "frame_indices": d["frame_indices"].copy(),
                    }
                    median_t = np.median(np.linalg.norm(cam_t, axis=-1))
                    logging.info(
                        f"Geometric ReID [{scene_id}]: loaded bg {vid_id}/P{pid} "
                        f"— {len(cam_t)} frames, median |cam_t|={median_t:.2f} m"
                    )
            if persons:
                background_data[vid_id] = persons
            else:
                logging.info(
                    f"Geometric ReID [{scene_id}]: {vid_id} — no background persons "
                    f"(fg_pids={sorted(fg_pids)}, all_pids={[int(p.stem.split('_')[1]) for p in sorted(body_dir.glob('person_*.npz'))]})"
                )

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
        for (vid_a, vid_b), (align_frames, R_arr, t_arr) in alignment.items():
            bg_a = background_data.get(vid_a, {})
            bg_b = background_data.get(vid_b, {})
            if not bg_a or not bg_b:
                logging.info(
                    f"Geometric ReID [{scene_id}]: skipping pair {vid_a}↔{vid_b} "
                    f"— no background persons in {'both cameras' if not bg_a and not bg_b else vid_a if not bg_a else vid_b}"
                )
                continue

            delta = offsets.get(vid_a, 0) - offsets.get(vid_b, 0)
            align_lut = {int(fi): idx for idx, fi in enumerate(align_frames)}
            logging.info(
                f"Geometric ReID [{scene_id}]: checking pair {vid_a}↔{vid_b} "
                f"— {len(align_frames)} alignment frames, delta={delta} frames, "
                f"bg_a={sorted(bg_a)}, bg_b={sorted(bg_b)}"
            )

            for pid_a, data_a in bg_a.items():
                frames_a = {int(fi): idx for idx, fi in enumerate(data_a["frame_indices"])}
                for pid_b, data_b in bg_b.items():
                    frames_b = {int(fi): idx for idx, fi in enumerate(data_b["frame_indices"])}

                    # Keep only frames that have a body detection in both cameras
                    # AND a Kabsch estimate for that frame.
                    common = [
                        (fi, frames_a[fi], frames_b[fi + delta])
                        for fi in frames_a
                        if (fi + delta) in frames_b and fi in align_lut
                    ]
                    if len(common) < self.min_overlap_frames:
                        logging.info(
                            f"Geometric ReID [{scene_id}]: {vid_a}/P{pid_a} ↔ {vid_b}/P{pid_b} "
                            f"— only {len(common)} overlapping frames (need ≥{self.min_overlap_frames}) → skipping"
                        )
                        continue

                    fi_list  = [c[0] for c in common]
                    rows_a   = [c[1] for c in common]
                    rows_b   = [c[2] for c in common]
                    pos_a = data_a["pred_cam_t"][rows_a]  # (N, 3)
                    pos_b = data_b["pred_cam_t"][rows_b]  # (N, 3)

                    # Per-frame transform: X_b = R @ X_a + t
                    align_idxs = [align_lut[fi] for fi in fi_list]
                    R_frames = R_arr[align_idxs]           # (N, 3, 3)
                    t_frames = t_arr[align_idxs]           # (N, 3)
                    pos_a_in_b = np.einsum("nij,nj->ni", R_frames, pos_a) + t_frames  # (N, 3)
                    dists = np.linalg.norm(pos_a_in_b - pos_b, axis=-1)
                    median_dist = float(np.median(dists))
                    mean_dist   = float(np.mean(dists))
                    min_dist    = float(np.min(dists))

                    logging.info(
                        f"Geometric ReID [{scene_id}]: {vid_a}/P{pid_a} ↔ {vid_b}/P{pid_b} "
                        f"— {len(common)} frames, dist: median={median_dist:.3f} m, "
                        f"mean={mean_dist:.3f} m, min={min_dist:.3f} m "
                        f"(threshold={self.distance_threshold} m)"
                        + (" → MERGE" if median_dist < self.distance_threshold else " → no merge")
                    )

                    if median_dist < self.distance_threshold:
                        _union((vid_a, pid_a), (vid_b, pid_b))

        # ── Collect merge groups ──────────────────────────────────────────────
        all_nodes: set[tuple[str, int]] = set(parent.keys()) | set(parent.values())
        groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for node in all_nodes:
            root = _find(node)
            groups.setdefault(root, []).append(node)

        # ── Resolve within-camera conflicts ──────────────────────────────────
        # A conflict is when ≥2 members of the same group share the same camera.
        # The member with better cross-camera evidence (lower mean median distance
        # to the other cameras in the group) is kept; the rest are evicted.
        # An evicted node is re-matched to another group if possible; otherwise
        # it is *displaced* — given a fresh free PID so its data is not lost and
        # it does not collide with the kept member.

        def _pair_dist(
            vid_x: str, pid_x: int, vid_y: str, pid_y: int
        ) -> "float | None":
            """Median 3-D distance (metres) between two background nodes."""
            if pid_x not in background_data.get(vid_x, {}):
                return None
            if pid_y not in background_data.get(vid_y, {}):
                return None
            dx    = background_data[vid_x][pid_x]
            dy    = background_data[vid_y][pid_y]
            delta = offsets.get(vid_x, 0) - offsets.get(vid_y, 0)
            if (vid_x, vid_y) in alignment:
                R, t = alignment[(vid_x, vid_y)]
                # X_y = R @ X_x + t  (row notation: px @ R.T + t)
                xform = lambda px, R=R, t=t: px @ R.T + t[None, :]
            elif (vid_y, vid_x) in alignment:
                R, t = alignment[(vid_y, vid_x)]
                # X_x = R @ X_y + t  →  X_y = R^T (X_x − t)  (row: (px−t) @ R)
                xform = lambda px, R=R, t=t: (px - t[None, :]) @ R
            else:
                return None
            fx = {int(fi): i for i, fi in enumerate(dx["frame_indices"])}
            fy = {int(fi): i for i, fi in enumerate(dy["frame_indices"])}
            common = [(fx[fi], fy[fi + delta]) for fi in fx if (fi + delta) in fy]
            if len(common) < self.min_overlap_frames:
                return None
            px = dx["pred_cam_t"][[r for r, _ in common]]
            py = dy["pred_cam_t"][[r for _, r in common]]
            return float(np.median(np.linalg.norm(xform(px) - py, axis=-1)))

        def _group_score(
            node: tuple[str, int], members: list[tuple[str, int]]
        ) -> float:
            """Mean median distance from node to all cross-camera group members."""
            dists = [
                d
                for ov, op in members
                if ov != node[0]
                for d in [_pair_dist(node[0], node[1], ov, op)]
                if d is not None
            ]
            return float(np.median(dists)) if dists else float("inf")

        # Collect used PIDs per camera so we can allocate fresh ones for displaced nodes
        used_pids: dict[str, set[int]] = {
            vid: set(background_data[vid].keys()) for vid in background_data
        }
        for vid, fps in foreground.items():
            used_pids.setdefault(vid, set()).update(fps)

        def _alloc_pid(vid: str) -> int:
            used    = used_pids.setdefault(vid, set())
            new_pid = max(used, default=-1) + 1
            used.add(new_pid)
            return new_pid

        # Work on a flat list of mutable sets; root is recomputed at remap time
        group_sets: list[set[tuple[str, int]]] = [
            set(members) for members in groups.values()
        ]
        # (vid, old_pid, new_pid): within-camera displacement — apply BEFORE group remaps
        displacements: list[tuple[str, int, int]] = []

        conflict_found = True
        while conflict_found:
            conflict_found = False
            for gi, group in enumerate(group_sets):
                cam_members: dict[str, list[tuple[str, int]]] = {}
                for node in group:
                    cam_members.setdefault(node[0], []).append(node)

                for vid_id, conflicting in cam_members.items():
                    if len(conflicting) <= 1:
                        continue

                    members_list = list(group)
                    scores = {n: _group_score(n, members_list) for n in conflicting}
                    best         = min(conflicting, key=lambda n: scores[n])
                    evicted_list = [n for n in conflicting if n != best]

                    logging.info(
                        f"Geometric ReID [{scene_id}]: within-camera conflict — "
                        f"{vid_id} has PIDs {[n[1] for n in conflicting]} in same group. "
                        f"Keeping P{best[1]} (score={scores[best]:.3f} m), "
                        f"evicting {[(n[1], f'{scores[n]:.3f} m') for n in evicted_list]}"
                    )

                    for ev in evicted_list:
                        group.discard(ev)

                        # Try to re-match to another group within the distance threshold
                        best_gi, best_dist = -1, float("inf")
                        for other_gi, other_group in enumerate(group_sets):
                            if other_gi == gi:
                                continue
                            if any(m[0] == ev[0] for m in other_group):
                                continue  # would create another conflict
                            dists = [
                                d
                                for ov, op in other_group
                                if ov != ev[0]
                                for d in [_pair_dist(ev[0], ev[1], ov, op)]
                                if d is not None
                            ]
                            if not dists:
                                continue
                            mean_d = float(np.mean(dists))
                            if mean_d < self.distance_threshold and mean_d < best_dist:
                                best_dist = mean_d
                                best_gi   = other_gi

                        if best_gi >= 0:
                            group_sets[best_gi].add(ev)
                            logging.info(
                                f"Geometric ReID [{scene_id}]: evicted {ev[0]}/P{ev[1]} "
                                f"re-matched to group {best_gi} "
                                f"(mean dist={best_dist:.3f} m)"
                            )
                        else:
                            new_pid = _alloc_pid(ev[0])
                            displacements.append((ev[0], ev[1], new_pid))
                            logging.info(
                                f"Geometric ReID [{scene_id}]: evicted {ev[0]}/P{ev[1]} "
                                f"displaced to new free ID P{new_pid} (no cross-camera match)"
                            )

                    conflict_found = True
                    break
                if conflict_found:
                    break

        # ── Second proximity pass: displaced nodes + singletons ───────────────
        # After conflict resolution some nodes were displaced (couldn't fit in
        # any group) and others never participated in a union at all (singletons).
        # Run a second proximity check among these using a looser threshold —
        # their 3-D estimates tend to be noisier because visual ReID may have
        # mislabelled them.

        nodes_in_groups: set[tuple[str, int]] = {n for g in group_sets for n in g}
        displaced_orig: set[tuple[str, int]] = {
            (vid_id, old_pid) for vid_id, old_pid, _ in displacements
        }
        second_pass_nodes: set[tuple[str, int]] = set()
        for vid_id, persons in background_data.items():
            for pid in persons:
                node = (vid_id, pid)
                if node not in nodes_in_groups and node not in displaced_orig:
                    second_pass_nodes.add(node)  # singleton
        second_pass_nodes |= displaced_orig

        if len(second_pass_nodes) >= 2:
            sp_parent: dict[tuple[str, int], tuple[str, int]] = {}

            def _sp_find(x: tuple[str, int]) -> tuple[str, int]:
                while sp_parent.get(x, x) != x:
                    sp_parent[x] = sp_parent.get(sp_parent[x], sp_parent[x])
                    x = sp_parent[x]
                return x

            def _sp_union(x: tuple[str, int], y: tuple[str, int]) -> None:
                rx, ry = _sp_find(x), _sp_find(y)
                if rx == ry:
                    return
                if rx[1] > ry[1]:
                    rx, ry = ry, rx
                sp_parent[ry] = rx

            sp_by_cam: dict[str, list[int]] = {}
            for vid_id, pid in second_pass_nodes:
                sp_by_cam.setdefault(vid_id, []).append(pid)

            sp_cams = sorted(sp_by_cam.keys())
            for i, vid_a in enumerate(sp_cams):
                for vid_b in sp_cams[i + 1:]:
                    for pid_a in sp_by_cam[vid_a]:
                        for pid_b in sp_by_cam[vid_b]:
                            d = _pair_dist(vid_a, pid_a, vid_b, pid_b)
                            if d is None:
                                continue
                            logging.info(
                                f"Geometric ReID [{scene_id}] (2nd pass): "
                                f"{vid_a}/P{pid_a} ↔ {vid_b}/P{pid_b} "
                                f"— median dist={d:.3f} m "
                                f"(threshold={self.second_pass_threshold} m)"
                                + (" → MERGE" if d < self.second_pass_threshold else " → no merge")
                            )
                            if d < self.second_pass_threshold:
                                _sp_union((vid_a, pid_a), (vid_b, pid_b))

            # Collect second-pass groups with ≥2 members
            sp_all: set[tuple[str, int]] = set(sp_parent.keys()) | set(sp_parent.values())
            sp_groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
            for node in sp_all:
                sp_groups.setdefault(_sp_find(node), []).append(node)

            sp_matched: set[tuple[str, int]] = set()
            for sp_group in sp_groups.values():
                if len(sp_group) <= 1:
                    continue

                # Resolve within-camera conflicts: keep the lower pid per camera
                cam_to_best: dict[str, tuple[str, int]] = {}
                for node in sp_group:
                    vid, pid = node
                    if vid not in cam_to_best or pid < cam_to_best[vid][1]:
                        cam_to_best[vid] = node
                clean_group = set(cam_to_best.values())

                # Log any conflicts that were silently resolved
                for node in sp_group:
                    if node not in clean_group:
                        logging.info(
                            f"Geometric ReID [{scene_id}] (2nd pass): "
                            f"within-camera conflict — dropped {node[0]}/P{node[1]} "
                            f"from group (lower-pid member kept)"
                        )

                group_sets.append(clean_group)
                sp_matched |= clean_group
                logging.info(
                    f"Geometric ReID [{scene_id}] (2nd pass): "
                    f"new group {sorted(clean_group)}"
                )

            # Displaced nodes that found a match no longer need displacement
            displacements = [
                (vid_id, old_pid, new_pid)
                for vid_id, old_pid, new_pid in displacements
                if (vid_id, old_pid) not in sp_matched
            ]

        has_cross_cam = any(len(g) > 1 for g in group_sets)
        if not has_cross_cam and not displacements:
            logging.info(f"Geometric ReID [{scene_id}]: no background merges found")
            return

        # ── Build per-camera remaps ───────────────────────────────────────────
        per_cam_remap: dict[str, dict[int, int]] = {}

        # Cross-camera group remaps (only groups with >1 member need a remap)
        for group in group_sets:
            if len(group) <= 1:
                continue
            root      = min(group, key=lambda n: n[1])
            _, new_pid = root
            for vid_id, old_pid in group:
                if old_pid == new_pid:
                    continue
                per_cam_remap.setdefault(vid_id, {})[old_pid] = new_pid

        # Displacement remaps (tracked separately so they are applied first)
        displacement_keys: set[tuple[str, int]] = set()
        for vid_id, old_pid, new_pid in displacements:
            per_cam_remap.setdefault(vid_id, {})[old_pid] = new_pid
            displacement_keys.add((vid_id, old_pid))

        # ── Apply remaps ──────────────────────────────────────────────────────
        for vid_id, remap in per_cam_remap.items():
            if not remap:
                continue
            vid_dir  = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"

            fg_pids_cam = foreground.get(vid_id, set())

            # Separate displacement remaps from group remaps.
            # Displacements must be applied FIRST so that when a group remap later
            # tries to rename A → B, the file at B has already been moved out of the
            # way (avoiding an incorrect merge of two different people's data).
            disp_here = {
                old: new for old, new in remap.items()
                if (vid_id, old) in displacement_keys
            }
            grp_here  = {
                old: new for old, new in remap.items()
                if (vid_id, old) not in displacement_keys
            }

            # 1) Displacements — target IDs are freshly allocated, no collisions
            tmp_disp: list[tuple[Path, Path]] = []
            for old_id, new_id in disp_here.items():
                if old_id in fg_pids_cam or new_id in fg_pids_cam:
                    logging.warning(
                        f"Geometric ReID [{scene_id}]: {vid_id} — "
                        f"skipping displacement {old_id}→{new_id}: foreground person involved"
                    )
                    continue
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.disp.npz"
                    src.rename(tmp)
                    tmp_disp.append((tmp, body_dir / f"person_{new_id}.npz"))
                    logging.info(
                        f"Geometric ReID [{scene_id}]: {vid_id} — "
                        f"displacing P{old_id} → P{new_id}"
                    )
            for tmp, dst in tmp_disp:
                tmp.rename(dst)

            # 2) Group remaps — split into merge (target exists) and rename (target free)
            merge_remap:  dict[int, int] = {}
            rename_remap: dict[int, int] = {}
            for old_id, new_id in grp_here.items():
                if new_id in fg_pids_cam or old_id in fg_pids_cam:
                    logging.warning(
                        f"Geometric ReID [{scene_id}]: {vid_id} — "
                        f"skipping {old_id}→{new_id}: foreground person involved"
                    )
                    continue
                if (body_dir / f"person_{new_id}.npz").exists():
                    merge_remap[old_id] = new_id
                else:
                    rename_remap[old_id] = new_id

            # Merge: concatenate frames of old track into existing target file.
            for old_id, new_id in merge_remap.items():
                src = body_dir / f"person_{old_id}.npz"
                dst = body_dir / f"person_{new_id}.npz"
                if src.exists():
                    logging.info(
                        f"Geometric ReID [{scene_id}]: {vid_id} — "
                        f"merging {old_id}→{new_id} (target already exists)"
                    )
                    self._merge_npz(src, dst)

            # Rename via tmp to avoid clobbering when IDs swap.
            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in rename_remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.geotmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    # Collision introduced by a swap within this remap — merge.
                    logging.info(
                        f"Geometric ReID [{scene_id}]: {vid_id} — "
                        f"merging {tmp.name} → {dst.name} (swap collision)"
                    )
                    self._merge_npz(tmp, dst)
                else:
                    tmp.rename(dst)

            # Full remap (displacements + group remaps) for metadata updates
            full_remap = {**disp_here, **merge_remap, **rename_remap}

            summary_path = body_dir / "body_params_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                new_persons: dict[str, object] = {}
                for str_id, info in summary.get("persons", {}).items():
                    new_persons[str(full_remap.get(int(str_id), int(str_id)))] = info
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
                        new_key = f"{full_remap.get(old_pid, old_pid)}_conf"
                    else:
                        old_pid = int(k)
                        new_key = str(full_remap.get(old_pid, old_pid))
                    new_gallery[new_key] = gdata[k]
                np.savez(str(gallery_path), **new_gallery)

            CrossVideoReidentifier.apply_reid_remap(vid_dir, full_remap)
            logging.info(f"  {vid_id}: geometric ReID — {len(full_remap)} remap(s): {full_remap}")
