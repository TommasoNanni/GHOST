"""Cross-view person re-identification for multi-camera scenes.

Assigns consistent global person IDs across all camera views using a hybrid
appearance + shape + pose descriptor and Hungarian matching.  A second geometric
validation pass uses inter-person distance profiles to veto spurious matches.

Also exposes :meth:`CrossVideoReidentifier.apply_reid_remap`, the shared
utility that rewrites ``mask_data.npz`` and ``json_data/`` after any ReID step.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.video_dataset import Scene


class CrossVideoReidentifier:
    """Assign consistent global person IDs across camera views in a scene.

    Parameters
    ----------
    threshold : float
        Minimum hybrid cosine similarity to accept a cross-view match.
    appearance_weight : float
        Weight for the DINOv3 appearance component of the hybrid descriptor.
    shape_weight : float
        Weight for the SMPL-X shape (beta) component.
    pose_weight : float
        Weight for the canonical joint sequence (root-relative, global-orient-free).
    """

    _THRESHOLD: float = 0.4
    _APPEARANCE_WEIGHT: float = 0.5
    _SHAPE_WEIGHT: float = 0.2
    _POSE_WEIGHT: float = 0.3

    def __init__(
        self,
        threshold: float | None = None,
        appearance_weight: float | None = None,
        shape_weight: float | None = None,
        pose_weight: float | None = None,
    ):
        self.threshold = threshold if threshold is not None else self._THRESHOLD
        self.appearance_weight = appearance_weight if appearance_weight is not None else self._APPEARANCE_WEIGHT
        self.shape_weight = shape_weight if shape_weight is not None else self._SHAPE_WEIGHT
        self.pose_weight = pose_weight if pose_weight is not None else self._POSE_WEIGHT

    def match_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
    ) -> set[tuple[str, int]]:
        """Assign consistent global person IDs across all camera views in a scene.

        Returns
        -------
        foreground_nodes : set of (video_id, person_id) tuples
            Foreground persons after ReID (post-remap global IDs).  Empty set
            if ReID was skipped (already done).  Callers should pass this to
            temporal sync and camera alignment.
        """
        scene_id = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent

        if (scene_dir / "cross_view_reid.json").exists():
            logging.info(f"  Scene {scene_id}: cross-view ReID already done, skipping")
            try:
                with open(scene_dir / "cross_view_reid.json") as _f:
                    _saved = json.load(_f)
                _fg_raw = _saved.get("foreground", {})
                return {(vid, int(pid)) for vid, pids in _fg_raw.items() for pid in pids}
            except Exception:
                return set()

        return self._cross_view_reid_core(
            video_dirs=video_dirs,
            scene_id=scene_id,
            scene_dir=scene_dir,
            cross_view_reid_threshold=self.threshold,
            appearance_weight=self.appearance_weight,
            shape_weight=self.shape_weight,
            pose_weight=self.pose_weight,
        )

    @staticmethod
    def apply_reid_remap(
        video_dir: Path,
        id_remap: dict[int, int],
        gpu_label: str = "",
    ) -> None:
        """Rewrite mask_data.npz and json_data/*.json with re-identified IDs.

        Uses the same stream-remap logic as PersonSegmenter._apply_id_mapping
        so that the segmentation files stay consistent with body_data/ after
        within-video re-identification.

        Parameters
        ----------
        video_dir : Path
            Root output directory for one video (contains mask_data.npz and
            json_data/).
        id_remap : dict[int, int]
            Mapping of raw SAM2 ID → canonical person ID discovered during
            body parameter estimation.
        """
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

    @staticmethod
    def _cross_view_reid_core(
        video_dirs: dict[str, Path],
        scene_id: str,
        scene_dir: Path,
        cross_view_reid_threshold: float,
        appearance_weight: float,
        shape_weight: float,
        pose_weight: float,
    ) -> set[tuple[str, int]]:
        """Cross-view re-identification logic. Called by match_across_views.

        Returns
        -------
        foreground_nodes : set of (video_id, person_id) tuples
            Persons whose cross-view component spans at least (max_cameras - 1)
            distinct cameras.  These are the reliable subjects of the scene and
            should be used exclusively for temporal sync and camera alignment.
        """
        video_ids = list(video_dirs.keys())
        person_descs: dict[str, dict[int, tuple[np.ndarray | None, np.ndarray | None]]] = {}
        person_pids: dict[str, list[int]] = {}
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
                feat_keys = [k for k in gdata.files if not k.endswith("_conf")]
                for k in feat_keys:
                    pid_key = int(k)
                    conf_key = f"{k}_conf"
                    feats = gdata[k]
                    confs = gdata[conf_key] if conf_key in gdata.files else np.ones(len(feats), dtype=np.float32)
                    app_gallery[pid_key] = (feats, confs)

            descs: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
            for pid in pids:
                npz_path = body_dir / f"person_{pid}.npz"
                if not npz_path.exists():
                    continue

                shape_feat: np.ndarray | None = None
                with np.load(str(npz_path)) as pdata:
                    if "smplx_betas" in pdata:
                        shape_vecs = pdata["smplx_betas"]
                        if len(shape_vecs) > 0:
                            conf = pdata.get("pred_joint_confidence")
                            if conf is not None and len(conf) == len(shape_vecs):
                                frame_conf = np.mean(conf, axis=-1).astype(np.float32)
                                total_conf = frame_conf.sum()
                                if total_conf > 0:
                                    shape_med = (frame_conf[:, None] * shape_vecs).sum(0) / total_conf
                                else:
                                    shape_med = np.median(shape_vecs, axis=0).astype(np.float32)
                            else:
                                shape_med = np.median(shape_vecs, axis=0).astype(np.float32)
                            norm = np.linalg.norm(shape_med)
                            shape_feat = shape_med / norm if norm > 0 else shape_med
                    if "pred_cam_t" in pdata and "frame_indices" in pdata:
                        person_cam_t.setdefault(vid_id, {})[pid] = (
                            pdata["frame_indices"].copy(),
                            pdata["pred_cam_t"].copy(),
                        )

                pose_feat: tuple[np.ndarray, np.ndarray] | None = None
                with np.load(str(npz_path)) as pdata2:
                    if "pred_keypoints_3d" in pdata2 and "smplx_global_orient" in pdata2:
                        from scipy.spatial.transform import Rotation as _Rot
                        kps = pdata2["pred_keypoints_3d"].astype(np.float32)
                        gorient = pdata2["smplx_global_orient"].astype(np.float32)
                        conf_kps = pdata2.get("pred_joint_confidence")
                        N_kps = len(kps)
                        if N_kps > 0:
                            kps_rel = kps - kps[:, 0:1, :]
                            rot_mats = _Rot.from_rotvec(gorient).inv().as_matrix()
                            kps_canon = np.einsum('nij,nkj->nki', rot_mats, kps_rel).astype(np.float32)
                            pose_vecs = kps_canon.reshape(N_kps, -1)
                            norms_p = np.linalg.norm(pose_vecs, axis=1, keepdims=True)
                            pose_vecs = np.where(norms_p > 1e-6, pose_vecs / norms_p, pose_vecs)
                            frame_conf_p = (
                                np.mean(conf_kps, axis=-1).astype(np.float32)
                                if conf_kps is not None and len(conf_kps) == N_kps
                                else np.ones(N_kps, dtype=np.float32)
                            )
                            pose_feat = (pose_vecs, frame_conf_p)

                app_feat: tuple[np.ndarray, np.ndarray] | None = app_gallery.get(pid)
                if app_feat is None and shape_feat is None and pose_feat is None:
                    continue
                descs[pid] = (app_feat, shape_feat, pose_feat)

            if descs:
                person_descs[vid_id] = descs
                person_pids[vid_id] = sorted(descs.keys())

        active_vids = [v for v in video_ids if v in person_descs]

        if len(active_vids) < 2:
            logging.info(
                f"Scene {scene_id}: fewer than 2 videos with descriptors, "
                f"skipping cross-view re-ID"
            )
            return

        # ── Union-Find with edge tracking ─────────────────────────────
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

        # ── All-pairs Hungarian matching ───────────────────────────────
        def _xcorr_sim(
            feats_a: np.ndarray,
            feats_b: np.ndarray,
            min_overlap: int = 30,
        ) -> tuple[float, int]:
            N, M = len(feats_a), len(feats_b)
            # Remove the temporal mean (DC component) from each sequence so that
            # the similarity captures pose *variation* rather than absolute pose.
            # Without this, all per-frame L2-normalised pose vectors cluster near
            # the "average human skeleton" direction, giving cosine ~0.93 for every
            # person pair and making xcorr non-discriminative.
            fa = feats_a - feats_a.mean(axis=0)
            fb = feats_b - feats_b.mean(axis=0)
            na = np.linalg.norm(fa, axis=1, keepdims=True)
            nb = np.linalg.norm(fb, axis=1, keepdims=True)
            fa = np.where(na > 1e-6, fa / na, fa)
            fb = np.where(nb > 1e-6, fb / nb, fb)
            S = fa @ fb.T
            best_score, best_offset = -1.0, 0
            for tau in range(-(M - 1), N):
                diag = np.diagonal(S, offset=-tau)
                if len(diag) < min_overlap:
                    continue
                score = float(diag.mean())
                if score > best_score:
                    best_score = score
                    best_offset = tau
            return float(max(0.0, best_score)), best_offset

        def _chamfer_sim(
            feats_a: np.ndarray,
            confs_a: np.ndarray,
            feats_b: np.ndarray,
            confs_b: np.ndarray,
        ) -> float:
            S = feats_a @ feats_b.T
            return 0.5 * (float(S.max(axis=1).mean()) + float(S.max(axis=0).mean()))

        def _weighted_sim_mat(
            pids_a: list[int],
            pids_b: list[int],
            descs_a: dict[int, tuple],
            descs_b: dict[int, tuple],
            w_app: float,
            w_shape: float,
            w_pose: float,
            vid_a: str = "",
            vid_b: str = "",
        ) -> np.ndarray:
            Na, Nb = len(pids_a), len(pids_b)
            sim_mat    = np.zeros((Na, Nb), dtype=np.float32)
            weight_mat = np.zeros((Na, Nb), dtype=np.float32)

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
                    sim_mat[i, j]    += w_app * s
                    weight_mat[i, j] += w_app

            shape_a = [descs_a[p][1] for p in pids_a]
            shape_b = [descs_b[p][1] for p in pids_b]
            mask_a = np.array([f is not None for f in shape_a], dtype=np.float32)
            mask_b = np.array([f is not None for f in shape_b], dtype=np.float32)
            if mask_a.any() and mask_b.any():
                dim = next(f for f in shape_a if f is not None).shape[0]
                zero = np.zeros(dim, dtype=np.float32)
                mat_a = np.stack([f if f is not None else zero for f in shape_a])
                mat_b = np.stack([f if f is not None else zero for f in shape_b])
                shape_sim = mat_a @ mat_b.T
                shape_w   = np.outer(mask_a, mask_b) * w_shape
                sim_mat    += shape_w * shape_sim
                weight_mat += shape_w

            for i, pa in enumerate(pids_a):
                pose_a = descs_a[pa][2]
                if pose_a is None:
                    continue
                feats_a, confs_a = pose_a
                for j, pb in enumerate(pids_b):
                    pose_b = descs_b[pb][2]
                    if pose_b is None:
                        continue
                    feats_b, confs_b = pose_b
                    s, offset = _xcorr_sim(feats_a, feats_b)
                    logging.info(
                        f"  pose xcorr  {vid_a}:P{pa} vs {vid_b}:P{pb}"
                        f"  sim={s:.3f}  best_offset={offset:+d} frames"
                    )
                    sim_mat[i, j]    += w_pose * s
                    weight_mat[i, j] += w_pose

            return np.where(weight_mat > 0, sim_mat / weight_mat, 0.0)

        # ── Geometric consistency helper ───────────────────────────────
        GEO_CORR_THRESHOLD = 0.3
        GEO_MIN_OVERLAP    = 30

        def _geo_consistency(
            frames_i: np.ndarray, cam_t_i: np.ndarray,
            frames_anc_a: np.ndarray, cam_t_anc_a: np.ndarray,
            frames_j: np.ndarray, cam_t_j: np.ndarray,
            frames_anc_b: np.ndarray, cam_t_anc_b: np.ndarray,
        ) -> float | None:
            common_a = np.intersect1d(frames_i, frames_anc_a)
            common_b = np.intersect1d(frames_j, frames_anc_b)
            if len(common_a) < GEO_MIN_OVERLAP or len(common_b) < GEO_MIN_OVERLAP:
                return None
            d_a = np.linalg.norm(
                cam_t_i[np.searchsorted(frames_i, common_a)] -
                cam_t_anc_a[np.searchsorted(frames_anc_a, common_a)], axis=1)
            d_b = np.linalg.norm(
                cam_t_j[np.searchsorted(frames_j, common_b)] -
                cam_t_anc_b[np.searchsorted(frames_anc_b, common_b)], axis=1)

            def _norm(s: np.ndarray) -> np.ndarray:
                std = s.std()
                return (s - s.mean()) / std if std > 1e-6 else np.zeros_like(s)

            d_a, d_b = _norm(d_a), _norm(d_b)
            from scipy.signal import correlate as _correlate
            cc   = _correlate(d_a, d_b, mode="full")
            norm = float(np.linalg.norm(d_a) * np.linalg.norm(d_b))
            return float(cc.max() / norm) if norm > 1e-9 else None

        def _geo_score_for_match(
            pa: int, pb: int,
            vid_a: str, vid_b: str,
            anchors: list[tuple[int, int]],
        ) -> float | None:
            cam_t_a = person_cam_t.get(vid_a, {})
            cam_t_b = person_cam_t.get(vid_b, {})
            if pa not in cam_t_a or pb not in cam_t_b:
                return None
            fi_a, ct_a = cam_t_a[pa]
            fi_b, ct_b = cam_t_b[pb]
            scores = []
            for anc_a, anc_b in anchors:
                if anc_a not in cam_t_a or anc_b not in cam_t_b:
                    continue
                fa_a, ct_anc_a = cam_t_a[anc_a]
                fa_b, ct_anc_b = cam_t_b[anc_b]
                g = _geo_consistency(fi_a, ct_a, fa_a, ct_anc_a,
                                     fi_b, ct_b, fa_b, ct_anc_b)
                if g is not None:
                    scores.append(g)
            return max(scores) if scores else None

        def _match_with_geo_validation(vid_a: str, vid_b: str) -> list[tuple[int, int, float]]:
            pids_a = person_pids[vid_a]
            pids_b = person_pids[vid_b]
            sim_mat = _weighted_sim_mat(
                pids_a, pids_b,
                person_descs[vid_a], person_descs[vid_b],
                appearance_weight, shape_weight, pose_weight,
                vid_a=vid_a, vid_b=vid_b,
            )
            row_ind, col_ind = linear_sum_assignment(1.0 - sim_mat)
            assignment: list[tuple[int, int, float]] = [
                (pids_a[r], pids_b[c], float(sim_mat[r, c]))
                for r, c in zip(row_ind, col_ind)
            ]
            if len(assignment) <= 1:
                return assignment

            geo_removals = 0
            for _ in range(len(pids_a) * len(pids_b) + 1):
                geo_scores: list[float | None] = []
                for i, (pa, pb, _) in enumerate(assignment):
                    anchors = [(a, b) for j, (a, b, _) in enumerate(assignment) if j != i]
                    geo_scores.append(_geo_score_for_match(pa, pb, vid_a, vid_b, anchors))
                scored = [(i, g) for i, g in enumerate(geo_scores) if g is not None]
                if not scored:
                    break
                worst_i, worst_geo = min(scored, key=lambda x: x[1])
                if worst_geo >= GEO_CORR_THRESHOLD:
                    break
                if geo_removals >= 1:
                    break

                pa_fail, pb_fail, _ = assignment[worst_i]
                logging.info(
                    f"[GEO] {scene_id}  {vid_a}↔{vid_b}  "
                    f"veto P{pa_fail}↔P{pb_fail} (geo={worst_geo:.3f} < {GEO_CORR_THRESHOLD})"
                )
                assignment.pop(worst_i)
                claimed_b = {pb for _, pb, _ in assignment}
                ia = pids_a.index(pa_fail)
                candidates = sorted(
                    [(float(sim_mat[ia, ib]), pids_b[ib])
                     for ib in range(len(pids_b)) if pids_b[ib] not in claimed_b],
                    reverse=True,
                )
                replaced = False
                for cand_sim, cand_pb in candidates:
                    if cand_sim < cross_view_reid_threshold:
                        break
                    anchors = [(a, b) for a, b, _ in assignment]
                    geo = _geo_score_for_match(pa_fail, cand_pb, vid_a, vid_b, anchors)
                    if geo is not None and geo >= GEO_CORR_THRESHOLD:
                        assignment.append((pa_fail, cand_pb, cand_sim))
                        logging.info(
                            f"[GEO] {scene_id}  {vid_a}↔{vid_b}  "
                            f"P{pa_fail}→P{cand_pb} accepted (geo={geo:.3f})"
                        )
                        replaced = True
                        break
                geo_removals += 1
                if not replaced:
                    logging.info(
                        f"[cross-view reid] {scene_id}  "
                        f"{vid_a}/P{pa_fail} left unmatched after geo validation"
                    )
            return assignment

        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1:]:
                matches = _match_with_geo_validation(vid_a, vid_b)
                for pa, pb, sim in matches:
                    if sim >= cross_view_reid_threshold:
                        edges.append((sim, (vid_a, pa), (vid_b, pb)))
                        _union((vid_a, pa), (vid_b, pb))

        # ── Conflict detection and resolution ─────────────────────────
        def _get_components() -> dict[tuple, list[tuple]]:
            comps: dict[tuple, list[tuple]] = {}
            for _vid in active_vids:
                for _pid in person_pids[_vid]:
                    _node = (_vid, _pid)
                    _root = _find(_node)
                    comps.setdefault(_root, []).append(_node)
            return comps

        def _find_path_min_edge(src: tuple, dst: tuple) -> int:
            conflict_root = _find(src)
            adj: dict[tuple, list[tuple]] = {}
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
            path_edge_indices = []
            cur = dst
            while prev[cur] is not None:
                par, ei = prev[cur]
                path_edge_indices.append(ei)
                cur = par
            return min(path_edge_indices, key=lambda ei: edges[ei][0])

        def _resolve_conflicts() -> None:
            for _ in range(len(edges) + 1):
                comps = _get_components()
                conflict_found = False
                for members in comps.values():
                    vid_to_nodes: dict[str, list[tuple]] = {}
                    for node in members:
                        vid_to_nodes.setdefault(node[0], []).append(node)
                    for vid_id, nodes in vid_to_nodes.items():
                        if len(nodes) < 2:
                            continue
                        conflict_found = True
                        worst_idx = _find_path_min_edge(nodes[0], nodes[1])
                        if worst_idx >= 0:
                            removed_sim, removed_na, removed_nb = edges[worst_idx]
                            logging.info(
                                f"Scene {scene_id}: removed conflicting edge "
                                f"{removed_na[0]}/P{removed_na[1]} ↔ {removed_nb[0]}/P{removed_nb[1]} "
                                f"(sim={removed_sim:.3f}) — same-video duplicate in {vid_id}"
                            )
                            edges.pop(worst_idx)
                            parent.clear()
                            rank_uf.clear()
                            for _vid in active_vids:
                                for _pid in person_pids[_vid]:
                                    _find((_vid, _pid))
                            for _sim, _na, _nb in edges:
                                _union(_na, _nb)
                        break
                if not conflict_found:
                    break

        _resolve_conflicts()

        # ── Consolidation pass with component centroids ────────────────
        consolidation_threshold = max(0.15, cross_view_reid_threshold - 0.15)
        comps = _get_components()
        multi_view_comps: dict[tuple, list[tuple]] = {
            root: members
            for root, members in comps.items()
            if len({v for v, _ in members}) >= 2
        }
        isolated: list[tuple] = [
            (vid, pid)
            for root, members in comps.items()
            if root not in multi_view_comps
            for vid, pid in members
        ]
        comp_centroids: dict[tuple, tuple[np.ndarray | None, np.ndarray | None]] = {}
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
                all_feats = np.concatenate([f for f, _ in app_vecs], axis=0)
                all_confs = np.concatenate([c for _, c in app_vecs], axis=0)
                total_w = all_confs.sum()
                m = ((all_confs[:, None] * all_feats).sum(0) / total_w
                     if total_w > 0 else all_feats.mean(0))
                n = np.linalg.norm(m)
                app_c = (m / n if n > 0 else m).astype(np.float32)
            if shape_vecs:
                m = np.mean(np.stack(shape_vecs), axis=0).astype(np.float32)
                n = np.linalg.norm(m)
                shape_c = m / n if n > 0 else m
            if app_c is not None or shape_c is not None:
                comp_centroids[root] = (app_c, shape_c)

        def _weighted_sim_scalar(
            feat: tuple[np.ndarray | None, np.ndarray | None],
            centroid: tuple[np.ndarray | None, np.ndarray | None],
            w_app: float,
            w_shape: float,
        ) -> float:
            sim, weight = 0.0, 0.0
            if feat[0] is not None and centroid[0] is not None:
                feats, confs = feat[0]
                total_w = confs.sum()
                mean_feat = ((confs[:, None] * feats).sum(0) / total_w
                             if total_w > 0 else feats.mean(0))
                sim += w_app * float(np.dot(mean_feat, centroid[0]))
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
                sim = _weighted_sim_scalar(feat, centroid, appearance_weight, shape_weight)
                if sim > best_sim:
                    best_sim, best_root = sim, root
            if best_root is not None and best_sim >= consolidation_threshold:
                comp_member = multi_view_comps[best_root][0]
                edges.append((best_sim, (vid, pid), comp_member))
                _union((vid, pid), comp_member)
                consolidation_edges_added = True
                logging.info(
                    f"Scene {scene_id}: consolidation linked "
                    f"{vid}/P{pid} → component {best_root} "
                    f"(centroid_sim={best_sim:.3f})"
                )

        if consolidation_edges_added:
            # Simplified conflict resolution for consolidation edges
            for _ in range(len(edges) + 1):
                comps = _get_components()
                conflict_found = False
                for members in comps.values():
                    vid_to_nodes: dict[str, list[tuple]] = {}
                    for node in members:
                        vid_to_nodes.setdefault(node[0], []).append(node)
                    for vid_id, nodes in vid_to_nodes.items():
                        if len(nodes) < 2:
                            continue
                        conflict_found = True
                        conflict_root = _find(nodes[0])
                        worst_sim, worst_idx = float("inf"), -1
                        for ei, (sim, na, nb) in enumerate(edges):
                            if _find(na) == conflict_root and sim < worst_sim:
                                worst_sim, worst_idx = sim, ei
                        if worst_idx >= 0:
                            edges.pop(worst_idx)
                            parent.clear()
                            rank_uf.clear()
                            for _vid in active_vids:
                                for _pid in person_pids[_vid]:
                                    _find((_vid, _pid))
                            for _sim, _na, _nb in edges:
                                _union(_na, _nb)
                            logging.info(
                                f"Scene {scene_id}: consolidation conflict "
                                f"resolved — removed edge (sim={worst_sim:.3f})"
                            )
                        break
                if not conflict_found:
                    break

        # ── Foreground detection ───────────────────────────────────────
        _comps_for_fg = _get_components()
        _cam_counts = {
            root: len({v for v, _ in members})
            for root, members in _comps_for_fg.items()
        }
        _max_cams = max(_cam_counts.values()) if _cam_counts else 0
        _fg_threshold = max(1, _max_cams - 1)
        foreground_nodes: set[tuple[str, int]] = set()
        for root, members in _comps_for_fg.items():
            if _cam_counts[root] >= _fg_threshold:
                foreground_nodes.update(members)

        _covered = {v for v, _ in foreground_nodes}
        _uncovered = set(active_vids) - _covered
        if _uncovered:
            _remaining = sorted(
                [(root, members) for root, members in _comps_for_fg.items()
                 if _cam_counts[root] < _fg_threshold],
                key=lambda x: -_cam_counts[x[0]],
            )
            for root, members in _remaining:
                _comp_cams = {v for v, _ in members}
                if _comp_cams & _uncovered:
                    foreground_nodes.update(members)
                    _uncovered -= _comp_cams
                if not _uncovered:
                    break

        logging.info(
            f"Scene {scene_id}: foreground detection — "
            f"max_cams={_max_cams}, threshold≥{_fg_threshold}, "
            f"{len(foreground_nodes)} foreground node(s) covering "
            f"{len({v for v,_ in foreground_nodes})}/{len(active_vids)} camera(s)"
        )

        # ── Assign global IDs ──────────────────────────────────────────
        comps = _get_components()
        used_global_ids: set[int] = set()
        global_remap: dict[str, dict[int, int]] = {v: {} for v in active_vids}

        pending_single: list[tuple[int, list[tuple]]] = []
        global_counter = 1
        for members in comps.values():
            if len({v for v, _ in members}) >= 2:
                global_id = global_counter
                global_counter += 1
                used_global_ids.add(global_id)
                for vid_id, pid in members:
                    if pid != global_id:
                        global_remap[vid_id][pid] = global_id
            else:
                proposed_id = min(pid for (_, pid) in members)
                pending_single.append((proposed_id, list(members)))

        next_new_id = global_counter
        for proposed_id, members in sorted(pending_single):
            global_id = next_new_id
            next_new_id += 1
            used_global_ids.add(global_id)
            for vid_id, pid in members:
                if pid != global_id:
                    global_remap[vid_id][pid] = global_id

        total_remaps = sum(len(m) for m in global_remap.values())
        logging.info(
            f"Scene {scene_id}: cross-view re-ID → {len(comps)} global "
            f"person(s), {total_remaps} local-ID remap(s) across "
            f"{len(active_vids)} view(s)"
        )

        # ── Apply remaps ───────────────────────────────────────────────
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
                        f"{vid_id}: cross-view remap — {dst.name} already exists, "
                        f"discarding duplicate track from {tmp.name}"
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
                        new_key = f"{remap.get(old_pid, old_pid)}_conf"
                    else:
                        old_pid = int(k)
                        new_key = str(remap.get(old_pid, old_pid))
                    new_gallery[new_key] = gdata[k]
                np.savez(str(gallery_path), **new_gallery)

            mapping_path = body_dir / "cross_view_id_mapping.json"
            with open(mapping_path, "w") as _f:
                json.dump({str(k): v for k, v in remap.items()}, _f, indent=2)

            CrossVideoReidentifier.apply_reid_remap(vid_dir, remap)
            print(f"  {vid_id}: cross-view re-ID — {len(remap)} ID remap(s): {remap}")

        # ── Write scene-level summary ──────────────────────────────────
        reid_summary_path = scene_dir / "cross_view_reid.json"
        fg_serialised: dict[str, list[int]] = {}
        for vid_id, pid in foreground_nodes:
            global_pid = global_remap.get(vid_id, {}).get(pid, pid)
            fg_serialised.setdefault(vid_id, []).append(global_pid)
        reid_summary = {
            "remaps": {
                vid_id: {str(k): v for k, v in remap.items()}
                for vid_id, remap in global_remap.items()
            },
            "foreground": fg_serialised,
        }
        with open(reid_summary_path, "w") as _f:
            json.dump(reid_summary, _f, indent=2)

        return foreground_nodes
