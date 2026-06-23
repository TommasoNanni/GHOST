"""
Cross-view person ReID v4.

Delta estimation: identical to v2/v3 (weighted xcorr + ALS per candidate + consensus clocks).
Assignment at fixed delta: tiered approach.

  Tier 1 — Split persons by motion (pose_std > STD_THRESHOLD).
            If nobody exceeds the threshold, fall back to top-TIER1_FALLBACK_N by std.
            Match high-motion persons via canonical joint similarity (camera-invariant,
            no geometry required).

  Geometry — Fit (R, t, {λ_i}) from Tier 1 accepted pairs only (well-conditioned ALS).

  Tier 2 — Match static persons geometrically.
            For each candidate pair (k, l): given fixed (R, t), solve a 2×2 regularised
            least-squares for (λ_k, λ_l) with L2 penalty α*(λ-1)² on each scale.
            Data term normalised by T·J so α=0.1 is scale-independent.
            Accept pairs below RMSE_THR.

No second pass, no appearance rescue.
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

# ── Constants (delta estimation — identical to v2/v3) ─────────────────────────
_MIN_OVERLAP              = 30
_MIN_DELTA_OVERLAP_FRAMES = 100
_OVERLAP_PENALTY_K        = 30.0
_OVERLAP_REF_FRAMES       = 500
_DELTA_PRIOR_WEIGHT       = 0.0
_DELTA_PRIOR_SCALE        = 30.0

# ── Constants (v4 assignment) ─────────────────────────────────────────────────
_STD_THRESHOLD    = 0.05   # m — raw kp3d std over time; above = high-motion
_TIER1_FALLBACK_N = 2      # take top-N per side if nobody exceeds _STD_THRESHOLD
_CANON_THR        = 0.75   # Tier-1 acceptance: canonical sim must exceed this
_RMSE_THR         = 0.15   # Tier-2 acceptance: regularised RMSE must be below this (m)
_LAMBDA_ALPHA     = 0.1    # L2 penalty weight on (λ-1)² in the normalised objective


class CrossViewReidentifierV4:
    """Cross-view ReID with motion-split tiered assignment."""

    _THRESHOLD         = 0.50
    _APPEARANCE_WEIGHT = 0.30
    _SHAPE_WEIGHT      = 0.30
    _POSE_WEIGHT       = 0.40
    _MATCH_RMSE_THR    = 0.60

    def __init__(
        self,
        threshold: float = _THRESHOLD,
        appearance_weight: float = _APPEARANCE_WEIGHT,
        shape_weight: float = _SHAPE_WEIGHT,
        pose_weight: float = _POSE_WEIGHT,
        match_rmse_thr: float = _MATCH_RMSE_THR,
    ) -> None:
        self.threshold         = threshold
        self.appearance_weight = appearance_weight
        self.shape_weight      = shape_weight
        self.pose_weight       = pose_weight
        self.match_rmse_thr    = match_rmse_thr

    def match_across_views(
        self,
        scene: Scene,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path] | None = None,
        dry_run: bool = False,
    ) -> None:
        scene_id  = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent
        if not dry_run and (scene_dir / "cross_view_reid_v4.json").exists():
            logging.info(f"  Scene {scene_id}: cross-view ReID v4 already done, skipping")
            return
        self._core(video_dirs=video_dirs, scene_id=scene_id, scene_dir=scene_dir, dry_run=dry_run)

    # ── Core ──────────────────────────────────────────────────────────────────

    def _core(
        self,
        video_dirs: dict[str, Path],
        scene_id: str,
        scene_dir: Path,
        dry_run: bool,
    ) -> None:
        video_ids = list(video_dirs.keys())

        # ── Data loading (identical to v2/v3) ─────────────────────────────────
        person_descs  : dict[str, dict[int, tuple]] = {}
        person_pids   : dict[str, list[int]]        = {}
        person_kpts3d : dict[str, dict[int, tuple]] = {}

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

        # ── Appearance helpers (identical to v2/v3) ────────────────────────────

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

        # ── Geometry helpers (identical to v2/v3) ──────────────────────────────

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
            return float(valid.std(axis=0).mean()) if len(valid) > 0 else 0.0

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

        # ── Delta estimation (identical to v2/v3) ──────────────────────────────

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

        # ── v4 assignment helpers ──────────────────────────────────────────────

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

        def _regularized_pair_rmse(
            R: np.ndarray, t: np.ndarray,
            pid_b: int, vid_b: str,
            pid_a: int, vid_a: str,
            delta: int,
        ) -> float:
            """
            Given fixed (R, t) mapping cam_b → cam_a, fit (λ_a, λ_b) for the pair via:

              minimise  (1/M) ‖λ_a · dst  −  λ_b · V‖²  +  α·(λ_a−1)²  +  α·(λ_b−1)²

            where V = R @ src + t  (cam_b joints projected into cam_a frame),
            dst = cam_a joints, M = T·J·3 (normalises for track length / joint count).

            Returns the RMSE after fitting (in metres), or inf if no co-visible frames.
            """
            src, dst = _get_aligned_flat(pid_b, vid_b, pid_a, vid_a, delta)
            if src is None:
                return float("inf")
            valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
            if valid.sum() < _MIN_OVERLAP:
                return float("inf")
            src, dst = src[valid], dst[valid]   # [N, 3], N = T*J

            V = (R @ src.T).T + t               # [N, 3]
            d = dst.flatten()                   # [N*3]
            v = V.flatten()                     # [N*3]
            M = float(len(d))

            dd = float(d @ d) / M
            vv = float(v @ v) / M
            dv = float(d @ v) / M

            # 2×2 system: [dd+α, -dv; -dv, vv+α][λ_a; λ_b] = [α; α]
            det = (dd + _LAMBDA_ALPHA) * (vv + _LAMBDA_ALPHA) - dv * dv
            if abs(det) < 1e-10:
                return float("inf")

            lam_a = ((vv + _LAMBDA_ALPHA) * _LAMBDA_ALPHA + dv * _LAMBDA_ALPHA) / det
            lam_b = ((dd + _LAMBDA_ALPHA) * _LAMBDA_ALPHA + dv * _LAMBDA_ALPHA) / det

            lam_a = max(0.1, min(10.0, lam_a))
            lam_b = max(0.1, min(10.0, lam_b))

            resid = lam_a * dst - lam_b * V     # [N, 3]
            rmse  = float(np.sqrt((resid ** 2).sum(1).mean()))
            return rmse if np.isfinite(rmse) else float("inf")

        def _tiered_assign(
            vid_a: str, vid_b: str, pdata: dict, delta_bar: int,
        ) -> list[tuple[int, int, float]]:
            """
            Two-tier assignment at fixed delta_bar.

            Tier 1 — high-motion persons: canonical sim → Hungarian → accept ≥ CANON_THR.
                      If no person exceeds STD_THRESHOLD, fall back to top-TIER1_FALLBACK_N.
            Geometry — ALS fit from Tier-1 anchors only.
            Tier 2 — static persons: regularised per-pair LS → RMSE → Hungarian → accept < RMSE_THR.
            """
            pids_a_s = pdata["pids_a_s"]
            pids_b_s = pdata["pids_b_s"]
            kd_a     = person_kpts3d.get(vid_a, {})
            kd_b     = person_kpts3d.get(vid_b, {})

            # ── Motion split ───────────────────────────────────────────────────
            stds_a = {pa: _track_std(vid_a, pa) for pa in pids_a_s}
            stds_b = {pb: _track_std(vid_b, pb) for pb in pids_b_s}

            high_a = [pa for pa in pids_a_s if stds_a[pa] > _STD_THRESHOLD]
            high_b = [pb for pb in pids_b_s if stds_b[pb] > _STD_THRESHOLD]

            if not high_a:
                high_a = sorted(pids_a_s, key=lambda p: -stds_a[p])[:_TIER1_FALLBACK_N]
                logging.info(f"  [{vid_a}↔{vid_b}] Tier-1 fallback (cam_a): top-{_TIER1_FALLBACK_N} by std")
            if not high_b:
                high_b = sorted(pids_b_s, key=lambda p: -stds_b[p])[:_TIER1_FALLBACK_N]
                logging.info(f"  [{vid_a}↔{vid_b}] Tier-1 fallback (cam_b): top-{_TIER1_FALLBACK_N} by std")

            high_a_set = set(high_a)
            high_b_set = set(high_b)
            static_a   = [pa for pa in pids_a_s if pa not in high_a_set]
            static_b   = [pb for pb in pids_b_s if pb not in high_b_set]

            logging.info(
                f"  [{vid_a}↔{vid_b}] motion split: "
                f"Tier1 {len(high_a)}a/{len(high_b)}b  "
                f"Tier2 {len(static_a)}a/{len(static_b)}b"
            )

            # ── Tier 1: canonical similarity ───────────────────────────────────
            Na_h, Nb_h = len(high_a), len(high_b)
            canon_sim = np.zeros((Na_h, Nb_h), dtype=np.float32)
            for i, pa in enumerate(high_a):
                pose_a = person_descs[vid_a][pa][2]
                if pose_a is None or pa not in kd_a:
                    continue
                frames_a = kd_a[pa][0]
                for j, pb in enumerate(high_b):
                    pose_b = person_descs[vid_b][pb][2]
                    if pose_b is None or pb not in kd_b:
                        continue
                    frames_b = kd_b[pb][0]
                    canon_sim[i, j] = _canon_sim_at_delta(
                        pose_a, frames_a, pose_b, frames_b, delta_bar
                    )

            ri, ci = linear_sum_assignment(1.0 - canon_sim)
            tier1_pairs: list[tuple[int, int]] = []
            for r, c in zip(ri, ci):
                if canon_sim[r, c] >= _CANON_THR:
                    pa, pb = high_a[r], high_b[c]
                    tier1_pairs.append((pa, pb))
                    logging.info(
                        f"  Tier-1 accept {vid_a}:P{pa} ↔ {vid_b}:P{pb}  "
                        f"canon_sim={canon_sim[r,c]:.3f}"
                    )

            accepted: list[tuple[int, int, float]] = [
                (pa, pb, float(canon_sim[
                    high_a.index(pa), high_b.index(pb)
                ]))
                for pa, pb in tier1_pairs
            ]

            # ── Geometry: ALS fit from Tier-1 anchors only ────────────────────
            R_ref, t_ref = None, None
            if tier1_pairs:
                res = _constrained_fit(
                    [(pb, pa) for pa, pb in tier1_pairs], vid_b, vid_a, delta_bar
                )
                if res is not None:
                    R_ref, t_ref = res[0], res[1]
                    logging.info(
                        f"  [{vid_a}↔{vid_b}] Tier-1 ALS  "
                        f"({len(tier1_pairs)} anchors)  rmse={res[3]:.3f}"
                    )
            else:
                logging.info(
                    f"  [{vid_a}↔{vid_b}] no Tier-1 accepted pairs — Tier-2 skipped"
                )

            # ── Tier 2: static persons via regularised LS ──────────────────────
            if R_ref is not None and static_a and static_b:
                Na_s, Nb_s = len(static_a), len(static_b)
                rmse_mat = np.full((Na_s, Nb_s), float("inf"), dtype=np.float64)
                for i, pa in enumerate(static_a):
                    for j, pb in enumerate(static_b):
                        rmse_mat[i, j] = _regularized_pair_rmse(
                            R_ref, t_ref, pb, vid_b, pa, vid_a, delta_bar
                        )

                ri2, ci2 = linear_sum_assignment(
                    np.where(np.isfinite(rmse_mat), rmse_mat, 1e6)
                )
                for r, c in zip(ri2, ci2):
                    if not np.isfinite(rmse_mat[r, c]) or rmse_mat[r, c] >= _RMSE_THR:
                        continue
                    pa, pb = static_a[r], static_b[c]
                    conf = float(1.0 - rmse_mat[r, c] / _RMSE_THR)
                    accepted.append((pa, pb, conf))
                    logging.info(
                        f"  Tier-2 accept {vid_a}:P{pa} ↔ {vid_b}:P{pb}  "
                        f"rmse={rmse_mat[r,c]:.3f}  conf={conf:.3f}"
                    )

            return accepted

        # ── All-pairs: delta candidates → consensus → tiered assign ───────────
        camera_pair_offsets: dict[str, int] = {}
        pair_data: dict[tuple, dict] = {}
        for ii, vid_a in enumerate(active_vids):
            for vid_b in active_vids[ii + 1:]:
                pair_data[(vid_a, vid_b)] = _pair_delta_candidates(vid_a, vid_b)

        consensus = _solve_consensus_offsets(pair_data)
        for (vid_a, vid_b), pdata in pair_data.items():
            delta_bar = consensus[(vid_a, vid_b)]
            matches   = _tiered_assign(vid_a, vid_b, pdata, delta_bar)
            camera_pair_offsets[f"{vid_a}→{vid_b}"] = delta_bar
            for pa, pb, conf in matches:
                edges.append((conf, (vid_a, pa), (vid_b, pb)))
                _union((vid_a, pa), (vid_b, pb))

        # ── Conflict resolution (identical to v2/v3) ───────────────────────────
        def _get_components() -> dict[tuple, list[tuple]]:
            comps: dict[tuple, list[tuple]] = {}
            for _v in active_vids:
                for _p in person_pids[_v]:
                    _node = (_v, _p)
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
            return min(path_ei, key=lambda ei: edges[ei][0])

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

        # ── Global ID assignment (identical to v2/v3) ──────────────────────────
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
            f"Scene {scene_id}: v4 → {n_comps} global person(s), "
            f"{n_remaps} remap(s) across {len(active_vids)} view(s)"
        )

        if dry_run:
            print(
                f"\n[DRY RUN v4] Scene {scene_id}: {n_comps} global persons, "
                f"camera pair offsets: {camera_pair_offsets}"
            )
            for vid_id, remap in global_remap.items():
                if remap:
                    print(f"  {vid_id}: would remap {remap}")
                else:
                    print(f"  {vid_id}: no remaps needed")
            return

        # ── Apply remaps (identical to v3) ────────────────────────────────────
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir  = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"
            tmp_renames: list[tuple[Path, Path]] = []
            for old_id, new_id in remap.items():
                src = body_dir / f"person_{old_id}.npz"
                if src.exists():
                    tmp = body_dir / f"person_{old_id}.v4tmp.npz"
                    src.rename(tmp)
                    tmp_renames.append((tmp, body_dir / f"person_{new_id}.npz"))
            for tmp, dst in tmp_renames:
                if dst.exists():
                    logging.warning(f"{vid_id}: v4 remap — {dst.name} already exists, discarding")
                    tmp.unlink()
                else:
                    tmp.rename(dst)
            from preprocessing.cross_view_reid_v2 import CrossVideoReidentifierV2
            CrossVideoReidentifierV2.apply_reid_remap(vid_dir, remap)

        (scene_dir / "cross_view_reid_v4.json").write_text(
            json.dumps({"status": "done", "n_global": n_comps}, indent=2)
        )
