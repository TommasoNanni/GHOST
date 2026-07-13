"""Stage-A camera pass for cross-view ReID v7.

Estimates the camera rig of a scene from the STATIC BACKGROUND only — no GT
calibration, no person correspondences — and caches everything the (CPU-only)
v7 matcher needs into ``<scene_dir>/reid_cameras.npz``.

Pipeline per scene
------------------
1.  Tuple selection: per camera, pick ``n_tuples`` frames with the lowest
    person-pixel coverage (from our own ``mask_data.npz``), one per time bin.
    Videos are asynchronous — tuples deliberately do NOT align frame indices
    across cameras; only the static background is assumed shared.
2.  VGGT-Omega per tuple → per-tuple extrinsics/intrinsics/depth.
3.  Per-tuple scale normalisation (each VGGT run has its own arbitrary global
    scale): translations are divided by the mean pairwise camera-centre
    distance of that tuple; the tuple's depth is divided by the same factor.
4.  Consensus rig: choose the reference camera with the most consistent
    estimates, re-root every tuple to it, then per camera cluster the tuple
    estimates (rotation geodesic + centre distance) and chordal-mean the
    LARGEST cluster only.  Never mean-then-orthonormalise over outliers — a
    single mirrored tuple silently corrupts an unclustered mean.
5.  Scale: the shared frame stays in (normalized) VGGT units by default —
    matching only needs a CONSISTENT scale, and the v7 matcher converts its
    metric thresholds via the median per-track λ (SAM3D depth is metric-ish,
    so humans act as the ruler).  MapAnything "baselines" metric scale is
    available behind ``metric_scale_mode="mapanything"`` for ablations only.
6.  Person depth anchors: for every track (``body_data/person_N.npz``), read
    the VGGT depth under the person's mask∩bbox in the tuple frames → sparse
    ``(frame_idx, z_metric, n_px)`` samples used by v7 to calibrate the
    per-track monocular-depth scale λ.  Extra "anchor tuples" are run for
    tracks not covered by the main tuples.
7.  Moving cameras (egocentric): DROID-SLAM per video (sync-free) → local
    trajectory; anchored into the shared frame with a RANSAC-guarded Umeyama
    similarity fitted on VGGT tuples of [static frames + ego frame at τ].
    Statics are frozen first; a bad ego frame can only invalidate itself.

The cache is self-contained: after this one GPU pass, the v7 matcher (and its
dry-run loop) never touches images or models again.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_INT_RE = re.compile(r"\d+")


# ── Small geometry helpers ────────────────────────────────────────────────────

def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = np.clip((np.trace(R1 @ R2.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1
        out = U @ Vt
    return out


def _cam_center(ext: np.ndarray) -> np.ndarray:
    """Extrinsic (3,4) cam-from-world → camera centre in world coords."""
    return -ext[:3, :3].T @ ext[:3, 3]


def _reroot(ext: np.ndarray, ref_ext: np.ndarray) -> np.ndarray:
    """Re-express one cam-from-world extrinsic so that ``ref_ext`` = [I|0]."""
    R0, t0 = ref_ext[:3, :3], ref_ext[:3, 3]
    R, t = ext[:3, :3], ext[:3, 3]
    out = np.empty((3, 4), dtype=np.float64)
    out[:3, :3] = R @ R0.T
    out[:3, 3] = t - out[:3, :3] @ t0
    return out


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) minimising ‖s·R@src + t − dst‖²."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


# ── ReidCameraPass ────────────────────────────────────────────────────────────

class ReidCameraPass:
    """One-shot GPU camera pass for ReID v7 (see module docstring).

    Parameters
    ----------
    vggt_weights      : local ``.pt`` for VGGT-Omega.
    droid_weights     : DROID-SLAM checkpoint (needed only for moving cams).
    droid_root        : DROID-SLAM repo root.
    device            : torch device string.
    n_tuples          : number of main frame-tuples.
    min_anchor_frames : per-track minimum depth-anchor samples before extra
                        anchor tuples are scheduled.
    force             : recompute even when reid_cameras.npz exists.
    """

    # tuple-consensus clustering
    _ROT_CLUSTER_DEG = 10.0
    _CENTER_CLUSTER = 0.15      # normalized units (mean baseline == 1)
    # depth-anchor gates
    _MIN_MASK_PX = 40
    _DEPTH_CONF_THR = 0.5
    _RING_REL_MARGIN = 0.05     # person ≥5% in front of background ring (scale-free)
    # moving-cam anchoring
    _N_EGO_ANCHORS = 8
    _ANCHOR_STATIC_ROT_DEG = 10.0
    _EGO_RESID_MAX_REL = 0.10   # Sim(3) residual / mean anchor spread (scale-free)
    # metric scale
    _N_SCALE_TUPLES = 3
    # coverage scan step (mask decompression is the cost)
    _COVERAGE_STEP = 5

    def __init__(
        self,
        vggt_weights: str,
        droid_weights: str | None = None,
        droid_root: str = "/users/tnanni/ghost/DROID-SLAM",
        device: str = "cuda:0",
        n_tuples: int = 16,
        min_anchor_frames: int = 3,
        max_anchor_tuples: int = 24,
        metric_scale_mode: str = "none",
        force: bool = False,
    ):
        if metric_scale_mode not in ("none", "mapanything"):
            raise ValueError(f"metric_scale_mode must be 'none' or 'mapanything', got {metric_scale_mode!r}")
        self.vggt_weights = vggt_weights
        self.droid_weights = droid_weights
        self.droid_root = droid_root
        self.device = device
        self.n_tuples = n_tuples
        self.min_anchor_frames = min_anchor_frames
        self.max_anchor_tuples = max_anchor_tuples
        self.metric_scale_mode = metric_scale_mode
        self.force = force
        self._vggt = None  # lazy

    # ── lazy model ────────────────────────────────────────────────────────────

    def _get_vggt(self):
        if self._vggt is None:
            from preprocessing.run_vggt import VGGTPreprocessor
            self._vggt = VGGTPreprocessor(self.vggt_weights, device=self.device)
        return self._vggt

    # ── frame / mask indexing ─────────────────────────────────────────────────

    @staticmethod
    def _index_frames(frames_dir: Path) -> dict[int, Path]:
        """Map frame index → image path.  Index = first integer in the stem
        (covers ``000042.jpg`` and RICH-centered ``00042_06.jpg``)."""
        out: dict[int, Path] = {}
        for p in sorted(frames_dir.iterdir()):
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            m = _INT_RE.search(p.stem)
            if m:
                out[int(m.group(0))] = p
        return out

    @staticmethod
    def _mask_keys(mask_npz: Path) -> dict[int, str]:
        """Map frame index → key stem inside mask_data.npz."""
        if not mask_npz.exists():
            return {}
        out: dict[int, str] = {}
        with zipfile.ZipFile(str(mask_npz), "r") as zf:
            for name in zf.namelist():
                m = _INT_RE.search(Path(name).stem)
                if m:
                    out[int(m.group(0))] = Path(name).stem
        return out

    @staticmethod
    def _load_mask(mask_npz: Path, stem: str) -> np.ndarray | None:
        with zipfile.ZipFile(str(mask_npz), "r") as zf:
            key = stem + ".npy"
            if key not in zf.namelist():
                return None
            with zf.open(key) as f:
                return np.load(io.BytesIO(f.read()))

    def _person_coverage(self, video_dir: Path, frame_ids: list[int]) -> dict[int, float]:
        """Fraction of person pixels per frame (subsampled scan).  Frames
        without a mask entry get coverage 0 (they are the best candidates)."""
        mask_npz = video_dir / "mask_data.npz"
        keys = self._mask_keys(mask_npz)
        cov = {fi: 0.0 for fi in frame_ids}
        scan = frame_ids[:: self._COVERAGE_STEP]
        for fi in scan:
            stem = keys.get(fi)
            if stem is None:
                continue
            m = self._load_mask(mask_npz, stem)
            if m is not None and m.size:
                cov[fi] = float(np.count_nonzero(m)) / m.size
        # fill unscanned frames with nearest scanned value (vectorised)
        scanned = np.asarray(sorted(scan))
        if scanned.size:
            all_ids = np.asarray(frame_ids)
            pos = np.clip(np.searchsorted(scanned, all_ids), 0, scanned.size - 1)
            prev = np.clip(pos - 1, 0, scanned.size - 1)
            nearest = np.where(
                np.abs(scanned[pos] - all_ids) <= np.abs(scanned[prev] - all_ids),
                scanned[pos], scanned[prev],
            )
            scan_set = set(scan)
            for fi, nf in zip(frame_ids, nearest):
                if fi not in scan_set:
                    cov[fi] = cov[int(nf)]
        return cov

    def _select_tuple_frames(
        self,
        cams: list[str],
        video_dirs: dict[str, Path],
        frame_maps: dict[str, dict[int, Path]],
    ) -> list[dict[str, int]]:
        """Per camera: one lowest-coverage frame per time bin → n_tuples
        tuples of {cam: frame_idx}.  Bins are per-camera (async videos)."""
        per_cam_picks: dict[str, list[int]] = {}
        for cam in cams:
            fids = sorted(frame_maps[cam])
            if not fids:
                per_cam_picks[cam] = []
                continue
            cov = self._person_coverage(video_dirs[cam], fids)
            bins = np.array_split(np.asarray(fids), self.n_tuples)
            picks = [int(min(b, key=lambda fi: cov[fi])) for b in bins if len(b)]
            per_cam_picks[cam] = picks
        n = max((len(v) for v in per_cam_picks.values()), default=0)
        tuples = []
        for i in range(n):
            tup = {c: per_cam_picks[c][i] for c in cams if i < len(per_cam_picks[c])}
            if len(tup) >= 2:
                tuples.append(tup)
        return tuples

    # ── VGGT tuple execution ──────────────────────────────────────────────────

    def _run_tuple(
        self,
        tup: dict[str, int],
        cams: list[str],
        frame_maps: dict[str, dict[int, Path]],
    ) -> dict | None:
        """Run VGGT on one tuple.  Returns per-cam extrinsic/intrinsic/depth
        (K-slot arrays with NaN for absent cams), normalized to unit mean
        baseline, plus the normalisation factor."""
        paths = [frame_maps[c].get(tup[c]) if c in tup else None for c in cams]
        if sum(p is not None for p in paths) < 2:
            return None
        res = self._get_vggt().run_frame(paths)
        K = len(cams)
        ext = np.full((K, 3, 4), np.nan, dtype=np.float64)
        intr = np.full((K, 3, 3), np.nan, dtype=np.float64)
        size = np.zeros((K, 2), dtype=np.int32)
        present = res["present_indices"].tolist()
        for j, k in enumerate(present):
            ext[k] = res["extrinsics"][j].astype(np.float64)
            intr[k] = res["intrinsics"][j].astype(np.float64)
            size[k] = res["original_size"][j]
        centers = np.array([_cam_center(ext[k]) for k in present])
        dists = [
            float(np.linalg.norm(centers[i] - centers[j]))
            for i in range(len(present))
            for j in range(i + 1, len(present))
        ]
        factor = float(np.mean(dists)) if dists else 0.0
        if factor < 1e-6:
            return None
        ext[:, :, 3] /= factor
        return {
            "tup": tup,
            "ext": ext,
            "intr": intr,
            "size": size,
            "present": present,
            "factor": factor,
            "depth": res["depth"] / factor,          # (K_present, H, W)
            "depth_conf": res["depth_conf"],
            "vggt_hw": res["depth"].shape[-2:],
        }

    # ── consensus ─────────────────────────────────────────────────────────────

    def _pick_reference(self, tuples: list[dict], cams: list[str]) -> int:
        """Reference camera = the one whose choice as world origin makes the
        per-camera tuple estimates most consistent."""
        best_ref, best_score = 0, np.inf
        for r in range(len(cams)):
            present = [t for t in tuples if r in t["present"]]
            if len(present) < max(2, int(0.5 * len(tuples))):
                continue
            spreads = []
            for k in range(len(cams)):
                Rs = [
                    _reroot(t["ext"][k], t["ext"][r])[:3, :3]
                    for t in present
                    if k in t["present"]
                ]
                if len(Rs) < 2:
                    continue
                med = np.median(
                    [_geodesic_deg(Rs[i], Rs[j])
                     for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
                )
                spreads.append(med)
            if spreads and float(np.mean(spreads)) < best_score:
                best_score, best_ref = float(np.mean(spreads)), r
        return best_ref

    def _consensus_extrinsics(
        self, tuples: list[dict], cams: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[int, list[int]]]:
        """Cluster-then-average per camera.  Returns (extrinsics (K,3,4),
        inlier_frac (K,), pair_sigma (K,K), ref index, inlier tuple ids)."""
        K = len(cams)
        ref = self._pick_reference(tuples, cams)
        # re-root every tuple to the chosen reference
        rerooted: list[np.ndarray] = []
        usable: list[dict] = []
        for t in tuples:
            if ref not in t["present"]:
                continue
            E = np.full((K, 3, 4), np.nan)
            for k in t["present"]:
                E[k] = _reroot(t["ext"][k], t["ext"][ref])
            rerooted.append(E)
            usable.append(t)

        ext_out = np.full((K, 3, 4), np.nan)
        inlier_frac = np.zeros(K, dtype=np.float32)
        inlier_ids: dict[int, list[int]] = {}
        for k in range(K):
            idxs = [i for i, E in enumerate(rerooted) if np.isfinite(E[k]).all()]
            if not idxs:
                continue
            # ball-medoid clustering: the tuple agreeing with the most others wins
            def _agree(i: int, j: int) -> bool:
                Ei, Ej = rerooted[i][k], rerooted[j][k]
                return (
                    _geodesic_deg(Ei[:3, :3], Ej[:3, :3]) <= self._ROT_CLUSTER_DEG
                    and np.linalg.norm(_cam_center(Ei) - _cam_center(Ej))
                    <= self._CENTER_CLUSTER
                )
            clusters = {i: [j for j in idxs if _agree(i, j)] for i in idxs}
            medoid = max(clusters, key=lambda i: len(clusters[i]))
            members = clusters[medoid]
            inlier_frac[k] = len(members) / len(idxs)
            inlier_ids[k] = members
            R_mean = _orthonormalize(
                np.mean([rerooted[i][k][:3, :3] for i in members], axis=0)
            )
            C_mean = np.mean([_cam_center(rerooted[i][k]) for i in members], axis=0)
            ext_out[k, :3, :3] = R_mean
            ext_out[k, :3, 3] = -R_mean @ C_mean

        # per-pair positional sigma: spread of relative centre distance
        pair_sigma = np.zeros((K, K), dtype=np.float32)
        for a in range(K):
            for b in range(a + 1, K):
                common = [
                    i for i in inlier_ids.get(a, []) if i in inlier_ids.get(b, [])
                ]
                if len(common) >= 2:
                    ds = [
                        np.linalg.norm(
                            _cam_center(rerooted[i][a]) - _cam_center(rerooted[i][b])
                        )
                        for i in common
                    ]
                    pair_sigma[a, b] = pair_sigma[b, a] = float(np.std(ds))
                else:
                    pair_sigma[a, b] = pair_sigma[b, a] = np.nan
        return ext_out, inlier_frac, pair_sigma, ref, inlier_ids

    # ── metric scale ──────────────────────────────────────────────────────────

    def _metric_scale(
        self,
        tuples: list[dict],
        cams: list[str],
        frame_maps: dict[str, dict[int, Path]],
    ) -> tuple[float, bool]:
        """MapAnything images-only baselines vs normalized consensus baselines."""
        try:
            from preprocessing.run_mapanything import MapAnythingScaleEstimator, PATCH
        except Exception as e:  # mapanything not installed on this node
            logger.warning(f"[v7-CAM] MapAnything unavailable ({e}); metric scale = 1.0")
            return 1.0, False

        est = MapAnythingScaleEstimator(device=self.device, scale_from="baselines")
        est._load_model()

        picks = tuples[:: max(1, len(tuples) // self._N_SCALE_TUPLES)][: self._N_SCALE_TUPLES]
        ratios: list[float] = []
        for t in picks:
            present = t["present"]
            if len(present) < 2:
                continue
            # MA input dims: multiples of PATCH, from the first image's aspect
            first = frame_maps[cams[present[0]]][t["tup"][cams[present[0]]]]
            from PIL import Image
            with Image.open(first) as img:
                W0, H0 = img.size
            scale = 518 / max(W0, H0)
            W_ma = max(PATCH, int(round(W0 * scale / PATCH)) * PATCH)
            H_ma = max(PATCH, int(round(H0 * scale / PATCH)) * PATCH)
            cam_file_lists = {
                k: [frame_maps[cams[k]][t["tup"][cams[k]]]] for k in present
            }
            vggt_exts = t["ext"][None, :, :, :]  # (1, K, 3, 4), normalized
            res = est._run_batch_baselines(
                [0], present, cam_file_lists, vggt_exts, H_ma, W_ma
            )
            if 0 in res and np.isfinite(res[0]):
                ratios.append(res[0])
        if not ratios:
            logger.warning("[v7-CAM] no metric-scale ratios; scale = 1.0")
            return 1.0, False
        scale = float(np.median(ratios))
        mad = float(np.median(np.abs(np.asarray(ratios) - scale)))
        ok = len(ratios) >= 2 and mad / max(scale, 1e-6) < 0.15
        logger.info(
            f"[v7-CAM] metric scale = {scale:.3f} "
            f"({len(ratios)} tuples, rel-MAD {mad / max(scale, 1e-6):.3f}, ok={ok})"
        )
        return scale, ok

    # ── person depth anchors ──────────────────────────────────────────────────

    @staticmethod
    def _load_track_meta(video_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """pid → (frame_indices, bbox (N,4) xyxy) from body_data."""
        out = {}
        body = video_dir / "body_data"
        if not body.is_dir():
            return out
        for npz_path in sorted(body.glob("person_*.npz")):
            try:
                pid = int(npz_path.stem.split("_")[1])
            except ValueError:
                continue
            try:
                with np.load(npz_path) as d:
                    if "frame_indices" not in d or "bbox" not in d:
                        continue
                    out[pid] = (d["frame_indices"].astype(int), d["bbox"].astype(float))
            except (OSError, zipfile.BadZipFile) as e:
                logger.warning(f"[v7-CAM] unreadable {npz_path}: {e}")
        return out

    def _harvest_depth(
        self,
        t: dict,
        cams: list[str],
        video_dirs: dict[str, Path],
        track_meta: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
        metric_scale: float,
        samples: dict[tuple[str, int], list[tuple[int, float, float]]],
        only_cam: str | None = None,
    ) -> None:
        """Read person depth from one tuple's depth maps.

        Association is by bbox overlap with mask pixels (mask ids and
        body_data pids desync — never trust id equality here).
        """
        Hv, Wv = t["vggt_hw"]
        for j, k in enumerate(t["present"]):
            cam = cams[k]
            if only_cam is not None and cam != only_cam:
                continue
            fi = t["tup"].get(cam)
            if fi is None:
                continue
            mask_npz = video_dirs[cam] / "mask_data.npz"
            keys = self._mask_keys(mask_npz)
            stem = keys.get(fi)
            if stem is None:
                continue
            mask = self._load_mask(mask_npz, stem)
            if mask is None:
                continue
            H0, W0 = mask.shape
            # nearest-neighbour resize of the mask to VGGT resolution
            yy = (np.arange(Hv) * H0 / Hv).astype(int)
            xx = (np.arange(Wv) * W0 / Wv).astype(int)
            mask_v = mask[yy][:, xx]
            person_v = mask_v > 0
            # 1-px erosion (kills boundary bleed onto the background)
            er = person_v.copy()
            er[1:] &= person_v[:-1]
            er[:-1] &= person_v[1:]
            er[:, 1:] &= person_v[:, :-1]
            er[:, :-1] &= person_v[:, 1:]
            depth = t["depth"][j]
            conf = t["depth_conf"][j]
            sx, sy = Wv / W0, Hv / H0
            for pid, (frames, bboxes) in track_meta.get(cam, {}).items():
                pos = np.nonzero(frames == fi)[0]
                if not len(pos):
                    continue
                x1, y1, x2, y2 = bboxes[pos[0]]
                xa, xb = int(x1 * sx), int(np.ceil(x2 * sx))
                ya, yb = int(y1 * sy), int(np.ceil(y2 * sy))
                xa, ya = max(0, xa), max(0, ya)
                xb, yb = min(Wv, xb), min(Hv, yb)
                if xb <= xa or yb <= ya:
                    continue
                sel = np.zeros_like(er)
                sel[ya:yb, xa:xb] = er[ya:yb, xa:xb]
                sel &= conf >= self._DEPTH_CONF_THR
                n_px = int(sel.sum())
                if n_px < self._MIN_MASK_PX:
                    continue
                z_person = float(np.median(depth[sel]))
                # background ring: bbox dilated 25%, person pixels excluded
                mx, my = int(0.25 * (xb - xa)), int(0.25 * (yb - ya))
                rxa, rya = max(0, xa - mx), max(0, ya - my)
                rxb, ryb = min(Wv, xb + mx), min(Hv, yb + my)
                ring = np.zeros_like(er)
                ring[rya:ryb, rxa:rxb] = True
                ring[ya:yb, xa:xb] = False
                ring &= ~person_v
                if ring.sum() >= self._MIN_MASK_PX:
                    ring_q25 = float(np.quantile(depth[ring], 0.25))
                    if z_person > ring_q25 * (1.0 - self._RING_REL_MARGIN):
                        continue  # not clearly in front of background → reject
                samples.setdefault((cam, pid), []).append(
                    (fi, z_person * metric_scale, float(n_px))
                )

    # ── moving cameras ────────────────────────────────────────────────────────

    def _run_moving_cam(
        self,
        cam: str,
        cams: list[str],
        frame_maps: dict[str, dict[int, Path]],
        video_dirs: dict[str, Path],
        consensus_ext: np.ndarray,
        consensus_intr: np.ndarray,
        orig_size: np.ndarray,
        vggt_hw: tuple[int, int],
        ref: int,
        metric_scale: float,
        static_low_cov: dict[str, int],
    ) -> dict | None:
        """DROID-SLAM trajectory + Umeyama anchoring into the shared frame."""
        if not self.droid_weights:
            logger.warning(f"[v7-CAM] {cam}: moving cam but no droid_weights — skipped")
            return None
        frames_dir_map = frame_maps[cam]
        if not frames_dir_map:
            return None
        frames_dir = next(iter(frames_dir_map.values())).parent
        k_ego = cams.index(cam)

        # intrinsics for SLAM: VGGT-estimated at VGGT resolution → scale to
        # the original resolution (the SLAM stream rescales from original).
        intr = consensus_intr[k_ego]
        if not np.isfinite(intr).all() or not orig_size[k_ego].any():
            logger.warning(f"[v7-CAM] {cam}: no VGGT intrinsics — skipped")
            return None
        W0, H0 = float(orig_size[k_ego][0]), float(orig_size[k_ego][1])
        Hv, Wv = float(vggt_hw[0]), float(vggt_hw[1])
        fx, cx = intr[0, 0] * W0 / Wv, intr[0, 2] * W0 / Wv
        fy, cy = intr[1, 1] * H0 / Hv, intr[1, 2] * H0 / Hv

        poses_result = _droid_run(
            frames_dir, float(fx), float(fy), float(cx), float(cy),
            self.droid_weights, self.droid_root,
        )
        if poses_result is None:
            return None
        poses, frame_ids_sorted = poses_result
        traj_std = float(np.std(poses[:, :3]))
        if traj_std < 0.05:
            logger.info(f"[v7-CAM] {cam}: SLAM traj std {traj_std:.3f} — treating as static")
            return {"static": True}

        # anchor tuples: [static low-coverage frames + ego frame at τ]
        n = len(frame_ids_sorted)
        taus = [frame_ids_sorted[i] for i in
                np.linspace(0, n - 1, self._N_EGO_ANCHORS).astype(int)]
        anchors_slam, anchors_world = [], []
        for tau in taus:
            tup = dict(static_low_cov)
            tup[cam] = tau
            res = self._run_tuple(tup, cams, frame_maps)
            if res is None or k_ego not in res["present"] or ref not in res["present"]:
                continue
            # sanity: the ego frame must not perturb the static consensus
            E = {k: _reroot(res["ext"][k], res["ext"][ref]) for k in res["present"]}
            bad = False
            for k in res["present"]:
                if k == k_ego or not np.isfinite(consensus_ext[k]).all():
                    continue
                if _geodesic_deg(E[k][:3, :3], consensus_ext[k][:3, :3]) > self._ANCHOR_STATIC_ROT_DEG:
                    bad = True
                    break
            if bad:
                logger.info(f"[v7-CAM] {cam}: anchor τ={tau} rejected (static perturbation)")
                continue
            slam_idx = frame_ids_sorted.index(tau)
            pose = poses[slam_idx]
            if not np.isfinite(pose).all():
                continue
            anchors_slam.append(pose[:3].astype(np.float64))          # SLAM c2w centre
            anchors_world.append(_cam_center(E[k_ego]) * metric_scale)
        if len(anchors_slam) < 3:
            logger.warning(f"[v7-CAM] {cam}: only {len(anchors_slam)} valid anchors — unreliable")
            return {"static": False, "reliable": False}

        s, R_al, t_al = _umeyama(np.asarray(anchors_slam), np.asarray(anchors_world))
        resid = float(np.mean(np.linalg.norm(
            (s * (R_al @ np.asarray(anchors_slam).T).T + t_al) - np.asarray(anchors_world),
            axis=1,
        )))
        aw = np.asarray(anchors_world)
        spread = float(np.mean(np.linalg.norm(aw - aw.mean(axis=0), axis=1)))
        reliable = resid <= self._EGO_RESID_MAX_REL * max(spread, 1e-9)
        logger.info(
            f"[v7-CAM] {cam}: Sim(3) anchored on {len(anchors_slam)} anchors, "
            f"residual {resid:.3f} m, reliable={reliable}"
        )

        # per-frame world extrinsics: SLAM pose (c2w: t=centre, q=rotation)
        T = len(frame_ids_sorted)
        ego_ext = np.full((T, 3, 4), np.nan, dtype=np.float32)
        for i in range(T):
            pose = poses[i]
            if not np.isfinite(pose).all():
                continue
            R_wc = _quat_to_rot(pose[3:])
            C = s * (R_al @ pose[:3]) + t_al
            R_world = R_al @ R_wc
            ego_ext[i, :3, :3] = R_world.T
            ego_ext[i, :3, 3] = -R_world.T @ C
        return {
            "static": False,
            "reliable": reliable,
            "ego_ext": ego_ext,
            "ego_frames": np.asarray(frame_ids_sorted, dtype=np.int32),
            "residual": resid,
        }

    # ── orchestrator ──────────────────────────────────────────────────────────

    def run_scene(
        self,
        scene_dir: Path,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path],
        moving_cams: list[str] | None = None,
    ) -> Path | None:
        scene_dir = Path(scene_dir)
        out_path = scene_dir / "reid_cameras.npz"
        if out_path.exists() and not self.force:
            logger.info(f"[v7-CAM] {scene_dir.name}: reid_cameras.npz exists — skipping")
            return out_path

        moving = set(moving_cams or [])
        cams = sorted(video_dirs)
        statics = [c for c in cams if c not in moving]
        frame_maps = {
            c: self._index_frames(Path(frames_dirs[c])) for c in cams if c in frames_dirs
        }
        for c in cams:
            if c not in frame_maps or not frame_maps[c]:
                logger.warning(f"[v7-CAM] {scene_dir.name}: no frames for {c}")
                return None

        # 1–2. tuples on static cams only
        tuples_sel = self._select_tuple_frames(statics, video_dirs, frame_maps)
        tuples = []
        for tup in tuples_sel:
            res = self._run_tuple(tup, cams, frame_maps)
            if res is not None:
                tuples.append(res)
        logger.info(f"[v7-CAM] {scene_dir.name}: {len(tuples)}/{len(tuples_sel)} tuples ran")
        if len(tuples) < 3:
            logger.warning(f"[v7-CAM] {scene_dir.name}: too few tuples — aborting")
            return None

        # 3–4. consensus rig
        ext, inlier_frac, pair_sigma, ref, inlier_ids = self._consensus_extrinsics(
            tuples, cams
        )
        for k, c in enumerate(cams):
            logger.info(
                f"[v7-CAM] {c}: inlier_frac={inlier_frac[k]:.2f}"
                + (" (UNRELIABLE)" if inlier_frac[k] < 0.6 else "")
            )

        # consensus intrinsics = median over inlier tuples
        intr = np.full((len(cams), 3, 3), np.nan)
        size = np.zeros((len(cams), 2), dtype=np.int32)
        for k in range(len(cams)):
            mats = [tuples[i]["intr"][k] for i in inlier_ids.get(k, [])
                    if np.isfinite(tuples[i]["intr"][k]).all()]
            if mats:
                intr[k] = np.median(np.stack(mats), axis=0)
            sizes = [tuples[i]["size"][k] for i in range(len(tuples))
                     if tuples[i]["size"][k].any()]
            if sizes:
                size[k] = sizes[0]

        # 5. scale — default: stay in normalized VGGT units (scale=1); the v7
        # matcher converts metric thresholds via median per-track λ.
        if self.metric_scale_mode == "mapanything":
            scale, scale_ok = self._metric_scale(tuples, cams, frame_maps)
        else:
            scale, scale_ok = 1.0, False
        ext_metric = ext.copy()
        ext_metric[:, :, 3] *= scale
        pair_sigma_m = pair_sigma * scale

        # 6. person depth anchors from main tuples
        track_meta = {c: self._load_track_meta(video_dirs[c]) for c in cams}
        samples: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
        for t in tuples:
            self._harvest_depth(t, cams, video_dirs, track_meta, scale, samples)

        # 6b. anchor top-up for uncovered tracks
        # lowest-coverage frame per static cam (reuse first tuple's picks)
        static_low_cov = {c: tuples[0]["tup"][c] for c in statics if c in tuples[0]["tup"]}
        extra = 0
        for cam in cams:
            for pid, (frames, bboxes) in track_meta[cam].items():
                have = len(samples.get((cam, pid), []))
                if have >= self.min_anchor_frames or extra >= self.max_anchor_tuples:
                    continue
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                order = np.argsort(-areas)
                for oi in order[: self.min_anchor_frames - have]:
                    fi = int(frames[oi])
                    if fi not in frame_maps[cam]:
                        continue
                    tup = dict(static_low_cov)
                    tup[cam] = fi
                    res = self._run_tuple(tup, cams, frame_maps)
                    extra += 1
                    if res is not None:
                        self._harvest_depth(
                            res, cams, video_dirs, track_meta, scale, samples,
                            only_cam=cam,
                        )
                    if extra >= self.max_anchor_tuples:
                        break
        n_cov = sum(1 for v in samples.values() if len(v) >= self.min_anchor_frames)
        logger.info(
            f"[v7-CAM] {scene_dir.name}: depth anchors for {n_cov} tracks "
            f"(+{extra} anchor tuples)"
        )

        # 7. moving cams
        ego_data: dict[str, dict] = {}
        is_moving = np.zeros(len(cams), dtype=bool)
        for cam in cams:
            if cam not in moving:
                continue
            res = self._run_moving_cam(
                cam, cams, frame_maps, video_dirs, ext, intr, size,
                tuples[0]["vggt_hw"], ref, scale, static_low_cov,
            )
            if res is None:
                is_moving[cams.index(cam)] = True  # moving but no data → unreliable
                continue
            if res.get("static"):
                continue  # SLAM says it never moved; consensus extrinsic stands
            is_moving[cams.index(cam)] = True
            ego_data[cam] = res

        # ── save ──────────────────────────────────────────────────────────────
        arrays: dict[str, np.ndarray] = {
            "camera_names": np.array(cams, dtype="S64"),
            "is_moving": is_moving,
            "extrinsics_static": ext_metric.astype(np.float32),
            "intrinsics": intr.astype(np.float32),
            "original_size": size,
            "vggt_hw": np.asarray(tuples[0]["vggt_hw"], dtype=np.int32),
            "cam_inlier_frac": inlier_frac,
            "pair_sigma": pair_sigma_m.astype(np.float32),
            "metric_scale": np.float32(scale),
            "metric_scale_ok": np.bool_(scale_ok),
            "ref_cam": np.int32(ref),
        }
        for (cam, pid), rows in samples.items():
            arrays[f"person_depth_{cam}_{pid}"] = np.asarray(rows, dtype=np.float32)
        for cam, res in ego_data.items():
            if "ego_ext" in res:
                arrays[f"ego_extrinsics_{cam}"] = res["ego_ext"]
                arrays[f"ego_frame_indices_{cam}"] = res["ego_frames"]
                arrays[f"ego_anchor_residual_{cam}"] = np.float32(res["residual"])
                arrays[f"ego_reliable_{cam}"] = np.bool_(res["reliable"])
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(out_path)

        meta = {
            "n_tuples_ran": len(tuples),
            "tuple_frames": [t["tup"] for t in tuples],
            "ref_cam": cams[ref],
            "cam_inlier_frac": {c: float(inlier_frac[k]) for k, c in enumerate(cams)},
            "metric_scale": scale,
            "metric_scale_ok": bool(scale_ok),
            "metric_scale_mode": self.metric_scale_mode,
            "tracks_with_anchors": n_cov,
            "extra_anchor_tuples": extra,
            "moving_cams": sorted(moving),
        }
        (scene_dir / "reid_cameras_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[v7-CAM] {scene_dir.name}: wrote {out_path}")
        return out_path


# ── DROID-SLAM runners (copied from cross_view_v4, adapted to return frame ids)

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ], dtype=np.float64)


def _droid_run(
    frames_dir: Path, fx: float, fy: float, cx: float, cy: float,
    weights: str, droid_root: str,
) -> tuple[np.ndarray, list[int]] | None:
    """Run DROID-SLAM over a full video.  Returns (poses (T,7) c2w
    [tx,ty,tz,qx,qy,qz,qw], sorted frame indices parsed from filenames)."""
    import sys as _sys

    import cv2
    import torch as _torch

    if droid_root not in _sys.path:
        _sys.path.insert(0, str(droid_root))
        _sys.path.insert(0, str(Path(droid_root) / "droid_slam"))
    from droid import Droid
    try:
        _torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    img_files = sorted(
        p for p in Path(frames_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    if not img_files:
        logger.warning(f"DROID-SLAM: no images in {frames_dir}")
        return None
    frame_ids = []
    for p in img_files:
        m = _INT_RE.search(p.stem)
        frame_ids.append(int(m.group(0)) if m else len(frame_ids))

    def _stream():
        for t, imfile in enumerate(img_files):
            image = cv2.imread(str(imfile))
            h0, w0 = image.shape[:2]
            sc = np.sqrt((384 * 512) / (h0 * w0))
            h1, w1 = int(h0 * sc), int(w0 * sc)
            image = cv2.resize(image, (w1, h1))
            image = image[: h1 - h1 % 8, : w1 - w1 % 8]
            image = _torch.as_tensor(image).permute(2, 0, 1)
            intr = _torch.tensor([fx * w1 / w0, fy * h1 / h0, cx * w1 / w0, cy * h1 / h0])
            yield t, image[None], intr

    stream = list(_stream())
    H_slam, W_slam = stream[0][1].shape[2], stream[0][1].shape[3]
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
        logger.warning(f"DROID-SLAM backend failed ({e}); skipping.")
        del droid
        _torch.cuda.empty_cache()
        return None
    del droid
    _torch.cuda.empty_cache()
    return np.asarray(poses), frame_ids


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from configuration import CONFIG

    ap = argparse.ArgumentParser(description="ReID v7 camera pass (Stage A)")
    ap.add_argument("--output_dir", type=Path, default=Path(CONFIG.data.output_directory))
    ap.add_argument("--frames_root", type=Path, required=True,
                    help="Root with <scene>/<cam>/ image folders")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--moving_cams", nargs="*", default=[],
                    help="Camera names treated as moving (e.g. aria01)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_tuples", type=int, default=16)
    ap.add_argument("--metric_scale", choices=["none", "mapanything"], default="none")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    vggt_ckpt = CONFIG.data.vggt_omega_checkpoint
    droid_w = getattr(CONFIG.data, "droid_weights", None)
    runner = ReidCameraPass(
        vggt_weights=vggt_ckpt, droid_weights=droid_w,
        device=args.device, n_tuples=args.n_tuples,
        metric_scale_mode=args.metric_scale, force=args.force,
    )
    scene_dirs = (
        [args.output_dir / s for s in args.scenes]
        if args.scenes
        else sorted(p for p in args.output_dir.iterdir() if p.is_dir())
    )
    for scene_dir in scene_dirs:
        video_dirs = {
            p.name: p for p in sorted(scene_dir.iterdir())
            if p.is_dir() and (p / "body_data").is_dir()
        }
        if len(video_dirs) < 2:
            logger.info(f"{scene_dir.name}: <2 cameras with body_data — skipped")
            continue
        frames_dirs = {
            c: args.frames_root / scene_dir.name / c for c in video_dirs
        }
        frames_dirs = {c: d for c, d in frames_dirs.items() if d.is_dir()}
        try:
            runner.run_scene(scene_dir, video_dirs, frames_dirs,
                             moving_cams=args.moving_cams)
        except Exception:
            logger.exception(f"{scene_dir.name}: camera pass FAILED")


if __name__ == "__main__":
    main()
