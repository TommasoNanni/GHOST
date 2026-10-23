"""Cross-view + within-video ReID v7 — geometry-first via background cameras.

FROZEN-GEOMETRY INVARIANT
-------------------------
The camera rig — extrinsics/intrinsics from ``reid_cameras.npz`` (estimated
from the static background by ``preprocessing/reid_cameras.py``) and the
per-camera time offsets δ — is frozen BEFORE the first correspondence decision
and is NEVER refit from correspondences.  Every previous version (v1–v6)
failed by deriving geometry from the people, making correspondences ↔
transform ↔ δ circular.

DEPTH-FREE ASSOCIATION (triangulation)
--------------------------------------
We do NOT estimate any per-person depth (SAM3D ``pred_cam_t.z`` is off by
±40 % per person → the v7-first-cut δ was garbage).  Instead: given the rig,
two tracks are the same person iff their 2D pelvis sightlines actually cross
in 3D.  For a candidate pair at time offset δ, each aligned frame's two pelvis
rays are triangulated (2-view DLT) to a single 3D point and reprojected; the
residual (normalised by focal length → radians) is ~pixel-noise when the rays
truly meet (same person) and large when they pass metres apart (different
people).  No depth, no λ, no scale — only "do the sightlines cross?".

Algorithm
---------
1.  Load the cached rig.  Missing cache → log + no-op (fail-open).
2.  Load tracks (v6 loader: kp2d-vs-bbox QA, median betas, TransReID/DINOv3
    appearance) + hip-midpoint 2D pelvis.  Build per-camera projection
    matrices P_k = K_k·[R_k|t_k] (per-frame for moving cams).
3.  Within-video repair (single camera → 2D only): duplicate removal, gap
    merges with 2D-velocity extrapolation + appearance/betas agreement.
4.  δ per camera pair: Hungarian assignment of triangulation-reprojection cost
    over dynamic tracks, coarse→fine scan, plateau detection; global consensus
    via the Synchronizer least-squares solver + cycle-consistency weights;
    per-pair referee.
5.  Association: two tracks match iff their pelvis sightlines cross with low
    reprojection residual over the aligned overlap (dynamic pairs at consensus
    δ; static pairs are δ-invariant → median pelvis).  Appearance
    (TransReID+betas) fires only where geometry is silent (no temporal
    overlap, unreliable camera, uncertain δ) at v6's strong-edge gates.
    Geometry VETOES any merge whose sightlines demonstrably do not cross.
6.  Greedy Union-Find with source (≤1 track/cam/cluster), size (≤ n_cams) and
    veto constraints → global ids → remap npz/masks/json (v6 apply block).
    Sentinel: ``cross_view_reid.json``.

Reliability asymmetry: a track on an unreliable camera can neither confirm NOR
veto geometrically — only the appearance strong-edge path can merge it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# MHR70 keypoint indices (SAM3D pred_keypoints_2d ordering)
_HIP_L, _HIP_R = 9, 10

_BIG = 1e9
# reprojection penalty (radians) for a frame whose rays don't intersect in
# front of both cameras — inflates a wrong pair's cost, trimmed out for a
# correct pair that has a few bad frames.
_BIG_RAD = 0.5


# ── small helpers ─────────────────────────────────────────────────────────────

def _cos01(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine similarity mapped to [0, 1]; None if either vector is missing."""
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    return float((np.dot(a, b) / (na * nb) + 1.0) / 2.0)


def _bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return float(inter / (a1 + a2 - inter + 1e-9))


def _trimmed_mean(vals, trim: float = 0.30) -> float:
    """Mean over the best (1-trim) fraction of values."""
    a = np.sort(np.asarray(vals, dtype=float))
    if a.size == 0:
        return float("inf")
    keep = a[: max(1, int(np.ceil(a.size * (1.0 - trim))))]
    return float(np.mean(keep))


def _bbox_diag(b: np.ndarray) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


# ── triangulation (2-view DLT + reprojection residual) ──────────────────────

def _stack_P(geo: "_Geo", cam: str, frames: np.ndarray) -> np.ndarray | None:
    """(M,3,4) projection matrices for `cam` at the given frames.
    Static cam → broadcast; moving cam → per-frame gather (None if any miss)."""
    if cam in geo.P:
        return np.broadcast_to(geo.P[cam], (len(frames), 3, 4))
    d = geo.Pf.get(cam)
    if not d:
        return None
    out = []
    for f in frames:
        P = d.get(int(f))
        if P is None:
            return None
        out.append(P)
    return np.asarray(out, dtype=np.float64)


def _triangulate_batch(Pa: np.ndarray, Pb: np.ndarray,
                       xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """Vectorised 2-view DLT + reprojection residual (pixels) over M frames.

    Pa,Pb: (M,3,4)  xa,xb: (M,2) → residual (M,), inf where the point is not
    in front of both cameras."""
    M = xa.shape[0]
    r1 = xa[:, 0:1] * Pa[:, 2, :] - Pa[:, 0, :]      # (M,4)
    r2 = xa[:, 1:2] * Pa[:, 2, :] - Pa[:, 1, :]
    r3 = xb[:, 0:1] * Pb[:, 2, :] - Pb[:, 0, :]
    r4 = xb[:, 1:2] * Pb[:, 2, :] - Pb[:, 1, :]
    A = np.stack([r1, r2, r3, r4], axis=1)           # (M,4,4)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return np.full(M, np.inf)
    Xh = Vt[:, -1, :]                                # (M,4)
    w = Xh[:, 3]
    valid = np.abs(w) > 1e-9
    w = np.where(valid, w, 1.0)
    Xh = Xh / w[:, None]                             # homogeneous, last=1
    pa = np.einsum("mij,mj->mi", Pa, Xh)             # (M,3)
    pb = np.einsum("mij,mj->mi", Pb, Xh)
    front = (pa[:, 2] > 1e-6) & (pb[:, 2] > 1e-6) & valid
    ea = np.hypot(pa[:, 0] / pa[:, 2] - xa[:, 0], pa[:, 1] / pa[:, 2] - xa[:, 1])
    eb = np.hypot(pb[:, 0] / pb[:, 2] - xb[:, 0], pb[:, 1] / pb[:, 2] - xb[:, 1])
    res = 0.5 * (ea + eb)
    res[~front] = np.inf
    return res


class _Geo:
    """Projection matrices derived from the rig (VGGT pixel space)."""

    __slots__ = ("rig", "P", "Pf", "focal", "Wv", "Hv")

    def __init__(self, rig: "_Rig"):
        self.rig = rig
        self.P: dict[str, np.ndarray] = {}          # static cam → (3,4)
        self.Pf: dict[str, dict[int, np.ndarray]] = {}   # moving cam → {frame:(3,4)}
        self.focal: dict[str, float] = {}
        self.Hv, self.Wv = int(rig.vggt_hw[0]), int(rig.vggt_hw[1])

    def build(self) -> None:
        rig = self.rig
        for cam in rig.cams:
            k = rig.k_of(cam)
            K = rig.intr[k].astype(np.float64)
            if not np.isfinite(K).all():
                continue
            self.focal[cam] = float(K[0, 0])
            if rig.is_moving[k] and cam in rig.ego:
                ext = rig.ego[cam]["ext"]
                self.Pf[cam] = {
                    f: K @ ext[row] for f, row in rig.ego[cam]["row"].items()
                    if np.isfinite(ext[row]).all()
                }
            elif np.isfinite(rig.ext[k]).all():
                self.P[cam] = K @ rig.ext[k].astype(np.float64)

    def proj(self, cam: str, frame: int) -> np.ndarray | None:
        if cam in self.P:
            return self.P[cam]
        d = self.Pf.get(cam)
        return d.get(int(frame)) if d else None

    def focal_mean(self, va: str, vb: str) -> float:
        return 0.5 * (self.focal.get(va, 1000.0) + self.focal.get(vb, 1000.0))


class _Track:
    """Per-(camera, pid) track state.  Plain attribute bag."""

    __slots__ = (
        "frames", "pelvis2d", "pel_v", "bbox", "betas", "app",
        "frame_pos", "conf", "geo_ok", "strict", "dyn",
    )

    def __init__(self):
        self.frames = np.empty(0, int)
        self.pelvis2d = np.empty((0, 2), np.float32)   # original image coords
        self.pel_v = None                              # VGGT pixel coords (N,2)
        self.bbox = np.empty((0, 4), np.float32)
        self.betas = None
        self.app = None
        self.conf = np.empty(0)
        self.geo_ok = False      # camera reliable + projectable
        self.strict = False      # camera reliable (can veto)
        self.dyn = False         # enough 2D image motion to disambiguate δ
        self.frame_pos: dict[int, int] = {}


class _Rig:
    """Camera rig loaded from reid_cameras.npz."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.cams = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in d["camera_names"]]
        self.ext = d["extrinsics_static"].astype(np.float64)      # (K,3,4)
        self.intr = d["intrinsics"].astype(np.float64)             # (K,3,3)
        self.size = d["original_size"].astype(int)                 # (K,2) [W,H]
        self.vggt_hw = tuple(int(v) for v in d["vggt_hw"])
        self.inlier_frac = d["cam_inlier_frac"].astype(float)
        self.is_moving = d["is_moving"].astype(bool)
        self.ego: dict[str, dict] = {}
        for cam in self.cams:
            if f"ego_extrinsics_{cam}" in d.files:
                fi = d[f"ego_frame_indices_{cam}"].astype(int)
                self.ego[cam] = {
                    "ext": d[f"ego_extrinsics_{cam}"].astype(np.float64),
                    "row": {int(f): i for i, f in enumerate(fi)},
                    "reliable": bool(d.get(f"ego_reliable_{cam}", False)),
                }

    def k_of(self, cam: str) -> int:
        return self.cams.index(cam)

    def cam_geo_reliable(self, cam: str, min_inlier_frac: float) -> bool:
        k = self.k_of(cam)
        if self.is_moving[k]:
            return cam in self.ego and self.ego[cam]["reliable"]
        return (
            float(self.inlier_frac[k]) >= min_inlier_frac
            and np.isfinite(self.ext[k]).all()
        )


# ── main class ────────────────────────────────────────────────────────────────

class CrossViewReidentifierV7:
    """Geometry-first cross-view + within-video ReID (see module docstring)."""

    def __init__(
        self,
        reid_ckpt: str | None = None,
        fps: float = 15.0,
        # camera reliability
        cam_min_inlier_frac: float = 0.60,
        # δ estimation
        delta_max: int = 600,
        delta_coarse_step: int = 5,
        delta_min_overlap: int = 100,
        overlap_penalty_k: float = 30.0,
        delta_betas_pen: float = 0.0,    # betas-mismatch tiebreak (off: too noisy)
        dyn_2d_frac: float = 0.02,      # pelvis 2D std / image-diagonal
        plateau_margin: float = 0.10,
        referee_margin: float = 0.10,
        # association (reprojection residual, radians)
        geo_confirm_rad: float = 0.03,  # ~1.7°  → same person
        geo_veto_rad: float = 0.10,     # ~5.7°  → provably different
        min_geo_overlap: int = 30,
        app_strong_edge: float = 0.87,
        app_ratio: float = 1.05,
        # within-video repair (2D)
        repair_gap_max: int = 90,
        repair_gap_2d_frac: float = 0.5,   # predicted-vs-actual / bbox-diag
        repair_betas_min: float = 0.90,
        dup_2d_frac: float = 0.30,         # pelvis 2D dist / bbox-diag
        dup_iou: float = 0.50,
        dup_min_overlap: int = 15,
        repair_enabled: bool = True,
        body_subdir: str = "body_data",
    ):
        self.reid_ckpt = reid_ckpt
        self.body_subdir = body_subdir
        self.fps = fps
        self.cam_min_inlier_frac = cam_min_inlier_frac
        self.delta_max = delta_max
        self.delta_coarse_step = delta_coarse_step
        self.delta_min_overlap = delta_min_overlap
        self.overlap_penalty_k = overlap_penalty_k
        self.delta_betas_pen = delta_betas_pen
        self.dyn_2d_frac = dyn_2d_frac
        self.plateau_margin = plateau_margin
        self.referee_margin = referee_margin
        self.geo_confirm_rad = geo_confirm_rad
        self.geo_veto_rad = geo_veto_rad
        self.min_geo_overlap = min_geo_overlap
        self.app_strong_edge = app_strong_edge
        self.app_ratio = app_ratio
        self.repair_gap_max = repair_gap_max
        self.repair_gap_2d_frac = repair_gap_2d_frac
        self.repair_betas_min = repair_betas_min
        self.dup_2d_frac = dup_2d_frac
        self.dup_iou = dup_iou
        self.dup_min_overlap = dup_min_overlap
        self.repair_enabled = repair_enabled
        self._reid = None

    # ── TransReID (copied pattern from v6) ────────────────────────────────────

    def _get_reid(self):
        if self._reid is None and self.reid_ckpt:
            from preprocessing.transreid_extractor import TransReIDExtractor
            self._reid = TransReIDExtractor(self.reid_ckpt)
        return self._reid

    # ── entry point (same signature as v5/v6) ─────────────────────────────────

    def match_across_views(
        self,
        scene,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path] | None = None,
        intrinsics_map=None,
        dry_run: bool = False,
    ) -> list[set] | None:
        """Returns the final clusters (list of sets of (vid, pid) nodes) for
        scoring harnesses; None when the scene was skipped."""
        scene_id = scene.scene_id
        scene_dir = Path(next(iter(video_dirs.values()))).parent
        if not dry_run and (scene_dir / "cross_view_reid.json").exists():
            logger.info(f"Scene {scene_id}: cross-view ReID v7 already done, skipping")
            return None
        if frames_dirs is None:
            frames_dirs = {}
            for video in scene.videos:
                if getattr(video, "frames_home", None) is not None:
                    frames_dirs[video.video_id] = video.frames_home

        rig_path = scene_dir / "reid_cameras.npz"
        if not rig_path.exists():
            logger.warning(
                f"[v7-CAM] Scene {scene_id}: no reid_cameras.npz — run "
                f"preprocessing/reid_cameras.py first.  No-op."
            )
            return None
        rig = _Rig(rig_path)

        tracks, geo = self._build_geometry(rig, video_dirs, frames_dirs)
        if len({v for v, _ in tracks}) < 2:
            logger.info(f"Scene {scene_id}: fewer than 2 views with tracks, skipping")
            return None

        # ── Stage: within-video repair (2D, before δ/association) ────────────
        if self.repair_enabled:
            ops = self._within_video_repair(tracks)
            if ops and not dry_run:
                self._apply_repair_ops(ops, video_dirs)
                tracks, geo = self._build_geometry(rig, video_dirs, frames_dirs)
            elif ops:
                logger.info(
                    f"[v7-REPAIR] dry_run: {len(ops)} op(s) printed only; "
                    f"matching continues on unrepaired tracks"
                )

        active_vids = sorted({v for v, _ in tracks})
        person_pids = {
            v: sorted(p for vv, p in tracks if vv == v) for v in active_vids
        }

        # ── Stage: δ per camera pair + global consensus ───────────────────────
        deltas, delta_state = self._solve_deltas(tracks, active_vids, geo)

        # ── Stage: association ────────────────────────────────────────────────
        edges, vetoes = self._build_edges(
            tracks, active_vids, deltas, delta_state, geo
        )

        clusters, n_merged = self._cluster(
            edges, vetoes, tracks, active_vids, person_pids
        )

        # ── Global ID assignment (v6 pattern) ─────────────────────────────────
        global_remap: dict[str, dict[int, int]] = {v: {} for v in active_vids}
        gid = 1
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

        n_comps = len(clusters)
        n_remaps = sum(len(m) for m in global_remap.values())
        logger.info(
            f"Scene {scene_id}: v7 → {n_comps} global person(s), "
            f"{n_merged} merge(s), {n_remaps} remap(s), {len(vetoes)} veto pair(s) "
            f"across {len(active_vids)} view(s)"
        )

        if dry_run:
            print(f"\n[DRY RUN v7] Scene {scene_id}: {n_comps} global persons")
            for cl in sorted(clusters, key=lambda c: -len({v for v, _ in c})):
                cams = sorted({v for v, _ in cl})
                tag = f"{len(cams)}-cam" if len(cams) >= 2 else "singleton"
                mstr = " ".join(f"{v.split('_')[-1]}:P{p}" for v, p in sorted(cl))
                print(f"  [{tag}] {mstr}")
            return clusters

        # ── Apply remaps (v6 block, .v7tmp.npz staging) ────────────────────────
        from preprocessing.cross_view_reid_v2 import CrossVideoReidentifierV2
        for vid_id, remap in global_remap.items():
            if not remap:
                continue
            vid_dir = Path(video_dirs[vid_id])
            body_dir = vid_dir / "body_data"
            tmp: list[tuple[Path, Path]] = []
            for old, new in remap.items():
                src_path = body_dir / f"person_{old}.npz"
                if src_path.exists():
                    tmp_path = body_dir / f"person_{old}.v7tmp.npz"
                    src_path.rename(tmp_path)
                    tmp.append((tmp_path, body_dir / f"person_{new}.npz"))
            for tmp_path, dst_path in tmp:
                if dst_path.exists():
                    tmp_path.unlink()
                else:
                    tmp_path.rename(dst_path)
            CrossVideoReidentifierV2.apply_reid_remap(vid_dir, remap)

        (scene_dir / "cross_view_reid.json").write_text(
            json.dumps({"status": "done", "n_global": n_comps}, indent=2)
        )
        return clusters

    # ── track loading (v6 loader, lifted parts) ───────────────────────────────

    def _load_tracks(
        self,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path],
    ) -> dict[tuple[str, int], _Track]:
        reid = self._get_reid()
        tracks: dict[tuple[str, int], _Track] = {}

        for vid_id, vid_dir in video_dirs.items():
            body_dir = Path(vid_dir) / self.body_subdir
            # Derive pids from the actual person_*.npz files — NOT from
            # body_params_summary.json, whose "persons" keys go stale after
            # within-video ops rename files (e.g. files person_4/7/10 while the
            # summary still lists 1/2/3 → tracks silently dropped).
            pids = sorted(int(p.stem.split("_", 1)[1])
                          for p in body_dir.glob("person_*.npz"))
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

            fdir = frames_dirs.get(vid_id)
            jdir = Path(vid_dir) / "json_data"
            cam_suffix = str(vid_id).split("_")[-1]

            for pid in pids:
                npz = body_dir / f"person_{pid}.npz"
                if not npz.exists():
                    continue
                node = (vid_id, pid)
                tr = _Track()

                with np.load(str(npz)) as d:
                    if not ("frame_indices" in d and "pred_keypoints_2d" in d
                            and "bbox" in d and "pred_cam_t" in d):
                        continue
                    fi = d["frame_indices"].astype(int)
                    kp2d = d["pred_keypoints_2d"].astype(np.float32)   # (T,J,2)
                    bb = d["bbox"].astype(np.float32)                  # (T,4)
                    ct = d["pred_cam_t"].astype(np.float32)            # (T,3)
                    if len(fi) == 0 or len(kp2d) != len(fi) or len(ct) != len(fi):
                        continue

                    # per-frame QA: ≤30% of 2D kps outside bbox (v6 filter —
                    # SAM3D hallucinates full bodies from partial crops)
                    J = kp2d.shape[1]
                    n_out = (
                        (kp2d[:, :, 0] < bb[:, None, 0])
                        | (kp2d[:, :, 0] > bb[:, None, 2])
                        | (kp2d[:, :, 1] < bb[:, None, 1])
                        | (kp2d[:, :, 1] > bb[:, None, 3])
                    ).sum(axis=1)
                    good = (n_out / J <= 0.30) & (ct[:, 2] > 0.5)
                    n_good = int(good.sum())
                    logger.info(f"  [v7-QA] {node}: {n_good}/{len(fi)} good frames")
                    if n_good < 15:
                        logger.info(f"  [v7-QA] {node}: DISCARD — only {n_good} good")
                        continue

                    order = np.argsort(fi[good])
                    tr.frames = fi[good][order]
                    tr.pelvis2d = (
                        0.5 * (kp2d[good][:, _HIP_L, :2] + kp2d[good][:, _HIP_R, :2])
                    )[order]
                    tr.bbox = bb[good][order]
                    tr.frame_pos = {int(f): i for i, f in enumerate(tr.frames)}

                    if "smplx_betas" in d and len(d["smplx_betas"]) == len(fi):
                        sm = np.median(d["smplx_betas"].astype(np.float32)[good], axis=0)
                        nm = np.linalg.norm(sm)
                        if nm > 0:
                            tr.betas = sm / nm

                # appearance: TransReID preferred; DINOv3 gallery-mean fallback
                av = None
                transreid_attempted = reid is not None and fdir is not None
                if transreid_attempted:
                    try:
                        av = reid.person_feature(fdir, jdir, cam_suffix, pid)
                    except Exception as e:
                        logger.warning(f"  [v7-APP] {node}: TransReID error ({e}) "
                                       f"→ DINOv3 fallback")
                        transreid_attempted = False
                if not transreid_attempted and av is None and pid in gallery:
                    f, c = gallery[pid]
                    w = c / (c.sum() + 1e-8)
                    m = f.T @ w
                    n = np.linalg.norm(m)
                    av = (m / n).astype(np.float32) if n > 1e-8 else None
                tr.app = av
                tracks[node] = tr
        return tracks

    # ── geometry setup (projection matrices, reliability, motion) ─────────────

    def _build_geometry(
        self,
        rig: _Rig,
        video_dirs: dict[str, Path],
        frames_dirs: dict[str, Path],
    ) -> tuple[dict[tuple[str, int], _Track], _Geo]:
        tracks = self._load_tracks(video_dirs, frames_dirs)
        geo = _Geo(rig)
        geo.build()
        diag2d = float(np.hypot(geo.Wv, geo.Hv))
        for (vid, pid), tr in tracks.items():
            k = rig.k_of(vid)
            W0, H0 = float(rig.size[k][0]), float(rig.size[k][1])
            if W0 > 0 and H0 > 0:
                tr.pel_v = tr.pelvis2d * np.array([geo.Wv / W0, geo.Hv / H0])
            tr.conf = np.ones(len(tr.frames))
            cam_ok = rig.cam_geo_reliable(vid, self.cam_min_inlier_frac)
            has_P = (vid in geo.P) or (vid in geo.Pf)
            tr.geo_ok = bool(cam_ok and has_P and tr.pel_v is not None)
            tr.strict = bool(cam_ok and has_P)
            if tr.pel_v is not None and len(tr.pel_v) >= 5:
                mot = float(np.linalg.norm(np.std(tr.pel_v, axis=0)))
                tr.dyn = mot / diag2d > self.dyn_2d_frac
            logger.info(f"  [v7-GEO] ({vid},P{pid}): geo_ok={tr.geo_ok} "
                        f"dyn={tr.dyn} cam_ok={cam_ok}")
        return tracks, geo

    # ── triangulation costs ───────────────────────────────────────────────────

    def _pair_reproj_cost(
        self, ta: _Track, tb: _Track, va: str, vb: str, geo: _Geo, delta: int,
    ) -> tuple[float, int]:
        """Trimmed-mean reprojection residual (radians) of the triangulated
        pelvis over aligned frames.  Convention: f_a = f_b + δ."""
        ia, ib = [], []
        for idx, f in enumerate(ta.frames):
            j = tb.frame_pos.get(int(f) - delta)
            if j is None:
                continue
            if ta.conf[idx] <= 0 or tb.conf[j] <= 0:
                continue
            ia.append(idx)
            ib.append(j)
        if not ia:
            return float("inf"), 0
        ia = np.asarray(ia)
        ib = np.asarray(ib)
        Pa = _stack_P(geo, va, ta.frames[ia])
        Pb = _stack_P(geo, vb, tb.frames[ib])
        if Pa is None or Pb is None:
            return float("inf"), 0
        res_px = _triangulate_batch(Pa, Pb, ta.pel_v[ia], tb.pel_v[ib])
        fm = geo.focal_mean(va, vb)
        res = np.where(np.isfinite(res_px), res_px / fm, _BIG_RAD)
        return _trimmed_mean(res), len(res)

    def _static_reproj_cost(
        self, ta: _Track, tb: _Track, va: str, vb: str, geo: _Geo,
    ) -> float:
        """δ-invariant residual (radians) for two static tracks: triangulate
        their median pelvis rays."""
        pa = np.median(ta.pel_v[ta.conf > 0], axis=0)
        pb = np.median(tb.pel_v[tb.conf > 0], axis=0)
        Pa = geo.proj(va, int(ta.frames[len(ta.frames) // 2]))
        Pb = geo.proj(vb, int(tb.frames[len(tb.frames) // 2]))
        if Pa is None or Pb is None:
            return float("inf")
        res_px = _triangulate_batch(
            Pa[None], Pb[None], pa[None], pb[None])[0]
        if not np.isfinite(res_px):
            return _BIG_RAD
        return float(res_px / geo.focal_mean(va, vb))

    # ── within-video repair (single camera → 2D only) ─────────────────────────

    def _within_video_repair(self, tracks: dict) -> list[dict]:
        ops: list[dict] = []
        by_cam: dict[str, list[tuple[int, _Track]]] = {}
        for (vid, pid), tr in tracks.items():
            by_cam.setdefault(vid, []).append((pid, tr))

        for vid, plist in by_cam.items():
            plist.sort()
            for ai in range(len(plist)):
                for bi in range(ai + 1, len(plist)):
                    pa, ta = plist[ai]
                    pb, tb = plist[bi]
                    common = np.intersect1d(ta.frames, tb.frames)
                    if len(common) >= self.dup_min_overlap:
                        # duplicate-track check (2D pelvis + bbox IoU)
                        ia = [ta.frame_pos[int(f)] for f in common]
                        ib = [tb.frame_pos[int(f)] for f in common]
                        d = np.linalg.norm(
                            ta.pelvis2d[ia] - tb.pelvis2d[ib], axis=1)
                        diag = np.array([
                            0.5 * (_bbox_diag(ta.bbox[i]) + _bbox_diag(tb.bbox[j]))
                            for i, j in zip(ia, ib)
                        ])
                        iou = [
                            _bbox_iou(ta.bbox[i], tb.bbox[j])
                            for i, j in zip(ia, ib)
                        ]
                        rel = np.median(d / np.maximum(diag, 1e-6))
                        if rel < self.dup_2d_frac and np.median(iou) > self.dup_iou:
                            drop = pb if len(tb.frames) <= len(ta.frames) else pa
                            ops.append({"op": "remove", "vid": vid, "pid": drop})
                            logger.info(
                                f"[v7-REPAIR] {vid}: duplicate P{pa}/P{pb} "
                                f"(rel2d={rel:.2f} IoU={np.median(iou):.2f}) "
                                f"→ remove P{drop}"
                            )
                        continue
                    if len(common):
                        continue
                    # gap-merge check (order by time)
                    if ta.frames[-1] < tb.frames[0]:
                        first, fp, second, sp = ta, pa, tb, pb
                    elif tb.frames[-1] < ta.frames[0]:
                        first, fp, second, sp = tb, pb, ta, pa
                    else:
                        continue
                    gap = int(second.frames[0] - first.frames[-1])
                    if not (0 < gap <= self.repair_gap_max):
                        continue
                    tail = min(5, len(first.frames) - 1)
                    if tail < 1:
                        continue
                    dt = first.frames[-1] - first.frames[-1 - tail]
                    vel = (first.pelvis2d[-1] - first.pelvis2d[-1 - tail]) / max(dt, 1)
                    p_pred = first.pelvis2d[-1] + vel * gap
                    d_gap = float(np.linalg.norm(p_pred - second.pelvis2d[0]))
                    diag = 0.5 * (_bbox_diag(first.bbox[-1]) + _bbox_diag(second.bbox[0]))
                    if d_gap / max(diag, 1e-6) > self.repair_gap_2d_frac:
                        continue
                    b_sim = _cos01(first.betas, second.betas)
                    a_sim = _cos01(first.app, second.app)
                    if not ((b_sim is not None and b_sim >= self.repair_betas_min)
                            or (a_sim is not None and a_sim >= self.app_strong_edge)):
                        continue
                    ops.append({"op": "merge", "vid": vid,
                                "id_a": fp, "id_b": sp})
                    logger.info(
                        f"[v7-REPAIR] {vid}: gap-merge P{fp}+P{sp} "
                        f"(gap={gap}fr, rel2d={d_gap / max(diag, 1e-6):.2f}, "
                        f"betas={b_sim}, app={a_sim})"
                    )
        return ops

    @staticmethod
    def _apply_repair_ops(ops: list[dict], video_dirs: dict[str, Path]) -> None:
        from utilities.within_reid_operations import merge_tracks, remove_track
        # removes first, then merges (an id consumed by a merge must survive removal)
        merge_ids = {(o["vid"], o["id_a"]) for o in ops if o["op"] == "merge"}
        merge_ids |= {(o["vid"], o["id_b"]) for o in ops if o["op"] == "merge"}
        for o in ops:
            if o["op"] == "remove" and (o["vid"], o["pid"]) not in merge_ids:
                remove_track(Path(video_dirs[o["vid"]]), o["pid"])
        for o in ops:
            if o["op"] == "merge":
                merge_tracks(Path(video_dirs[o["vid"]]), o["id_a"], o["id_b"],
                             output_id=min(o["id_a"], o["id_b"]))

    # ── δ estimation ──────────────────────────────────────────────────────────

    def _pair_delta_cost(
        self, dyn_a: list[_Track], dyn_b: list[_Track],
        va: str, vb: str, geo: _Geo, delta: int,
    ) -> float:
        """Assignment cost (radians) of dynamic tracks at offset δ.
        Frame alignment: f_a = f_b + δ."""
        if not dyn_a or not dyn_b:
            return float("inf")
        C = np.full((len(dyn_a), len(dyn_b)), _BIG)
        for i, ta in enumerate(dyn_a):
            for j, tb in enumerate(dyn_b):
                cost, n = self._pair_reproj_cost(ta, tb, va, vb, geo, delta)
                if n < self.delta_min_overlap:
                    continue
                C[i, j] = cost * (1.0 + self.overlap_penalty_k / np.sqrt(n))
        ri, ci = linear_sum_assignment(C)
        # betas-mismatch tiebreak: a period-alias δ pairs a DIFFERENT person
        # (body shape differs) while the true δ pairs the same person.  Adding
        # a small betas penalty to matched pairs pushes the argmin off aliases.
        costs = []
        for i, j in zip(ri, ci):
            if C[i, j] >= _BIG:
                continue
            c = C[i, j]
            b_sim = _cos01(dyn_a[i].betas, dyn_b[j].betas)
            if b_sim is not None:
                c += self.delta_betas_pen * (1.0 - b_sim)
            costs.append(c)
        return float(np.mean(costs)) if costs else float("inf")

    def _solve_deltas(
        self, tracks: dict, active_vids: list[str], geo: _Geo,
    ) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], dict]]:
        """Per-pair δ (f_a = f_b + δ) + state {local, cost curves, uncertain}."""
        rig = geo.rig
        dyn: dict[str, list[_Track]] = {v: [] for v in active_vids}
        for (vid, _pid), tr in tracks.items():
            if tr.geo_ok and tr.dyn and tr.pel_v is not None:
                dyn[vid].append(tr)

        Kc = len(active_vids)
        state: dict[tuple[str, str], dict] = {}
        for a in range(Kc):
            for b in range(a + 1, Kc):
                va, vb = active_vids[a], active_vids[b]
                st = {"local": None, "local_cost": float("inf"),
                      "uncertain": True, "known": False}
                state[(va, vb)] = st
                if not (rig.cam_geo_reliable(va, self.cam_min_inlier_frac)
                        and rig.cam_geo_reliable(vb, self.cam_min_inlier_frac)):
                    continue
                if not dyn[va] or not dyn[vb]:
                    continue
                coarse: dict[int, float] = {}
                for delta in range(-self.delta_max, self.delta_max + 1,
                                   self.delta_coarse_step):
                    coarse[delta] = self._pair_delta_cost(
                        dyn[va], dyn[vb], va, vb, geo, delta)
                finite = {d: c for d, c in coarse.items() if np.isfinite(c)}
                if not finite:
                    continue
                # refine around the 3 best coarse minima
                best3 = sorted(finite, key=finite.get)[:3]
                fine: dict[int, float] = dict(finite)
                for d0 in best3:
                    for delta in range(d0 - self.delta_coarse_step,
                                       d0 + self.delta_coarse_step + 1):
                        if delta not in fine:
                            fine[delta] = self._pair_delta_cost(
                                dyn[va], dyn[vb], va, vb, geo, delta)
                best = min((d for d in fine if np.isfinite(fine[d])),
                           key=lambda d: fine[d])
                costs = np.asarray([c for c in fine.values() if np.isfinite(c)])
                p10 = float(np.percentile(costs, 10))
                uncertain = fine[best] > (1.0 - self.plateau_margin) * p10
                st.update({"local": int(best), "local_cost": float(fine[best]),
                           "uncertain": bool(uncertain), "known": True})
                logger.info(
                    f"[v7-DELTA] {va}↔{vb}: δ*={best} cost={fine[best]:.4f}rad "
                    f"(p10={p10:.4f}, uncertain={uncertain}, "
                    f"{len(costs)} evaluated)"
                )

        # global consensus (Synchronizer LS — shared infra)
        consensus: dict[str, float] = {v: 0.0 for v in active_vids}
        known_pairs = [k for k, st in state.items() if st["known"]]
        if len(known_pairs) >= 1 and Kc >= 2:
            import torch
            from synchronize_videos.synchronizer import Synchronizer
            sync = Synchronizer(device="cpu", verbose=False)
            O = torch.zeros(Kc, Kc)
            Wt = torch.zeros(Kc, Kc)
            vidx = {v: i for i, v in enumerate(active_vids)}
            for (va, vb), st in state.items():
                if not st["known"]:
                    continue
                # our δ: f_a = f_b + δ ⇒ offset_matrix[a,b] = δ
                i, j = vidx[va], vidx[vb]
                O[i, j] = float(st["local"])
                O[j, i] = -float(st["local"])
                Wt[i, j] = Wt[j, i] = 1.0
            cyc = sync.cycle_consistency_weights(O)
            weights = cyc * Wt if cyc is not None else Wt
            t = sync.estimate_initial_times(O, weights)
            for v, i in vidx.items():
                consensus[v] = float(t[i])
            logger.info("[v7-DELTA] consensus clocks: "
                        + " ".join(f"{v}={consensus[v]:.1f}" for v in active_vids))

        # referee: local wins only if it beats consensus on evidence
        deltas: dict[tuple[str, str], int] = {}
        for (va, vb), st in state.items():
            d_cons = int(round(consensus[vb] - consensus[va]))
            if st["known"] and not st["uncertain"]:
                d_local = st["local"]
                if d_local != d_cons:
                    c_cons = self._pair_delta_cost(
                        dyn[va], dyn[vb], va, vb, geo, d_cons)
                    if st["local_cost"] <= c_cons * (1.0 - self.referee_margin):
                        deltas[(va, vb)] = d_local
                        logger.info(f"[v7-DELTA] {va}↔{vb}: referee keeps "
                                    f"local δ={d_local} (vs consensus {d_cons})")
                        continue
                deltas[(va, vb)] = d_cons
            else:
                deltas[(va, vb)] = d_cons
        return deltas, state

    # ── association ───────────────────────────────────────────────────────────

    def _build_edges(
        self,
        tracks: dict,
        active_vids: list[str],
        deltas: dict[tuple[str, str], int],
        delta_state: dict[tuple[str, str], dict],
        geo: _Geo,
    ) -> tuple[list[tuple[float, tuple, tuple]], set[frozenset]]:
        edges: list[tuple[float, tuple, tuple]] = []
        vetoes: set[frozenset] = set()
        by_cam: dict[str, list[tuple[tuple, _Track]]] = {v: [] for v in active_vids}
        for node, tr in tracks.items():
            by_cam[node[0]].append((node, tr))

        for ai in range(len(active_vids)):
            for bi in range(ai + 1, len(active_vids)):
                va, vb = active_vids[ai], active_vids[bi]
                delta = deltas.get((va, vb), 0)
                st = delta_state.get((va, vb), {"uncertain": True, "known": False})
                delta_usable = st["known"] and not st["uncertain"]
                nodes_a, nodes_b = by_cam[va], by_cam[vb]
                geo_had_say: set[tuple[int, int]] = set()

                # ── triangulation cost matrix ────────────────────────────────
                costs: dict[tuple[int, int], tuple[float, int]] = {}
                for i, (na, ta) in enumerate(nodes_a):
                    for j, (nb, tb) in enumerate(nodes_b):
                        if not (ta.geo_ok and tb.geo_ok):
                            continue
                        static_pair = (not ta.dyn) and (not tb.dyn)
                        if static_pair:
                            cost = self._static_reproj_cost(ta, tb, va, vb, geo)
                            n = min(len(ta.frames), len(tb.frames))
                            if not np.isfinite(cost):
                                continue
                        else:
                            if not delta_usable:
                                continue
                            cost, n = self._pair_reproj_cost(
                                ta, tb, va, vb, geo, delta)
                            if n < self.min_geo_overlap:
                                continue
                        costs[(i, j)] = (cost, n)

                if costs:
                    C = np.full((len(nodes_a), len(nodes_b)), _BIG)
                    for (i, j), (cost, _n) in costs.items():
                        C[i, j] = cost
                    ri, ci = linear_sum_assignment(C)
                    matched = {(i, j) for i, j in zip(ri, ci) if C[i, j] < _BIG}
                    for (i, j), (cost, _n) in costs.items():
                        na, ta = nodes_a[i]
                        nb, tb = nodes_b[j]
                        geo_had_say.add((i, j))
                        if cost <= self.geo_confirm_rad and (i, j) in matched:
                            sim = 2.0 - min(cost / self.geo_confirm_rad, 1.0)
                            edges.append((sim, na, nb))
                            logger.info(f"[v7-GEO] {na} ↔ {nb}: "
                                        f"resid={cost:.4f}rad → edge sim={sim:.3f}")
                        elif (ta.strict and tb.strict
                              and cost > self.geo_veto_rad):
                            vetoes.add(frozenset((na, nb)))
                            logger.info(f"[v7-VETO] {na} ⊘ {nb}: "
                                        f"resid={cost:.4f}rad")

                # ── appearance strong edges where geometry is silent ─────────
                app_sims = np.full((len(nodes_a), len(nodes_b)), -1.0)
                for i, (na, ta) in enumerate(nodes_a):
                    for j, (nb, tb) in enumerate(nodes_b):
                        if (i, j) in geo_had_say:
                            continue  # geometry confirmed or rejected
                        b_sim = _cos01(ta.betas, tb.betas)
                        a_sim = _cos01(ta.app, tb.app)
                        if a_sim is None:
                            continue
                        app_sims[i, j] = (0.5 * b_sim + 0.5 * a_sim
                                          if b_sim is not None else a_sim)
                for i, (na, ta) in enumerate(nodes_a):
                    for j, (nb, tb) in enumerate(nodes_b):
                        s = app_sims[i, j]
                        if s < self.app_strong_edge:
                            continue
                        if frozenset((na, nb)) in vetoes:
                            continue
                        # mutual best + ratio gates (v5/v6 pattern)
                        if (s < app_sims[i].max() or s < app_sims[:, j].max()):
                            continue
                        run_i = sorted(app_sims[i][app_sims[i] >= 0])[-2:]
                        run_j = sorted(app_sims[:, j][app_sims[:, j] >= 0])[-2:]
                        ru = max(
                            run_i[0] if len(run_i) == 2 else 0.0,
                            run_j[0] if len(run_j) == 2 else 0.0,
                        )
                        if ru > 0 and s / ru < self.app_ratio:
                            continue
                        edges.append((s, na, nb))
                        logger.info(f"[v7-APP] {na} ↔ {nb}: strong edge "
                                    f"sim={s:.3f}")
        return edges, vetoes

    # ── clustering (v6 greedy UF + veto awareness) ────────────────────────────

    def _cluster(
        self,
        edges: list[tuple[float, tuple, tuple]],
        vetoes: set[frozenset],
        tracks: dict,
        active_vids: list[str],
        person_pids: dict[str, list[int]],
    ) -> tuple[list[set], int]:
        nodes = list(tracks.keys())
        parent = {n: n for n in nodes}

        def _find(x: tuple) -> tuple:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _members(root: tuple) -> list[tuple]:
            return [n for n in nodes if _find(n) == root]

        n_cams = len(active_vids)
        n_merged = 0
        # per-node frame set (fragment-aware source constraint below)
        fset = {n: set(int(f) for f in tr.frames) for n, tr in tracks.items()}
        edges.sort(key=lambda e: -e[0])
        for s, ni, nj in edges:
            ri, rj = _find(ni), _find(nj)
            if ri == rj:
                continue
            mi, mj = _members(ri), _members(rj)
            ci, cj = {n[0] for n in mi}, {n[0] for n in mj}
            # Fragment-aware source constraint: two tracks from the SAME camera
            # may share a cluster ONLY if they are time-disjoint (fragments of
            # one person the raw tracker split).  If they overlap in time they
            # are simultaneously present → different people → forbid.  This
            # reproduces manual within-video gap-merges using the cross-view
            # geometry that already linked the fragments.
            conflict = False
            for cam in (ci & cj):
                anc = [n for n in mi if n[0] == cam]
                bnc = [n for n in mj if n[0] == cam]
                if any(fset[na] & fset[nb] for na in anc for nb in bnc):
                    conflict = True
                    break
            if conflict:
                logger.info(f"[v7-UF] REJECT {ni} ↔ {nj} sim={s:.3f} "
                            f"(same-cam time overlap → different people)")
                continue
            if any(frozenset((x, y)) in vetoes for x in mi for y in mj):
                logger.info(f"[v7-UF] REJECT {ni} ↔ {nj} sim={s:.3f} (veto)")
                continue
            parent[rj] = ri
            n_merged += 1
            logger.info(f"[v7-UF] merge {ni} ↔ {nj} sim={s:.3f}")

        root_to_idx: dict[tuple, int] = {}
        clusters: list[set] = []
        for n in nodes:
            r = _find(n)
            if r not in root_to_idx:
                root_to_idx[r] = len(clusters)
                clusters.append(set())
            clusters[root_to_idx[r]].add(n)
        return clusters, n_merged
