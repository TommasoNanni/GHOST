"""
3D animation of geometric cross-view ReID using the v2 constrained rigid model.

Two-phase animation:
  1. Camera-local view — each camera's tracks in their own coordinate frame,
     spread along the X axis so they don't overlap; coloured by camera.
  2. Aligned view — constrained rigid transform maps each non-reference camera
     into the reference camera's space; matched persons share a colour.

Alignment algorithm (mirrors cross_view_reid_v2.py):
  Model:  abs_a_i  ≈  λ_i · (R @ abs_b_i + t)
    R, t  — shared rotation and translation for the camera pair.
    λ_i   — per-person depth-scale scalar (accounts for monocular scale error).
  1. xcorr on pelvis-Z signal across all pairs → coarse delta.
  2. Scan coarse ± 5 frames:
       a. ALS pass 1: seed from the (ta, tb) pair with the most common frames.
       b. Hungarian re-assignment on per-pair RMSE under (R, t) from pass 1.
       c. ALS pass 2: fit on geo-assigned pairs.
  3. Accept best-delta result; accept individual pairs with RMSE < _MATCH_THR.

Requirements:
    pip install manim scipy  (manim community edition >= 0.18)

Usage:
    SCENE_DIR=/path/to/ghost_outputs/rich_train/ParkingLot2_016_stretching1 \\
        manim -pql visualize/reid_animation.py ReIDAnimation

    # High quality:
    SCENE_DIR=... manim -pqh visualize/reid_animation.py ReIDAnimation

Optional env vars:
    MAX_TRACKS        cap on total tracks loaded            (default: 20)
    SKELETON_INTERVAL frames between skeleton ghost poses   (default: 150)
    ALIGN_DELTA       max temporal offset (frames) searched (default: 30)
    SKIP_ALIGN        set to 1 to render only Phase 1       (default: 0)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from manim import *

# ── MHR70 joint indices ───────────────────────────────────────────────────────
JOINT_IDX: dict[str, int] = {
    "left_hip": 9,       "right_hip": 10,
    "left_knee": 11,     "right_knee": 12,
    "left_ankle": 13,    "right_ankle": 14,
    "neck": 69,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 62,    "right_wrist": 41,
}
JOINT_NAMES: list[str] = list(JOINT_IDX.keys()) + ["pelvis"]
N_JOINTS = len(JOINT_NAMES)
_JI: dict[str, int] = {n: i for i, n in enumerate(JOINT_NAMES)}

BONES: list[tuple[str, str]] = [
    ("neck",          "left_shoulder"),  ("neck",          "right_shoulder"),
    ("left_shoulder", "left_elbow"),     ("right_shoulder", "right_elbow"),
    ("left_elbow",    "left_wrist"),     ("right_elbow",   "right_wrist"),
    ("neck",          "pelvis"),
    ("pelvis",        "left_hip"),       ("pelvis",        "right_hip"),
    ("left_hip",      "left_knee"),      ("right_hip",     "right_knee"),
    ("left_knee",     "left_ankle"),     ("right_knee",    "right_ankle"),
]

CAM_COLORS  = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
PERS_COLORS = ["#FFFFFF", "#FFD700", "#00CED1", "#FF6347",
               "#9370DB", "#32CD32", "#FF69B4", "#20B2AA"]

SCENE_DIR         = Path(os.environ.get("SCENE_DIR", "."))
MAX_TRACKS        = int(os.environ.get("MAX_TRACKS",        "20"))
SKELETON_INTERVAL = int(os.environ.get("SKELETON_INTERVAL", "150"))
ALIGN_DELTA       = int(os.environ.get("ALIGN_DELTA",       "30"))
SKIP_ALIGN        = os.environ.get("SKIP_ALIGN", "0") != "0"
# Override reference camera (default: camera with the most tracks).
REF_CAM           = os.environ.get("REF_CAM", "")

# Per-joint RMSE threshold (metres) to accept a matched pair.
# Mirrors _MATCH_RMSE_THR in CrossVideoReidentifierV2.
_MATCH_THR   = 0.50
_MIN_OVERLAP = 30   # minimum common frames to attempt a fit


# ── Data loading ──────────────────────────────────────────────────────────────

def _extract_joints(kp3d: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """
    kp3d : (T, 70, 3)  root-centred keypoints in camera space
    cam_t: (T, 3)      camera-local root position (pred_cam_t)
    Returns (T, N_JOINTS, 3) absolute camera-space positions.
    """
    pos = np.zeros((len(kp3d), N_JOINTS, 3), dtype=np.float32)
    for name, idx in JOINT_IDX.items():
        pos[:, _JI[name]] = cam_t + kp3d[:, idx]
    pos[:, _JI["pelvis"]] = (
        pos[:, _JI["left_hip"]] + pos[:, _JI["right_hip"]]
    ) / 2.0
    return pos


def load_tracks(scene_dir: Path) -> list[dict]:
    """Glob body_data/person_*.npz under scene_dir; return list of track dicts."""
    tracks = []
    for body_file in sorted(scene_dir.glob("*/body_data/person_*.npz")):
        cam_id = body_file.parents[1].name
        data = np.load(body_file, allow_pickle=False)
        if "pred_keypoints_3d" not in data.files:
            continue
        kp3d  = data["pred_keypoints_3d"]
        cam_t = (data["pred_cam_t"]    if "pred_cam_t"    in data.files
                 else np.zeros((len(kp3d), 3), np.float32))
        fi    = (data["frame_indices"] if "frame_indices" in data.files
                 else np.arange(len(kp3d)))
        joints = _extract_joints(kp3d, cam_t)
        tracks.append({
            "cam_id":        cam_id,
            "pid":           body_file.stem,          # e.g. "person_1"
            "frame_indices": np.asarray(fi, np.int64),
            "joints":        joints,
            "pelvis":        joints[:, _JI["pelvis"]],
        })
    return tracks[:MAX_TRACKS]


# ── Constrained rigid + per-person scale alignment ────────────────────────────
#
# Model:  abs_a_i  ≈  λ_i · (R @ abs_b_i + t)
#
#   R  — 3×3 rotation (SO(3)), shared across all persons in the pair.
#   t  — 3-D translation, shared.
#   λ_i — scalar per person (monocular depth scale).
#
# ALS alternates:
#   (A) Fix λ_i: solve R, t via Procrustes on (src_i, dst_i / λ_i).
#   (B) Fix R, t: solve λ_i via closed-form 1-D LS.


def _apply_constrained(
    pts: np.ndarray, R: np.ndarray, t_vec: np.ndarray, lam: float
) -> np.ndarray:
    """Apply λ · (R @ pts + t) to any (..., 3) array."""
    shape = pts.shape
    flat  = pts.reshape(-1, 3).astype(np.float64)
    return (lam * ((R @ flat.T).T + t_vec)).reshape(shape)


def _common_joints(
    ta: dict, tb: dict, delta: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Return (src, dst) with matching frames under the given delta.
    delta: frame f of camera B corresponds to frame f+delta of camera A.
    Both outputs are (N_common_frames * N_JOINTS, 3) float64.
    """
    fa_map = {int(f): i for i, f in enumerate(ta["frame_indices"])}
    ib, ia = [], []
    for j, fb in enumerate(tb["frame_indices"]):
        fa = int(fb) + delta
        if fa in fa_map:
            ib.append(j)
            ia.append(fa_map[fa])
    if len(ib) < _MIN_OVERLAP:
        return None, None
    src = tb["joints"][ib].reshape(-1, 3).astype(np.float64)  # camera B
    dst = ta["joints"][ia].reshape(-1, 3).astype(np.float64)  # camera A (ref)
    return src, dst


def _constrained_als(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    max_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """ALS: abs_a ≈ λ_i · (R @ abs_b + t).  Returns (R, t, lambdas, avg_rmse)."""
    if not pairs:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), [], float("inf")

    R       = np.eye(3, dtype=np.float64)
    t_vec   = np.zeros(3, dtype=np.float64)
    lambdas = [1.0] * len(pairs)

    for _ in range(max_iter):
        # (A) Fix λ_i → solve R, t via Procrustes on (src_i, dst_i / λ_i)
        all_s = np.concatenate([s for s, _ in pairs])
        all_d = np.concatenate([d / lam for (_, d), lam in zip(pairs, lambdas)])
        mu_s, mu_d = all_s.mean(0), all_d.mean(0)
        H = (all_s - mu_s).T @ (all_d - mu_d)
        U, _, Vt = np.linalg.svd(H)
        R     = Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T
        t_vec = mu_d - R @ mu_s

        # (B) Fix R, t → solve each λ_i via closed-form 1-D LS
        for k, (src, dst) in enumerate(pairs):
            rsrc = (R @ src.T).T + t_vec
            den  = float(np.sum(rsrc * rsrc))
            lam  = float(np.sum(rsrc * dst)) / den if den > 1e-10 else 1.0
            lambdas[k] = float(np.clip(lam, 0.1, 10.0))

    rmses = []
    for k, (src, dst) in enumerate(pairs):
        pred = lambdas[k] * ((R @ src.T).T + t_vec)
        rmses.append(float(np.sqrt(((pred - dst) ** 2).mean())))
    return R, t_vec, lambdas, float(np.mean(rmses)) if rmses else float("inf")


def _xcorr_best_delta(
    fi_a: np.ndarray, pa: np.ndarray,
    fi_b: np.ndarray, pb: np.ndarray,
    max_delta: int,
) -> int:
    """Estimate frame offset via cross-correlation of pelvis depth (Z) signals."""
    f_min = min(int(fi_a.min()), int(fi_b.min()))
    f_max = max(int(fi_a.max()), int(fi_b.max()))
    n = f_max - f_min + 1
    sig_a, sig_b = np.zeros(n), np.zeros(n)
    for i, f in enumerate(fi_a):
        sig_a[int(f) - f_min] = pa[i, 2]
    for i, f in enumerate(fi_b):
        sig_b[int(f) - f_min] = pb[i, 2]
    sig_a -= sig_a.mean()
    sig_b -= sig_b.mean()
    corr = np.correlate(sig_a, sig_b, mode="full")
    lags = np.arange(-(n - 1), n)
    mask = np.abs(lags) <= max_delta
    return int(lags[mask][np.argmax(corr[mask])])


def compute_alignment(tracks: list[dict], ref_cam: str) -> dict:
    """
    Compute constrained rigid transforms from each non-ref camera into ref_cam.

    Returns {
        cam_b: {
            "R":       (3, 3) ndarray,
            "t":       (3,)   ndarray,
            "lambdas": {pid_str: float},    # only matched persons
            "matches": [(pid_b, pid_ref)],  # accepted (pid_b, pid_a) pairs
            "delta":   int,
            "rmse":    float,
        }
    }
    """
    ref_tracks = [t for t in tracks if t["cam_id"] == ref_cam]
    by_cam: dict[str, list[dict]] = {}
    for t in tracks:
        if t["cam_id"] != ref_cam:
            by_cam.setdefault(t["cam_id"], []).append(t)

    result: dict = {}
    for cam_b, b_tracks in by_cam.items():

        # ── Coarse delta: xcorr over all cross-camera pairs ───────────────────
        coarse_list = [
            _xcorr_best_delta(
                ta["frame_indices"], ta["pelvis"],
                tb["frame_indices"], tb["pelvis"],
                ALIGN_DELTA,
            )
            for ta in ref_tracks for tb in b_tracks
        ]
        coarse = int(np.median(coarse_list)) if coarse_list else 0

        best_rmse, best_result = float("inf"), None

        for delta in range(coarse - 5, coarse + 6):
            if abs(delta) > ALIGN_DELTA:
                continue

            # ── ALS pass 1: seed from the pair with the most common frames ────
            seed_pair: tuple[np.ndarray, np.ndarray] | None = None
            seed_n = 0
            for ta in ref_tracks:
                for tb in b_tracks:
                    src, dst = _common_joints(ta, tb, delta)
                    if src is not None and len(src) > seed_n:
                        seed_n, seed_pair = len(src), (src, dst)

            if seed_pair is None:
                continue

            R1, t1, _, _ = _constrained_als([seed_pair])

            # ── Hungarian re-assignment on RMSE under R1, t1 ─────────────────
            Na, Nb = len(ref_tracks), len(b_tracks)
            mat = np.full((Nb, Na), 1e6, dtype=np.float64)
            for j, tb in enumerate(b_tracks):
                for i, ta in enumerate(ref_tracks):
                    src, dst = _common_joints(ta, tb, delta)
                    if src is None:
                        continue
                    rsrc = (R1 @ src.T).T + t1
                    den  = float(np.sum(rsrc * rsrc))
                    lam  = float(np.clip(
                        float(np.sum(rsrc * dst)) / den if den > 1e-10 else 1.0,
                        0.1, 10.0,
                    ))
                    r = float(np.sqrt(((lam * rsrc - dst) ** 2).mean()))
                    if np.isfinite(r):
                        mat[j, i] = r

            rows, cols = linear_sum_assignment(mat)
            geo_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            geo_idx:  list[tuple[int, int]] = []
            for j, i in zip(rows, cols):
                if mat[j, i] > 1e5:
                    continue
                src, dst = _common_joints(ref_tracks[i], b_tracks[j], delta)
                if src is not None:
                    geo_pairs.append((src, dst))
                    geo_idx.append((j, i))

            if not geo_pairs:
                continue

            # ── ALS pass 2: full geo-assigned pairs ───────────────────────────
            R2, t2, _, rmse2 = _constrained_als(geo_pairs)
            if rmse2 >= best_rmse:
                continue

            best_rmse = rmse2

            # Re-evaluate per-person λ under final (R2, t2); accept by threshold
            matches: list[tuple[str, str]] = []
            lambdas: dict[str, float] = {}
            for j, i in geo_idx:
                ta = ref_tracks[i]
                tb = b_tracks[j]
                src, dst = _common_joints(ta, tb, delta)
                if src is None:
                    continue
                rsrc = (R2 @ src.T).T + t2
                den  = float(np.sum(rsrc * rsrc))
                lam  = float(np.clip(
                    float(np.sum(rsrc * dst)) / den if den > 1e-10 else 1.0,
                    0.1, 10.0,
                ))
                r = float(np.sqrt(((lam * rsrc - dst) ** 2).mean()))
                if r < _MATCH_THR:
                    matches.append((tb["pid"], ta["pid"]))
                    lambdas[tb["pid"]] = lam

            best_result = {
                "R": R2, "t": t2,
                "lambdas": lambdas,
                "matches": matches,
                "delta":   delta,
                "rmse":    rmse2,
            }

        if best_result is not None:
            result[cam_b] = best_result
            print(
                f"  {cam_b} → {ref_cam}: δ={best_result['delta']}"
                f"  RMSE={best_result['rmse']:.3f} m"
                f"  matches={best_result['matches']}"
            )
        else:
            print(f"  {cam_b} → {ref_cam}: no alignment found")

    return result


# ── Manim geometry helpers ─────────────────────────────────────────────────────
#
# Coordinate mapping (OpenCV → Manim 3D):
#   cam(x, y, z) → manim(x, z, -y)
#   so depth (Z_cam) → Manim Y, and image-down (Y_cam) → Manim -Z (up).

def _m3(pt: np.ndarray) -> np.ndarray:
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    return np.array([x, z, -y])


def _make_traj(pelvis: np.ndarray, color, width: float = 2.5) -> VMobject:
    """Pelvis trajectory as a 3D polyline. pelvis: (T, 3)."""
    obj = VMobject(stroke_color=color, stroke_width=width)
    obj.set_points_as_corners([_m3(p) for p in pelvis])
    return obj


def _make_skel(joints: np.ndarray, color, stroke_width: float = 1.5) -> VGroup:
    """Single-frame skeleton. joints: (N_JOINTS, 3)."""
    vg = VGroup()
    for a, b in BONES:
        pa = _m3(joints[_JI[a]])
        pb = _m3(joints[_JI[b]])
        vg.add(Line(pa, pb, stroke_color=color, stroke_width=stroke_width))
    return vg


def _norm(pts: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Subtract center and scale. Broadcasts over any leading dims."""
    return (pts - center) * scale


# ── Manim scene ───────────────────────────────────────────────────────────────

class ReIDAnimation(ThreeDScene):
    def construct(self) -> None:
        # ── Load ──────────────────────────────────────────────────────────────
        print(f"\nLoading tracks from: {SCENE_DIR}")
        tracks = load_tracks(SCENE_DIR)
        if not tracks:
            self.add(Text(f"No tracks found.\nSet SCENE_DIR env var.", font_size=28))
            self.wait(2)
            return
        print(f"  Loaded {len(tracks)} tracks.")

        cam_ids = sorted({t["cam_id"] for t in tracks})
        n_cams  = len(cam_ids)
        cam_col = {c: CAM_COLORS[i % len(CAM_COLORS)] for i, c in enumerate(cam_ids)}

        # ── Global normalisation ───────────────────────────────────────────────
        all_p  = np.vstack([t["pelvis"] for t in tracks])
        center = all_p.mean(axis=0)
        spread = np.percentile(np.linalg.norm(all_p - center, axis=1), 95) + 1e-6
        scale  = 2.0 / spread

        spacing = max(5.0, 4.5)
        cam_off = {
            c: np.array([(i - (n_cams - 1) / 2.0) * spacing, 0.0, 0.0])
            for i, c in enumerate(cam_ids)
        }

        # ── Camera orientation ────────────────────────────────────────────────
        self.set_camera_orientation(phi=65 * DEGREES, theta=-60 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.07)

        # ── Phase 1: camera-local trajectories ────────────────────────────────
        title1 = Text("Before alignment — camera-local space",
                      font_size=22, color=WHITE)
        self.add_fixed_in_frame_mobjects(title1)
        title1.to_edge(UP)
        self.play(FadeIn(title1))

        traj_mob: dict[tuple, VMobject] = {}
        skel_mob: dict[tuple, VGroup]   = {}
        creates: list = []

        for t in tracks:
            key   = (t["cam_id"], t["pid"])
            color = cam_col[t["cam_id"]]
            off   = cam_off[t["cam_id"]]

            p_n  = _norm(t["pelvis"], center, scale) + off
            traj = _make_traj(p_n, color)
            traj_mob[key] = traj
            creates.append(Create(traj))

            skels = VGroup()
            for fi in range(0, len(t["joints"]), SKELETON_INTERVAL):
                jn = _norm(t["joints"][fi], center, scale) + off
                skels.add(_make_skel(jn, color))
            skel_mob[key] = skels
            creates.append(FadeIn(skels, run_time=1))

        self.play(*creates, run_time=4, lag_ratio=0.04)
        self.wait(2)

        if SKIP_ALIGN:
            print("\nSKIP_ALIGN=1 — stopping after Phase 1.")
            self.stop_ambient_camera_rotation()
            return

        # ── Compute alignment (v2 constrained rigid model) ───────────────────
        # Use the camera with the most tracks so multi-person pairs can be
        # aligned; override with REF_CAM env var if set.
        by_cam_count = {c: sum(1 for t in tracks if t["cam_id"] == c) for c in cam_ids}
        ref_cam = REF_CAM if REF_CAM in cam_ids else max(by_cam_count, key=by_cam_count.get)
        print(f"\nComputing alignment (v2) relative to reference camera: {ref_cam} …")
        alignment = compute_alignment(tracks, ref_cam)
        n_matched = sum(len(v["matches"]) for v in alignment.values())
        print(f"  Matched {n_matched} track(s) across {n_cams - 1} non-ref camera(s).")

        # Assign a colour per reference-camera person; matched tracks inherit it.
        ref_persons = [t for t in tracks if t["cam_id"] == ref_cam]
        pers_col = {t["pid"]: PERS_COLORS[i % len(PERS_COLORS)]
                    for i, t in enumerate(ref_persons)}

        # ── Phase 2: animate into aligned space ───────────────────────────────
        title2 = Text("After alignment — common space  (λ·(R·x + t))",
                      font_size=22, color=WHITE)
        self.add_fixed_in_frame_mobjects(title2)
        title2.to_edge(UP)

        anims: list = [ReplacementTransform(title1, title2)]

        for t in tracks:
            key = (t["cam_id"], t["pid"])

            if t["cam_id"] == ref_cam:
                # Reference camera: remove grid offset, recolour by person.
                color = pers_col.get(t["pid"], WHITE)
                p_n   = _norm(t["pelvis"], center, scale)
                anims.append(Transform(traj_mob[key], _make_traj(p_n, color)))
                skels2 = VGroup()
                for fi in range(0, len(t["joints"]), SKELETON_INTERVAL):
                    skels2.add(_make_skel(_norm(t["joints"][fi], center, scale), color))
                anims.append(Transform(skel_mob[key], skels2))

            elif t["cam_id"] in alignment:
                cam_info = alignment[t["cam_id"]]
                pid      = t["pid"]

                if pid in cam_info["lambdas"]:
                    R_align = cam_info["R"]
                    t_trans = cam_info["t"]
                    lam     = cam_info["lambdas"][pid]
                    # Look up which ref-camera person this pid was matched to.
                    matched_ref = dict(cam_info["matches"]).get(pid)
                    color = pers_col.get(matched_ref, WHITE)

                    p_aff = _apply_constrained(t["pelvis"], R_align, t_trans, lam)
                    anims.append(Transform(
                        traj_mob[key],
                        _make_traj(_norm(p_aff, center, scale), color),
                    ))
                    skels2 = VGroup()
                    for fi in range(0, len(t["joints"]), SKELETON_INTERVAL):
                        j_aff = _apply_constrained(t["joints"][fi], R_align, t_trans, lam)
                        skels2.add(_make_skel(_norm(j_aff, center, scale), color))
                    anims.append(Transform(skel_mob[key], skels2))

                else:
                    # Camera aligned but this track was below threshold: fade out.
                    anims.append(FadeOut(traj_mob[key]))
                    anims.append(FadeOut(skel_mob[key]))

            else:
                # No alignment found for this camera at all: fade out.
                anims.append(FadeOut(traj_mob[key]))
                anims.append(FadeOut(skel_mob[key]))

        self.play(*anims, run_time=5)
        self.wait(3)
        self.stop_ambient_camera_rotation()
