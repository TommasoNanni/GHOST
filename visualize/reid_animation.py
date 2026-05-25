"""
3D animation of geometric cross-view ReID trajectory alignment using Manim.

Two-phase animation:
  1. Camera-local view — each camera's tracks in their own coordinate frame,
     spread along the X axis so they don't overlap; coloured by camera.
  2. Aligned view — affine transforms map each camera into the reference
     camera's space; matched persons share a colour.

Requirements:
    pip install manim scipy  (manim community edition >= 0.18)

Usage:
    SCENE_DIR=/path/to/ghost_outputs/Pavallion_002_plankjack \\
        manim -pql visualize/reid_animation.py ReIDAnimation

    # High quality:
    SCENE_DIR=... manim -pqh visualize/reid_animation.py ReIDAnimation

Optional env vars:
    MAX_TRACKS        cap on total tracks loaded            (default: 20)
    SKELETON_INTERVAL frames between skeleton ghost poses   (default: 60)
    ALIGN_DELTA       max temporal offset (frames) searched (default: 150)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from manim import *

# ── MHR70 joint indices ───────────────────────────────────────────────────────
# From the _SMPLX_TO_MHR70 mapping in scripts/check_joint_offset.py.
# MHR70 joint 0 is a head landmark — pelvis is synthesised as the hip midpoint.

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
# Set SKIP_ALIGN=1 to render only Phase 1 (fast, no alignment computation).
SKIP_ALIGN        = os.environ.get("SKIP_ALIGN", "0") != "0"

_MATCH_THR   = 0.25   # metres RMSE to accept a match
_MIN_OVERLAP = 30     # minimum common frames to attempt a fit


# ── Data loading ──────────────────────────────────────────────────────────────

def _extract_joints(kp3d: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """
    kp3d : (T, 70, 3)  root-centred keypoints in camera space
    cam_t: (T, 3)      camera-local root position
    Returns (T, N_JOINTS, 3) absolute camera-space positions for the
    selected joints.
    """
    pos = np.zeros((len(kp3d), N_JOINTS, 3), dtype=np.float32)
    for name, idx in JOINT_IDX.items():
        pos[:, _JI[name]] = cam_t + kp3d[:, idx]
    pos[:, _JI["pelvis"]] = (
        pos[:, _JI["left_hip"]] + pos[:, _JI["right_hip"]]
    ) / 2.0
    return pos


def load_tracks(scene_dir: Path) -> list[dict]:
    """
    Glob all body_data/person_*.npz files under scene_dir and return a list
    of track dicts with keys: cam_id, pid, frame_indices, joints, pelvis.
    """
    tracks = []
    for body_file in sorted(scene_dir.glob("*/body_data/person_*.npz")):
        cam_id = body_file.parents[1].name
        data = np.load(body_file, allow_pickle=False)
        if "pred_keypoints_3d" not in data.files:
            continue
        kp3d  = data["pred_keypoints_3d"]
        cam_t = (data["pred_cam_t"]      if "pred_cam_t"      in data.files
                 else np.zeros((len(kp3d), 3), np.float32))
        fi    = (data["frame_indices"]   if "frame_indices"   in data.files
                 else np.arange(len(kp3d)))
        joints = _extract_joints(kp3d, cam_t)
        tracks.append({
            "cam_id":        cam_id,
            "pid":           body_file.stem,
            "frame_indices": np.asarray(fi, np.int64),
            "joints":        joints,
            "pelvis":        joints[:, _JI["pelvis"]],
        })
    return tracks[:MAX_TRACKS]


# ── Affine alignment ──────────────────────────────────────────────────────────

def _affine_fit(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine: Y ≈ (A @ X.T).T + t. X, Y: (N, 3)."""
    Xh = np.hstack([X, np.ones((len(X), 1))])
    W, _, _, _ = np.linalg.lstsq(Xh, Y, rcond=None)
    return W[:3].T.astype(np.float64), W[3].astype(np.float64)


def _apply_affine(pts: np.ndarray, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply affine to any (..., 3) array."""
    shape = pts.shape
    flat = pts.reshape(-1, 3).astype(np.float64)
    return ((A @ flat.T).T + t).reshape(shape)


def _build_frame_map(fi: np.ndarray) -> dict[int, int]:
    return {int(f): i for i, f in enumerate(fi)}


def _common_idx(fi_a: np.ndarray, fi_b: np.ndarray,
                delta: int) -> tuple[np.ndarray, np.ndarray]:
    fa_map = _build_frame_map(fi_a)
    ia, ib = [], []
    for j, fb in enumerate(fi_b):
        fa = int(fb) + delta
        if fa in fa_map:
            ia.append(fa_map[fa])
            ib.append(j)
    return np.array(ia, np.int64), np.array(ib, np.int64)


def _xcorr_best_delta(fi_a: np.ndarray, pa: np.ndarray,
                      fi_b: np.ndarray, pb: np.ndarray,
                      max_delta: int) -> int:
    """
    Fast delta estimate via cross-correlation of the pelvis Z (depth) signal.
    Projects each track onto a dense frame grid, cross-correlates, and returns
    the lag (in frames) that maximises correlation. Much faster than fitting
    an affine at every delta.
    """
    f_min = min(int(fi_a.min()), int(fi_b.min()))
    f_max = max(int(fi_a.max()), int(fi_b.max()))
    n = f_max - f_min + 1

    sig_a = np.zeros(n)
    sig_b = np.zeros(n)
    for i, f in enumerate(fi_a):
        sig_a[int(f) - f_min] = pa[i, 2]   # depth channel
    for i, f in enumerate(fi_b):
        sig_b[int(f) - f_min] = pb[i, 2]

    # Remove DC so correlation isn't dominated by mean offset
    sig_a -= sig_a.mean()
    sig_b -= sig_b.mean()

    corr = np.correlate(sig_a, sig_b, mode="full")
    lags = np.arange(-(n - 1), n)
    mask = np.abs(lags) <= max_delta
    best_lag = int(lags[mask][np.argmax(corr[mask])])
    return best_lag


def _best_delta_fit(ta: dict, tb: dict) -> tuple | None:
    """
    Find best delta via xcorr (fast), then scan ±5 frames around it for the
    affine fit with lowest RMSE. Returns (A, t, delta, rmse) or None.
    """
    fi_a, fi_b = ta["frame_indices"], tb["frame_indices"]
    ja,   jb   = ta["joints"],        tb["joints"]

    # Coarse delta from cross-correlation — O(T log T) instead of O(T * 2*delta)
    coarse = _xcorr_best_delta(fi_a, ta["pelvis"], fi_b, tb["pelvis"], ALIGN_DELTA)

    best_rmse, best = np.inf, None
    for delta in range(coarse - 5, coarse + 6):
        if abs(delta) > ALIGN_DELTA:
            continue
        ia, ib = _common_idx(fi_a, fi_b, delta)
        if len(ia) < _MIN_OVERLAP:
            continue
        X = jb[ib].reshape(-1, 3).astype(np.float64)
        Y = ja[ia].reshape(-1, 3).astype(np.float64)
        A, t_ = _affine_fit(X, Y)
        rmse = float(np.sqrt(np.mean((_apply_affine(X, A, t_) - Y) ** 2)))
        if rmse < best_rmse:
            best_rmse, best = rmse, (A, t_, delta, rmse)
    return best


def compute_alignment(tracks: list[dict], ref_cam: str) -> dict:
    """
    Hungarian matching: for each non-ref camera, find the best affine from
    each of its tracks into ref_cam space.

    Returns {(cam_b, pid_b): {"A", "t", "matched_pid", "rmse"}}
    """
    ref = [t for t in tracks if t["cam_id"] == ref_cam]
    by_cam: dict[str, list] = {}
    for t in tracks:
        if t["cam_id"] != ref_cam:
            by_cam.setdefault(t["cam_id"], []).append(t)

    result: dict = {}
    for cam_b, bt in by_cam.items():
        na, nb = len(ref), len(bt)
        mat: np.ndarray = np.full((nb, na), np.inf)
        cache: dict[tuple, tuple] = {}
        for j, tb in enumerate(bt):
            for i, ta in enumerate(ref):
                al = _best_delta_fit(ta, tb)
                if al is not None:
                    mat[j, i] = al[3]
                    cache[(j, i)] = al
        rows, cols = linear_sum_assignment(mat)
        for j, i in zip(rows, cols):
            if mat[j, i] < _MATCH_THR and (j, i) in cache:
                A, t_, _, rmse = cache[(j, i)]
                result[(cam_b, bt[j]["pid"])] = {
                    "A": A, "t": t_,
                    "matched_pid": ref[i]["pid"],
                    "rmse": rmse,
                }
    return result


# ── Manim geometry helpers ─────────────────────────────────────────────────────
#
# Coordinate mapping:
#   Camera space (OpenCV): X right, Y down, Z into scene (depth).
#   Manim 3D:              X right, Y up,   Z out of screen.
#   Map: cam(x, y, z) → manim(x, z, -y)
#     so depth (Z_cam) becomes the horizontal Y_manim axis and
#     vertical height (−Y_cam) becomes Z_manim pointing up.

def _m3(pt: np.ndarray) -> np.ndarray:
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    return np.array([x, z, -y])


def _make_traj(pelvis: np.ndarray, color, width: float = 2.5) -> VMobject:
    """Pelvis trajectory as a 3D polyline. pelvis: (T, 3)."""
    obj = VMobject(stroke_color=color, stroke_width=width)
    obj.set_points_as_corners([_m3(p) for p in pelvis])
    return obj


def _make_skel(joints: np.ndarray, color, stroke_width: float = 1.5) -> VGroup:
    """Single-frame skeleton. Uses lightweight Line (not Line3D) to avoid mesh overhead."""
    vg = VGroup()
    for a, b in BONES:
        pa = _m3(joints[_JI[a]])
        pb = _m3(joints[_JI[b]])
        line = Line(pa, pb, stroke_color=color, stroke_width=stroke_width)
        vg.add(line)
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
            msg = Text(f"No tracks found.\nSet SCENE_DIR env var.", font_size=28)
            self.add(msg)
            self.wait(2)
            return
        print(f"  Loaded {len(tracks)} tracks.")

        cam_ids = sorted({t["cam_id"] for t in tracks})
        n_cams  = len(cam_ids)
        cam_col = {c: CAM_COLORS[i % len(CAM_COLORS)]
                   for i, c in enumerate(cam_ids)}

        # ── Global normalisation ───────────────────────────────────────────────
        all_p  = np.vstack([t["pelvis"] for t in tracks])
        center = all_p.mean(axis=0)
        spread = np.percentile(np.linalg.norm(all_p - center, axis=1), 95) + 1e-6
        scale  = 2.0 / spread  # 95th-percentile pelvis distance → 2 Manim units

        # Camera grid: cameras spread along scene X with enough gap so their
        # normalised trajectories (radius ≈ 2 units) don't overlap.
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

            # Pelvis trajectory
            p_n = _norm(t["pelvis"], center, scale) + off
            traj = _make_traj(p_n, color)
            traj_mob[key] = traj
            creates.append(Create(traj))

            # Skeleton ghost frames
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

        # ── Compute alignment ─────────────────────────────────────────────────
        ref_cam = cam_ids[0]
        print(f"\nComputing alignment relative to reference camera: {ref_cam} …")
        alignment = compute_alignment(tracks, ref_cam)
        print(f"  Matched {len(alignment)} tracks across {n_cams - 1} non-ref camera(s).")

        # Assign a colour per reference-camera person; matched tracks inherit it.
        ref_persons = [t for t in tracks if t["cam_id"] == ref_cam]
        pers_col = {t["pid"]: PERS_COLORS[i % len(PERS_COLORS)]
                    for i, t in enumerate(ref_persons)}

        # ── Phase 2: animate into aligned space ───────────────────────────────
        title2 = Text("After alignment — common space",
                      font_size=22, color=WHITE)
        self.add_fixed_in_frame_mobjects(title2)
        title2.to_edge(UP)

        anims: list = [ReplacementTransform(title1, title2)]

        for t in tracks:
            key = (t["cam_id"], t["pid"])

            if t["cam_id"] == ref_cam:
                # Reference camera: just remove the grid offset.
                color = pers_col.get(t["pid"], WHITE)
                p_n   = _norm(t["pelvis"], center, scale)
                anims.append(Transform(traj_mob[key], _make_traj(p_n, color)))
                skels2 = VGroup()
                for fi in range(0, len(t["joints"]), SKELETON_INTERVAL):
                    skels2.add(_make_skel(
                        _norm(t["joints"][fi], center, scale), color))
                anims.append(Transform(skel_mob[key], skels2))

            elif key in alignment:
                # Non-ref camera matched track: apply affine then normalise.
                info  = alignment[key]
                A, t_ = info["A"], info["t"]
                color = pers_col.get(info["matched_pid"], WHITE)

                p_aff = _apply_affine(t["pelvis"], A, t_)
                anims.append(Transform(
                    traj_mob[key],
                    _make_traj(_norm(p_aff, center, scale), color),
                ))
                skels2 = VGroup()
                for fi in range(0, len(t["joints"]), SKELETON_INTERVAL):
                    j_aff = _apply_affine(t["joints"][fi], A, t_)
                    skels2.add(_make_skel(
                        _norm(j_aff, center, scale), color))
                anims.append(Transform(skel_mob[key], skels2))

            else:
                # Unmatched track: fade out.
                anims.append(FadeOut(traj_mob[key]))
                anims.append(FadeOut(skel_mob[key]))

        self.play(*anims, run_time=5)
        self.wait(3)
        self.stop_ambient_camera_rotation()
