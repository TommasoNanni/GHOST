"""Presentation videos for the temporal-synchronisation demo.

Takes one scene, injects random per-camera shifts (the exact recipe the sync
benchmark uses), runs the GHOST Synchronizer on the shifted tracks, and writes
three stages of mp4s:

  01_shifted          the frames each camera contributes AFTER the injected
                      shift and end cut — i.e. the desynchronised recording
  02_shifted_overlay  same frames with the per-camera SAM3D body reprojected
                      (SMPL-X mesh, z-buffer rasteriser)
  03_synced           the same streams re-cut by the offsets the synchroniser
                      recovered, so all cameras start on the same instant
                      (mesh overlay on by default, --no-sync-overlay to drop it)

Each stage also gets a grid.mp4 tiling all cameras — that is the panel that
shows desync -> sync at a glance.

Nothing outside --out-dir is written; every dataset root is opened read-only.

The shift/anchor logic is copied from evaluation/alignment_experiments_multi.py
(and its egohumans twin) and the mesh rasteriser from visualize/visualize_smplx.py
rather than imported, so this script cannot drift the benchmark or the viewer.

Usage
-----
    # EgoHumans (frames already on scratch)
    pixi run python visualize/visualize_for_presentation.py \
        --dataset egohumans --scene 01_tagging/007_tagging --max-shift 75

    # RICH (needs centered_train.sqsh mounted at --frames-root)
    pixi run python visualize/visualize_for_presentation.py \
        --dataset rich --scene BBQ_001_guitar --max-shift 75 \
        --frames-root /tmp/centered_train
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import multiprocessing as mp

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from synchronize_videos.synchronizer import Synchronizer          # noqa: E402
from utilities.body_data import load_person_smplx_pose            # noqa: E402

logger = logging.getLogger("presentation")

# ── roots ────────────────────────────────────────────────────────────────────
# Output lives in its own tree: the script never writes next to the pipeline
# outputs it reads.
DEFAULT_OUT_ROOT = Path("/iopsstor/scratch/cscs/tnanni/presentation_videos")

RICH_SCENES_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_FRAMES_ROOT = Path("/tmp/centered_train")   # centered_train.sqsh mount point

EGO_SCENES_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
EGO_FRAMES_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted")
EGO_INNER        = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

# ── person ids, copied from the two experiment modules ───────────────────────
# Same source of truth as the benchmark: GT-verified for RICH
# (scripts/verify_sync_scenes_gt.py), manual_reid.json for EgoHumans.
RICH_SCENE_PIDS: dict[str, list[int]] = {
    "BBQ_001_guitar":              [1],
    "BBQ_001_juggle":              [1],
    "ParkingLot1_002_burpee3":     [1],
    "ParkingLot1_002_overfence1":  [1],
    "ParkingLot1_002_overfence2":  [1],
    "ParkingLot1_002_pushup1":     [1],
    "ParkingLot1_002_stretching1": [1],
    "ParkingLot1_004_burpeejump1": [1],
    "ParkingLot1_005_overfence1":  [1],
    "ParkingLot2_008_pushup1":     [1],
    "ParkingLot2_008_overfence1":  [1],
    "ParkingLot2_014_pushup2":     [1],
    "ParkingLot1_005_burpeejump2": [1],
    "ParkingLot2_008_burpeejump1": [1],
    "ParkingLot2_014_burpeejump1": [1],
    "ParkingLot2_015_overfence1":  [1],
    "ParkingLot2_016_stretching1": [1],
    "Pavallion_003_plankjack":     [1],
}
RICH_SKIP_CAMERAS: dict[str, list[str]] = {
    "Pavallion_003_018_tossball":    ["cam_06"],
    "ParkingLot2_008_pushup2":       ["cam_03"],
    "ParkingLot2_014_takingphotos2": ["cam_01"],
}

EGO_SCENE_PIDS: dict[str, list[int]] = {
    "01_tagging/007_tagging":       [1, 2, 3, 4],
    "01_tagging/011_tagging":       [1, 2, 3, 4],
    "02_lego/001_legoassemble":     [1, 2, 3],
    "02_lego/003_legoassemble":     [1, 2, 3],
    "03_fencing/005_fencing":       [1, 2, 3],
    "03_fencing/006_fencing":       [1, 2, 3],
    "04_basketball/001_basketball": [1, 2, 3, 4],
    "04_basketball/011_basketball": [1, 2, 3, 4],
    "05_volleyball/004_volleyball": [1, 2, 3, 4],
    "05_volleyball/011_volleyball": [1, 2, 3, 4],
    "06_badminton/022_badminton":   [1, 2, 3, 4],
    "06_badminton/031_badminton":   [1, 2, 3, 4],
    "07_tennis/007_tennis":         [1, 2],
    "07_tennis/012_tennis":         [1, 2],
}
EGO_SKIP_CAMERAS: dict[str, list[str]] = {}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_NUM_RE = re.compile(r"(\d+)")


# ── dataset layout ───────────────────────────────────────────────────────────
class Layout:
    """Where a dataset keeps its body tracks and its frames."""

    def __init__(self, dataset: str, scenes_root: Path, frames_root: Path):
        self.dataset = dataset
        self.scenes_root = scenes_root
        self.frames_root = frames_root
        # RICH runs the live pipeline output; EgoHumans reads the manually
        # cleaned ids (body_data_clean), which is what the benchmark scores.
        self.body_dir = "body_data" if dataset == "rich" else "body_data_clean"

    def scene_dir(self, scene: str) -> Path:
        return self.scenes_root / scene

    def frames_dir(self, scene: str, cam: str) -> Path:
        if self.dataset == "rich":
            return self.frames_root / scene / cam
        act, seq = scene.split("/")
        return (self.frames_root / act / EGO_INNER / act / seq
                / "exo" / cam / "images_undistorted" / "frames")

    def scene_pids(self, scene: str) -> list[int] | None:
        table = RICH_SCENE_PIDS if self.dataset == "rich" else EGO_SCENE_PIDS
        return table.get(scene)

    def skip_cameras(self, scene: str) -> list[str]:
        table = RICH_SKIP_CAMERAS if self.dataset == "rich" else EGO_SKIP_CAMERAS
        return table.get(scene, [])


# ── track loading (copied from the alignment experiments) ────────────────────
def _load_anchored(npz_path: Path):
    """Load one person and re-anchor its sequence to absolute frame 0.

    utilities.body_data anchors at the camera's own first detection.  Both
    datasets are hardware synced, so absolute frame index is true time: a
    camera whose person appears 27 frames late would otherwise sit 27 frames
    early in its array, and the slicing below would turn that into a real
    offset the ground truth does not model.  Left-padding with zero pose and
    zero confidence makes array index == absolute frame in every camera.
    """
    result = load_person_smplx_pose(npz_path)
    if result is None:
        return None
    seq, conf = result
    with np.load(str(npz_path)) as d:
        if "frame_indices" not in d.files:
            raise KeyError(f"{npz_path}: no frame_indices — cannot anchor to absolute time")
        start = int(d["frame_indices"].astype(int).min())
    if start > 0:
        seq = torch.cat([seq.new_zeros((start, *seq.shape[1:])), seq], dim=0)
        conf = torch.cat([conf.new_zeros((start, *conf.shape[1:])), conf], dim=0)
    return seq, conf


def _pad_to_common_length(cam_data: dict) -> None:
    """Right-pad every sequence to the longest one, in place (zero confidence)."""
    lengths = [s.shape[0] for persons in cam_data.values() for s, _ in persons.values()]
    if not lengths:
        return
    T_max = max(lengths)
    for persons in cam_data.values():
        for pid, (seq, conf) in persons.items():
            pad = T_max - seq.shape[0]
            if pad > 0:
                persons[pid] = (
                    torch.cat([seq, seq.new_zeros((pad, *seq.shape[1:]))], dim=0),
                    torch.cat([conf, conf.new_zeros((pad, *conf.shape[1:]))], dim=0),
                )


def _fill_missing_persons(cam_data: dict, pids: list[int]) -> dict[int, int]:
    """Give every camera an entry for every requested person, in place.

    estimate_couple_offset takes index-aligned person lists of equal length, so
    a person missing from one view is inserted as zero pose / zero confidence:
    its cost matrix is all +inf and it is skipped in exactly the pairs that
    involve this camera, leaving the other pairs at full weight.
    """
    seen = {pid: sum(pid in persons for persons in cam_data.values()) for pid in pids}
    T_ref = max((s.shape[0] for persons in cam_data.values() for s, _ in persons.values()),
                default=0)
    shapes = next(((s.shape[1:], c.shape[1:])
                   for persons in cam_data.values() for s, c in persons.values()), None)
    if shapes is None:
        return seen
    seq_shape, conf_shape = shapes
    for persons in cam_data.values():
        for pid in pids:
            if pid not in persons:
                persons[pid] = (torch.zeros((T_ref, *seq_shape)),
                                torch.zeros((T_ref, *conf_shape)))
    return seen


def load_scene(scene_dir: Path, body_dir: str, exclude_cameras: list[str]) -> dict:
    """{cam_id: {pid: (rotations T×51×3, conf T×51)}} anchored to absolute time."""
    cam_data: dict[str, dict[int, tuple]] = {}
    for cam_dir in sorted(scene_dir.iterdir()):
        bdir = cam_dir / body_dir
        if not cam_dir.is_dir() or not bdir.exists():
            continue
        if cam_dir.name in exclude_cameras:
            logger.info(f"  {cam_dir.name}: skipped (SKIP_CAMERAS)")
            continue
        persons: dict[int, tuple] = {}
        for npz_path in sorted(bdir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            result = _load_anchored(npz_path)
            if result is None:
                logger.warning(f"  {npz_path.name}: missing pose params, skipping")
                continue
            persons[pid] = result
        if persons:
            cam_data[cam_dir.name] = persons
            logger.info(f"  {cam_dir.name}: {len(persons)} person(s), "
                        f"T={next(iter(persons.values()))[0].shape[0]}")
    _pad_to_common_length(cam_data)
    return cam_data


def select_pids(layout: Layout, scene: str, cam_data: dict,
                override: list[int] | None) -> list[int]:
    """Person ids to synchronise: --pids, else the benchmark table, else the
    intersection over cameras (with a warning — an intersection is a ReID
    product, not a statement about who the subject is)."""
    if override:
        return list(override)
    listed = layout.scene_pids(scene)
    if listed:
        return list(listed)
    common = set.intersection(*(set(p) for p in cam_data.values())) if cam_data else set()
    if not common:
        raise KeyError(f"{scene}: no SCENE_PIDS entry and no person id common to all "
                       f"cameras — pass --pids explicitly")
    logger.warning(f"  {scene}: not in the benchmark table, falling back to the "
                   f"cross-camera intersection {sorted(common)}")
    return sorted(common)


# ── shift injection (same draw as evaluation/sync_ablation_runner.py) ────────
def draw_shifts(cam_ids: list[str], max_shift: int, seed: int, trial: int,
                use_end_cuts: bool) -> tuple[dict[str, int], dict[str, int]]:
    """Trial `trial` of the benchmark's rng stream for this scene.

    The runner draws trials in sequence from one generator per scene, so
    drawing and discarding `trial` times reproduces the same numbers.
    """
    rng = np.random.default_rng(seed)
    shifts = end_cuts = None
    for _ in range(trial + 1):
        raw = [0] + rng.integers(-max_shift, max_shift + 1, size=len(cam_ids) - 1).tolist()
        shifts = {c: int(s) for c, s in zip(cam_ids, raw)}
        cuts = rng.integers(0, max_shift + 1, size=len(cam_ids)).tolist()
        end_cuts = {c: (int(e) if use_end_cuts else 0) for c, e in zip(cam_ids, cuts)}
    return shifts, end_cuts


def camera_windows(cam_data: dict, shifts: dict[str, int], end_cuts: dict[str, int],
                   pids: list[int], min_overlap: int) -> dict[str, tuple[int, int]]:
    """{cam: (start, end)} absolute frame window each camera contributes.

    Camera c starts at max_shift - shift_c, so the camera that was shifted the
    most starts at absolute frame 0 and the others start later — after the cut,
    aligning the content again means dropping (shift_c - min shift) frames.
    """
    max_s = max(shifts.values())
    windows: dict[str, tuple[int, int]] = {}
    for cam_id, persons in cam_data.items():
        T = persons[pids[0]][0].shape[0]
        start = max_s - shifts[cam_id]
        end = T - end_cuts[cam_id] if end_cuts[cam_id] > 0 else T
        if end - start < min_overlap:
            raise RuntimeError(
                f"{cam_id}: only {end - start} frames after shift "
                f"(need >= {min_overlap}) — lower --max-shift or pick another scene")
        windows[cam_id] = (start, end)
    return windows


def dump_cost_matrix(cam_data: dict, pid: int, windows: dict[str, tuple[int, int]],
                     cams: tuple[str, str], device: str, out_path: Path,
                     min_overlap: int, max_shift: int) -> None:
    """Save the pairwise cost matrix of one camera pair, with its frame axes.

    This is Synchronizer._compute_cost_matrix itself, on the same shifted windows
    the estimate ran on, so the animation shows the surface the algorithm
    actually searched.  Rows/cols are saved as ABSOLUTE frame indices: the matrix
    is window-local, and a figure that mixes the two silently mislabels itself.
    """
    a, b = cams
    sync = Synchronizer(device=device, min_overlap=min_overlap, max_shift=max_shift)
    sa, ea = windows[a]
    sb, eb = windows[b]
    cost = sync._compute_cost_matrix(
        cam_data[a][pid][0][sa:ea].to(device), cam_data[b][pid][0][sb:eb].to(device),
        cam_data[a][pid][1][sa:ea].to(device), cam_data[b][pid][1][sb:eb].to(device))
    C = cost.cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, C=C, row_frames=np.arange(sa, ea), col_frames=np.arange(sb, eb),
             cams=np.array([a, b]), pid=pid)
    finite = np.isfinite(C)
    logger.info(f"  cost matrix {C.shape} -> {out_path} "
                f"({100 * finite.mean():.1f}% finite, min {C[finite].min():.3f})")


def run_sync(cam_data: dict, pids: list[int], windows: dict[str, tuple[int, int]],
             cam_ids: list[str], max_shift: int, min_overlap: int,
             device: str):
    """Synchronizer on the sliced windows.

    Returns (t_k frames to DROP from camera k, pairwise offset matrix, cycle
    weights).  The pairwise matrix is what the solver is given: offset[i, j] is
    how many frames camera i starts after camera j, estimated from that pair
    alone, before the global least squares reconciles them.
    """
    joints_list, confs_list = [], []
    for cam_id in cam_ids:
        s, e = windows[cam_id]
        joints_list.append([cam_data[cam_id][pid][0][s:e].to(device) for pid in pids])
        confs_list.append([cam_data[cam_id][pid][1][s:e].to(device) for pid in pids])

    sync = Synchronizer(device=device, min_overlap=min_overlap, max_shift=max_shift)
    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
    weights = sync.cycle_consistency_weights(offset_mat)
    estimated = sync.estimate_initial_times(offset_mat, weights).cpu()
    if torch.isnan(estimated).any():
        raise RuntimeError("synchroniser returned NaN — an isolated camera, no global solve")
    return estimated - estimated.min(), offset_mat.cpu(), weights.cpu()


# ── frames on disk ───────────────────────────────────────────────────────────
def frame_index_map(frames_dir: Path) -> dict[int, Path]:
    """{absolute frame index: file}.  The index is the first integer group of
    the stem, which covers RICH (00042_00.jpg) and EgoHumans (00042.jpg)."""
    if not frames_dir.is_dir():
        raise FileNotFoundError(
            f"frames not found: {frames_dir}\n"
            f"(RICH needs centered_train.sqsh mounted at --frames-root)")
    out: dict[int, Path] = {}
    for p in sorted(frames_dir.iterdir()):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        m = _NUM_RE.search(p.stem)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"no images in {frames_dir}")
    return out


# ── SMPL-X mesh overlay (copied from visualize/visualize_smplx.py) ───────────
_PALETTE: list[tuple[int, int, int]] = [
    (60, 80, 220), (60, 200, 60), (220, 80, 60), (40, 210, 210),
    (210, 60, 210), (210, 210, 40), (40, 140, 220), (160, 60, 160),
]


def _color(pid: int) -> tuple[int, int, int]:
    return _PALETTE[pid % len(_PALETTE)]


def _recover_cx_cy(kpts_cam: np.ndarray, kpts_2d: np.ndarray,
                   focal_length: float) -> tuple[float, float]:
    """Principal point from the stored 3D->2D correspondences.

    SAM3D stores pred_keypoints_2d in full-image pixels and pred_keypoints_3d
    body-centric, so cx, cy fall out of the projection equation and the crop
    offsets never enter.
    """
    valid = kpts_cam[:, 2] > 1e-5
    if not valid.any():
        return 0.0, 0.0
    z = kpts_cam[valid, 2]
    cx = float(np.median(kpts_2d[valid, 0] - kpts_cam[valid, 0] / z * focal_length))
    cy = float(np.median(kpts_2d[valid, 1] - kpts_cam[valid, 1] / z * focal_length))
    return cx, cy


def _project(verts_cam: np.ndarray, focal_length: float, cx: float, cy: float) -> np.ndarray:
    z = np.maximum(verts_cam[:, 2], 1e-6)
    return np.stack([verts_cam[:, 0] / z * focal_length + cx,
                     verts_cam[:, 1] / z * focal_length + cy], axis=-1)


_BODY_FACES_CACHE: dict[int, np.ndarray] = {}


def body_faces_only(faces: np.ndarray) -> np.ndarray:
    """Faces of the largest connected component — SMPL-X minus its eyeballs.

    The two eyeballs are separate closed components (546 vertices each) sitting
    in open sockets.  Painter's algorithm has no z-buffer, so they fight with
    the eyelid rim and read as dark holes; nobody wants eyeballs in a figure
    anyway.  Dropping them leaves the sockets open, which _close_alpha_holes
    paints over.
    """
    key = id(faces)
    if key not in _BODY_FACES_CACHE:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        n_v = int(faces.max()) + 1
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_v, n_v))
        _, lab = connected_components(adj, directed=False)
        keep = np.bincount(lab).argmax()
        _BODY_FACES_CACHE[key] = faces[lab[faces[:, 0]] == keep]
    return _BODY_FACES_CACHE[key]


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(verts)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    for c in range(3):
        np.add.at(n, faces[:, c], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def _outward_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(V,3) vertex normals, flipped if the winding points them inwards."""
    n = _vertex_normals(verts, faces)
    if np.median(np.einsum("ij,ij->i", n, verts - verts.mean(0))) < 0:
        n = -n
    return n


def render_mesh_zbuffer(entries: list[tuple[np.ndarray, np.ndarray, tuple]],
                        fl: float, cx: float, cy: float, out_w: int, out_h: int,
                        ss: int = 1) -> np.ndarray:
    """Scanline-rasterise the meshes with a real z-buffer -> (out_px,out_px,4) BGRA.

    Sorting whole triangles by mean depth (painter's algorithm) cannot resolve a
    surface against itself: the far side of a limb, or the inside of the head,
    lands on top of the skin.  Here every pixel keeps the nearest fragment, and
    the normal is interpolated across the triangle (Gouraud) rather than being
    constant per face, so the shading is smooth.

    entries are (vertices in OpenCV camera space, faces, BGR colour).  pytorch3d
    would do this on the GPU, but this environment has it without its compiled
    _C extension, so the rasteriser is unavailable.
    """
    w, h = out_w * max(1, ss), out_h * max(1, ss)
    # the caller sized the projection for out_w/out_h; the raster is ss times that.
    fl, cx, cy = fl * max(1, ss), cx * max(1, ss), cy * max(1, ss)
    zbuf = np.full((h, w), np.inf, np.float64)
    rgb = np.zeros((h, w, 3), np.float64)

    for verts, faces, bgr in entries:
        v = np.asarray(verts, np.float64)
        vn = _outward_normals(v, faces)
        uv = _project(v, fl, cx, cy)
        z = np.maximum(v[:, 2], 1e-6)
        base = np.asarray(bgr, np.float64)
        per_vertex = base.ndim == 2      # (V,3) instead of one BGR triple

        cen = v[faces].mean(1)
        fn = vn[faces].mean(1)
        # camera sits at the origin looking down +z: a face whose outward normal
        # points along the ray to it is turned away.
        front = np.einsum("ij,ij->i", fn, cen) < 0
        for f in faces[front]:
            p2 = uv[f]
            x0 = max(int(np.floor(p2[:, 0].min())), 0)
            x1 = min(int(np.ceil(p2[:, 0].max())), w - 1)
            y0 = max(int(np.floor(p2[:, 1].min())), 0)
            y1 = min(int(np.ceil(p2[:, 1].max())), h - 1)
            if x1 < x0 or y1 < y0:
                continue
            (ax, ay), (bx, by), (cxx, cyy) = p2
            det = (by - cyy) * (ax - cxx) + (cxx - bx) * (ay - cyy)
            if abs(det) < 1e-12:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5,
                               np.arange(y0, y1 + 1) + 0.5)
            l0 = ((by - cyy) * (X - cxx) + (cxx - bx) * (Y - cyy)) / det
            l1 = ((cyy - ay) * (X - cxx) + (ax - cxx) * (Y - cyy)) / det
            l2 = 1.0 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            iz = np.stack([l0, l1, l2], -1) / z[f]          # perspective weights
            depth = 1.0 / np.maximum(iz.sum(-1), 1e-12)
            win = inside & (depth < zbuf[y0:y1 + 1, x0:x1 + 1])
            if not win.any():
                continue
            nrm = np.einsum("hwk,kj->hwj", iz, vn[f]) * depth[..., None]
            nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
            lam = np.clip(-nrm[..., 2], 0.0, 1.0)           # lit from the camera
            shade = 0.35 + 0.65 * lam
            # colour is interpolated across the triangle exactly like the normal,
            # so a highlighted region has a smooth edge rather than a facet one
            face_col = (np.einsum("hwk,kj->hwj", iz, base[f]) * depth[..., None]
                        if per_vertex else base)
            zbuf[y0:y1 + 1, x0:x1 + 1] = np.where(win, depth,
                                                  zbuf[y0:y1 + 1, x0:x1 + 1])
            patch = rgb[y0:y1 + 1, x0:x1 + 1]
            rgb[y0:y1 + 1, x0:x1 + 1] = np.where(win[..., None],
                                                 face_col * shade[..., None], patch)

    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    img[:, :, 3] = np.where(np.isfinite(zbuf), 255, 0).astype(np.uint8)
    return _downsample_rgba(img, out_w, out_h) if ss > 1 else img


def _person_projection(entry: dict, row: int) -> tuple[float, float, float]:
    """(focal, cx, cy) for one person-frame; principal point recovered analytically."""
    fl = float(entry["fl"][row])
    kpts_cam = entry["kp3d"][row] + entry["cam_t"][row][None, :]
    cx, cy = _recover_cx_cy(kpts_cam, entry["kp2d"][row], fl)
    return fl, cx, cy


def project_bbox(cache: dict, frame_idx: int, draw_pids: list[int] | None):
    """(u0, v0, u1, v1) pixel bounds of the drawn bodies, unzoomed, or None."""
    us, vs = [], []
    for pid, entry in cache.items():
        if draw_pids is not None and pid not in draw_pids:
            continue
        row = entry["frame_to_row"].get(frame_idx)
        if row is None or "faces" not in entry:
            continue
        fl, cx, cy = _person_projection(entry, row)
        uv = _project(entry["verts"][row].astype(np.float64), fl, cx, cy)
        us.append(uv[:, 0])
        vs.append(uv[:, 1])
    if not us:
        return None
    u, v = np.concatenate(us), np.concatenate(vs)
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def fit_from_bbox(bbox, canvas_w: int, canvas_h: int, margin: float = 0.06):
    """(s, dx, dy) mapping a projected bbox onto a canvas.

    Scale and offset must come from the SAME bbox: mixing a scale from one frame
    with a centre from another puts the body partly outside the canvas.
    """
    u0, v0, u1, v1 = bbox
    s = min(canvas_w * (1 - 2 * margin) / max(u1 - u0, 1e-6),
            canvas_h * (1 - 2 * margin) / max(v1 - v0, 1e-6))
    return s, canvas_w / 2 - s * (u0 + u1) / 2, canvas_h / 2 - s * (v0 + v1) / 2


def fit_to_canvas(cache: dict, frame_idx: int, draw_pids: list[int] | None,
                  canvas_w: int, canvas_h: int, margin: float = 0.06):
    """(s, dx, dy) placing the drawn bodies on a canvas of this size.

    u' = u * s + dx is identical to projecting with focal * s and principal point
    cx * s + dx, so the zoom stays an exact perspective projection: the mesh is
    rasterised at the final resolution rather than an upscaled image.
    """
    bbox = project_bbox(cache, frame_idx, draw_pids)
    return None if bbox is None else fit_from_bbox(bbox, canvas_w, canvas_h, margin)


def _downsample_rgba(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Box-filter an RGBA canvas, premultiplied so the edges keep their colour.

    Averaging straight alpha would pull the transparent black background into
    every silhouette pixel and leave a dark fringe around the body.
    """
    f = img.astype(np.float32)
    a = f[:, :, 3:4] / 255.0
    f[:, :, :3] *= a
    f = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    a2 = f[:, :, 3:4] / 255.0
    f[:, :, :3] = np.where(a2 > 1e-6, f[:, :, :3] / np.maximum(a2, 1e-6), 0.0)
    return np.clip(f, 0, 255).astype(np.uint8)


def _build_smplx_model(smplx_model_path: Path, batch_size: int):
    import smplx as smplx_lib
    p = Path(smplx_model_path)
    kw = {"model_path": str(p), "ext": p.suffix.lstrip(".")} if p.is_file() \
        else {"model_path": str(p), "ext": "pkl"}
    return smplx_lib.create(**kw, model_type="smplx", gender="neutral",
                            use_pca=False, num_betas=10, batch_size=batch_size)


def build_mesh_cache(body_dir: Path, pids: list[int], window: tuple[int, int],
                     smplx_model_path: Path) -> dict:
    """Per-person camera-space vertices for every frame of the window.

    SMPL-X runs once per person over the whole window (transl baked into the
    vertices), so the renderer projects them with no further translation.
    """
    start, end = window
    cache: dict[int, dict] = {}
    faces = None
    for pid in pids:
        npz_path = body_dir / f"person_{pid}.npz"
        if not npz_path.exists():
            continue
        with np.load(str(npz_path)) as f:
            data = {k: f[k] for k in f.files}
        need = ("smplx_betas", "smplx_body_pose", "smplx_global_orient", "smplx_transl",
                "pred_keypoints_2d", "pred_keypoints_3d", "pred_cam_t",
                "focal_length", "frame_indices")
        missing = [k for k in need if k not in data]
        if missing:
            logger.warning(f"    person_{pid}: missing {missing} — no mesh")
            continue
        fi = data["frame_indices"].astype(int)
        rows = np.nonzero((fi >= start) & (fi < end))[0]
        if rows.size == 0:
            continue
        import torch
        model = _build_smplx_model(smplx_model_path, len(rows))

        def _t(arr):
            return torch.from_numpy(np.asarray(arr[rows], dtype=np.float32)).reshape(len(rows), -1)

        with torch.no_grad():
            out = model(betas=_t(data["smplx_betas"]),
                        body_pose=_t(data["smplx_body_pose"]),
                        global_orient=_t(data["smplx_global_orient"]),
                        transl=_t(data["smplx_transl"]),
                        left_hand_pose=_t(data["smplx_left_hand_pose"])
                        if "smplx_left_hand_pose" in data else None,
                        right_hand_pose=_t(data["smplx_right_hand_pose"])
                        if "smplx_right_hand_pose" in data else None,
                        return_verts=True)
        verts = out.vertices.numpy()                      # (n, V, 3) camera space
        joints = out.joints.numpy()[:, :55]               # (n, 55, 3) same space
        faces = body_faces_only(model.faces.astype(np.int32))
        del model
        cache[pid] = {
            "frame_to_row": {int(fi[r]): i for i, r in enumerate(rows)},
            "verts": verts,
            "joints": joints,
            "kp2d": data["pred_keypoints_2d"][rows].astype(np.float64),
            "kp3d": data["pred_keypoints_3d"][rows].astype(np.float64),
            "cam_t": data["pred_cam_t"][rows].astype(np.float64),
            "fl": data["focal_length"][rows].astype(np.float64).reshape(len(rows), -1)[:, 0],
        }
        logger.info(f"    person_{pid}: mesh ready ({len(rows)} frames)")
    if faces is not None:
        for entry in cache.values():
            entry["faces"] = faces
    return cache


_HIGHLIGHT: dict = {}


def _vertex_colours(entry: dict, row: int, pid: int) -> tuple | np.ndarray:
    """Body colour, or per-vertex colours with one joint or bone picked out.

    Vertices are selected geometrically rather than by skinning weight: the
    weights of a joint spread down the whole limb it drives, which is not the
    same shape as the limb itself.
    """
    if not _HIGHLIGHT or "joints" not in entry:
        return _color(pid)
    idx = _HIGHLIGHT["joints"]
    joints = entry["joints"][row]
    if max(idx) >= joints.shape[0]:
        return _color(pid)
    v = entry["verts"][row].astype(np.float64)
    cols = np.tile(np.asarray(_color(pid), np.float64), (v.shape[0], 1))
    if len(idx) == 1:
        dist = np.linalg.norm(v - joints[idx[0]][None, :], axis=1)
    else:
        # distance to the bone segment, not to its ends: a limb is the tube
        # between two joints, and two spheres would miss its middle
        a, b = joints[idx[0]], joints[idx[1]]
        ab = b - a
        t = np.clip(((v - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        dist = np.linalg.norm(v - (a + t[:, None] * ab), axis=1)
    cols[dist <= _HIGHLIGHT["radius"]] = _HIGHLIGHT["colour"]
    return cols


def mesh_rgba(cache: dict, frame_idx: int, draw_pids: list[int] | None,
              out_w: int, out_h: int, ss: int = 1,
              fit: tuple[float, float, float] | None = None) -> np.ndarray | None:
    """Rasterise the people of one frame -> (out_h, out_w, 4) BGRA, or None if empty.

    Every person shares one camera: the per-person focal and principal point come
    from the same analytic recovery on the same frame, so the first one drawn
    defines the scene camera.  fit applies the zoom-to-canvas transform.
    """
    entries, cam = [], None
    for pid, entry in cache.items():
        if draw_pids is not None and pid not in draw_pids:
            continue
        row = entry["frame_to_row"].get(frame_idx)
        if row is None or "faces" not in entry:
            continue
        if cam is None:
            fl, cx, cy = _person_projection(entry, row)
            if fit is not None:
                s_, dx, dy = fit
                fl, cx, cy = fl * s_, cx * s_ + dx, cy * s_ + dy
            cam = (fl, cx, cy)
        entries.append((entry["verts"][row], entry["faces"],
                        _vertex_colours(entry, row, pid)))
    if not entries:
        return None
    return render_mesh_zbuffer(entries, *cam, out_w, out_h, ss)


def composite(frame: np.ndarray, rgba: np.ndarray, alpha: float) -> None:
    """Alpha-blend a rendered BGRA layer onto a BGR frame, in place."""
    a = (rgba[:, :, 3:4] / 255.0) * float(alpha)
    frame[:] = (rgba[:, :, :3] * a + frame * (1.0 - a)).astype(np.uint8)


_MESH_CTX: dict[str, dict] = {}


def _render_frame_task(task):
    """One video frame in a worker process -> (position, encoded jpg bytes).

    The mesh cache is inherited through fork rather than pickled: the vertex
    arrays run to tens of megabytes per person and would dominate the transfer.
    """
    pos, cam_id, fi, path_str, label, blank_bg, alpha, draw_pids, ss, use_mesh = task
    cv2.setNumThreads(1)
    frame = cv2.imread(path_str)
    if frame is None:
        return pos, None
    h, w = frame.shape[:2]
    if blank_bg:
        frame = np.zeros_like(frame)
    if use_mesh:
        layer = mesh_rgba(_MESH_CTX[cam_id], fi, draw_pids, w, h, ss)
        if layer is not None:
            composite(frame, layer, alpha)
    _label(frame, label)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return pos, (buf.tobytes() if ok else None)


def _label(frame: np.ndarray, text: str) -> None:
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, fh), _ = cv2.getTextSize(text, font, fs, th)
    cv2.rectangle(frame, (8, 8), (8 + tw + 16, 8 + fh + 16), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (16, 8 + fh + 6), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def _still_task(task):
    """One blank-bg still in a worker; the mesh cache arrives through fork."""
    cam_id, fi, fit, fill, ss, draw_pids, path_str = task
    cv2.setNumThreads(1)
    if fit is None:      # per-frame fit: this body fills its own canvas
        fit = fit_to_canvas(_MESH_CTX[cam_id], fi, draw_pids, fill, fill)
        if fit is None:
            return 0
    img = mesh_rgba(_MESH_CTX[cam_id], fi, draw_pids, fill, fill, ss, fit)
    if img is None:
        return 0
    cv2.imwrite(path_str, img)
    return 1


def render_still_range(stage_dir: Path, cam_ids: list[str],
                       indices: dict[str, list[int]], meshes: dict[str, dict],
                       draw_pids: list[int] | None, fill: int, ss: int,
                       jobs: int = 1, per_frame_fit: bool = False) -> None:
    """One blank-bg RGBA PNG per frame of each camera's window.

    The zoom is fitted ONCE over the whole range by default: a per-frame fit
    rescales and recentres the body every frame, which reads as jitter when the
    PNGs are played back in place.

    per_frame_fit trades that away for resolution.  The union window has to hold
    the person's whole path through the scene, so any single body covers a small
    part of it — fine for playback, wasteful when the consumer re-crops and
    normalises the height itself (visualize/animate_cost_matrix.py does), where
    it just throws away most of the pixels.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    for cam_id in cam_ids:
        cache = meshes.get(cam_id, {})
        boxes = [b for b in (project_bbox(cache, fi, draw_pids)
                             for fi in indices[cam_id]) if b is not None]
        if not boxes:
            logger.warning(f"  {cam_id}: no body in the whole range — skipped")
            continue
        # one fit from the bbox that contains every frame, so the body never
        # leaves the canvas and never changes scale between frames
        b = np.asarray(boxes)
        fit = fit_from_bbox((b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()),
                            fill, fill)
        tasks = [(cam_id, fi, None if per_frame_fit else fit, fill, ss, draw_pids,
                  str(stage_dir / f"{cam_id}_f{fi}.png")) for fi in indices[cam_id]]
        _MESH_CTX.update(meshes)
        if jobs > 1:
            with mp.get_context("fork").Pool(jobs) as pool:
                written = sum(pool.imap_unordered(_still_task, tasks, chunksize=2))
        else:
            written = sum(map(_still_task, tasks))
        logger.info(f"  {cam_id}: {written} mesh PNGs -> {stage_dir}")


def render_stage(stage_dir: Path, cam_ids: list[str], indices: dict[str, list[int]],
                 frame_maps: dict[str, dict[int, Path]], labels: dict[str, str],
                 meshes: dict[str, dict] | None, fps: int, alpha: float,
                 grid: bool, cell_width: int, still: bool = False,
                 blank_bg: bool = False, draw_pids: list[int] | None = None,
                 fill: int = 0, ss: int = 1, jobs: int = 1) -> None:
    """One mp4 per camera plus, optionally, a tiled grid.mp4.

    still=True writes just the last frame of each camera's window, as a PNG.
    blank_bg drops the photo and keeps only the bodies — transparent RGBA for a
    still, black for video.  fill (stills only) puts them on a square canvas of
    that many pixels, zoomed to fill it and rasterised at ss times the size.

    Bodies always come from render_mesh_zbuffer, so a still and a video frame are
    the same picture at different sizes.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    if still:
        for cam_id in cam_ids:
            fi = indices[cam_id][-1]
            path = frame_maps[cam_id].get(fi)
            frame = cv2.imread(str(path)) if path is not None else None
            if frame is None:
                logger.warning(f"  {cam_id}: frame {fi} unreadable or missing — skipped")
                continue
            h, w = frame.shape[:2]
            cache = meshes[cam_id] if meshes else {}
            if blank_bg and fill:
                fit = fit_to_canvas(cache, fi, draw_pids, fill, fill)
                if fit is None:
                    logger.warning(f"  {cam_id}: no body to draw at f{fi} — skipped")
                    continue
                out = mesh_rgba(cache, fi, draw_pids, fill, fill, ss, fit)
            else:
                layer = mesh_rgba(cache, fi, draw_pids, w, h, ss)
                if blank_bg:
                    out = layer if layer is not None else np.zeros((h, w, 4), np.uint8)
                else:
                    if layer is not None:
                        composite(frame, layer, alpha)
                    _label(frame, f"{cam_id}  {labels[cam_id]}  f{fi}")
                    out = frame
            out_path = stage_dir / f"{cam_id}_f{fi}.png"
            cv2.imwrite(str(out_path), out)
            logger.info(f"  wrote {out_path}")
        return

    n_grid = min(len(indices[c]) for c in cam_ids) if grid else 0
    cols = int(np.ceil(np.sqrt(len(cam_ids))))
    rows = int(np.ceil(len(cam_ids) / cols))
    cells: dict[str, list[np.ndarray]] = {c: [] for c in cam_ids} if grid else {}
    cell_h = None

    if meshes:
        _MESH_CTX.update(meshes)
    pool = mp.get_context("fork").Pool(jobs) if jobs > 1 else None

    for cam_id in cam_ids:
        writer, size = None, None
        out_path = stage_dir / f"{cam_id}.mp4"
        tasks = [(pos, cam_id, fi, str(frame_maps[cam_id][fi]),
                  f"{cam_id}  {labels[cam_id]}  f{fi}", blank_bg, alpha,
                  draw_pids, ss, meshes is not None)
                 for pos, fi in enumerate(indices[cam_id]) if fi in frame_maps[cam_id]]
        missing = len(indices[cam_id]) - len(tasks)
        if missing:
            logger.warning(f"  {cam_id}: {missing} frame(s) missing on disk — skipped")
        results = (pool.imap(_render_frame_task, tasks, chunksize=2) if pool
                   else map(_render_frame_task, tasks))
        for pos, buf in results:
            if buf is None:
                logger.warning(f"  {cam_id}: frame at position {pos} unreadable — skipped")
                continue
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), size)
            elif (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            if grid and pos < n_grid:
                if cell_h is None:
                    cell_h = int(round(frame.shape[0] * cell_width / frame.shape[1]))
                cells[cam_id].append(cv2.resize(frame, (cell_width, cell_h)))
        if writer is not None:
            writer.release()
            logger.info(f"  wrote {out_path}")
    if pool is not None:
        pool.close()
        pool.join()

    if grid and cell_h is not None:
        grid_path = stage_dir / "grid.mp4"
        gw = cv2.VideoWriter(str(grid_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (cols * cell_width, rows * cell_h))
        blank = np.zeros((cell_h, cell_width, 3), np.uint8)
        for t in range(min(len(v) for v in cells.values())):
            tiles = [cells[c][t] for c in cam_ids]
            tiles += [blank] * (rows * cols - len(tiles))
            gw.write(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                                for r in range(rows)]))
        gw.release()
        logger.info(f"  wrote {grid_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["rich", "egohumans"], required=True)
    ap.add_argument("--scene", required=True,
                    help="RICH: BBQ_001_guitar   EgoHumans: 01_tagging/007_tagging")
    ap.add_argument("--scenes-root", type=Path, default=None,
                    help="ghost output root (read-only)")
    ap.add_argument("--frames-root", type=Path, default=None,
                    help="image root the pipeline consumed (read-only)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"video output root (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--max-shift", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial", type=int, default=0,
                    help="which draw of the rng stream to reproduce")
    ap.add_argument("--no-end-cuts", action="store_true",
                    help="inject start shifts only (benchmark also cuts the ends)")
    ap.add_argument("--pids", type=int, nargs="+", default=None)
    ap.add_argument("--stages", nargs="+", default=["shifted", "overlay", "synced"],
                    choices=["shifted", "overlay", "synced", "none"],
                    help="'none' renders no video — for --dump-cost / --still-range runs")
    ap.add_argument("--no-sync-overlay", action="store_true",
                    help="render stage 3 without the reprojected bodies")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--still", action="store_true",
                    help="write only the last frame of each camera's window, as PNG")
    ap.add_argument("--blank-bg", action="store_true",
                    help="draw the bodies on a blank canvas instead of the photo "
                         "(transparent RGBA with --still, black in video)")
    ap.add_argument("--draw-pids", type=int, nargs="+", default=None,
                    help="restrict which persons are drawn; --pids still drives the sync")
    ap.add_argument("--cams", nargs="+", default=None,
                    help="render only these cameras; the shift draw and the sync still "
                         "use every camera, so the offsets do not change")
    ap.add_argument("--still-fill", type=int, default=0, metavar="PX",
                    help="blank still: square canvas of PX pixels with the body zoomed "
                         "to fill it (e.g. 2048); 0 keeps the original frame geometry")
    ap.add_argument("--still-ss", type=int, default=2, metavar="N",
                    help="supersample factor (rasterise N x, box-filter down); 1 for video speed")
    ap.add_argument("--cell-width", type=int, default=640)
    ap.add_argument("--still-fit", choices=["range", "frame"], default="range",
                    help="'range' fits one window to the whole clip (steady playback); "
                         "'frame' fits each body to its own canvas (max resolution, "
                         "for consumers that re-crop)")
    ap.add_argument("--still-range", action="store_true",
                    help="write one blank-bg mesh PNG per frame of the window "
                         "(input for visualize/animate_cost_matrix.py)")
    ap.add_argument("--dump-cost", type=Path, default=None, metavar="NPZ",
                    help="save the DTW cost matrix of --cost-cams for --cost-pid")
    ap.add_argument("--cost-cams", nargs=2, default=None, metavar=("CAM_A", "CAM_B"))
    ap.add_argument("--cost-pid", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1),
                    help="worker processes for video frames (the rasteriser is CPU-bound)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap rendered frames per camera per stage")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--joint", type=int, default=None, metavar="IDX",
                    help="paint the body around this SMPL-X joint (0 pelvis, 15 head, "
                         "20 left wrist, 21 right wrist, 10/11 feet)")
    ap.add_argument("--limb", type=int, nargs=2, default=None, metavar=("J0", "J1"),
                    help="paint the bone between two joints, e.g. 18 20 = left "
                         "forearm, 16 18 = left upper arm, 4 7 = left shin")
    ap.add_argument("--joint-color", default="#c62828",
                    help="hex colour for --joint")
    ap.add_argument("--joint-radius", type=float, default=0.08,
                    help="metres around the joint to paint")
    ap.add_argument("--mesh-alpha", type=float, default=1.0,
                    help="1.0 = opaque mesh, <1 blends with the frame")
    ap.add_argument("--smplx-model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--min-overlap", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s  %(message)s")
    # OpenCV's own thread pool must not exist when render_stage forks its
    # workers: a child that inherits a lock held by a parent cv2 thread hangs on
    # the first imread, which looks exactly like a very slow render.
    cv2.setNumThreads(1)

    if args.joint is not None or args.limb is not None:
        h = args.joint_color.lstrip("#")
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        _HIGHLIGHT.update(joints=tuple(args.limb) if args.limb else (args.joint,),
                          radius=args.joint_radius,
                          colour=np.asarray(rgb[::-1], np.float64))   # BGR

    scenes_root = args.scenes_root or (RICH_SCENES_ROOT if args.dataset == "rich"
                                       else EGO_SCENES_ROOT)
    frames_root = args.frames_root or (RICH_FRAMES_ROOT if args.dataset == "rich"
                                       else EGO_FRAMES_ROOT)
    layout = Layout(args.dataset, scenes_root, frames_root)

    scene_dir = layout.scene_dir(args.scene)
    if not scene_dir.is_dir():
        sys.exit(f"scene not found: {scene_dir}")

    logger.info(f"Scene: {args.scene}")
    cam_data = load_scene(scene_dir, layout.body_dir, layout.skip_cameras(args.scene))
    if len(cam_data) < 2:
        sys.exit(f"{args.scene}: need >= 2 cameras with {layout.body_dir}, "
                 f"found {len(cam_data)}")

    pids = select_pids(layout, args.scene, cam_data, args.pids)
    seen = _fill_missing_persons(cam_data, pids)
    logger.info(f"  persons {pids} (cameras observing each: "
                f"{ {p: seen.get(p, 0) for p in pids} })")

    cam_ids = list(cam_data.keys())
    render_cams = list(args.cams) if args.cams else cam_ids
    unknown = [c for c in render_cams if c not in cam_ids]
    if unknown:
        sys.exit(f"--cams: {unknown} not among {cam_ids}")
    shifts, end_cuts = draw_shifts(cam_ids, args.max_shift, args.seed, args.trial,
                                   not args.no_end_cuts)
    logger.info(f"  injected shifts: {shifts}")
    logger.info(f"  end cuts:        {end_cuts}")

    windows = camera_windows(cam_data, shifts, end_cuts, pids, args.min_overlap)

    logger.info("Synchronising...")
    estimated, offset_mat, weights = run_sync(
        cam_data, pids, windows, cam_ids, args.max_shift, args.min_overlap, args.device)
    true_t = torch.tensor([float(shifts[c]) for c in cam_ids])
    true_t = true_t - true_t.min()
    errors = (estimated - true_t).abs()
    drops = {c: int(round(float(estimated[i]))) for i, c in enumerate(cam_ids)}
    for i, c in enumerate(cam_ids):
        logger.info(f"  {c}: true={true_t[i]:.0f}  estimated={estimated[i]:.2f}  "
                    f"err={errors[i]:.2f}")
    logger.info(f"  MAE = {errors.mean():.3f} frames")
    logger.info("  Pairwise offsets (estimated | true | error, frames):")
    for i in range(len(cam_ids)):
        for j in range(i + 1, len(cam_ids)):
            est = float(offset_mat[i, j])
            # the window of camera c starts at max_shift - shift_c, so a pair's
            # true start difference is the negated difference of their shifts
            tru = float(shifts[cam_ids[j]] - shifts[cam_ids[i]])
            logger.info(f"    {cam_ids[i]} -> {cam_ids[j]}: {est:+8.2f} | {tru:+7.1f} | "
                        f"{abs(est - tru):5.2f}   w={float(weights[i, j]):.3f}")

    out_dir = (args.out_dir / args.dataset / args.scene.replace("/", "_")
               / f"shift{args.max_shift}_seed{args.seed}_trial{args.trial}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_maps = {c: frame_index_map(layout.frames_dir(args.scene, c))
                  for c in render_cams}

    def _cap(seq: list[int]) -> list[int]:
        return seq[:args.max_frames] if args.max_frames else seq

    shifted_idx = {c: _cap(list(range(*windows[c]))) for c in cam_ids}
    # After the sync the streams share a start, so they also share a length:
    # the shortest one bounds the others, otherwise the grid drifts apart again.
    synced_full = {c: list(range(windows[c][0] + drops[c], windows[c][1])) for c in cam_ids}
    common_len = min(len(v) for v in synced_full.values())
    if args.max_frames:
        common_len = min(common_len, args.max_frames)
    synced_idx = {c: v[:common_len] for c, v in synced_full.items()}

    if args.dump_cost:
        pair = tuple(args.cost_cams) if args.cost_cams else tuple(cam_ids[:2])
        logger.info(f"Cost matrix for {pair[0]} x {pair[1]}, person {args.cost_pid}...")
        dump_cost_matrix(cam_data, args.cost_pid, windows, pair, args.device,
                         args.dump_cost, args.min_overlap, args.max_shift)

    if "none" in args.stages:
        args.stages = []
    need_mesh = (args.still_range or "overlay" in args.stages
                 or ("synced" in args.stages and not args.no_sync_overlay))
    meshes: dict[str, dict] = {}
    if need_mesh:
        if not args.smplx_model.exists():
            sys.exit(f"SMPL-X model not found: {args.smplx_model}")
        logger.info("Building SMPL-X meshes...")
        for cam_id in render_cams:
            logger.info(f"  {cam_id}")
            meshes[cam_id] = build_mesh_cache(scene_dir / cam_id / layout.body_dir,
                                              pids, windows[cam_id], args.smplx_model)

    if args.still_range:
        full_window = {c: list(range(*windows[c])) for c in render_cams}
        render_still_range(out_dir / "mesh_frames", render_cams, full_window, meshes,
                           args.draw_pids, args.still_fill or 1024, args.still_ss,
                           args.jobs, args.still_fit == "frame")

    grid = not args.no_grid and not args.still
    extra = dict(still=args.still, blank_bg=args.blank_bg, draw_pids=args.draw_pids,
                 fill=args.still_fill, ss=args.still_ss, jobs=args.jobs)
    if "shifted" in args.stages:
        logger.info("Rendering 01_shifted...")
        render_stage(out_dir / "01_shifted", render_cams, shifted_idx, frame_maps,
                     {c: f"shift {shifts[c]:+d}" for c in cam_ids}, None,
                     args.fps, args.mesh_alpha, grid, args.cell_width, **extra)
    if "overlay" in args.stages:
        logger.info("Rendering 02_shifted_overlay...")
        render_stage(out_dir / "02_shifted_overlay", render_cams, shifted_idx, frame_maps,
                     {c: f"shift {shifts[c]:+d}" for c in cam_ids}, meshes,
                     args.fps, args.mesh_alpha, grid, args.cell_width, **extra)
    if "synced" in args.stages:
        logger.info("Rendering 03_synced...")
        render_stage(out_dir / "03_synced", render_cams, synced_idx, frame_maps,
                     {c: f"synced (dropped {drops[c]})" for c in cam_ids},
                     None if args.no_sync_overlay else meshes,
                     args.fps, args.mesh_alpha, grid, args.cell_width, **extra)

    meta = {
        "dataset": args.dataset, "scene": args.scene, "cameras": cam_ids,
        "rendered_cameras": render_cams,
        "pids": pids, "max_shift": args.max_shift, "seed": args.seed,
        "trial": args.trial, "end_cuts_enabled": not args.no_end_cuts,
        "true_shifts": shifts, "end_cuts": end_cuts,
        "windows": {c: list(windows[c]) for c in cam_ids},
        "true_times": true_t.tolist(), "estimated_times": estimated.tolist(),
        "errors": errors.tolist(), "mae": float(errors.mean()),
        "frames_dropped_after_sync": drops,
        "pairwise_offsets": offset_mat.tolist(),
        "pairwise_weights": weights.tolist(),
        "scenes_root": str(scenes_root), "frames_root": str(frames_root),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
