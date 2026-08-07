"""scripts/sync_demo_rerun.py — before/after temporal-synchronisation .rrd figure.

Single-frame Rerun recording that shows what temporal synchronisation buys, on an
EgoHumans (or RICH) scene.  Two ``.rrd`` files are written:

  ``<out>/<scene>_desync.rrd``
      Per-camera delays δ_k ∈ [0, max_shift] are injected (camera 0 anchors,
      δ_0 = 0).  At the chosen instant t*, camera k contributes the frame
      t* + δ_k, so each view reconstructs the moving people at a *different*
      moment → the depth reconstructions ghost across several positions and the
      video billboards disagree.

  ``<out>/<scene>_sync.rrd``
      The Synchronizer estimates δ̂_k from the desynced 3D joint tracks; applying
      the correction leaves camera k at t* + (δ_k − δ̂_k) ≈ t*, so every view
      collapses back onto one crisp person.

Both recordings log, at a single frame as static data:
  * ``world/scene/<cam>``  — per-camera VGGT depth back-projected to a textured
    mesh, people KEPT (they are the moving subject that reveals the sync);
  * ``world/<cam>``        — that camera's pinhole + pose + the single RGB frame;
  * ``world/person_<p>/pred`` — the fused SMPL-X body at t* (shared reference).

Only one frame is logged, so each recording is small and fast to build (no video
column, no per-frame depth stream).

Offset injection / worst-frame pick / Synchronizer are reused from
``scripts/sync_demo.py``; mesh + depth + SMPL-X rendering from
``visualize/visualize_rerun.py``.

Example
-------
    EXO=/iopsstor/scratch/cscs/tnanni/temp_egohumans/03_fencing/media/rawalk/\\
disk1/rawalk/datasets/ego_exo/camera_ready/03_fencing/001_fencing/exo
    pixi run python scripts/sync_demo_rerun.py \\
        --scene 001_fencing \\
        --scenes-root /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new/03_fencing \\
        --predictions fusion_outputs/001_fencing.npz \\
        --frames-dir "$EXO" \\
        --out-dir /capstor/scratch/cscs/tnanni/ghost_renders
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tyro

# offset logic (reused)
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from sync_demo import (
    _camera_names, _pick_frame, _sample_offsets, _estimate_offsets,
    _build_view, _build_model, _forward, _mesh_at, _R_to_6d,
)

# rendering helpers (reused)
from visualize.visualize_rerun import (
    _load_depth_context, _depth_grid_mesh, _grid_vertex_colors, _load_rich_frame,
    _build_smplx_vertices, _vertex_normals, _log_pinhole_simple,
    _log_static_transform, _default_blueprint, _PALETTE, _jpeg,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Clean-track loader (EgoHumans uses global ids in body_data_clean)
# ══════════════════════════════════════════════════════════════════════════════

def _scale_only_ctx(scene_dir: Path):
    """Depth context with metric scale from MapAnything, no fusion predictions.

    ``_load_depth_context`` derives the scale from ‖t_pred‖/‖t_vggt‖; with no
    predictions we hand it fake 'predicted' cameras = VGGT extrinsics × the
    MapAnything per-frame scale, so it recovers exactly that scale.  Lets the sync
    demo run on any scene (depth + image planes) without a fusion forward pass.
    """
    vggt = np.load(Path(scene_dir) / "vggt_cameras_centered.npz", allow_pickle=False)
    extr = vggt["extrinsics"]                      # (T, K, 3, 4)
    T, K = extr.shape[:2]
    scale = None
    for fn in ("mapanything_scale_baseline.npy", "mapanything_scale_centered.npy"):
        p = Path(scene_dir) / fn
        if p.exists():
            scale = np.load(p).astype(np.float64)
            break
    scale = np.ones(T) if scale is None else np.broadcast_to(np.atleast_1d(scale), (T,)).copy()
    camera = np.zeros((T, K, 8), np.float32)
    camera[:, :, 0] = 1.0
    camera[:, :, 4:7] = extr[:, :, :3, 3] * scale[:, None, None]
    return _load_depth_context(Path(scene_dir), camera, T, K)


def _condition_body(
    tag, scene, scene_dir, cams, offsets, work_root, centered_root,
    model, device, body_split, t_star, half_window, all_people=False,
):
    """Fuse + place the body for one offset condition → (verts (P,V,3), faces).

    ``_build_view`` rewrites each camera's ``frame_indices`` by ``−offsets[k]``, so
    at global frame t* camera k contributes its physical frame ``t*+offsets[k]``.
    Feeding that to the fusion model reproduces what desynchronisation actually
    does to the method: the views disagree, so the fused pose degrades.  With the
    Synchronizer's correction applied the offsets collapse to ≈0 and the body
    comes back.

    Unlike ``sync_demo._run_condition`` this does NOT re-run VGGT — the existing
    per-frame cameras and MapAnything scale are symlinked in.  RICH exo cameras
    are static, so desync barely perturbs the extrinsics; the visible effect is
    on the body and the depth, which is what the figure shows.
    """
    w0 = max(0, t_star - half_window)
    w1 = t_star + half_window + 1
    view_dir = Path(work_root) / tag / scene
    _build_view(Path(scene_dir), view_dir, cams, offsets, w0, w1)
    # keep the original cameras / scale (no per-condition VGGT pass)
    for name in ("vggt_cameras_centered.npz", "mapanything_scale_baseline.npy"):
        src, dst = Path(scene_dir) / name, view_dir / name
        if src.exists() and not dst.exists():
            os.symlink(src, dst)

    from data.fusion_dataset import RICHFusionDatapoint
    from configuration import CONFIG
    from inference import _run_placer

    dp = RICHFusionDatapoint(
        scene_dir      = view_dir,
        rich_data_root = str(centered_root),
        rich_gt_dir    = CONFIG.data.rich_gt_dir,
        body_split     = body_split,
        # all_people keeps background people (no GT annotation) so the scene
        # shows everyone, not just the GT-matched subject.
        restrict_to_gt_persons = not all_people,
        min_foreground_cams    = 1 if all_people else None,
    )
    frame_start = dp._frame_start
    logger.info(f"[{tag}] datapoint frames [{dp._frame_start}, {dp._frame_end}) "
                f"{dp.num_cameras} cams, {dp.max_persons} persons")
    out = _forward(model, dp, device)
    pred_pose_54 = out["pred_pose_54"]
    all_pids = sorted({pid for pids in dp._pid_order for pid in pids})

    cam_dirs = sorted(d for d in view_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())
    raw_body = []
    for cam_dir in cam_dirs:
        per_pid = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            if pid not in all_pids:
                continue
            with np.load(npz_path, allow_pickle=False) as d:
                per_pid[pid] = {k: d[k] for k in d.files}
        raw_body.append(per_pid)

    logger.info(f"[{tag}] BodyPlacer …")
    root_translation, orient_R, _ = _run_placer(
        scene_dir        = view_dir,
        cam_dirs         = cam_dirs,
        raw              = raw_body,
        all_pids         = all_pids,
        frame_start      = frame_start,
        T                = pred_pose_54.shape[0],
        smplx_model_path = Path(CONFIG.data.smplx_model_path),
        fused_pose       = pred_pose_54,
        crop_meta_path   = Path(centered_root) / scene / "crop_meta.json",
    )
    result = {"pred_pose_54": pred_pose_54, "pred_shape": out["pred_shape"],
              "root_translation": root_translation, "orient_R": orient_R}
    t_local = t_star - frame_start
    if not (0 <= t_local < pred_pose_54.shape[0]):
        logger.warning(f"[{tag}] t*={t_star} outside datapoint window — no body")
        return None, None
    verts, faces, placed = _mesh_at(result, t_local,
                                    Path(CONFIG.data.smplx_model_path))
    logger.info(f"[{tag}] placed persons at t*: {placed}")
    return (verts if len(placed) else None), (faces if len(placed) else None)


def _condition_body_egohumans(
    tag, scene, scene_dir, cams, offsets, work_root,
    model, device, t_star, half_window,
):
    """EgoHumans counterpart of :func:`_condition_body`.

    ``RICHFusionDatapoint`` needs RICH GT, so EgoHumans uses the same Route B as
    ``scripts/infer_scene.py``: pack the fusion input straight from
    ``body_data_clean``.  The offset injection still goes through ``_build_view``
    (which rewrites ``body_data``), so we mount a clean view first — body_data ->
    body_data_clean — shift that, then expose the shifted tracks back under
    ``body_data_clean`` for the Route B loader.
    """
    from infer_scene import _clean_scene_view, _egohumans_forward
    from inference import _run_placer
    from configuration import CONFIG

    w0 = max(0, t_star - half_window)
    w1 = t_star + half_window + 1

    # clean view of the untouched scene: body_data -> body_data_clean
    clean = _clean_scene_view(Path(scene_dir), cams)
    try:
        view_dir = Path(work_root) / tag / scene
        _build_view(clean, view_dir, cams, offsets, w0, w1)   # shifts body_data
        for name in ("vggt_cameras_centered.npz", "mapanything_scale_baseline.npy",
                     "mapanything_scale_centered.npy"):
            src, dst = Path(scene_dir) / name, view_dir / name
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
        # Route B reads body_data_clean; point it at the shifted tracks.
        for cam in cams:
            bd, bc = view_dir / cam / "body_data", view_dir / cam / "body_data_clean"
            if bd.is_dir() and not bc.exists():
                os.symlink(bd, bc)

        raw_arrays, meta = _egohumans_forward(model, view_dir, cams, device)
        pred_pose_54 = raw_arrays["pred_pose_54"]
        frame_start = meta["frame_start"]
        all_pids = meta["pids"]
        logger.info(f"[{tag}] egohumans forward: T={pred_pose_54.shape[0]}, "
                    f"pids={all_pids}, frame_start={frame_start}")

        cam_dirs = sorted(d for d in view_dir.iterdir()
                          if d.is_dir() and (d / "body_data").is_dir())
        raw_body = []
        for cam_dir in cam_dirs:
            per_pid = {}
            for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(npz_path.stem.split("_")[1])
                if pid not in all_pids:
                    continue
                with np.load(npz_path, allow_pickle=False) as d:
                    per_pid[pid] = {k: d[k] for k in d.files}
            raw_body.append(per_pid)

        logger.info(f"[{tag}] BodyPlacer …")
        root_translation, orient_R, _ = _run_placer(
            scene_dir        = view_dir,
            cam_dirs         = cam_dirs,
            raw              = raw_body,
            all_pids         = all_pids,
            frame_start      = frame_start,
            T                = pred_pose_54.shape[0],
            smplx_model_path = Path(CONFIG.data.smplx_model_path),
            fused_pose       = pred_pose_54,
            crop_meta_path   = None,          # egohumans kp2d already centered
        )
        result = {"pred_pose_54": pred_pose_54, "pred_shape": raw_arrays["pred_shape"],
                  "root_translation": root_translation, "orient_R": orient_R}
        t_local = t_star - frame_start
        if not (0 <= t_local < pred_pose_54.shape[0]):
            logger.warning(f"[{tag}] t*={t_star} outside window — no body")
            return None, None
        verts, faces, placed = _mesh_at(result, t_local,
                                        Path(CONFIG.data.smplx_model_path))
        logger.info(f"[{tag}] placed persons at t*: {placed}")
        return (verts if len(placed) else None), (faces if len(placed) else None)
    finally:
        import shutil
        shutil.rmtree(clean, ignore_errors=True)


def _load_clean_tracks(scene_dir: Path, cams: list[str],
                       min_frames: int = 0) -> list[dict[int, dict]]:
    """tracks[k][global_pid] = {array_name: array} from body_data_clean.

    body_data_clean carries cross-view-consistent GLOBAL person ids, so the same
    id means the same person in every camera — required for the Synchronizer's
    shared-pid intersection.  Falls back to body_data if clean is absent (RICH).
    """
    tracks: list[dict[int, dict]] = []
    for cam in cams:
        per_pid: dict[int, dict] = {}
        sub = "body_data_clean" if (scene_dir / cam / "body_data_clean").is_dir() else "body_data"
        for npz_path in sorted((scene_dir / cam / sub).glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            with np.load(npz_path, allow_pickle=False) as d:
                arrays = {k: d[k] for k in d.files}
            # Sparsely observed persons corrupt the offset estimate: their
            # cross-correlation prefers whatever shift makes their two short
            # observation windows coincide, and the per-person sum lets that
            # outvote a fully tracked subject (measured: a 193-frame background
            # person dragged a synced scene to a bogus 50-frame offset).  They are
            # excluded from SYNC only — rendering still shows them via all_people.
            if len(arrays["frame_indices"]) < min_frames:
                continue
            per_pid[pid] = arrays
        tracks.append(per_pid)
    return tracks


# ══════════════════════════════════════════════════════════════════════════════
#  One camera's static depth mesh (people KEPT)
# ══════════════════════════════════════════════════════════════════════════════

def _log_depth_mesh(ent, ctx, k, depth_idx, scene, rich_root, frames_dir,
                    img_frame, W, H, stride, conf_thr, boxes=None):
    """Back-project camera k's depth at ``depth_idx`` into a textured mesh (static).

    When ``boxes`` is given, the people are carved out of the depth using this
    camera's own person boxes at its own effective frame (each camera contributes a
    different instant under desync, so the boxes must be looked up per camera).
    Without carving the depth-reconstructed people and the fused SMPL-X bodies
    occupy the same space and occlude each other.
    """
    depth_mm, conf_a = ctx["depth_mm"], ctx["depth_conf"]
    if not (ctx["depth_valid"][depth_idx, k] and ctx["cam_valid"][depth_idx, k]):
        return
    s = float(ctx["scale"][depth_idx])
    d = depth_mm[depth_idx, k][::stride, ::stride].astype(np.float32) / 1000.0 * s
    c = conf_a[depth_idx, k][::stride, ::stride].astype(np.float32)
    d = np.where((d > 1e-4) & (c >= conf_thr), d, np.nan)

    h_d, w_d = d.shape
    vv, uu = np.mgrid[0:h_d, 0:w_d].astype(np.float32)
    uu *= stride; vv *= stride
    x1, y1, x2, y2 = ctx["oc"][depth_idx, k]
    d = np.where((uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2), d, np.nan)

    if boxes:
        # Frame pixels -> depth grid: inverse of the colour-sampling transform.
        sx = float(x2 - x1) / max(1e-6, W)
        sy = float(y2 - y1) / max(1e-6, H)
        for bx1, by1, bx2, by2 in boxes.get(img_frame, ()):
            pw, ph = 0.04 * (bx2 - bx1), 0.04 * (by2 - by1)   # pad for limbs
            u1 = x1 + (bx1 - pw) * sx; u2 = x1 + (bx2 + pw) * sx
            v1 = y1 + (by1 - ph) * sy; v2 = y1 + (by2 + ph) * sy
            d = np.where((uu >= u1) & (uu <= u2) & (vv >= v1) & (vv <= v2), np.nan, d)

    res = _depth_grid_mesh(d, uu, vv, ctx["intr"][depth_idx, k],
                           ctx["extr"][depth_idx, k, :3, :3],
                           ctx["extr"][depth_idx, k, :3, 3] * s)
    if res is None:
        return
    verts, tris, used = res
    bgr = _load_rich_frame(rich_root, scene, ctx["names"][k], k, img_frame, frames_dir)
    colors = (_grid_vertex_colors(bgr, uu, vv, used, ctx["oc"][depth_idx, k], W, H)
              if bgr is not None else None)
    import rerun as rr
    rr.log(ent, rr.Mesh3D(vertex_positions=verts, triangle_indices=tris,
                          vertex_colors=colors), static=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Build one .rrd for a given per-camera offset vector
# ══════════════════════════════════════════════════════════════════════════════

def _build_rrd(
    out_path, label, cam_names, eff_frames, ctx,
    scene, rich_root, frames_dir, frame_start, W, H, stride, conf_thr,
    t_star, look_target, eye_position, fps, body_verts=None, body_faces=None,
    person_boxes=None,
):
    """One recording: static depth mesh + image + this condition's fused body."""
    import rerun as rr
    rr.init(f"sync_demo_{label}", spawn=False)

    # fused body meshes at t* for THIS condition (desync ones are degraded)
    nP = 0 if body_verts is None else len(body_verts)
    for p in range(nP):
        v = body_verts[p]
        if not np.isfinite(v).all():
            continue
        faces = body_faces
        col = np.broadcast_to(np.asarray(_PALETTE[p % len(_PALETTE)], np.uint8), (v.shape[0], 3))
        n = _vertex_normals(v[None], np.asarray(faces, np.int64))[0]
        rr.log(f"world/person_{p}/pred",
               rr.Mesh3D(vertex_positions=v.astype(np.float32),
                         triangle_indices=np.asarray(faces, np.uint32),
                         vertex_colors=col.copy(),
                         vertex_normals=n.astype(np.float32)),
               static=True)

    focal = ctx["intr"]
    for k, cam in enumerate(cam_names):
        eff = int(eff_frames[k])                      # absolute frame for camera k
        depth_idx = eff - frame_start
        depth_idx = int(np.clip(depth_idx, 0, ctx["depth_mm"].shape[0] - 1))
        ent = f"world/{cam}"

        # pinhole + pose (use this camera's own extrinsic at the shown frame)
        _log_pinhole_simple(ent, float(ctx["intr"][depth_idx, k, 0, 0]), W, H)
        _log_static_transform(ent, ctx["extr"][depth_idx, k, :3, :3],
                              ctx["extr"][depth_idx, k, :3, 3] * float(ctx["scale"][depth_idx]))

        # single RGB frame on the image plane
        bgr = _load_rich_frame(rich_root, scene, cam, k, eff, frames_dir)
        if bgr is not None:
            blob = _jpeg(bgr)
            if blob is not None:
                rr.log(ent, rr.EncodedImage(contents=blob, media_type="image/jpeg"),
                       static=True)

        # depth mesh; carve this camera's people at ITS effective frame
        _log_depth_mesh(f"world/scene/{cam}", ctx, k, depth_idx, scene, rich_root,
                        frames_dir, eff, W, H, stride, conf_thr,
                        boxes=(person_boxes or {}).get(cam))

    bp = _default_blueprint(fps, look_target, eye_position)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_path), default_blueprint=bp)
    logger.info(f"  {label}: saved → {out_path}  (frames per cam: {list(map(int, eff_frames))})")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(
    scene:        str,
    scenes_root:  Path,
    predictions:  Path | None = None,
    frames_dir:   Path | None = None,
    rich_data_root: Path | None = None,
    smplx_model_dir: Path = Path("body_models/SMPLX_NEUTRAL.pkl"),
    out_dir:      Path = Path("."),
    frame_start:  int  = 1,
    max_shift:    int  = 15,
    seed:         int  = 0,
    depth_stride: int  = 2,
    depth_conf_thr: float = 0.5,
    fps:          float = 30.0,
    device:       str  = "cpu",
    with_body:    bool = True,
    all_people:   bool = False,
    sync_min_frames: int = 600,
    dataset:      str = "rich",
    carve_people: bool = True,
    t_star_override: int | None = None,
    body_split:   str  = "test_body",
    checkpoint:   Path | None = None,
    half_window:  int  = 64,
    centered_root: Path | None = None,
) -> None:
    """Write ``<scene>_desync.rrd`` and ``<scene>_sync.rrd`` for a single frame.

    scene / scenes_root : ghost output scene (contains cam*/, vggt_*_centered.npz)
    predictions         : fusion npz (pose/shape/camera/body_transl_world)
    frames_dir          : image root (EgoHumans exo dir); None → rich_data_root/scene
    max_shift           : maximum injected per-camera delay in frames
    """
    import torch  # noqa: F401  (Synchronizer needs it via _estimate_offsets)

    scene_dir = Path(scenes_root) / scene
    cam_names = _camera_names(scene_dir)
    K = len(cam_names)
    _rich_root = str(rich_data_root) if rich_data_root is not None else str(scenes_root)
    # centered root supplies crop_meta.json to the placer (and defaults to the
    # image root, which for RICH is the centered mount holding both).
    _centered_root = Path(centered_root) if centered_root is not None else (
        Path(frames_dir).parent if frames_dir is not None else Path(_rich_root))
    logger.info(f"Scene {scene}: {K} cameras {cam_names}")

    # ── depth context + optional SMPL-X reference bodies ─────────────────────
    pred_verts = faces = transl = None
    if predictions is not None:
        d = dict(np.load(predictions, allow_pickle=True))
        pose, shape = d["pose"], d["shape"]
        camera, transl = d["camera"], d["body_transl_world"]
        T = pose.shape[0]
        logger.info("Running SMPL-X forward (reference bodies) …")
        pred_verts, faces = _build_smplx_vertices(pose, shape, transl, smplx_model_dir)
        ctx = _load_depth_context(scene_dir, camera, T, K)
    else:
        logger.info("No predictions — depth+images only, scale from MapAnything.")
        ctx = _scale_only_ctx(scene_dir)
    if ctx is None:
        raise RuntimeError(f"No VGGT depth/cameras in {scene_dir}")

    # ── tracks → inject offsets, pick worst frame, estimate correction ───────
    tracks = _load_clean_tracks(scene_dir, cam_names, sync_min_frames)
    T_scene = int(max(int(p["frame_indices"].max())
                      for per in tracks for p in per.values())) + 1
    delays = _sample_offsets(K, max_shift, seed)
    if t_star_override is not None:
        t_star = int(t_star_override)
        logger.info(f"  t* forced to {t_star} (override)")
    else:
        t_star = _pick_frame(tracks, T_scene, window=2, max_shift=max_shift, delays=delays)
    dhat = _estimate_offsets(tracks, delays, T_scene, device, max_shift)
    # estimate_initial_times returns −δ (see memory sync_sign_convention), so the
    # correction that cancels the injected delay is δ_inject + δ̂ (≈ 0), NOT a
    # subtraction — subtracting would double the desync.
    sync_shift = delays + dhat
    logger.info(f"  injected δ   = {list(map(int, delays))}")
    logger.info(f"  estimated δ̂  = {list(map(int, dhat))}  (≈ −δ)")
    logger.info(f"  sync residual= {list(map(int, sync_shift))}  (sync frames = t* + this ≈ t*)")

    # image size from the anchor camera's frame at t*
    sample = _load_rich_frame(_rich_root, scene, cam_names[0], 0, t_star, frames_dir)
    H, W = (sample.shape[:2] if sample is not None else (1080, 1920))

    # eye: centre on the bodies at t* if we have them, else on the rig centroid
    # (the capture volume where the people are); look from behind median camera.
    R_w2c = ctx["extr"][:, :, :3, :3]
    t_w2c = ctx["extr"][:, :, :3, 3] * ctx["scale"][:, None, None]
    cam_c = -np.einsum("tkij,tki->tkj", R_w2c, t_w2c).reshape(-1, 3)
    cam_med = np.median(cam_c, axis=0)
    if transl is not None:
        tr = transl[t_star - frame_start]
        tr = tr[~np.all(tr == 0, axis=-1)]
        look_target = np.median(tr, axis=0) if tr.size else cam_med
    else:
        look_target = cam_med                       # rig centroid ≈ people
    eye_position = look_target + 1.4 * (cam_med - look_target)
    if np.allclose(eye_position, look_target):      # no offset → back off along +Z
        eye_position = look_target + np.array([0.0, -0.5, -3.0])

    # ── per-condition fused body (desync one is genuinely degraded) ──────────
    bodies: dict[str, tuple] = {"desync": (None, None), "sync": (None, None)}
    if with_body:
        import torch as _torch
        import tempfile, shutil
        work_root = Path(tempfile.mkdtemp(prefix="syncdemo_rrd_"))
        try:
            model = _build_model()
            ck = Path(checkpoint) if checkpoint else Path(
                __import__("configuration").CONFIG.fusion.checkpoint_dir) / "best.pt"
            state = _torch.load(str(ck), map_location="cpu")
            model.load_state_dict(state["model"])
            logger.info(f"fusion checkpoint: {ck} (epoch {state.get('epoch','?')})")
            dev = _torch.device(device)
            for tag, offs in (("desync", delays), ("sync", sync_shift)):
                try:
                    if dataset == "egohumans":
                        bodies[tag] = _condition_body_egohumans(
                            tag, scene, scene_dir, cam_names, offs, work_root,
                            model, dev, t_star, half_window,
                        )
                    else:
                        bodies[tag] = _condition_body(
                            tag, scene, scene_dir, cam_names, offs, work_root,
                            _centered_root, model, dev, body_split, t_star,
                            half_window, all_people,
                        )
                except Exception as e:
                    logger.warning(f"[{tag}] body failed — {e!r}")
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    # Person boxes per camera, for carving the people out of the depth so the
    # reconstruction does not occlude the fused bodies.
    person_boxes = None
    if carve_people:
        from visualize.visualize_rerun import _load_person_boxes
        person_boxes = {cam: _load_person_boxes(scene_dir, cam) for cam in cam_names}
        logger.info("carving people from depth: "
                    + ", ".join(f"{c}:{len(b)}fr" for c, b in person_boxes.items()))

    common = dict(
        cam_names=cam_names, ctx=ctx, person_boxes=person_boxes,
        scene=scene, rich_root=_rich_root, frames_dir=frames_dir,
        frame_start=frame_start, W=W, H=H, stride=depth_stride,
        conf_thr=depth_conf_thr, t_star=t_star, look_target=look_target,
        eye_position=eye_position, fps=fps,
    )
    out_dir = Path(out_dir)
    _build_rrd(out_dir / f"{scene}_desync.rrd", "desync",
               eff_frames=t_star + delays,
               body_verts=bodies["desync"][0], body_faces=bodies["desync"][1],
               **common)
    _build_rrd(out_dir / f"{scene}_sync.rrd", "sync",
               eff_frames=t_star + sync_shift,
               body_verts=bodies["sync"][0], body_faces=bodies["sync"][1],
               **common)
    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
