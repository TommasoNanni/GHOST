"""Viser-based interactive viewer for ghost fusion predictions.

Shows animated SMPL-X body meshes (predicted + GT) together with all camera
frustums textured with the original video frames.

Usage
-----
    pixi run python visualize/visualize_fusion.py \\
        --predictions fusion_outputs/60980456_predictions.npz \\
        --scene_dir   test_outputs/rich10_segmentation_test/BBQ_001_guitar \\
        --frame_start 0 \\
        --port 8080

IMPORTANT — run as a persistent background process (otherwise it dies when the
shell exits):

    nohup bash -c 'CONDA_OVERRIDE_CUDA=12.6 pixi run python visualize/visualize_fusion.py --predictions fusion_outputs/<job_id>_predictions.npz --scene_dir test_outputs/rich10_segmentation_test/BBQ_001_guitar --smplx-model-dir body_models/SMPLX_NEUTRAL.pkl --port 8080' > ~/viser.log 2>&1 &

    NOTE: do NOT use line breaks in the nohup command — copy it as a single line.
    NOTE: use ~/viser.log (not /tmp/viser.log) to avoid permission issues.

Check it started:
    tail ~/viser.log
    lsof -t -i:8080         # should print a PID

To stop it:
    kill $(lsof -t -i:8080)

Port forwarding on Euler (VSCode + Windows)
-------------------------------------------
PROBLEM: euler.ethz.ch is a round-robin — each SSH connection may land on a
different login node. If the server runs on eu-login-08 but the tunnel goes to
eu-login-15, the browser gets "Connection refused".

SOLUTION: Always forward directly to the node the server is running on:

    # 1. Find which node the server is on (run on the cluster):
    hostname

    # 2. On your Windows laptop, forward to that specific node:
    ssh -L 8080:eu-login-08:8080 tnanni@euler.ethz.ch
    #                ^^^^^^^^^^^ replace with actual node

    # 3. Open in browser:
    http://localhost:8080

VSCode auto-tunneling also works IF VSCode is connected to the same node.
Check bottom-left corner of VSCode — it must match the node the server is on.
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configuration import CONFIG

import numpy as np
import torch
import tyro
import viser
import cv2
from scipy.spatial.transform import Rotation as SciR
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
)

# ── colour palette (one per person) ──────────────────────────────────────────
_PALETTE = [
    (220,  80,  60),
    ( 60, 200,  60),
    ( 60,  80, 220),
    (210, 210,  40),
    (210,  60, 210),
    ( 40, 210, 210),
]

# Viser expects Y-up; ghost / SMPL-X uses Y-down (OpenCV).
# A 180° rotation around X maps between the two conventions.
_ROT180 = np.diag([1., -1., -1.])


# ── helpers ───────────────────────────────────────────────────────────────────

def _6d_to_aa(pose_6d: np.ndarray) -> np.ndarray:
    """(*, 6) float32 6D rotation → (*, 3) axis-angle."""
    t = torch.from_numpy(pose_6d.astype(np.float32))
    mat = rotation_6d_to_matrix(t)
    aa  = matrix_to_axis_angle(mat)
    return aa.numpy()


def _build_smplx_vertices(
    pose:  np.ndarray,   # (T, P, J, 6)  world-frame 6D rotations
    shape: np.ndarray,   # (P, 10) or (T, P, 10)
    trans: np.ndarray,   # (T, P, 3)     world-frame root translation
    smplx_model_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SMPL-X forward and return (vertices, faces).

    Returns
    -------
    vertices : (T, P, V, 3) float32  world-space mesh vertices
    faces    : (F, 3)       int32
    """
    import smplx as smplx_lib

    T, P, J, _ = pose.shape

    if shape.ndim == 2:           # (P, 10) — constant over time
        shape = np.broadcast_to(shape[None], (T, P, 10)).copy()

    _p = smplx_model_dir
    _create_kwargs: dict = {"model_type": "smplx"}
    if _p.is_file():
        _create_kwargs["model_path"] = str(_p)
        _create_kwargs["ext"] = _p.suffix.lstrip(".")
    else:
        _create_kwargs["model_path"] = str(_p)
    model = smplx_lib.create(
        **_create_kwargs,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        flat_hand_mean=True,
        batch_size=T * P,
    )
    model.eval()

    global_orient_aa = _6d_to_aa(pose[:, :, 0, :])
    body_pose_aa     = _6d_to_aa(pose[:, :, 1:22, :])

    def _t(x): return torch.from_numpy(x.reshape(T * P, -1).astype(np.float32))

    with torch.no_grad():
        out = model(
            global_orient = _t(global_orient_aa),
            body_pose     = _t(body_pose_aa),
            betas         = _t(shape),
            transl        = _t(trans),
            return_verts  = True,
        )

    V = out.vertices.shape[1]
    verts = out.vertices.numpy().reshape(T, P, V, 3)
    faces = model.faces.copy()
    return verts, faces


def _load_rich_frame(
    rich_data_root: Path,
    scene_name: str,
    cam_idx: int,
    rich_frame_idx: int,
    frames_dir: Path | None = None,
) -> np.ndarray | None:
    """Load one RICH frame as BGR uint8, or None if not found.

    If frames_dir is given it is used directly as the scene root (e.g. a
    squashfuse mount point), bypassing rich_data_root / scene_name.
    """
    scene_root = frames_dir if frames_dir is not None else rich_data_root / scene_name
    cam_dir = scene_root / f"cam_{cam_idx:02d}"
    # Filename suffix matches camera index: 00000_01.jpg for cam_01
    search_dirs = [cam_dir, cam_dir / "frames"]
    for stem_suffix in (f"{rich_frame_idx:05d}_{cam_idx:02d}", f"{rich_frame_idx:05d}"):
        for search_dir in search_dirs:
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                q = search_dir / f"{stem_suffix}{ext}"
                if q.exists():
                    img = cv2.imread(str(q))
                    return img
    return None


def _cam_to_viser(
    R_w2c: np.ndarray,  # (3, 3)  world-to-camera rotation
    t_w2c: np.ndarray,  # (3,)    world-to-camera translation
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world-to-camera [R|t] to viser wxyz quaternion + position.

    Returns the camera-to-world pose in viser's Y-up convention.
    The returned wxyz is also suitable for client.camera.wxyz (same convention).
    """
    R_c2w = R_w2c.T
    t_c2w = -(R_w2c.T @ t_w2c)

    # Apply Y-flip to match viser's Y-up convention
    R_vis = _ROT180 @ R_c2w
    t_vis = _ROT180 @ t_c2w

    quat_xyzw = SciR.from_matrix(R_vis).as_quat()   # xyzw
    quat_wxyz  = np.concatenate([quat_xyzw[3:], quat_xyzw[:3]])
    return quat_wxyz, t_vis


# ── main viewer ───────────────────────────────────────────────────────────────

def run(
    predictions:     Path,
    scene_dir:       Path,
    rich_data_root:  Path       = Path(CONFIG.data.rich_data_root),
    smplx_model_dir: Path       = Path("body_models/SMPLX_NEUTRAL.pkl"),
    frame_start:     int        = 0,
    show_gt:         bool       = True,
    show_depth:      bool       = False,
    port:            int        = 9090,
    frames_dir:      Path | None = None,
) -> None:
    """Launch the interactive viser viewer.

    Parameters
    ----------
    predictions     : path to the *_predictions.npz saved by the trainer
    scene_dir       : ghost output directory for the scene
    rich_data_root  : root of the RICH dataset (contains cam_XX subdirs)
    smplx_model_dir : directory that contains SMPLX_NEUTRAL.pkl (or smplx/ subdir)
    frame_start     : first RICH frame index (used to load correct images)
    show_gt         : also render the GT body in wireframe
    show_depth      : also render the VGGT depth maps as world-space point clouds
                      (requires vggt_depth_centered.npz + vggt_cameras_centered.npz
                      in scene_dir)
    port            : viser WebSocket port
    """
    # ── load predictions ──────────────────────────────────────────────────────
    print(f"Loading {predictions} …")
    d = dict(np.load(predictions, allow_pickle=True))

    pose              = d["pose"]               # (T, P, J, 6)  predicted body pose (6D)
    shape             = d["shape"]              # (P, 10)        predicted SMPL-X betas
    camera            = d["camera"]             # (T, K, 8)      predicted [quat wxyz(4), trans(3), focal(1)]
    body_transl_world = d["body_transl_world"]  # (T, P, 3)      predicted body root in world (cam-0) frame

    T, P, J, _ = pose.shape
    K = camera.shape[1]
    scene_name = scene_dir.name

    print(f"  T={T} frames, P={P} persons, K={K} cameras")

    # Optional GT
    gt_body_pose         = d.get("gt_body_pose")
    gt_body_shape        = d.get("gt_body_shape")
    gt_body_transl_world = d.get("gt_body_transl_world")

    if gt_body_transl_world is not None:
        gt_valid = ~np.all(gt_body_transl_world == 0, axis=-1)   # (T, P)
        any_valid = gt_valid.any(axis=1)
        first_valid = int(any_valid.argmax()) if any_valid.any() else 0
        n_missing = int((~any_valid).sum())
        if n_missing:
            print(f"  {n_missing} frames have no GT annotation — GT body hidden there.")
    else:
        gt_valid  = np.zeros((T, P), dtype=bool)
        first_valid = 0

    # ── build SMPL-X meshes ───────────────────────────────────────────────────
    print("Running SMPL-X forward pass for predictions …")
    pred_verts, faces = _build_smplx_vertices(pose, shape, body_transl_world, smplx_model_dir)

    gt_verts = None
    if show_gt and gt_body_pose is not None and gt_body_transl_world is not None:
        print("Running SMPL-X forward pass for GT …")
        gt_shape_arr = gt_body_shape if gt_body_shape is not None else shape
        gt_verts, _ = _build_smplx_vertices(gt_body_pose, gt_shape_arr, gt_body_transl_world, smplx_model_dir)

    # Apply Y-flip for viser convention
    pred_verts_vis = pred_verts @ _ROT180.T
    gt_verts_vis   = gt_verts   @ _ROT180.T if gt_verts is not None else None

    # ── decode predicted cameras ──────────────────────────────────────────────
    quat_raw = camera[..., :4]
    norms    = np.linalg.norm(quat_raw, axis=-1, keepdims=True).clip(1e-8)
    quat_n   = quat_raw / norms
    R_w2c    = quaternion_to_matrix(
        torch.from_numpy(quat_n.reshape(-1, 4).astype(np.float32))
    ).numpy().reshape(T, K, 3, 3)
    t_w2c    = camera[..., 4:7]
    focal    = camera[0, :, 7]

    # ── decode GT cameras (if available) ─────────────────────────────────────
    gt_camera_data = d.get("gt_camera")
    gt_R_w2c = gt_t_w2c = gt_focal = None
    if gt_camera_data is not None:
        K_gt        = gt_camera_data.shape[1]   # may differ from K (missing cams)
        gt_quat_raw = gt_camera_data[..., :4]
        gt_norms    = np.linalg.norm(gt_quat_raw, axis=-1, keepdims=True).clip(1e-8)
        gt_quat_n   = gt_quat_raw / gt_norms
        gt_R_w2c    = quaternion_to_matrix(
            torch.from_numpy(gt_quat_n.reshape(-1, 4).astype(np.float32))
        ).numpy().reshape(T, K_gt, 3, 3)
        gt_t_w2c    = gt_camera_data[..., 4:7]
        gt_focal    = gt_camera_data[0, :, 7]

    # ── VGGT depth maps (optional) ────────────────────────────────────────────
    # Depth lives in VGGT 518-pixel space (same space as the npz intrinsics) and
    # is stored in VGGT units; the metric scale is recovered per frame from the
    # ratio between the predicted (scaled) camera translations and the raw VGGT
    # extrinsic translations.
    depth_mm = depth_conf = depth_valid_tk = None
    vggt_extr = vggt_intr = vggt_oc = vggt_cam_valid = None
    depth_scale = None
    if show_depth:
        depth_path = scene_dir / "vggt_depth_centered.npz"
        cam_path   = scene_dir / "vggt_cameras_centered.npz"
        if depth_path.exists() and cam_path.exists():
            depth_npz = np.load(depth_path, mmap_mode="r")
            depth_mm       = depth_npz["depth"]        # (T, K, h, w) uint16, VGGT units × 1000
            depth_conf     = depth_npz["depth_conf"]   # (T, K, h, w) float16
            depth_valid_tk = depth_npz["depth_valid"]  # (T, K) bool
            cam_npz = np.load(cam_path)
            vggt_extr      = cam_npz["extrinsics"]       # (T, K, 3, 4) cam-from-world
            vggt_intr      = cam_npz["intrinsics"]       # (T, K, 3, 3) 518-space
            vggt_oc        = cam_npz["original_coords"]  # (T, K, 4) [x1,y1,x2,y2]
            vggt_cam_valid = cam_npz["valid"]            # (T, K) bool
            vggt_cam_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in cam_npz["camera_names"]
            ]                                            # list[str] length K_vggt
            if not vggt_cam_valid.any():
                vggt_cam_valid = ~np.isnan(vggt_extr[:, :, 0, 0])

            # Per-frame metric scale: ||t_pred|| / ||t_vggt|| over valid,
            # non-reference cameras (the reference cam has t = 0).
            depth_scale = np.full(T, np.nan, dtype=np.float64)
            T_d = min(T, vggt_extr.shape[0])
            for t_s in range(T_d):
                ratios = []
                for k_s in range(vggt_extr.shape[1]):
                    if not vggt_cam_valid[t_s, k_s]:
                        continue
                    n_raw  = np.linalg.norm(vggt_extr[t_s, k_s, :3, 3])
                    n_pred = np.linalg.norm(camera[t_s, k_s, 4:7]) if k_s < K else 0.0
                    if n_raw > 1e-6 and n_pred > 1e-6:
                        ratios.append(n_pred / n_raw)
                if ratios:
                    depth_scale[t_s] = np.median(ratios)
            med = np.nanmedian(depth_scale)
            depth_scale = np.where(np.isfinite(depth_scale), depth_scale,
                                   med if np.isfinite(med) else 1.0)
            print(f"Depth point clouds enabled "
                  f"(metric scale median = {np.median(depth_scale):.4f} m/VGGT-unit)")
        else:
            print(f"WARNING: --show-depth requested but {depth_path.name} / "
                  f"{cam_path.name} not found in {scene_dir} — depth disabled.")
            show_depth = False

    # ── image size from first available frame ─────────────────────────────────
    sample_img = _load_rich_frame(rich_data_root, scene_name, 0, frame_start, frames_dir)
    if sample_img is not None:
        H, W = sample_img.shape[:2]
    else:
        H, W = 1080, 1920
    aspect = W / H

    # ── pre-load all video frames into RAM (parallel, ds=4 ~270×480) ──────────
    # All K cameras are loaded concurrently with threads (I/O bound).
    # On a network filesystem (cluster) this gives ~K× speedup.
    _DS = 4
    frames_cache: list[list[np.ndarray | None]] = [[None] * T for _ in range(K)]

    def _load_one(args: tuple[int, int]) -> tuple[int, int, np.ndarray | None]:
        k, t_idx = args
        img = _load_rich_frame(rich_data_root, scene_name, k, frame_start + t_idx, frames_dir)
        if img is not None:
            img = img[::_DS, ::_DS]
            return k, t_idx, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return k, t_idx, None

    all_tasks = [(k, t_idx) for k in range(K) for t_idx in range(T)]
    total = len(all_tasks)
    print(f"Pre-loading {T} frames × {K} cameras at 1/{_DS} resolution (parallel) …")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = min(64, total)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_load_one, task): task for task in all_tasks}
        done = 0
        for fut in as_completed(futs):
            k, t_idx, img = fut.result()
            frames_cache[k][t_idx] = img
            done += 1
            if done % max(1, total // 10) == 0:
                print(f"  {done}/{total} frames loaded …")

    for k in range(K):
        n_found = sum(f is not None for f in frames_cache[k])
        print(f"  cam {k}: {n_found}/{T} frames")
    print("Pre-loading complete.\n")

    # Placeholder black image for the background widget when not watching
    _BG_H, _BG_W = H // _DS, W // _DS
    _BLACK = np.zeros((_BG_H, _BG_W, 3), dtype=np.uint8)

    # ── start viser ──────────────────────────────────────────────────────────
    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")
    print(f"Viser viewer ready → http://localhost:{port}")
    print(f"On a cluster run:    ssh -L {port}:localhost:{port} <host>\n")

    # ── mutable state ─────────────────────────────────────────────────────────
    _playing     = False
    _last_frame  = [-1]          # list so inner functions can mutate it
    _clients: dict[int, viser.ClientHandle] = {}

    # ── GUI controls ──────────────────────────────────────────────────────────
    with server.gui.add_folder("Playback"):
        gui_frame       = server.gui.add_slider("Frame", min=0, max=T - 1,
                                                step=1, initial_value=first_valid)
        gui_play_button = server.gui.add_button("▶  Play")
        gui_fps         = server.gui.add_slider("FPS", min=1, max=60,
                                                step=1, initial_value=15)

    @gui_play_button.on_click
    def _toggle_play(_):
        nonlocal _playing
        _playing = not _playing
        gui_play_button.label = "⏸  Pause" if _playing else "▶  Play"

    with server.gui.add_folder("Visibility"):
        gui_show_pred       = server.gui.add_checkbox("Predicted body", True)
        gui_show_gt         = server.gui.add_checkbox("GT body", True)
        gui_show_frustum    = server.gui.add_checkbox("Camera frustums", True)
        gui_show_gt_frustum = server.gui.add_checkbox("GT camera frustums",
                                                       gt_R_w2c is not None)
        gui_show_frames     = server.gui.add_checkbox("Video frames in frustums", False)

    with server.gui.add_folder("Frustum"):
        gui_frustum_scale = server.gui.add_slider(
            "Scale", min=0.05, max=5.0, step=0.05, initial_value=0.15)
        gui_line_width    = server.gui.add_slider(
            "Line width", min=0.5, max=5.0, step=0.1, initial_value=1.5)
        # "all" moved to end — selecting one camera is the recommended path
        gui_video_cam     = server.gui.add_dropdown(
            "Video camera (frustum texture)",
            options=["none"] + [str(k) for k in range(K)] + ["all"],
            initial_value="none",
        )

    # Depth point-cloud controls (only when --show-depth)
    gui_show_depth = gui_depth_cam = gui_depth_point_size = None
    if show_depth:
        with server.gui.add_folder("Depth point cloud"):
            gui_show_depth = server.gui.add_checkbox("Show depth", True)
            gui_depth_cam  = server.gui.add_dropdown(
                "Depth camera",
                options=["all"] + [str(k) for k in range(K)],
                initial_value="all",
            )
            gui_depth_point_size = server.gui.add_slider(
                "Point size", min=0.001, max=0.1, step=0.001, initial_value=0.006)
            gui_depth_stride = server.gui.add_slider(
                "Pixel stride (1 = full res)", min=1, max=6, step=1, initial_value=2)

    # "Watch from camera" — snaps the viewer to a RICH camera each frame and
    # shows the live video feed as a sidebar background image.
    _watch_cam_options = ["Free"] + [f"pred cam {k}" for k in range(K)]
    if gt_R_w2c is not None:
        _watch_cam_options += [f"GT cam {k}" for k in range(K)]

    with server.gui.add_folder("Watch from Camera"):
        gui_watch_cam = server.gui.add_dropdown(
            "Camera", options=_watch_cam_options, initial_value="Free")
        gui_watch_info = server.gui.add_markdown(
            "Select a camera above to snap the viewer to it and show a live video feed below.")
        gui_bg_image = server.gui.add_image(
            _BLACK,
            label="Live feed  (updated each frame)",
            format="jpeg",
        )

    # ── frustum style callbacks ───────────────────────────────────────────────
    @gui_frustum_scale.on_update
    def _(_):
        for fh in _frustum_handles + _gt_frustum_handles:
            fh.scale = gui_frustum_scale.value

    @gui_line_width.on_update
    def _(_):
        for fh in _frustum_handles + _gt_frustum_handles:
            fh.line_width = gui_line_width.value

    @gui_show_frames.on_update
    def _(_):
        update_frame(gui_frame.value, force=True)

    @gui_video_cam.on_update
    def _(_):
        update_frame(gui_frame.value, force=True)

    # ── scene objects (created once, updated per frame) ───────────────────────
    _pred_mesh_handles = []
    for p in range(P):
        color = _PALETTE[p % len(_PALETTE)]
        h = server.scene.add_mesh_simple(
            f"/world/person_{p}/pred",
            vertices     = pred_verts_vis[0, p],
            faces        = faces,
            flat_shading = False,
            color        = color,
        )
        _pred_mesh_handles.append(h)

    _gt_mesh_handles = []
    if gt_verts_vis is not None:
        for p in range(P):
            color = (220, 220, 220)  # GT = light grey wireframe; pred uses _PALETTE
            h = server.scene.add_mesh_simple(
                f"/world/person_{p}/gt",
                vertices     = gt_verts_vis[0, p],
                faces        = faces,
                flat_shading = False,
                wireframe    = True,
                color        = color,
            )
            _gt_mesh_handles.append(h)

    # Predicted camera frustums
    _frustum_handles = []
    for k in range(K):
        vfov = 2 * np.arctan(H / 2 / max(focal[k], 1.0))
        wxyz, pos = _cam_to_viser(R_w2c[0, k], t_w2c[0, k])
        fh = server.scene.add_camera_frustum(
            f"/world/cam_{k:02d}",
            fov        = vfov,
            aspect     = aspect,
            scale      = gui_frustum_scale.value,
            line_width = gui_line_width.value,
            wxyz       = wxyz,
            position   = pos,
        )
        _frustum_handles.append(fh)

    # GT camera frustums — green
    _gt_frustum_handles = []
    if gt_R_w2c is not None:
        for k in range(gt_R_w2c.shape[1]):
            vfov_gt = 2 * np.arctan(H / 2 / max(gt_focal[k], 1.0))
            wxyz_gt, pos_gt = _cam_to_viser(gt_R_w2c[0, k], gt_t_w2c[0, k])
            fh_gt = server.scene.add_camera_frustum(
                f"/world/cam_gt_{k:02d}",
                fov        = vfov_gt,
                aspect     = aspect,
                scale      = gui_frustum_scale.value,
                line_width = gui_line_width.value,
                color      = (0, 200, 80),
                wxyz       = wxyz_gt,
                position   = pos_gt,
                visible    = gui_show_gt_frustum.value,
            )
            _gt_frustum_handles.append(fh_gt)

    # ── helpers for watch-from-camera mode ────────────────────────────────────
    def _parse_watch_cam() -> tuple[str, int] | None:
        """Parse gui_watch_cam.value → ('pred'|'gt', cam_idx) or None."""
        v = gui_watch_cam.value
        if v == "Free":
            return None
        if v.startswith("pred cam "):
            return ("pred", int(v.split()[-1]))
        if v.startswith("GT cam "):
            return ("gt", int(v.split()[-1]))
        return None

    def _snap_client_to_cam(client: viser.ClientHandle, t_idx: int) -> None:
        """Push this client's viewer to the selected RICH camera."""
        info = _parse_watch_cam()
        if info is None:
            return
        src, k = info
        if src == "pred":
            R, t, f_k = R_w2c[t_idx, k], t_w2c[t_idx, k], focal[k]
        else:
            if gt_R_w2c is None or k >= gt_R_w2c.shape[1]:
                return
            R, t, f_k = gt_R_w2c[t_idx, k], gt_t_w2c[t_idx, k], gt_focal[k]
        wxyz, pos = _cam_to_viser(R, t)
        vfov_k = float(2 * np.arctan(H / 2 / max(f_k, 1.0)))
        client.camera.wxyz     = wxyz
        client.camera.position = pos
        client.camera.fov      = vfov_k

    # ── client tracking ───────────────────────────────────────────────────────
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        _clients[client.client_id] = client
        # Snap new client to the currently selected camera immediately
        _snap_client_to_cam(client, gui_frame.value)

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        _clients.pop(client.client_id, None)

    # ── depth point clouds ────────────────────────────────────────────────────
    _depth_handles: dict[int, object] = {}
    _DEPTH_CONF_THR = 0.5
    # Lazily opened mask archives keyed by camera index k.
    _mask_npz: dict[int, object] = {}

    def _get_person_mask(abs_frame: int, k: int, h_full: int, w_full: int) -> np.ndarray | None:
        """Return a (h_full, w_full) bool array that is True where any person is present."""
        if k not in _mask_npz:
            cam_name = vggt_cam_names[k] if k < len(vggt_cam_names) else f"cam_{k:02d}"
            mpath = scene_dir / cam_name / "mask_data.npz"
            _mask_npz[k] = np.load(mpath, mmap_mode="r") if mpath.exists() else None
        npz = _mask_npz[k]
        if npz is None:
            return None
        cam_name = vggt_cam_names[k] if k < len(vggt_cam_names) else f"cam_{k:02d}"
        cam_idx  = int(cam_name.split("_")[-1])
        key = f"mask_{abs_frame:05d}_{cam_idx:02d}"
        if key not in npz:
            return None
        raw = npz[key].astype(np.uint16)
        if raw.shape == (h_full, w_full):
            return raw > 0
        from PIL import Image as _PIL
        resized = np.array(_PIL.fromarray(raw).resize((w_full, h_full), _PIL.NEAREST))
        return resized > 0

    def _depth_cloud(t_idx: int, k: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Unproject camera k's depth map at frame t into viser world space."""
        if (t_idx >= depth_mm.shape[0] or k >= depth_mm.shape[1]
                or not depth_valid_tk[t_idx, k] or not vggt_cam_valid[t_idx, k]):
            return None
        stride = int(gui_depth_stride.value)
        h_full, w_full = depth_mm[t_idx, k].shape
        d = depth_mm[t_idx, k][::stride, ::stride].astype(np.float32)
        d = d / 1000.0 * float(depth_scale[t_idx])   # → metres
        conf = depth_conf[t_idx, k][::stride, ::stride].astype(np.float32)

        h_d, w_d = d.shape
        vv, uu = np.mgrid[0:h_d, 0:w_d].astype(np.float32) * stride
        x1, y1, x2, y2 = vggt_oc[t_idx, k]

        # Exclude pixels occupied by any person so the depth cloud is background-only.
        # Mask is loaded at full depth resolution, then downsampled to match d's stride.
        person_mask_full = _get_person_mask(frame_start + t_idx, k, h_full, w_full)
        bg = True if person_mask_full is None else ~person_mask_full[::stride, ::stride]

        # Keep only valid-depth, confident, background pixels inside the un-padded region.
        mask = (
            (d > 1e-4) & (conf >= _DEPTH_CONF_THR)
            & (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2)
            & bg
        )
        if not mask.any():
            return None
        u, v, z = uu[mask], vv[mask], d[mask]

        intr = vggt_intr[t_idx, k]
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        pts_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)

        # World = R_w2c^T @ (x_cam - t_w2c), with t scaled to metres.
        R_d   = vggt_extr[t_idx, k, :3, :3]
        t_vec = vggt_extr[t_idx, k, :3, 3] * float(depth_scale[t_idx])
        pts_world = (pts_cam - t_vec) @ R_d
        pts_vis   = (pts_world @ _ROT180.T).astype(np.float32)

        # Colours sampled from the cached (downsampled) video frame.
        frame = frames_cache[k][t_idx] if k < len(frames_cache) else None
        if frame is not None:
            fh_, fw_ = frame.shape[:2]
            iu = np.clip(((u - x1) * W / (x2 - x1) / _DS).astype(np.int32), 0, fw_ - 1)
            iv = np.clip(((v - y1) * H / (y2 - y1) / _DS).astype(np.int32), 0, fh_ - 1)
            colors = frame[iv, iu]
        else:
            colors = np.full((len(z), 3), 128, dtype=np.uint8)
        return pts_vis, colors

    def _update_depth_clouds(t_idx: int) -> None:
        if not show_depth:
            return
        want: set[int] = set()
        if gui_show_depth.value:
            sel  = gui_depth_cam.value
            want = set(range(K)) if sel == "all" else {int(sel)}
        for k in list(_depth_handles):
            if k not in want:
                _depth_handles.pop(k).remove()
        for k in want:
            res = _depth_cloud(t_idx, k)
            if res is None:
                if k in _depth_handles:
                    _depth_handles.pop(k).remove()
                continue
            pts, cols = res
            _depth_handles[k] = server.scene.add_point_cloud(
                f"/world/depth/cam_{k:02d}",
                points     = pts,
                colors     = cols,
                point_size = gui_depth_point_size.value,
            )

    if show_depth:
        @gui_show_depth.on_update
        def _(_):
            _update_depth_clouds(gui_frame.value)

        @gui_depth_cam.on_update
        def _(_):
            _update_depth_clouds(gui_frame.value)

        @gui_depth_point_size.on_update
        def _(_):
            for h in _depth_handles.values():
                h.point_size = gui_depth_point_size.value

        @gui_depth_stride.on_update
        def _(_):
            _update_depth_clouds(gui_frame.value)

    # ── per-frame update ──────────────────────────────────────────────────────
    def update_frame(t_idx: int, force: bool = False) -> None:
        if t_idx == _last_frame[0] and not force:
            return
        _last_frame[0] = t_idx

        watch_info  = _parse_watch_cam()
        watching    = watch_info is not None
        vid_cam_str = gui_video_cam.value
        show_all    = vid_cam_str == "all"
        show_none   = vid_cam_str == "none"
        vid_cam_idx = -1 if vid_cam_str in ("all", "none") else int(vid_cam_str)

        with server.atomic():
            # ── bodies ───────────────────────────────────────────────────────
            for p, h in enumerate(_pred_mesh_handles):
                h.vertices = pred_verts_vis[t_idx, p]
                h.visible  = gui_show_pred.value
            for p, h in enumerate(_gt_mesh_handles):
                h.vertices = gt_verts_vis[t_idx, p]
                h.visible  = gui_show_gt.value and show_gt and bool(gt_valid[t_idx, p])

            # ── predicted camera frustums + optional frame texture ────────────
            for k, fh in enumerate(_frustum_handles):
                wxyz, pos = _cam_to_viser(R_w2c[t_idx, k], t_w2c[t_idx, k])
                fh.wxyz    = wxyz
                fh.position = pos
                fh.visible  = gui_show_frustum.value

                # Only send an image if explicitly requested (perf-sensitive)
                if gui_show_frames.value and not show_none:
                    if show_all or k == vid_cam_idx:
                        fh.image = frames_cache[k][t_idx] if frames_cache[k] else None
                    else:
                        fh.image = None
                else:
                    fh.image = None

            # ── GT frustums (static pose, only toggle visibility) ─────────────
            for fh_gt in _gt_frustum_handles:
                fh_gt.visible = gui_show_gt_frustum.value

            # ── background image for "watch from camera" mode ─────────────────
            if watching:
                src, k = watch_info
                cache_k = frames_cache[k]
                frame_img = cache_k[t_idx] if cache_k else None
                gui_bg_image.image = frame_img if frame_img is not None else _BLACK
            else:
                gui_bg_image.image = _BLACK

            # ── depth point clouds ─────────────────────────────────────────────
            _update_depth_clouds(t_idx)

    # Snap all clients when the user explicitly picks a camera — not every frame,
    # so the user can freely look around while the background image still updates.
    @gui_watch_cam.on_update
    def _(_):
        t_idx = gui_frame.value
        for client in list(_clients.values()):
            _snap_client_to_cam(client, t_idx)
        update_frame(t_idx, force=True)

    # Initial render
    update_frame(first_valid, force=True)

    # ── frame slider callback ─────────────────────────────────────────────────
    @gui_frame.on_update
    def _(_):
        update_frame(gui_frame.value)

    # ── event loop ────────────────────────────────────────────────────────────
    last_tick = time.time()
    try:
        while True:
            time.sleep(0.01)
            if _playing:
                now = time.time()
                if now - last_tick >= 1.0 / gui_fps.value:
                    last_tick = now
                    next_frame = (gui_frame.value + 1) % T
                    gui_frame.value = next_frame
                    update_frame(next_frame)
    except KeyboardInterrupt:
        pass


def main(
    predictions:     Path,
    scene_dir:       Path,
    rich_data_root:  Path        = Path(CONFIG.data.rich_data_root),
    smplx_model_dir: Path        = Path("body_models/SMPLX_NEUTRAL.pkl"),
    frame_start:     int         = 0,
    show_gt:         bool        = True,
    show_depth:      bool        = False,
    port:            int         = 9090,
    frames_dir:      Path | None = None,
) -> None:
    run(
        predictions     = predictions,
        scene_dir       = scene_dir,
        rich_data_root  = rich_data_root,
        smplx_model_dir = smplx_model_dir,
        frame_start     = frame_start,
        show_gt         = show_gt,
        show_depth      = show_depth,
        port            = port,
        frames_dir      = frames_dir,
    )


if __name__ == "__main__":
    tyro.cli(main)
