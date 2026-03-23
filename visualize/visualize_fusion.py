"""Viser-based interactive viewer for ghost fusion predictions.

Shows animated SMPL-X body meshes (predicted + GT) together with all camera
frustums textured with the original video frames.

Usage
-----
    pixi run python scripts/visualize_fusion.py \\
        --predictions fusion_outputs/60980456_predictions.npz \\
        --scene_dir   test_outputs/rich10_segmentation_test/BBQ_001_guitar \\
        --frame_start 0 \\
        --port 8080

IMPORTANT — run as a persistent background process (otherwise it dies when the
shell exits):

    nohup bash -c 'CONDA_OVERRIDE_CUDA=12.6 pixi run python scripts/visualize_fusion.py --predictions fusion_outputs/<job_id>_predictions.npz --scene_dir test_outputs/rich10_segmentation_test/BBQ_001_guitar --smplx-model-dir body_models/SMPLX_NEUTRAL.pkl --port 8080' > ~/viser.log 2>&1 &

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

import time
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
import viser.transforms as vtf
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
    mat = rotation_6d_to_matrix(t)          # (*, 3, 3)
    aa  = matrix_to_axis_angle(mat)         # (*, 3)
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

    # Normalise shape to (T, P, 10)
    if shape.ndim == 2:           # (P, 10) — constant over time
        shape = np.broadcast_to(shape[None], (T, P, 10)).copy()

    # Load model (neutral, no PCA for hands)
    # Support passing either a directory or a direct file path (pkl/npz)
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

    # Convert 6D → axis-angle
    global_orient_aa = _6d_to_aa(pose[:, :, 0, :])    # (T, P, 3)
    body_pose_aa     = _6d_to_aa(pose[:, :, 1:22, :]) # (T, P, 21, 3)

    def _t(x): return torch.from_numpy(x.reshape(T * P, -1).astype(np.float32))

    with torch.no_grad():
        out = model(
            global_orient = _t(global_orient_aa),            # (T*P, 3)
            body_pose     = _t(body_pose_aa),                # (T*P, 63)
            betas         = _t(shape),                       # (T*P, 10)
            transl        = _t(trans),                       # (T*P, 3)
            return_verts  = True,
        )

    V = out.vertices.shape[1]
    verts = out.vertices.numpy().reshape(T, P, V, 3)   # (T, P, V, 3)
    faces = model.faces.copy()                          # (F, 3)
    return verts, faces


def _load_rich_frame(
    rich_data_root: Path,
    scene_name: str,
    cam_idx: int,
    rich_frame_idx: int,
) -> np.ndarray | None:
    """Load one RICH frame as BGR uint8, or None if not found."""
    cam_dir = rich_data_root / scene_name / f"cam_{cam_idx:02d}"
    p = cam_dir / f"{rich_frame_idx:05d}_00.jpg"
    if not p.exists():
        # try without the _00 suffix
        for ext in (".jpg", ".png", ".bmp"):
            q = cam_dir / f"{rich_frame_idx:05d}{ext}"
            if q.exists():
                p = q
                break
    img = cv2.imread(str(p))
    return img   # None if still not found


def _cam_to_viser(
    R_w2c: np.ndarray,  # (3, 3)  world-to-camera rotation
    t_w2c: np.ndarray,  # (3,)    world-to-camera translation
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world-to-camera [R|t] to viser wxyz quaternion + position.

    Viser expects the camera-to-world transform, Y-up convention.
    """
    # cam-to-world
    R_c2w = R_w2c.T
    t_c2w = -(R_w2c.T @ t_w2c)

    # apply Y-flip to match viser's Y-up convention
    R_vis = _ROT180 @ R_c2w
    t_vis = _ROT180 @ t_c2w

    quat_xyzw = SciR.from_matrix(R_vis).as_quat()   # xyzw
    quat_wxyz  = np.concatenate([quat_xyzw[3:], quat_xyzw[:3]])
    return quat_wxyz, t_vis


# ── main viewer ───────────────────────────────────────────────────────────────

def run(
    predictions:     Path,
    scene_dir:       Path,
    rich_data_root:  Path  = Path("/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"),
    smplx_model_dir: Path  = Path("body_models/SMPLX_NEUTRAL.pkl"),
    frame_start:     int   = 0,
    show_gt:         bool  = True,
    port:            int   = 9090,
) -> None:
    """Launch the interactive viser viewer.

    Parameters
    ----------
    predictions     : path to the *_predictions.npz saved by the trainer
    scene_dir       : ghost output directory for the scene
                      (e.g. test_outputs/rich10_segmentation_test/BBQ_001_guitar)
    rich_data_root  : root of the RICH dataset (contains cam_XX subdirs)
    smplx_model_dir : directory that contains SMPLX_NEUTRAL.pkl (or smplx/ subdir)
    frame_start     : first RICH frame index (used to load correct images)
    show_gt         : also render the GT body in wireframe
    port            : viser WebSocket port
    """
    # ── load predictions ──────────────────────────────────────────────────────
    print(f"Loading {predictions} …")
    d = dict(np.load(predictions, allow_pickle=True))

    pose   = d["pose"]    # (T, P, J, 6)
    shape  = d["shape"]   # (P, 10)
    camera = d["camera"]  # (T, K, 8)  [quat wxyz(4), trans(3), focal(1)]
    trans  = d["trans"]   # (T, P, 3)

    T, P, J, _ = pose.shape
    K = camera.shape[1]
    scene_name = scene_dir.name

    print(f"  T={T} frames, P={P} persons, K={K} cameras")

    # Optionally load GT for side-by-side comparison
    gt_pose  = d.get("gt_pose")   # (T, P, J, 6) or None
    gt_shape = d.get("gt_shape")  # (T, P, 10) or None
    gt_trans = d.get("gt_trans")  # (T, P, 3)  or None — may be absent

    # Frames where gt_trans is all-zero have no RICH annotation — hide GT there.
    if gt_trans is not None:
        gt_valid = ~np.all(gt_trans.reshape(T, -1) == 0, axis=-1)  # (T,) bool
        first_valid = int(gt_valid.argmax()) if gt_valid.any() else 0
        n_missing = int((~gt_valid).sum())
        if n_missing:
            print(f"  {n_missing} frames have no GT annotation (gt_trans=0) — GT body hidden there.")
    else:
        gt_valid = np.zeros(T, dtype=bool)
        first_valid = 0

    # ── build SMPL-X meshes ───────────────────────────────────────────────────
    print("Running SMPL-X forward pass for predictions …")
    pred_verts, faces = _build_smplx_vertices(pose, shape, trans, smplx_model_dir)
    # pred_verts: (T, P, V, 3) in world (cam-0) space

    gt_verts = None
    if show_gt and gt_pose is not None and gt_trans is not None:
        print("Running SMPL-X forward pass for GT …")
        gt_shape_arr = gt_shape if gt_shape is not None else shape
        gt_verts, _ = _build_smplx_vertices(gt_pose, gt_shape_arr, gt_trans, smplx_model_dir)

    # Apply Y-flip for viser convention
    pred_verts_vis = pred_verts @ _ROT180.T   # (T, P, V, 3)
    gt_verts_vis   = gt_verts   @ _ROT180.T if gt_verts is not None else None

    # ── decode predicted cameras ──────────────────────────────────────────────
    # camera: (T, K, 8)  quat is wxyz, not unit (q_in + Δq stored raw)
    quat_raw = camera[..., :4]                              # (T, K, 4)
    # normalise
    norms = np.linalg.norm(quat_raw, axis=-1, keepdims=True).clip(1e-8)
    quat_n = quat_raw / norms                               # (T, K, 4) unit wxyz
    R_w2c  = quaternion_to_matrix(
        torch.from_numpy(quat_n.reshape(-1, 4).astype(np.float32))
    ).numpy().reshape(T, K, 3, 3)
    t_w2c  = camera[..., 4:7]                              # (T, K, 3)
    focal  = camera[0, :, 7]                               # (K,) constant

    # Image size from first available frame
    sample_img = _load_rich_frame(rich_data_root, scene_name, 0, frame_start)
    if sample_img is not None:
        H, W = sample_img.shape[:2]
    else:
        H, W = 1080, 1920
    aspect = W / H

    # ── start viser ──────────────────────────────────────────────────────────
    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")
    print(f"\nViser viewer ready → http://localhost:{port}")
    print(f"On a cluster run:    ssh -L {port}:localhost:{port} <host>\n")

    # ── GUI controls ──────────────────────────────────────────────────────────
    _playing = False   # mutable state for play/pause toggle

    with server.gui.add_folder("Playback"):
        gui_frame        = server.gui.add_slider("Frame", min=0, max=T - 1,
                                                 step=1, initial_value=first_valid)
        gui_play_button  = server.gui.add_button("▶  Play")
        gui_fps          = server.gui.add_slider("FPS", min=1, max=60,
                                                 step=1, initial_value=15)

    @gui_play_button.on_click
    def _toggle_play(_):
        nonlocal _playing
        _playing = not _playing
        gui_play_button.label = "⏸  Pause" if _playing else "▶  Play"

    with server.gui.add_folder("Visibility"):
        gui_show_pred    = server.gui.add_checkbox("Predicted body", True)
        gui_show_gt      = server.gui.add_checkbox("GT body", True)
        gui_show_frustum = server.gui.add_checkbox("Camera frustums", True)
        gui_show_frames  = server.gui.add_checkbox("Video frames", True)

    with server.gui.add_folder("Frustum"):
        gui_frustum_scale = server.gui.add_slider(
            "Scale", min=0.05, max=2.0, step=0.01, initial_value=0.3)
        gui_line_width    = server.gui.add_slider(
            "Line width", min=0.5, max=5.0, step=0.1, initial_value=1.5)

    # ── mesh handles (created once, updated per frame) ────────────────────────
    pred_mesh_handles = []
    for p in range(P):
        color = _PALETTE[p % len(_PALETTE)]
        h = server.scene.add_mesh_simple(
            f"/world/person_{p}/pred",
            vertices  = pred_verts_vis[0, p],
            faces     = faces,
            flat_shading = False,
            color     = color,
        )
        pred_mesh_handles.append(h)

    gt_mesh_handles = []
    if gt_verts_vis is not None:
        for p in range(P):
            color = _PALETTE[p % len(_PALETTE)]
            h = server.scene.add_mesh_simple(
                f"/world/person_{p}/gt",
                vertices  = gt_verts_vis[0, p],
                faces     = faces,
                flat_shading = False,
                wireframe    = True,
                color     = color,
            )
            gt_mesh_handles.append(h)

    # ── frustum handles (updated per frame because cameras may be time-varying) ─
    frustum_handles = []
    for k in range(K):
        vfov = 2 * np.arctan(H / 2 / max(focal[k], 1.0))
        wxyz, pos = _cam_to_viser(R_w2c[0, k], t_w2c[0, k])
        fh = server.scene.add_camera_frustum(
            f"/world/cam_{k:02d}",
            fov          = vfov,
            aspect       = aspect,
            scale        = gui_frustum_scale.value,
            line_width   = gui_line_width.value,
            wxyz         = wxyz,
            position     = pos,
        )
        frustum_handles.append(fh)

    # ── frustum style callbacks ───────────────────────────────────────────────
    @gui_frustum_scale.on_update
    def _(_):
        for fh in frustum_handles:
            fh.scale = gui_frustum_scale.value

    @gui_line_width.on_update
    def _(_):
        for fh in frustum_handles:
            fh.line_width = gui_line_width.value

    # ── per-frame update ──────────────────────────────────────────────────────
    last_frame = -1

    def update_frame(t_idx: int) -> None:
        nonlocal last_frame
        if t_idx == last_frame:
            return
        last_frame = t_idx

        rich_frame_idx = frame_start + t_idx

        with server.atomic():
            # Bodies
            for p, h in enumerate(pred_mesh_handles):
                h.vertices = pred_verts_vis[t_idx, p]
                h.visible  = gui_show_pred.value
            for p, h in enumerate(gt_mesh_handles):
                h.vertices = gt_verts_vis[t_idx, p]
                h.visible  = gui_show_gt.value and show_gt and bool(gt_valid[t_idx])

            # Cameras + images
            for k, fh in enumerate(frustum_handles):
                wxyz, pos = _cam_to_viser(R_w2c[t_idx, k], t_w2c[t_idx, k])
                fh.wxyz    = wxyz
                fh.position = pos
                fh.visible  = gui_show_frustum.value

                if gui_show_frames.value:
                    img_bgr = _load_rich_frame(rich_data_root, scene_name,
                                               k, rich_frame_idx)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        fh.image = img_rgb
                else:
                    fh.image = None

    # first render — start at first annotated frame
    update_frame(first_valid)

    # ── event loop ────────────────────────────────────────────────────────────
    @gui_frame.on_update
    def _(_):
        update_frame(gui_frame.value)

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
    rich_data_root:  Path = Path("/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"),
    smplx_model_dir: Path = Path("body_models/SMPLX_NEUTRAL.pkl"),
    frame_start:     int  = 0,
    show_gt:         bool = True,
    port:            int  = 9090,
) -> None:
    run(
        predictions     = predictions,
        scene_dir       = scene_dir,
        rich_data_root  = rich_data_root,
        smplx_model_dir = smplx_model_dir,
        frame_start     = frame_start,
        show_gt         = show_gt,
        port            = port,
    )


if __name__ == "__main__":
    tyro.cli(main)
