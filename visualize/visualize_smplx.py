"""Visualise SMPL-X body mesh overlaid on video frames.

For each person with stored smplx_* parameters, runs the SMPL-X model in
batch to recover mesh vertices for every frame, then rasterises the mesh
onto the frame as a solid (opaque) coloured surface using a painter's
algorithm (depth-sorted triangle fill via cv2).

Falls back to a capsule skeleton when smplx_* keys are absent.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ── Colour palette (BGR, one per person, cycles) ─────────────────────────────
_PALETTE: list[tuple[int, int, int]] = [
    ( 60,  80, 220),  # red
    ( 60, 200,  60),  # green
    (220,  80,  60),  # blue
    ( 40, 210, 210),  # yellow
    (210,  60, 210),  # magenta
    (210, 210,  40),  # cyan
    ( 40, 140, 220),  # orange
    (160,  60, 160),  # purple
]


def _color(person_id: int) -> tuple[int, int, int]:
    return _PALETTE[person_id % len(_PALETTE)]


# ── Camera projection ─────────────────────────────────────────────────────────

def _recover_cx_cy(
    kpts_cam: np.ndarray,   # (J, 3) camera-space joints
    kpts_2d: np.ndarray,    # (J, 2) pixel-space joints
    focal_length: float,
) -> tuple[float, float]:
    """Recover principal point analytically from known 3D→2D correspondences."""
    valid = kpts_cam[:, 2] > 1e-5
    if not valid.any():
        return 0.0, 0.0
    z = kpts_cam[valid, 2]
    cx = float(np.median(kpts_2d[valid, 0] - kpts_cam[valid, 0] / z * focal_length))
    cy = float(np.median(kpts_2d[valid, 1] - kpts_cam[valid, 1] / z * focal_length))
    return cx, cy


def _project(verts_cam: np.ndarray, focal_length: float,
             cx: float, cy: float) -> np.ndarray:
    """Perspective-project (N,3) camera-space points → (N,2) pixel coords."""
    z = np.maximum(verts_cam[:, 2], 1e-6)
    u = verts_cam[:, 0] / z * focal_length + cx
    v = verts_cam[:, 1] / z * focal_length + cy
    return np.stack([u, v], axis=-1)


# ── SMPL-X model ──────────────────────────────────────────────────────────────

def _smplx_create_kwargs(model_path: str) -> dict:
    """Auto-detect pkl vs npz and return the right model_path + ext for smplx.create."""
    p = Path(model_path)
    if p.is_file():
        return {"model_path": str(p), "ext": p.suffix.lstrip(".")}
    return {"model_path": model_path, "ext": "pkl"}


def _build_smplx_model(smplx_model_path: str, batch_size: int):
    """Create an SMPL-X model for the given batch size."""
    import smplx as smplx_lib
    return smplx_lib.create(
        **_smplx_create_kwargs(smplx_model_path),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=batch_size,
    )


def _run_smplx_batch(model, betas, body_pose, global_orient, transl,
                     left_hand_pose=None, right_hand_pose=None, expression=None):
    """Run SMPL-X forward; return (T, V, 3) vertices numpy array."""
    import torch
    def _t(arr, shape=None):
        x = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        if shape is not None:
            x = x.reshape(shape)
        return x

    T = len(betas)
    kwargs = dict(
        betas=_t(betas, (T, -1)),
        body_pose=_t(body_pose, (T, -1)),
        global_orient=_t(global_orient, (T, -1)),
        transl=_t(transl, (T, -1)),
        return_verts=True,
    )
    if left_hand_pose is not None:
        kwargs["left_hand_pose"] = _t(left_hand_pose, (T, -1))
    if right_hand_pose is not None:
        kwargs["right_hand_pose"] = _t(right_hand_pose, (T, -1))
    if expression is not None:
        kwargs["expression"] = _t(expression, (T, -1))
    with torch.no_grad():
        out = model(**kwargs)
    return out.vertices.cpu().numpy()  # (T, V, 3)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_persons(body_dir: Path, smplx_model_path: str | None) -> dict[int, dict]:
    """Load person_*.npz files; run SMPL-X forward if mesh params present."""
    persons: dict[int, dict] = {}
    for p in sorted(body_dir.glob("person_*.npz")):
        try:
            with np.load(str(p)) as f:
                data = dict(f)
            pid = int(p.stem.split("_")[-1])
            if "frame_indices" not in data:
                continue
            data["_frame_to_row"] = {int(v): i for i, v in enumerate(data["frame_indices"])}

            # Try to build SMPL-X mesh vertices for every frame
            mesh_keys = ("smplx_betas", "smplx_body_pose", "smplx_global_orient", "smplx_transl")
            missing = [k for k in mesh_keys if k not in data]
            if not smplx_model_path:
                print(f"  {p.name}: no smplx_model_path → skeleton fallback")
            elif missing:
                # Show which smplx_* keys are present vs absent so you can
                # diagnose conversion failures from body estimation.
                present = [k for k in data if k.startswith("smplx_")]
                if present:
                    print(f"  {p.name}: missing SMPL-X keys {missing}, "
                          f"present: {present} → skeleton fallback")
                else:
                    print(f"  {p.name}: no smplx_* keys found "
                          f"(SMPL-X conversion likely failed during estimation) "
                          f"→ skeleton fallback")
            else:
                try:
                    T = len(data["frame_indices"])
                    model = _build_smplx_model(smplx_model_path, T)
                    data["_vertices"] = _run_smplx_batch(
                        model,
                        data["smplx_betas"],
                        data["smplx_body_pose"],
                        data["smplx_global_orient"],
                        data["smplx_transl"],
                        left_hand_pose=data.get("smplx_left_hand_pose"),
                        right_hand_pose=data.get("smplx_right_hand_pose"),
                        expression=data.get("smplx_expression"),
                    )  # (T, V, 3) — camera space
                    data["_faces"] = model.faces.astype(np.int32)  # (F, 3)
                    del model  # free memory before next person
                    transl_z = data["smplx_transl"][:, 2]
                    print(f"  {p.name}: SMPL-X mesh ready ({T} frames, "
                          f"{data['_vertices'].shape[1]} verts, "
                          f"transl_z=[{transl_z.min():.2f}, {transl_z.max():.2f}]m)")
                except Exception as exc:
                    print(f"  {p.name}: SMPL-X forward failed — {exc}")
            persons[pid] = data
        except Exception:
            continue
    return persons


# ── Mesh rasteriser ───────────────────────────────────────────────────────────

def _draw_mesh(
    frame: np.ndarray,
    verts_body: np.ndarray,   # (V, 3) body-centric
    faces: np.ndarray,        # (F, 3) int
    cam_t: np.ndarray,        # (3,)
    focal_length: float,
    cx: float,
    cy: float,
    color: tuple[int, int, int],
) -> None:
    """Rasterise the SMPL-X mesh onto *frame* in-place (opaque, painter's algo)."""
    verts_cam = verts_body + cam_t[None, :]             # (V, 3) camera space
    verts_2d  = _project(verts_cam, focal_length, cx, cy)  # (V, 2) pixels

    # Face depths (mean z) — only keep front-facing faces
    face_z = verts_cam[faces, 2].mean(axis=1)           # (F,)
    valid  = face_z > 0.0

    # Backface cull via 2-D cross product (CCW winding = front)
    v0 = verts_2d[faces[valid, 0]]
    v1 = verts_2d[faces[valid, 1]]
    v2 = verts_2d[faces[valid, 2]]
    cross_z = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
            - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    front = cross_z > 0

    valid_idx = np.where(valid)[0][front]
    depths    = face_z[valid_idx]

    # Sort back-to-front (painter's algorithm)
    order = np.argsort(-depths)
    draw_faces = faces[valid_idx[order]]                # (K, 3)

    # Build list of triangle contours and fill in one cv2 call
    tris = verts_2d[draw_faces].astype(np.int32)        # (K, 3, 2)
    pts  = list(tris)                                   # list of (3, 2) arrays
    cv2.fillPoly(frame, pts, color)


# ── Fallback: capsule skeleton ────────────────────────────────────────────────

_BODY_EDGES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (0, 1), (0, 2),
    (12, 13), (12, 14), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21),
    (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
]


def _draw_skeleton(frame: np.ndarray, joints2d: np.ndarray,
                   bbox: np.ndarray, color: tuple) -> None:
    body_h = max(1, float(bbox[3] - bbox[1]))
    r = max(3, int(body_h / 24))
    for a, b in _BODY_EDGES:
        if a < len(joints2d) and b < len(joints2d):
            cv2.line(frame, tuple(joints2d[a].astype(int)),
                     tuple(joints2d[b].astype(int)), color, r * 2 + 1, cv2.LINE_AA)
    for x, y in joints2d:
        cv2.circle(frame, (int(x), int(y)), max(2, r // 2), color, -1, cv2.LINE_AA)
    if len(joints2d) > 15:
        neck = joints2d[12]; head = joints2d[15]
        cv2.circle(frame, tuple(head.astype(int)),
                   max(r, int(np.linalg.norm(head - neck) * 0.6)), color, -1, cv2.LINE_AA)


# ── Core visualize function ───────────────────────────────────────────────────

def visualize_smplx(
    video_dir: Path,
    fps: int = 30,
    frames_dir: Path | None = None,
    smplx_model_path: str | None = None,
) -> Path:
    """Render an mp4 with the SMPL-X mesh (solid, opaque) per person.

    Parameters
    ----------
    video_dir : Path
        Root output directory for one video (contains mask_data.npz,
        json_data/, body_data/).
    fps : int
        Frame rate of the output mp4.
    frames_dir : Path | None
        Directory with extracted JPEG frames.  Falls back to
        ``video_dir/frames/`` when not provided.
    smplx_model_path : str | None
        Path to SMPLX_NEUTRAL.npz.  Falls back to CONFIG if not given.
    """
    # Resolve smplx model path
    if smplx_model_path is None:
        try:
            from configuration import CONFIG
            smplx_model_path = getattr(CONFIG.data, "smplx_model_path", None)
        except Exception:
            pass

    video_dir = Path(video_dir)
    frame_dir = frames_dir if frames_dir is not None else video_dir / "frames"
    npz_path  = video_dir / "mask_data.npz"
    json_dir  = video_dir / "json_data"
    body_dir  = video_dir / "body_data"

    persons = _load_persons(body_dir, smplx_model_path)
    if not persons:
        print(f"  No person_*.npz files found in {body_dir}, skipping")
        return video_dir / "smplx_video.mp4"

    has_mesh = any("_vertices" in d for d in persons.values())
    if not has_mesh:
        print(f"  WARNING: No smplx_* params found in {body_dir}. "
              "Re-run the pipeline with smplx_model_path + mhr_model_path configured. "
              "Falling back to skeleton.")

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON metadata files found in {json_dir}")
    if not npz_path.exists():
        raise FileNotFoundError(f"mask_data.npz not found at {npz_path}")

    _FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def _find_frame(fi_str: str) -> Path | None:
        for ext in _FRAME_EXTS:
            p = frame_dir / f"{fi_str}{ext}"
            if p.exists():
                return p
        return None

    H, W = 0, 0
    for jf in json_files:
        p = _find_frame(jf.stem.replace("mask_", ""))
        if p is not None:
            sample = cv2.imread(str(p))
            if sample is not None:
                H, W = sample.shape[:2]
                break
    if H == 0:
        raise FileNotFoundError(f"No readable frames found in {frame_dir}")

    out_path = video_dir / "smplx_video.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H),
    )

    with zipfile.ZipFile(str(npz_path), "r") as zf:
        npz_keys = set(zf.namelist())

        for json_path in tqdm(json_files, desc=f"Rendering {video_dir.name}", leave=False):
            fi_str     = json_path.stem.replace("mask_", "")
            frame_path = _find_frame(fi_str)
            if frame_path is None:
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            with open(json_path) as f:
                labels: dict = json.load(f).get("labels", {})

            frame_idx = int(fi_str.split("_")[0])

            # Load per-frame mask
            mask_key = json_path.stem + ".npy"
            mask_img: np.ndarray | None = None
            if mask_key in npz_keys:
                with zf.open(mask_key) as mf:
                    mask_img = np.load(io.BytesIO(mf.read()))

            # 1. Draw SMPL-X mesh (or skeleton fallback) per person
            for pid, data in persons.items():
                row = data["_frame_to_row"].get(frame_idx)
                if row is None:
                    continue
                color   = _color(pid)
                cam_t   = data["pred_cam_t"][row]            # (3,)
                fl      = float(data["focal_length"][row])
                kp2d    = data["pred_keypoints_2d"][row]     # (J, 2)
                kp3d    = data["pred_keypoints_3d"][row]     # (J, 3)
                kpts_cam = kp3d + cam_t[None, :]
                cx, cy  = _recover_cx_cy(kpts_cam, kp2d, fl)

                if "_vertices" in data:
                    _draw_mesh(
                        frame,
                        verts_body=data["_vertices"][row],
                        faces=data["_faces"],
                        cam_t=np.zeros(3, dtype=np.float32),  # transl already baked into verts by SMPL-X forward pass
                        focal_length=fl,
                        cx=cx, cy=cy,
                        color=color,
                    )
                else:
                    _draw_skeleton(frame, kp2d[:22], data["bbox"][row], color)

            # 2. Bounding boxes and ID labels
            for str_id, info in labels.items():
                canon_id = int(str_id)
                color    = _color(canon_id)
                x1, y1   = int(info["x1"]), int(info["y1"])
                x2, y2   = int(info["x2"]), int(info["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"P{canon_id}"
                font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                (tw, fh), _ = cv2.getTextSize(label, font, fs, th)
                ty = max(y1 - 6, fh + 4)
                cv2.rectangle(frame, (x1, ty - fh - 4), (x1 + tw + 6, ty + 2), color, cv2.FILLED)
                cv2.putText(frame, label, (x1 + 3, ty - 1), font, fs, (255, 255, 255), th, cv2.LINE_AA)

            writer.write(frame)

    writer.release()
    print(f"  SMPL-X mesh visualisation saved: {out_path}")
    return out_path


def visualize_scene(
    scene_output_dir: Path,
    data_root: Path,
    fps: int = 30,
    smplx_model_path: str | None = None,
) -> None:
    """Run visualize_smplx() for every camera in a scene output directory."""
    scene_output_dir = Path(scene_output_dir)
    data_root        = Path(data_root)
    scene_name       = scene_output_dir.name
    # data_root may already point directly to the scene folder (e.g. .../train/BBQ_001_juggle)
    # or to its parent (e.g. .../train/).  Detect and handle both.
    if data_root.name == scene_name:
        scene_data = data_root
    else:
        scene_data = data_root / scene_name

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    cam_dirs = sorted(
        d for d in scene_output_dir.iterdir()
        if d.is_dir() and (d / "body_data").is_dir()
    )
    if not cam_dirs:
        print(f"No camera directories with body_data/ found in {scene_output_dir}")
        return

    for cam_dir in cam_dirs:
        cam_id = cam_dir.name

        frames_dir = None
        for candidate in [scene_data / cam_id / "frames", scene_data / cam_id]:
            if candidate.is_dir() and any(
                p.suffix.lower() in _IMG_EXTS for p in candidate.iterdir()
            ):
                frames_dir = candidate
                break

        if frames_dir is None:
            print(f"  {cam_id}: no source frames found, skipping")
            continue

        print(f"  {cam_id}: generating smplx_video.mp4 …")
        try:
            visualize_smplx(cam_dir, fps=fps, frames_dir=frames_dir,
                            smplx_model_path=smplx_model_path)
        except Exception as exc:
            print(f"  {cam_id}: failed — {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise SMPL-X body mesh overlaid on video frames."
    )
    parser.add_argument("--scene-dir",        type=Path, help="Scene output directory (e.g. test_outputs/.../BBQ_001_guitar)")
    parser.add_argument("--data-root",        type=Path, help="Data root with source frames (required with --scene-dir)")
    parser.add_argument("--video-dir",        type=Path, help="Single-camera output directory (single-camera mode)")
    parser.add_argument("--frames-dir",       type=Path, help="Source frames directory (overrides auto-detection)")
    parser.add_argument("--smplx-model-path", type=str,  default=None, help="Path to SMPLX_NEUTRAL.npz (falls back to CONFIG)")
    parser.add_argument("--fps",              type=int,  default=30)
    args = parser.parse_args()

    if args.scene_dir is not None:
        if args.data_root is None:
            parser.error("--data-root is required with --scene-dir")
        visualize_scene(args.scene_dir, args.data_root, args.fps,
                        smplx_model_path=args.smplx_model_path)
    elif args.video_dir is not None:
        visualize_smplx(args.video_dir, fps=args.fps, frames_dir=args.frames_dir,
                        smplx_model_path=args.smplx_model_path)
    else:
        parser.error("Provide --scene-dir or --video-dir")


if __name__ == "__main__":
    main()
