"""Run Sapiens 2D pose estimation on a ghost-pipeline scene (centered_train split).

Reads bounding boxes from json_data/ per camera, crops each person from the
RICH centered_train frames, runs Sapiens, and saves per-camera per-person keypoint files:
    {cam_dir}/sapiens_centered_kps_person_{pid}.npz
        keypoints    (T, 308, 3) float32  [x, y, conf] in CENTERED (cropped)
                                          image pixels — matches vggt_*_centered
        frame_indices (T,)        int32

Sapiens outputs 308 COCO-WholeBody keypoints:
    0-16:   COCO body (same order as ViTPose COCO-17)
    17-22:  feet
    23-90:  face
    91-111: left hand
    112-132: right hand

Usage:
    pixi run python scripts/sapiens_centered_preprocess.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich/centered_train \\
        --model_size 0.3b \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ImageNet normalisation
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

_MODEL_IDS = {
    "0.3b": ("facebook/sapiens-pose-0.3b-torchscript",
             "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"),
    "0.6b": ("facebook/sapiens-pose-0.6b-torchscript",
             "sapiens_0.6b_goliath_best_goliath_AP_600_torchscript.pt2"),
    "1b":   ("facebook/sapiens-pose-1b-torchscript",
             "sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2"),
    "2b":   ("facebook/sapiens-pose-2b-torchscript",
             "sapiens_2b_goliath_best_goliath_AP_673_torchscript.pt2"),
}

_INPUT_H = 1024   # Sapiens traced resolution (portrait)
_INPUT_W = 768
_BBOX_PAD   = 0.20  # 20% padding around each bbox


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(model_size: str, device: torch.device) -> torch.jit.ScriptModule:
    from huggingface_hub import hf_hub_download
    repo_id, filename = _MODEL_IDS[model_size]
    print(f"Downloading / loading {repo_id}/{filename} …")
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"  Cached at {local}")
    model = torch.jit.load(local, map_location=device)
    model.eval()
    return model


def _load_frame(rich_root: str, scene_name: str, cam_name: str,
                frame_idx: int) -> np.ndarray | None:
    """Load a RICH frame as HxWx3 uint8."""
    cam_suffix = cam_name.split("_")[-1]
    cam_dir = Path(rich_root) / scene_name / cam_name
    for ext in (".jpg", ".jpeg", ".bmp", ".png"):
        p = cam_dir / f"{frame_idx:05d}_{cam_suffix}{ext}"
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    return None


def _load_crop_offset(rich_root: str, scene_name: str, cam_name: str) -> tuple[int, int]:
    """Return the (off_x, off_y) crop offset for one camera from crop_meta.json.

    center_images.py writes <rich_root>/<scene>/crop_meta.json recording, per
    camera, the top-left corner of the crop in the source-image coordinate
    system.  SAM3 bboxes (json_data) live in source coordinates, so they must
    be shifted by -(off_x, off_y) to land correctly on the centered frames.
    Returns (0, 0) if no metadata is found (uncropped / legacy case).
    """
    meta_path = Path(rich_root) / scene_name / "crop_meta.json"
    if not meta_path.exists():
        print(f"  [warn] no crop_meta.json at {meta_path} — assuming offset (0,0)",
              file=sys.stderr)
        return 0, 0
    with open(meta_path) as f:
        meta = json.load(f)
    cam = meta.get("cameras", {}).get(cam_name)
    if cam is None:
        print(f"  [warn] {cam_name} not in {meta_path} — assuming offset (0,0)",
              file=sys.stderr)
        return 0, 0
    return int(cam["off_x"]), int(cam["off_y"])


def _pad_bbox(x1, y1, x2, y2, pad: float, W: int, H: int):
    """Add proportional padding and clamp to image bounds."""
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * pad, bh * pad
    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(W, int(x2 + dx))
    y2 = min(H, int(y2 + dy))
    return x1, y1, x2, y2


def _expand_bbox_to_aspect(x1, y1, x2, y2, target_w, target_h, W_img, H_img):
    """Expand bbox to match target_w:target_h aspect ratio without distortion.

    Returns the expanded (x1, y1, x2, y2) clamped to image bounds,
    and the actual padded (pad_x1, pad_y1) offset within the expanded crop
    (always 0,0 when expansion fits inside image).
    """
    bw, bh = x2 - x1, y2 - y1
    target_ratio = target_w / target_h  # 768/1024 = 0.75
    current_ratio = bw / max(bh, 1)

    if current_ratio < target_ratio:
        # Too narrow — expand width
        new_bw = bh * target_ratio
        cx = (x1 + x2) / 2
        x1 = max(0, int(cx - new_bw / 2))
        x2 = min(W_img, int(cx + new_bw / 2))
    else:
        # Too wide — expand height
        new_bh = bw / target_ratio
        cy = (y1 + y2) / 2
        y1 = max(0, int(cy - new_bh / 2))
        y2 = min(H_img, int(cy + new_bh / 2))

    return int(x1), int(y1), int(x2), int(y2)


def _preprocess_crop(img_np: np.ndarray, x1, y1, x2, y2,
                     device: torch.device) -> torch.Tensor:
    """Crop → expand to 3:4 aspect → resize to 768×1024 → normalise → (1,3,H,W)."""
    H_img, W_img = img_np.shape[:2]
    x1, y1, x2, y2 = _expand_bbox_to_aspect(x1, y1, x2, y2, _INPUT_W, _INPUT_H, W_img, H_img)
    crop = img_np[y1:y2, x1:x2]
    pil  = Image.fromarray(crop).resize((_INPUT_W, _INPUT_H), Image.BILINEAR)
    t    = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0)  # (H,W,3)
    t    = (t - _MEAN) / _STD
    t    = t.permute(2, 0, 1).unsqueeze(0).to(device)                 # (1,3,1024,768)
    return t, (x1, y1, x2 - x1, y2 - y1)  # also return expanded crop box


def _decode_heatmaps(out: torch.Tensor, crop_x1: int, crop_y1: int,
                     crop_w: int, crop_h: int) -> np.ndarray:
    """Convert (1, 308, 256, 192) heatmaps to (308, 3) [x, y, conf] in centered-image pixels."""
    hm = out.squeeze(0).float()   # (308, 256, 192)
    K, H, W = hm.shape
    flat = hm.reshape(K, -1)
    idx  = flat.argmax(-1)                      # (K,) peak location
    conf = torch.sigmoid(flat[torch.arange(K), idx])  # sigmoid of raw peak → [0,1]
    # Keypoint in model input space (768×1024)
    col  = (idx %  W).float() / W * _INPUT_W   # x in [0, 768]
    row  = (idx // W).float() / H * _INPUT_H   # y in [0, 1024]
    # Scale to original image space via the expanded crop box
    x_orig = col / _INPUT_W * crop_w + crop_x1
    y_orig = row / _INPUT_H * crop_h + crop_y1
    kps = torch.stack([x_orig, y_orig, conf], dim=-1).cpu().numpy()
    return kps.astype(np.float32)


# ── main processing ───────────────────────────────────────────────────────────

def process_camera(cam_dir: Path, scene_name: str, rich_root: str,
                   model: torch.jit.ScriptModule, device: torch.device,
                   overwrite: bool = False, max_frames: int | None = None) -> None:
    json_dir  = cam_dir / "json_data"
    body_dir  = cam_dir / "body_data"
    cam_name  = cam_dir.name
    cam_suffix = cam_name.split("_")[-1]

    # SAM3 bboxes (json_data) are in source-image pixels; the centered frames
    # are cropped, so shift each bbox by -(off_x, off_y) to align them.
    off_x, off_y = _load_crop_offset(rich_root, scene_name, cam_name)

    if not json_dir.is_dir() or not body_dir.is_dir():
        print(f"  [{cam_name}] missing json_data or body_data — skipping")
        return

    # Collect all pids tracked in this camera
    pid_files = sorted(body_dir.glob("person_*.npz"))
    if not pid_files:
        return

    # Build pid → frame_indices map
    pid_frames: dict[int, np.ndarray] = {}
    for pf in pid_files:
        pid = int(pf.stem.split("_")[1])
        d   = np.load(pf, allow_pickle=False)
        pid_frames[pid] = d["frame_indices"].astype(int)

    pids = sorted(pid_frames)

    # Check if all outputs already exist
    if not overwrite and all(
        (cam_dir / f"sapiens_centered_kps_person_{pid}.npz").exists() for pid in pids
    ):
        print(f"  [{cam_name}] all sapiens_centered_kps already exist — skipping (use --overwrite)")
        return

    # Accumulate keypoints per pid
    kps_accum:  dict[int, list[np.ndarray]] = {pid: [] for pid in pids}
    fi_accum:   dict[int, list[int]]        = {pid: [] for pid in pids}

    # All global frames across all pids (sorted)
    all_frames: set[int] = set()
    for fi in pid_frames.values():
        all_frames.update(fi.tolist())

    frame_list = sorted(all_frames)
    if max_frames is not None:
        frame_list = frame_list[:max_frames]
    total = len(frame_list)
    print(f"  [{cam_name}] {total} frames × {len(pids)} pid(s) …")

    cached_frame: tuple[int, np.ndarray | None] = (-1, None)

    for frame_idx in frame_list:
        # Lazy-load frame (reuse across pids in same frame)
        if cached_frame[0] != frame_idx:
            img = _load_frame(rich_root, scene_name, cam_name, frame_idx)
            cached_frame = (frame_idx, img)
        else:
            img = cached_frame[1]

        if img is None:
            # Frame not found — skip (accumulate zeros)
            for pid in pids:
                if frame_idx in pid_frames[pid]:
                    kps_accum[pid].append(np.zeros((308, 3), dtype=np.float32))
                    fi_accum[pid].append(frame_idx)
            continue

        H_img, W_img = img.shape[:2]

        # Load JSON for this frame
        json_path = json_dir / f"mask_{frame_idx:05d}_{cam_suffix}.json"
        if not json_path.exists():
            for pid in pids:
                if frame_idx in pid_frames[pid]:
                    kps_accum[pid].append(np.zeros((308, 3), dtype=np.float32))
                    fi_accum[pid].append(frame_idx)
            continue

        with open(json_path) as f:
            jdata = json.load(f)
        labels = jdata.get("labels", {})

        for pid in pids:
            if frame_idx not in pid_frames[pid]:
                continue

            pid_str = str(pid)
            if pid_str not in labels:
                kps_accum[pid].append(np.zeros((308, 3), dtype=np.float32))
                fi_accum[pid].append(frame_idx)
                continue

            bb = labels[pid_str]
            # Shift from source-image coords into the cropped (centered) frame,
            # then pad; _pad_bbox clamps to the centered image bounds.
            x1, y1, x2, y2 = _pad_bbox(
                bb["x1"] - off_x, bb["y1"] - off_y,
                bb["x2"] - off_x, bb["y2"] - off_y,
                _BBOX_PAD, W_img, H_img,
            )
            if x2 - x1 < 2 or y2 - y1 < 2:
                kps_accum[pid].append(np.zeros((308, 3), dtype=np.float32))
                fi_accum[pid].append(frame_idx)
                continue

            inp, (cx1, cy1, cw, ch) = _preprocess_crop(img, x1, y1, x2, y2, device)

            with torch.no_grad():
                out = model(inp)

            kps = _decode_heatmaps(out, cx1, cy1, cw, ch)
            kps_accum[pid].append(kps)
            fi_accum[pid].append(frame_idx)

    # Save per-pid
    for pid in pids:
        out_path = cam_dir / f"sapiens_centered_kps_person_{pid}.npz"
        if not kps_accum[pid]:
            continue
        np.savez_compressed(
            str(out_path),
            keypoints    = np.stack(kps_accum[pid]).astype(np.float32),  # (T, 308, 3)
            frame_indices = np.array(fi_accum[pid], dtype=np.int32),
        )
        print(f"  [{cam_name}] pid={pid}: saved {len(fi_accum[pid])} frames → {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",
        default="/capstor/scratch/cscs/tnanni/datasets/rich/centered_train")
    ap.add_argument("--model_size",  default="0.3b",
        choices=list(_MODEL_IDS), help="Sapiens model size")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite",   action="store_true")
    ap.add_argument("--cameras",     default="",
        help="Comma-separated camera names to process (default: all)")
    ap.add_argument("--max_frames",  type=int, default=None,
        help="Process only first N frames per camera (for testing)")
    args = ap.parse_args()

    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    device     = torch.device(args.device)

    # Load model
    model = _load_model(args.model_size, device)

    # Discover camera directories
    cam_dirs = sorted(
        d for d in scene_dir.iterdir()
        if d.is_dir() and (d / "body_data").is_dir()
    )
    if args.cameras:
        keep = {c.strip() for c in args.cameras.split(",")}
        cam_dirs = [d for d in cam_dirs if d.name in keep]

    print(f"Scene: {scene_name}  |  {len(cam_dirs)} camera(s)  |  model: {args.model_size}")
    for cam_dir in cam_dirs:
        process_camera(cam_dir, scene_name, args.rich_root, model, device,
                       args.overwrite, args.max_frames)

    print("Done.")


if __name__ == "__main__":
    main()
