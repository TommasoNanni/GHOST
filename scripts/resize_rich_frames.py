"""Resize RICH image sequences before squashing.

Walk data_root/<scene>/<cam>/*.{jpg,bmp,...} and write aspect-ratio-preserving
resized copies to output_dir (same sub-tree layout).  Only images whose
longest dimension exceeds --max_side are touched; smaller ones are copied as-is.

Typical usage (resize all cameras in a dataset):

    python scripts/resize_rich_frames.py \\
        --data_root /path/to/RICH \\
        --output_dir /path/to/RICH_resized \\
        --max_side 840 \\
        --workers 8

The output directory mirrors data_root exactly:

    output_dir/<scene>/<cam>/<frame>.jpg

so it can be squash-mounted at the same paths the pipeline expects.
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


def _list_image_files(cam_dir: Path) -> list[Path]:
    return sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _resize_camera(
    cam_dir: Path,
    out_cam_dir: Path,
    max_side: int,
    skip_existing: bool,
) -> tuple[int, int]:
    """Resize all images in cam_dir → out_cam_dir.  Returns (processed, skipped)."""
    images = _list_image_files(cam_dir)
    if not images:
        return 0, 0

    out_cam_dir.mkdir(parents=True, exist_ok=True)

    # Check first image to see if this camera even needs resizing.
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        print(f"  WARNING: cannot read {images[0]}, skipping camera.")
        return 0, 0

    h, w = first_img.shape[:2]
    needs_resize = max(h, w) > max_side

    processed = skipped = 0
    for src in images:
        dst = out_cam_dir / src.name
        if skip_existing and dst.exists():
            skipped += 1
            continue

        if not needs_resize:
            shutil.copy2(src, dst)
            processed += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"  WARNING: cannot read {src}, skipping.")
            continue
        ih, iw = img.shape[:2]
        scale = max_side / max(ih, iw)
        img = cv2.resize(img, (round(iw * scale), round(ih * scale)),
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img)
        processed += 1

    return processed, skipped


def _collect_cameras(data_root: Path) -> list[tuple[Path, str, str]]:
    """Return list of (cam_dir, scene_id, cam_id) for every camera found."""
    cameras = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            if any(p.suffix.lower() in IMAGE_EXTS for p in cam_dir.iterdir()):
                cameras.append((cam_dir, scene_dir.name, cam_dir.name))
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize RICH frames for squashing.")
    parser.add_argument("--data_root", required=True,
                        help="Root directory with RICH scenes.")
    parser.add_argument("--output_dir", required=True,
                        help="Destination root (mirrored layout).")
    parser.add_argument("--max_side", type=int, default=840,
                        help="Longest dimension after resize (default: 840).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel camera workers (default: 4).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already exist in output_dir.")
    parser.add_argument("--scene", default=None,
                        help="Process only this scene (optional filter).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    cameras = _collect_cameras(data_root)
    if args.scene:
        cameras = [(cd, sc, cam) for cd, sc, cam in cameras if sc == args.scene]

    if not cameras:
        print("No cameras found. Check --data_root and --scene.")
        return

    print(f"Found {len(cameras)} cameras across "
          f"{len({sc for _, sc, _ in cameras})} scenes.")
    print(f"Resizing to max_side={args.max_side} → {output_dir}")

    total_processed = total_skipped = 0

    def _task(item):
        cam_dir, scene_id, cam_id = item
        out_cam_dir = output_dir / scene_id / cam_id
        return _resize_camera(cam_dir, out_cam_dir, args.max_side, args.skip_existing)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, item): item for item in cameras}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                desc="Cameras"):
            cam_dir, scene_id, cam_id = futures[future]
            try:
                p, s = future.result()
                total_processed += p
                total_skipped += s
            except Exception as exc:
                print(f"  ERROR in {scene_id}/{cam_id}: {exc}")

    print(f"\nDone. Processed: {total_processed} images, "
          f"skipped (existing): {total_skipped} images.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
