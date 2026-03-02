"""Convert RICH dataset BMP images to JPEG in-place.

Walks all .bmp files under RICH_SPLIT_ROOT, converts each to JPEG at the
specified quality, and removes the original BMP.  Uses a multiprocessing
pool for throughput.

Usage:
    python -m utilities.convert_rich_bmp_to_jpeg \\
        --root /cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train \\
        --quality 92 \\
        --workers 16 \\
        --dry-run        # omit to actually convert
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not available — run: pip install Pillow")


# ---------------------------------------------------------------------------

def convert_one(args: tuple[Path, int, bool]) -> tuple[str, bool, str]:
    bmp_path, quality, dry_run = args
    jpg_path = bmp_path.with_suffix(".jpg")
    try:
        if not dry_run:
            img = Image.open(bmp_path)
            img.save(jpg_path, "JPEG", quality=quality, subsampling=0)
            bmp_path.unlink()
        return str(bmp_path), True, ""
    except Exception as exc:
        return str(bmp_path), False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RICH BMP → JPEG")
    parser.add_argument(
        "--root",
        default="/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train",
        help="Root directory to search recursively for .bmp files",
    )
    parser.add_argument(
        "--quality", type=int, default=92,
        help="JPEG quality 1–95 (default: 92)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel worker processes (default: 16)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be converted without doing anything",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Root not found: {root}")

    bmp_files = sorted(root.rglob("*.bmp"))
    if not bmp_files:
        print("No .bmp files found.")
        return

    total = len(bmp_files)
    total_mb = sum(p.stat().st_size for p in bmp_files) / 1024 / 1024
    print(f"Found {total} BMP files  ({total_mb:.1f} MB total)")
    if args.dry_run:
        print("DRY RUN — no files will be modified.")

    tasks = [(p, args.quality, args.dry_run) for p in bmp_files]

    ok = fail = 0
    with Pool(processes=args.workers) as pool:
        for i, (path, success, err) in enumerate(
            pool.imap_unordered(convert_one, tasks, chunksize=20), 1
        ):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FAILED {path}: {err}")
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}  ok={ok}  fail={fail}")

    print(f"\nDone. Converted: {ok}  Failed: {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
