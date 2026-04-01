"""One-off script to compact existing mask_data/ folders into mask_data.npz."""
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


def compact_mask_data(video_dir: Path) -> None:
    mask_data_dir = video_dir / "mask_data"
    if not mask_data_dir.exists():
        return

    npy_files = sorted(mask_data_dir.glob("*.npy"))
    if not npy_files:
        shutil.rmtree(str(mask_data_dir))
        return

    original_mb = sum(f.stat().st_size for f in npy_files) / 1024 / 1024
    npz_path = video_dir / "mask_data.npz"

    # Write one frame at a time into the zip to avoid loading 2.3 GB into RAM.
    with zipfile.ZipFile(str(npz_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in tqdm(npy_files, desc=f"  {video_dir.name}", leave=False):
            arr = np.load(str(f))
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(f.stem + ".npy", buf.getvalue())

    shutil.rmtree(str(mask_data_dir))

    compressed_mb = npz_path.stat().st_size / 1024 / 1024
    print(
        f"  {len(npy_files)} files, {original_mb:.0f} MB"
        f" → {npz_path.name} {compressed_mb:.0f} MB"
        f" ({100 * compressed_mb / original_mb:.0f}% of original)"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compact existing mask_data/ folders into mask_data.npz.")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Root output directory to scan for mask_data/ folders")
    args = parser.parse_args()
    test_outputs = args.output_dir

    print("=== Debug: raw rglob results ===")
    existing_npz = sorted(test_outputs.rglob("mask_data.npz"))
    print(f"mask_data.npz found ({len(existing_npz)}):")
    for p in existing_npz:
        print(f"  {p}  (exists={p.exists()}, is_file={p.is_file()}, size={p.stat().st_size if p.exists() else 'N/A'})")

    mask_dirs = sorted(p for p in test_outputs.rglob("mask_data") if p.is_dir())
    print(f"mask_data/ dirs found ({len(mask_dirs)}):")
    for d in mask_dirs:
        npy_count = len(list(d.glob("*.npy")))
        print(f"  {d}  (is_dir={d.is_dir()}, npy_count={npy_count})")
    print()

    # Dirs that already have an npz alongside them (leftover from a crashed run)
    orphan_dirs = [d for d in mask_dirs if (d.parent / "mask_data.npz").exists()]
    # Dirs that still need full compaction
    todo_dirs = [d for d in mask_dirs if d not in orphan_dirs]

    print("=== Current state ===")
    if existing_npz:
        print(f"\nAlready compacted ({len(existing_npz)} mask_data.npz):")
        for p in existing_npz:
            print(f"  {p}  ({p.stat().st_size / 1024 / 1024:.0f} MB)")
    if orphan_dirs:
        print(f"\nLeftover mask_data/ folders (npz already exists, just need deletion):")
        for d in orphan_dirs:
            size_mb = sum(f.stat().st_size for f in d.glob("*.npy")) / 1024 / 1024
            print(f"  {d}  ({size_mb:.0f} MB)")
    if todo_dirs:
        print(f"\nStill to compact ({len(todo_dirs)} mask_data/ folder(s)):")
        for d in todo_dirs:
            npy_count = len(list(d.glob("*.npy")))
            size_mb = sum(f.stat().st_size for f in d.glob("*.npy")) / 1024 / 1024
            print(f"  {d}  ({npy_count} files, {size_mb:.0f} MB)")
    if not existing_npz and not mask_dirs:
        print("  Nothing found under", test_outputs)
        return

    if orphan_dirs:
        print("\nDeleting leftover mask_data/ folders ...")
        for d in orphan_dirs:
            shutil.rmtree(str(d))
            print(f"  Deleted {d}")

    if todo_dirs:
        print()
        for d in todo_dirs:
            print(f"Compacting {d.parent.name}/{d.name} ...")
            compact_mask_data(d.parent)

    print("\nDone.")


if __name__ == "__main__":
    main()
