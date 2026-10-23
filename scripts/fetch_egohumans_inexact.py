#!/usr/bin/env python3
"""Fetch + undistort the EgoHumans scenes whose sync estimate was NOT exact.

Driver for the through-sync EgoHumans evaluation. Reads the manifest written by
evaluation/sync_dryrun_egohumans.py, takes the scenes in the INEXACT bucket (at
least one camera off by >=1 frame, so the multi-view reconstruction genuinely
changes and VGGT + MapAnything have to be re-run on the mis-aligned image
selection), and for each one, sequentially:

  1. streams its tarball off Drive and extracts `exo/` + `colmap/`
     (scripts/download_egohumans_gt.py --full --exclude ego processed_data)
  2. undistorts that scene's exo cameras in place
     (utilities/undistort_egohumans.py, which needs colmap/workplace/*.txt and
     writes exo/<cam>/images_undistorted/ + calibration.json)

Sequential on purpose: one scene's images land and are undistorted before the
next download starts, so a kill at any point leaves a consistent tree and the
peak disk cost is one tarball, not sixty.

`ego/` is excluded (never used -- the pipeline runs on exo only) and
`processed_data/` too (the GT is already staged at egohumans_gt_full). Note
that excluding costs nothing on the wire: a .tar.gz is sequential, so the bytes
are pulled either way, they just are not written.

Resumable: a scene is skipped when every camera in KEEP_CAMS already has an
`images_undistorted/` with as many images as `images/` (or when the raw images
have since been dropped but the undistorted ones are there), so re-running
after an interrupt costs one directory listing per scene.

Usage
-----
    nohup pixi run python scripts/fetch_egohumans_inexact.py \\
        --manifest eval_egohumans/sync_manifest_s75.json \\
        --data_root /iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted \\
        > logs/fetch_egohumans_inexact.log 2>&1 &

    # only the scenes whose sync error exceeds one frame (6.2 GiB, minutes):
    ... --min_error 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")

# Mirrors utilities/undistort_egohumans.py::KEEP_CAMS -- the exo cameras the
# pipeline actually ran on. Kept as a literal (not imported) so this driver
# stays runnable if that module's imports (cv2) are unavailable.
KEEP_CAMS: dict[str, list[str]] = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _n_images(d: Path) -> int:
    """Image count in d, looking through a frames/ subdir if the pipeline made one."""
    if not d.is_dir():
        return 0
    sub = d / "frames"
    if sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in sub.iterdir()):
        d = sub
    return sum(1 for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def scene_done(seq_dir: Path, cams: list[str]) -> bool:
    """True when every kept camera has a complete images_undistorted/.

    Complete means: at least as many undistorted images as raw ones. The `>=`
    (not `==`) matters because the raw images/ dir may have been deleted after
    a previous undistortion to save disk -- 0 raw and N undistorted is done,
    not broken.
    """
    checked = 0
    for cam in cams:
        cam_dir = seq_dir / "exo" / cam
        if not cam_dir.is_dir():
            # Not every scene ships every camera in KEEP_CAMS -- several badminton
            # scenes have only cam02/05/07. A camera the dataset does not contain
            # cannot be undistorted and must not make the scene look unfinished;
            # the manifest's cam_ids (what the reconstruction actually uses) never
            # names it either.
            continue
        checked += 1
        n_und = _n_images(cam_dir / "images_undistorted")
        if n_und == 0 or n_und < _n_images(cam_dir / "images"):
            return False
    # A scene that was never downloaded has NO camera directories at all, so the
    # loop above skips every one of them and would otherwise fall through to
    # "done". Requiring at least one verified camera keeps "nothing on disk" from
    # masquerading as "already complete".
    return checked > 0


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def undistort_scene(data_root: Path, activity: str, scene: str, seq_dir: Path,
                    workers: int) -> int:
    """Undistort ONE scene, via a scratch root that symlinks only that scene.

    utilities/undistort_egohumans.py walks every sequence under
    <data_root>/<activity>/<INNER>/<activity>/ and calls _parse_colmap_calibration
    on each, which raises FileNotFoundError on any sequence lacking
    colmap/workplace/cameras.txt. The scenes staged by earlier work have no
    colmap/ (only their undistorted images survived), so pointing the tool at the
    real root makes it die on a neighbour before it ever reaches the scene we
    just downloaded.

    Handing it a throwaway root whose camera_ready dir contains a single symlink
    to the target sequence keeps the walk down to that one scene. The tool writes
    into <seq_dir>/exo/<cam>/images_undistorted/, and since seq_dir is reached
    through the symlink, the output lands in the real tree.
    """
    with tempfile.TemporaryDirectory(prefix=f"undist_{activity}_{scene}_") as tmp:
        cam_ready = Path(tmp) / activity / INNER / activity
        cam_ready.mkdir(parents=True, exist_ok=True)
        (cam_ready / scene).symlink_to(seq_dir, target_is_directory=True)
        return run([sys.executable, "utilities/undistort_egohumans.py",
                    "--data_root", tmp, "--activity", activity,
                    "--workers", str(workers)])


def prune_unused_cams(seq_dir: Path, cams: list[str]) -> int:
    """Delete exo/<cam>/ for every camera the pipeline does not use.

    The tarball ships up to 8 exo cameras; ghost ran on the 4 in KEEP_CAMS, and
    only those are undistorted, so the rest are raw fisheye frames nothing will
    ever read. Deleting them cannot lose work: they are re-fetchable from the
    same tarball, and no output was derived from them.

    Bandwidth is unaffected -- a .tar.gz is sequential, so those bytes came over
    the wire regardless. This only reclaims disk.
    """
    exo = seq_dir / "exo"
    if not exo.is_dir():
        return 0
    n = 0
    for cam_dir in sorted(p for p in exo.iterdir() if p.is_dir()):
        if cam_dir.name in cams:
            continue
        shutil.rmtree(cam_dir, ignore_errors=True)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSON from evaluation/sync_dryrun_egohumans.py")
    ap.add_argument("--data_root", required=True, type=Path,
                    help="EgoHumans root holding <activity>/<INNER>/<activity>/<scene>/")
    ap.add_argument("--min_error", type=int, default=1,
                    help="Only fetch scenes whose max per-camera sync error is >= this "
                         "(1 = the whole inexact bucket).")
    ap.add_argument("--workers", type=int, default=16, help="Undistortion worker processes.")
    ap.add_argument("--keep_unused_cams", action="store_true",
                    help="Keep the exo cameras outside KEEP_CAMS. Off by default: they are "
                         "never undistorted and never read, so they are pure dead disk.")
    ap.add_argument("--drop_raw", action="store_true",
                    help="Delete exo/<cam>/images/ after a scene undistorts cleanly "
                         "(halves disk; the distorted frames are not used downstream).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    todo: list[tuple[str, str, int]] = []
    for key, entry in manifest["scenes"].items():
        worst = max((t["max_sync_error"] for t in entry["trials"] if not t["exact"]),
                    default=0)
        if worst >= args.min_error:
            activity, scene = key.split("/", 1)
            todo.append((activity, scene, worst))
    todo.sort(key=lambda r: (-r[2], r[0], r[1]))   # biggest errors first: they matter most

    print(f"{len(todo)} scene(s) with sync error >= {args.min_error} frame(s)")
    if args.dry_run:
        for a, s, e in todo:
            print(f"  {a}/{s}  max_err={e}")
        return 0

    n_ok = n_skip = n_fail = 0
    t_start = time.time()
    for i, (activity, scene, worst) in enumerate(todo, 1):
        cam_ready = args.data_root / activity / INNER / activity
        seq_dir = cam_ready / scene
        cams = KEEP_CAMS[activity]
        head = f"[{i}/{len(todo)}] {activity}/{scene} (max_err={worst})"

        if scene_done(seq_dir, cams):
            print(f"{head}: already undistorted -- skip", flush=True)
            n_skip += 1
            continue

        print(f"\n{'='*70}\n{head}  elapsed={time.time() - t_start:.0f}s\n{'='*70}", flush=True)
        cam_ready.mkdir(parents=True, exist_ok=True)

        rc = run([sys.executable, "scripts/download_egohumans_gt.py",
                  "--activity", activity, "--scenes", scene,
                  "--dest-root", str(cam_ready),
                  "--full", "--exclude", "ego", "processed_data"])
        if rc != 0:
            print(f"{head}: DOWNLOAD FAILED (rc={rc}) -- continuing", flush=True)
            n_fail += 1
            continue

        rc = undistort_scene(args.data_root, activity, scene, seq_dir, args.workers)
        if rc != 0 or not scene_done(seq_dir, cams):
            print(f"{head}: UNDISTORT INCOMPLETE (rc={rc}) -- continuing", flush=True)
            n_fail += 1
            continue

        if not args.keep_unused_cams:
            n_pruned = prune_unused_cams(seq_dir, cams)
            if n_pruned:
                print(f"{head}: pruned {n_pruned} unused camera dir(s)", flush=True)

        if args.drop_raw:
            for cam in cams:
                raw = seq_dir / "exo" / cam / "images"
                if raw.is_dir():
                    for p in raw.iterdir():
                        if p.is_file():
                            p.unlink()
                    raw.rmdir()
            print(f"{head}: dropped raw distorted images", flush=True)

        n_ok += 1
        print(f"{head}: OK  ({n_ok} done, {n_skip} skipped, {n_fail} failed, "
              f"{time.time() - t_start:.0f}s elapsed)", flush=True)

    print(f"\n{'='*70}\nDone. ok={n_ok} skipped={n_skip} failed={n_fail}  "
          f"total={time.time() - t_start:.0f}s\n{'='*70}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
