#!/usr/bin/env python
"""Download EgoHumans sequences from Drive: the ground truth only, or the lot.

EgoHumans ships one `.tar.gz` per subsequence on Google Drive, with
`processed_data/` bundled next to `exo/`, `ego/` and `colmap/`. There is no
per-file endpoint and no GT-only distribution (the repo's `benchmark/` folder is
COCO-2D for the three test activities only).

**GT mode (default).** `processed_data/` is the **first** member in every
archive. A `.tar.gz` must be read sequentially, but we never need the rest: this
script streams the download, extracts `processed_data/` on the fly, and **tears
down the connection at the first member outside it**. Measured on 001_badminton:
0.61 GB of a 21.57 GB tarball -> 2.8%, yielding 716 MB of GT (25843 files). The
GT is written next to the images already on iopsstor:

    <dest_root>/<scene>/processed_data/{smpl,poses2d,poses3d,bboxes,...}

Nothing outside `<dest_root>/<scene>/processed_data/` is created, modified or
removed, so the existing `exo/` image trees are untouched.

**Full mode (`--full`).** Every member is extracted and the stream is read to
EOF, rebuilding the whole scene tree (`processed_data/`, `exo/`, `ego/`,
`colmap/`) under `<dest_root>/<scene>/`. Budget **5-22 GB per scene**.
`--exclude ego` keeps a top-level dir off disk; it does *not* save bandwidth,
because tar members cannot be skipped without reading them.

Either way nothing lands in a temp file: bytes go from the socket into the
extracted files.

Resumable: a scene is skipped when its sentinel exists (`.gt_done` in GT mode,
`.full_done` in full mode) or, in GT mode, when its `processed_data/` already
holds files. Re-running after an interruption is safe. The pre-existing
`processed_data/` skeletons on iopsstor are *empty*, so they are correctly
treated as not-done.

For the pipeline you want the raw `images/`: the archived `images_undistorted/`
were produced with fisheye `balance=1.0` and carry black half-moon borders.
Re-undistort with `scripts/prep_undistort_egohumans.py` (balance=0).

Example - the scenes whose GT is lost to the corrupt sqsh part-a:

    pixi run python scripts/download_egohumans_gt.py \
        --activity 06_badminton \
        --dest-root /iopsstor/scratch/cscs/tnanni/backup/badminton_egohumans/06_badminton/\
media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/06_badminton \
        --scenes $(seq -f '%03g_badminton' 1 30)

Example - re-download whole scenes, images included, minus the ego videos:

    pixi run python scripts/download_egohumans_gt.py --full --exclude ego \
        --activity 02_lego --dest-root <.../camera_ready/02_lego> \
        --scenes 001_lego 002_lego
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import gdown
import requests

# Root Google Drive folder from the EgoHumans repo's assets/DOWNLOAD.md.
DEFAULT_FOLDER_ID = "1JD963urzuzV_R_6FOVOtlx8UupwUuknR"
DEFAULT_FOLDER_URL = f"https://drive.google.com/drive/folders/{DEFAULT_FOLDER_ID}"
GT_DIR = "processed_data"
SENTINEL = ".gt_done"
FULL_SENTINEL = ".full_done"
# Top-level dirs of a scene tree. Used to anchor member paths in full mode when
# the scene dir name itself is absent from the archive's nesting.
TOP_DIRS = frozenset({GT_DIR, "exo", "ego", "colmap"})
CHUNK = 1 << 20  # 1 MiB
PROGRESS_EVERY = 5000  # files, full mode only


class _DoneWithGT(Exception):
    """Raised to unwind out of tarfile iteration once GT has been read."""


class QuotaExceeded(Exception):
    """Drive refused the download: per-file public download quota is spent.

    Drive signals this with HTTP 200 + an HTML page, not an error status, so it
    must be detected from the content type. Retrying within the same run is
    pointless -- the quota resets on Google's own schedule (typically ~24 h).
    """


class HttpStream:
    """Minimal read-only file object over a streaming HTTP response."""

    def __init__(self, resp: requests.Response, chunk: int = CHUNK):
        self._it = resp.iter_content(chunk_size=chunk)
        self._buf = bytearray()
        self.n_read = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:  # tarfile's stream reader never does this, but be safe
            for b in self._it:
                self._buf += b
                self.n_read += len(b)
            out = bytes(self._buf)
            self._buf.clear()
            return out
        while len(self._buf) < size:
            try:
                b = next(self._it)
            except StopIteration:
                break
            self._buf += b
            self.n_read += len(b)
        out = bytes(self._buf[:size])
        del self._buf[:size]
        return out


class CountingReader:
    """Byte-counting passthrough over a binary stream (rclone's stdout).

    `rclone cat` reports no transfer stats on a pipe, so the pulled volume is
    measured here instead -- the number that tells you how much of the tarball
    actually crossed the wire before the GT-mode teardown.
    """

    def __init__(self, fh):
        self._fh = fh
        self.n_read = 0

    def read(self, size: int = -1) -> bytes:
        b = self._fh.read(size)
        self.n_read += len(b)
        return b


def list_activity_tarballs(folder_url: str, activity: str) -> list[tuple[str, str]]:
    """Return [(scene, file_id)] for every `<activity>/<scene>.tar.gz` on Drive."""
    entries = gdown.download_folder(folder_url, skip_download=True, quiet=True)
    if not entries:
        raise RuntimeError(f"Drive folder listing returned nothing: {folder_url}")
    prefix = f"{activity}/"
    out = [
        (Path(e.path).name[: -len(".tar.gz")], e.id)
        for e in entries
        if e.path.startswith(prefix) and e.path.endswith(".tar.gz")
    ]
    return sorted(out)


def already_done(scene_dir: Path, full: bool = False) -> bool:
    if (scene_dir / (FULL_SENTINEL if full else SENTINEL)).exists():
        return True
    if full:
        # A previous GT-only run leaves .gt_done and a populated processed_data/,
        # neither of which means the images are here -- so require image data.
        # (With --exclude exo the sentinel above is the only resume signal.)
        exo = scene_dir / "exo"
        return exo.is_dir() and any(p.is_file() for p in exo.rglob("*"))
    gt = scene_dir / GT_DIR
    if not gt.is_dir():
        return False
    # iopsstor holds an *empty* processed_data skeleton; only real files count.
    return any(p.is_file() for p in gt.rglob("*"))


def _rel_path(parts: tuple[str, ...], scene: str) -> Path | None:
    """Member path relative to the scene root, or None if it cannot be anchored.

    Archives nest the scene dir under a long `media/rawalk/.../camera_ready/`
    prefix whose depth is not guaranteed, so nothing here relies on a fixed
    --strip-components count: anchor on the scene dir when the archive carries
    it, else on a known top-level dir. Returns None for the scene dir itself
    (empty remainder) and for members matching neither anchor.
    """
    if scene in parts:
        rel = parts[parts.index(scene) + 1:]
    else:
        idx = next((i for i, p in enumerate(parts) if p in TOP_DIRS), None)
        if idx is None:
            return None
        rel = parts[idx:]
    return Path(*rel) if rel else None


def _extract_from_stream(fileobj, dest_root: Path, scene: str, *,
                         full: bool = False,
                         exclude: frozenset[str] = frozenset()) -> tuple[int, bool]:
    """Extract members of a sequential tar.gz stream into `<dest_root>/<scene>/`.

    GT mode keeps only `processed_data/` and stops (via _DoneWithGT) at the first
    member past it, so the caller can tear the transfer down. Full mode keeps
    every member whose top-level dir is not in `exclude`, and reads to EOF.

    Paths are rebuilt from each member's own suffix (see `_rel_path`) rather than
    a fixed --strip-components count, so a change in archive nesting cannot
    silently misplace files. Members are written explicitly (no extractall) and
    anything resolving outside dest_root/<scene> is rejected.
    """
    scene_root = (dest_root / scene).resolve()
    n_files = 0
    seen_gt = False
    unanchored = 0
    try:
        with tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
            for m in tf:
                parts = Path(m.name).parts
                if full:
                    rel = _rel_path(parts, scene)
                    if rel is None:
                        # The scene dir itself is expected; anything else is an
                        # archive layout we do not understand -- report, skip.
                        unanchored += scene not in parts
                        continue
                    if rel.parts[0] in exclude:
                        continue
                    seen_gt = seen_gt or rel.parts[0] == GT_DIR
                else:
                    if GT_DIR not in parts:
                        if seen_gt:
                            raise _DoneWithGT  # past the GT: stop pulling bytes
                        continue  # leading scene dir, before GT
                    seen_gt = True
                    rel = Path(*parts[parts.index(GT_DIR):])
                target = dest_root / scene / rel
                if not str(target.resolve()).startswith(str(scene_root)):
                    raise RuntimeError(f"refusing unsafe member path: {m.name}")
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n_files += 1
                    if full and n_files % PROGRESS_EVERY == 0:
                        print(f"    ... {n_files} files", flush=True)
    except _DoneWithGT:
        pass
    if unanchored:
        print(f"    warning: {unanchored} member(s) matched no anchor, skipped",
              flush=True)
    # seen_gt distinguishes "processed_data/ present but empty" (a legitimately
    # unannotated scene, e.g. 010_basketball) from "GT never appeared" (a
    # truncated/broken stream). The caller must not retry the former.
    return n_files, seen_gt


def stream_extract_rclone(scene: str, activity: str, dest_root: Path,
                          remote: str, root_id: str, *, full: bool = False,
                          exclude: frozenset[str] = frozenset()) -> tuple[int, bool, int]:
    """Stream the tarball through an authenticated `rclone cat` and extract it.

    Anonymous Drive downloads hit a per-downloader quota (HTTP 200 + a "Quota
    exceeded" HTML page) after a few tens of GB. An OAuth'd rclone remote is not
    subject to it, so this is the reliable path. In GT mode rclone is killed as
    soon as the GT has been read, so only a fraction of the tarball crosses the
    wire; in full mode the stream is drained to EOF and rclone's exit status is
    checked, so a transfer that died mid-tarball is not mistaken for a success.
    """
    src = f"{remote}{activity}/{scene}.tar.gz"
    cmd = ["rclone", "cat", "--drive-root-folder-id", root_id, src]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reader = CountingReader(proc.stdout)
    drained = False
    try:
        n_files, seen_gt = _extract_from_stream(reader, dest_root, scene,
                                                full=full, exclude=exclude)
        drained = True
    finally:
        # Full mode reaches EOF on its own; killing is for the GT-mode teardown
        # and for aborting a failed extraction without waiting out 20 GB.
        if not (full and drained):
            proc.kill()
        try:
            _, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            err = b""
    if full and proc.returncode:
        raise RuntimeError(f"rclone exited {proc.returncode} on {scene}: "
                           f"{err.decode('utf8', 'replace')[:200]}")
    # In GT mode, only a real failure if the GT dir never appeared in the stream.
    if not full and not seen_gt and err:
        raise RuntimeError(f"rclone gave no GT for {scene}: "
                           f"{err.decode('utf8', 'replace')[:200]}")
    return n_files, seen_gt, reader.n_read


def stream_extract_http(file_id: str, dest_root: Path, scene: str, *,
                        full: bool = False,
                        exclude: frozenset[str] = frozenset()) -> tuple[int, bool, int]:
    """Anonymous HTTP fallback. Subject to Drive's download quota."""
    url = (f"https://drive.usercontent.google.com/download"
           f"?id={file_id}&export=download&confirm=t")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        # Drive reports a spent download quota as 200 + text/html, so a status
        # check is not enough. Catch it here, else tarfile reports the useless
        # "not a gzip file".
        ctype = resp.headers.get("content-type", "")
        if "text/html" in ctype:
            head = next(resp.iter_content(chunk_size=4096), b"")
            text = head.decode("utf8", "replace")
            if "quota" in text.lower():
                raise QuotaExceeded(
                    f"Drive download quota exceeded for {scene} (id={file_id}). "
                    f"Use --backend rclone with an authenticated remote."
                )
            raise RuntimeError(f"expected tarball, got HTML for {scene}: {text[:200]}")

        stream = HttpStream(resp)
        n_files, seen_gt = _extract_from_stream(stream, dest_root, scene,
                                                full=full, exclude=exclude)
        return n_files, seen_gt, stream.n_read


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--activity", default="06_badminton")
    ap.add_argument("--dest-root", required=True, type=Path,
                    help="camera_ready/<activity> dir where the <scene>/ trees live")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="subset, e.g. $(seq -f '%%03g_badminton' 1 30)")
    ap.add_argument("--folder-url", default=DEFAULT_FOLDER_URL)
    ap.add_argument("--backend", choices=["rclone", "http"], default="rclone",
                    help="rclone = authenticated (no quota); http = anonymous")
    ap.add_argument("--rclone-remote", default="gdrive:",
                    help="OAuth'd rclone remote, e.g. gdrive:")
    ap.add_argument("--root-folder-id", default=DEFAULT_FOLDER_ID)
    ap.add_argument("--full", action="store_true",
                    help="extract the whole scene tree (images included), not "
                         "just processed_data/. 5-22 GB per scene.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    metavar="DIR",
                    help="--full only: top-level dirs to keep off disk, e.g. "
                         "'ego'. Does NOT save bandwidth (tar is sequential).")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if the scene already looks complete")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dest_root.is_dir():
        ap.error(f"--dest-root does not exist: {args.dest_root}")
    if args.exclude and not args.full:
        ap.error("--exclude only applies with --full")
    unknown = set(args.exclude) - TOP_DIRS
    if unknown:
        ap.error(f"--exclude: unknown top-level dir(s) {sorted(unknown)}; "
                 f"known: {sorted(TOP_DIRS)}")
    exclude = frozenset(args.exclude)
    sentinel = FULL_SENTINEL if args.full else SENTINEL
    kind = "scene" if args.full else "GT"

    print(f"Listing {args.activity} on Drive ...", flush=True)
    tarballs = list_activity_tarballs(args.folder_url, args.activity)
    if args.scenes:
        wanted = set(args.scenes)
        tarballs = [(s, i) for s, i in tarballs if s in wanted]
    if not tarballs:
        print(f"No tarballs matched activity={args.activity} scenes={args.scenes}")
        return 1

    todo, skipped = [], []
    for scene, fid in tarballs:
        (skipped if (not args.force
                     and already_done(args.dest_root / scene, args.full))
         else todo).append((scene, fid))

    if skipped:
        print(f"skip ({kind} present): {len(skipped)} -> "
              f"{', '.join(s for s, _ in skipped)}\n")
    if args.dry_run:
        print(f"DRY RUN - would fetch {kind} for:")
        for scene, fid in todo:
            print(f"  {scene}  id={fid}")
        print(f"\ntotal: {len(todo)} scene(s)")
        if args.full:
            print(f"est. {5 * len(todo)}-{22 * len(todo)} GB downloaded"
                  + (f", minus {sorted(exclude)} on disk" if exclude else ""))
        return 0
    if not todo:
        print("Nothing to do.")
        return 0
    if args.full:
        print(f"FULL mode: {len(todo)} scene(s), est. {5 * len(todo)}-"
              f"{22 * len(todo)} GB over the wire"
              + (f"; not writing {sorted(exclude)}" if exclude else "")
              + "\n", flush=True)

    ok, failed, empty, total_bytes = [], [], [], 0
    quota_hit = False
    for n, (scene, fid) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {scene}", flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                if args.backend == "rclone":
                    n_files, seen_gt, n_bytes = stream_extract_rclone(
                        scene, args.activity, args.dest_root,
                        args.rclone_remote, args.root_folder_id,
                        full=args.full, exclude=exclude)
                else:
                    n_files, seen_gt, n_bytes = stream_extract_http(
                        fid, args.dest_root, scene,
                        full=args.full, exclude=exclude)
            except QuotaExceeded as e:
                # No point retrying this scene or any later one in this run.
                print(f"  QUOTA: {e}", flush=True)
                quota_hit = True
                failed.append(scene)
                break
            except Exception as e:  # noqa: BLE001 - report and retry
                print(f"    attempt {attempt}: {type(e).__name__}: {e}", flush=True)
                if attempt < args.retries:
                    time.sleep(5 * attempt)
                continue

            if n_files == 0 and (args.full or not seen_gt):
                # Nothing landed -> truncated/broken stream (in GT mode: the GT
                # dir never even appeared). Retry. A full scene always has
                # images, so there is no legitimate empty case for --full.
                what = "members" if args.full else f"{GT_DIR}/ members"
                print(f"    attempt {attempt}: no {what} found", flush=True)
                if attempt < args.retries:
                    time.sleep(5 * attempt)
                continue

            # GT mode only: n_files == 0 with seen_gt is a legitimately
            # unannotated scene (empty processed_data/, e.g. 010_basketball):
            # mark done, do not retry.
            (args.dest_root / scene / sentinel).touch()
            if args.full and seen_gt:
                # A full scene carries the GT too, so let GT-mode runs skip it.
                (args.dest_root / scene / SENTINEL).touch()
            total_bytes += n_bytes
            if n_files == 0:
                empty.append(scene)
                print(f"  EMPTY {scene}: no GT upstream (unannotated scene)",
                      flush=True)
            else:
                ok.append(scene)
                print(f"  OK {scene}: {n_files} files, "
                      f"pulled {n_bytes/1e9:.2f} GB", flush=True)
            break
        else:
            failed.append(scene)
            print(f"  FAIL {scene}", flush=True)

        if quota_hit:
            remaining = [s for s, _ in todo[n:]]
            print(f"\nAborting run: Drive download quota is spent.", flush=True)
            if remaining:
                print(f"not attempted ({len(remaining)}): {', '.join(remaining)}")
            print("Quota resets on Google's schedule (typically ~24 h). "
                  "Re-run the same command; finished scenes are skipped.")
            break

    print(f"\ndone: {len(ok)} ok, {len(empty)} empty(no-GT), "
          f"{len(failed)} failed, {len(skipped)} skipped")
    print(f"total pulled: {total_bytes/1e9:.2f} GB")
    if empty:
        print("empty (no GT upstream): " + ", ".join(empty))
    if failed:
        print("failed: " + ", ".join(failed))
        return 2 if quota_hit else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
