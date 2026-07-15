"""
Extract specific frames from EgoExo4D takes directly from S3 using byte-range requests.

Downloads ~1-5 MB per video (moov atom + target GOP) instead of the full file.

Usage:
  pixi run python utilities/extract_egoexo4d_frames.py \
      --manifest /capstor/scratch/cscs/tnanni/datasets/egoexo4d/job_manifest.json \
      --out_dir  /capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames \
      [--take_idx 0]   # SLURM array index; omit to process all takes sequentially

Output: <out_dir>/<take_name>/<cam_id>/frame_<XXXXXX>.jpg
"""
import argparse
import json
import time
from pathlib import Path

import av
import boto3
import cv2


class S3SeekableFile:
    """File-like object backed by S3 byte-range GETs. Allows av.open() to seek."""

    def __init__(self, s3_client, bucket: str, key: str, size: int):
        self._s3 = s3_client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0
        self.bytes_downloaded = 0

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""
        end = self._pos + n - 1
        resp = self._s3.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={self._pos}-{end}"
        )
        data = resp["Body"].read()
        self.bytes_downloaded += len(data)
        self._pos += len(data)
        return data

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def extract_frame(s3, src_path: str, size: int, frame_idx: int) -> bytes | None:
    """Download ~1-5 MB from S3 and return the decoded frame as JPEG bytes."""
    # src_path: s3://<bucket>/<key>
    _, _, bucket_and_key = src_path.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")

    f = S3SeekableFile(s3, bucket, key, size)
    try:
        container = av.open(f)
        video = container.streams.video[0]
        fps = float(video.average_rate)
        # Seek to the nearest keyframe at/before the target, then decode FORWARD to
        # the exact frame. av.seek() lands on a keyframe; taking the first decoded
        # frame returns that keyframe (up to a GOP early), NOT frame_idx — a silent
        # temporal error that puts a moving subject up to ~0.5 m from ground truth
        # (frame_aligned_videos share keyframes, so every camera pulls the same
        # wrong instant). Decode until the frame whose pts reaches the target,
        # keeping the last frame <= target as an EOF fallback.
        target_pts = int(round(frame_idx / fps / video.time_base))
        container.seek(target_pts, stream=video)
        chosen = None
        for frame in container.decode(video):
            if frame.pts is None:
                continue
            chosen = frame
            if frame.pts >= target_pts:
                break
        img = chosen.to_ndarray(format="bgr24") if chosen is not None else None
        container.close()
    except Exception as e:
        print(f"    [ERR] {src_path}: {e}")
        return None, f.bytes_downloaded

    if img is None:
        return None, f.bytes_downloaded

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (buf.tobytes() if ok else None), f.bytes_downloaded


def process_take(take_name: str, info: dict, out_dir: Path, s3) -> None:
    frame_idx = info["frame"]
    t0 = time.time()
    total_dl = 0
    saved = 0
    for cam in info["cameras"]:
        cam_id = cam["cam_id"]
        out_path = out_dir / take_name / cam_id / f"frame_{frame_idx:06d}.jpg"
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jpg_bytes, dl = extract_frame(s3, cam["src"], cam["size"], frame_idx)
        total_dl += dl
        if jpg_bytes:
            out_path.write_bytes(jpg_bytes)
            saved += 1
        else:
            print(f"    [WARN] {take_name}/{cam_id}: frame extraction failed")
    print(
        f"  {take_name}: {saved}/{len(info['cameras'])} frames saved, "
        f"{total_dl/1e6:.1f} MB downloaded in {time.time()-t0:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--take_idx", type=int, default=None,
                    help="0-based index into the manifest (for SLURM array jobs)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    out_dir = Path(args.out_dir)
    s3 = boto3.client("s3")

    takes = list(manifest.items())
    if args.take_idx is not None:
        takes = [takes[args.take_idx]]

    for take_name, info in takes:
        process_take(take_name, info, out_dir, s3)


if __name__ == "__main__":
    main()
