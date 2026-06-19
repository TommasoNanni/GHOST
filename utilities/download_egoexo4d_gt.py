"""
Download GT annotations for 182 EgoExo4D takes — only the needed frames.

For each take:
  - Keypoint GT (annotation3D, 17 COCO joints): from ego_pose_pseudo_gt (171 takes)
    or annotations/ego_pose/val/body/annotation (11 takes — manual annotations)
  - Camera calibration: gopro_calibs.csv from take_trajectory

Output:
  <out_dir>/<take_name>/keypoints_gt.json   — {frame_idx: {joint: {x,y,z}}}
  <out_dir>/<take_name>/gopro_calibs.csv    — full calibration file (small, ~1KB)
"""
import json
from pathlib import Path

import boto3


def get_source_maps(s3):
    def load(key):
        return json.loads(s3.get_object(
            Bucket='ego4d-consortium-sharing', Key=key)['Body'].read().decode())

    pose_entries = load('egoexo-public/v2/ego_pose_pseudo_gt/manifest.json')
    uid_to_pose = {e['uid']: e['paths'][0]['source_path'] for e in pose_entries}

    ann_entries = load('egoexo-public/v2/annotations/manifest.json')
    uid_to_ann = {}
    for e in ann_entries:
        for p in e.get('paths', []):
            if 'body/annotation' in p['relative_path']:
                uid_to_ann[e['uid']] = p['source_path']

    traj_entries = load('egoexo-public/v2/take_trajectory/manifest.json')
    uid_to_calib = {}
    for e in traj_entries:
        for p in e['paths']:
            if 'gopro_calibs.csv' in p['relative_path']:
                uid_to_calib[e['uid']] = p['source_path']

    return uid_to_pose, uid_to_ann, uid_to_calib


def fetch_s3(s3, src_path: str) -> bytes:
    _, _, rest = src_path.partition('s3://')
    bucket, _, key = rest.partition('/')
    return s3.get_object(Bucket=bucket, Key=key)['Body'].read()


def download_gt(manifest_path: str, out_dir: str):
    s3 = boto3.client('s3')
    uid_to_pose, uid_to_ann, uid_to_calib = get_source_maps(s3)

    with open(manifest_path) as f:
        manifest = json.load(f)

    out = Path(out_dir)
    skipped = []

    for take_name, info in manifest.items():
        uid = info['uid']
        frame_idx = info['frames'][0]
        take_out = out / take_name
        take_out.mkdir(parents=True, exist_ok=True)

        # --- keypoint GT ---
        kp_path = take_out / 'keypoints_gt.json'
        if not kp_path.exists():
            src = uid_to_pose.get(uid) or uid_to_ann.get(uid)
            raw = json.loads(fetch_s3(s3, src))
            # keys are frame indices as strings
            frame_key = str(frame_idx)
            if frame_key not in raw:
                # find nearest available frame
                available = sorted(int(k) for k in raw.keys())
                nearest = min(available, key=lambda k: abs(k - frame_idx))
                print(f"  [WARN] {take_name}: frame {frame_idx} not in GT, using {nearest}")
                frame_key = str(nearest)
            entry = raw[frame_key]
            if isinstance(entry, list):
                if not entry:
                    available = [k for k in raw.keys() if raw[k] and raw[k] != []]
                    nearest = min(available, key=lambda k: abs(int(k) - frame_idx))
                    print(f"  [WARN] {take_name}: frame {frame_idx} empty, using {nearest}")
                    frame_key = nearest
                    entry = raw[frame_key]
                entry = entry[0]
            kp_path.write_text(json.dumps(
                {frame_key: entry['annotation3D']}, indent=2))

        # --- camera calibration ---
        calib_path = take_out / 'gopro_calibs.csv'
        if not calib_path.exists():
            src = uid_to_calib.get(uid)
            if src:
                calib_path.write_bytes(fetch_s3(s3, src))
            else:
                print(f"  [WARN] {take_name}: no calib found")
                skipped.append(take_name)

        print(f"  {take_name}: done")

    print(f"\nComplete: {len(manifest) - len(skipped)}/182, skipped: {skipped}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    download_gt(args.manifest, args.out_dir)
