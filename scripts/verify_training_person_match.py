"""Does the person we TRAIN on actually coincide with the GT person?

Every other checker asks a proxy question (is this id the same human across views?
does a projected root land near a bbox centre?). This asks the end-to-end one:
``RICHFusionDatapoint`` performs ReID -> visibility filtering -> ``_match_persons_to_gt``,
and whatever ghost pid survives that chain is what the loss is computed against. If that
pid is the wrong human, the model trains on mismatched supervision and nothing upstream
would show it.

So: build the datapoint exactly as training does, then for every GT person compare the
GT root translation against the matched ghost person's translation, per camera, in the
same world frame. Small distance = we train on the right human.

Crucially this is **per GT person**, so it is correct on multi-person scenes —
unlike ``verify_rich_subject_identity.py``, whose ``subject_global_id()`` assumes a
single subject and therefore mis-flags 2-person scenes (tossball,
greetingchattingeating1) as broken.

  dist < 0.35 m   correct person
  0.35 - 0.7 m    suspicious (could be a tracking offset or a neighbouring person)
  > 0.7 m         WRONG person -- training on mismatched supervision

    pixi run python scripts/verify_training_person_match.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \\
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich [--scenes A,B]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def scene_report(dp, name: str, bad: float, warn: float) -> list[tuple]:
    """Per GT person: distance between GT transl and the matched ghost person."""
    rows = []
    gt = dp._gt[0] if dp._gt else {}
    for pid, d in gt.items():
        gt_t = np.asarray(d.get("transl"), dtype=np.float64)      # (T, 3) world
        if gt_t.ndim != 2:
            continue
        # GT rows are NOT indexed by frame number: _gt_frame_lut[pid] maps
        # frame -> row (GT often starts at frame 5 and is shorter than the clip).
        lut = (dp._gt_frame_lut or {}).get(pid, {})
        if not lut:
            rows.append((name, pid, None, None, []))
            continue
        # Predictions live in each camera's OWN frame; GT is in cam-0 world space.
        # Transform cam-i -> cam-0 exactly as _match_persons_to_gt does, otherwise
        # every camera except the reference looks metres away from GT.
        ext0 = dp._cameras[0].get("extrinsics") if dp._cameras else None
        if ext0 is None:
            rows.append((name, pid, None, None, []))
            continue
        R0 = np.asarray(ext0)[:3, :3].astype(np.float64)
        t0 = np.asarray(ext0)[:3, 3].astype(np.float64)

        per_cam = []
        for ci, cam in enumerate(dp._raw):
            pd = cam.get(pid)
            if pd is None:
                continue
            tr = pd.get("smplx_transl")
            fi = pd.get("frame_indices")
            if tr is None or fi is None:
                continue
            ext_i = dp._cameras[ci].get("extrinsics") if ci < len(dp._cameras) else None
            if ext_i is None:
                continue
            R_i = np.asarray(ext_i)[:3, :3].astype(np.float64)
            t_i = np.asarray(ext_i)[:3, 3].astype(np.float64)
            R_i2w = R0 @ R_i.T
            d_i = t0 - R_i2w @ t_i
            tr = np.asarray(tr, dtype=np.float64) @ R_i2w.T + d_i
            fi = np.asarray(fi).astype(int)
            rows_idx = np.array([lut.get(int(f), -1) for f in fi])
            ok = (rows_idx >= 0) & (rows_idx < len(gt_t))
            if not ok.any():
                continue
            d3 = np.linalg.norm(tr[ok] - gt_t[rows_idx[ok]], axis=-1)
            d3 = d3[np.isfinite(d3)]
            if d3.size:
                per_cam.append((dp._cam_dirs[ci].name, float(np.median(d3))))
        if not per_cam:
            rows.append((name, pid, None, None, []))
            continue
        vals = [v for _, v in per_cam]
        worst = max(per_cam, key=lambda kv: kv[1])
        rows.append((name, pid, float(np.median(vals)), worst,
                     [f"{c}:{v:.2f}" for c, v in per_cam if v > bad]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--body_split", default="train_body")
    ap.add_argument("--bad", type=float, default=0.7, help="metres: wrong person")
    ap.add_argument("--warn", type=float, default=0.35, help="metres: suspicious")
    ap.add_argument("--scenes", default="")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    from data.fusion_dataset import RICHFusionDatapoint

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = [d for d in sorted(args.ghost_root.iterdir())
              if d.is_dir() and (d / "cross_view_reid.json").exists()]
    if wanted:
        scenes = [d for d in scenes if d.name in wanted]

    print(f"{'scene':<44}{'gtpid':>6}{'median':>9}{'worst':>9}  bad cameras")
    n_person = n_bad = n_warn = 0
    bad_list: list[str] = []
    for sd in scenes:
        try:
            dp = RICHFusionDatapoint(scene_dir=sd, rich_data_root=str(args.rich_root),
                                     rich_gt_dir=str(args.rich_root),
                                     body_split=args.body_split)
        except Exception as e:
            print(f"{sd.name:<44}   ERROR {type(e).__name__}: {e}")
            continue
        if not dp.has_gt or dp.num_frames == 0:
            print(f"{sd.name:<44}   (no gt / empty)")
            continue
        for name, pid, med, worst, bads in scene_report(dp, sd.name, args.bad, args.warn):
            n_person += 1
            if med is None:
                print(f"{name:<44}{pid:>6}   NO MATCHED GHOST DATA")
                n_bad += 1
                bad_list.append(f"{name} pid{pid} (no data)")
                continue
            tag = ""
            if med > args.bad:
                tag = "  <-- WRONG PERSON"
                n_bad += 1
                bad_list.append(f"{name} pid{pid} median={med:.2f}m")
            elif med > args.warn:
                tag = "  <-- suspicious"
                n_warn += 1
            print(f"{name:<44}{pid:>6}{med:9.2f}{worst[1]:9.2f}  "
                  f"{','.join(bads) if bads else ''}{tag}")

    print()
    print(f"  GT persons checked: {n_person}   wrong (>{args.bad}m): {n_bad}   "
          f"suspicious (>{args.warn}m): {n_warn}")
    for b in bad_list:
        print(f"    {b}")


if __name__ == "__main__":
    main()
