#!/usr/bin/env python
"""RICH-train version of probe_mapanything_standalone: validate the
baseline-ratio scale recipe on a normal-FOV dataset.

Compares, for one frame of a RICH train scene:
  - stored pipeline scale   (mapanything_scale_centered.npy, depth-ratio mode)
  - baseline-ratio scale    (MA images-only camera baselines / vggt baselines)
  - GT-optimal scale        (GT camera baselines / vggt baselines)

Usage (GPU node):
  pixi run python scripts/probe_mapanything_rich.py \
      --scene BBQ_001_guitar --img_root <mnt-of-centered_train.sqsh> \
      --rich_root /capstor/scratch/cscs/tnanni/datasets/rich --frame 200
"""
from __future__ import annotations
import argparse, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, "/users/tnanni/ghost")
from preprocessing.run_mapanything import DINOV2_MEAN, DINOV2_STD, HF_REPO, PATCH  # noqa

GHOST = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")


def load_img(path, H, W):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(img), DINOV2_MEAN, DINOV2_STD).unsqueeze(0)


def gt_centers(scene: str, rich_root: Path, cam_names):
    location = scene.split("_")[0]
    calib = rich_root / "scan_calibration" / location / "calibration"
    exts = []
    for xml_path in sorted(calib.glob("*.xml")):
        node = ET.parse(xml_path).getroot().find("CameraMatrix")
        vals = list(map(float, node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    out = {}
    for cn in cam_names:
        gidx = int(re.search(r"\d+", cn).group())
        if gidx < len(exts):
            E = exts[gidx]
            out[cn] = -E[:3, :3].T @ E[:3, 3]      # camera centre, metres
    return out


def baselines(centers):
    cams = sorted(centers)
    return {(a, b): float(np.linalg.norm(centers[a] - centers[b]))
            for i, a in enumerate(cams) for b in cams[i + 1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--img_root", required=True, help="centered train images root (scene/cam_XX/*.jpg)")
    ap.add_argument("--rich_root", default="/capstor/scratch/cscs/tnanni/datasets/rich")
    ap.add_argument("--frame", type=int, default=200)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = GHOST / args.scene
    v = np.load(S / "vggt_cameras_centered.npz", allow_pickle=False)
    cams = [n.decode() if isinstance(n, bytes) else n for n in v["camera_names"]]
    val = v["valid"][args.frame]
    ext = v["extrinsics"][args.frame].astype(np.float64)
    stored = np.load(S / "mapanything_scale_centered.npy")
    dz = np.load(S / "vggt_depth_centered.npz", mmap_mode="r")
    Hv, Wv = dz["depth"].shape[-2:]
    Hm, Wm = (Hv // PATCH) * PATCH, (Wv // PATCH) * PATCH

    # vggt centres (unscaled) for valid cams
    vc = {}
    for k, cn in enumerate(cams):
        if val[k]:
            R, t = ext[k, :, :3], ext[k, :, 3]
            vc[cn] = -R.T @ t
    vb = baselines(vc)

    gt = gt_centers(args.scene, Path(args.rich_root), cams)
    gb = baselines({c: gt[c] for c in vc if c in gt})
    opt = [gb[k] / vb[k] for k in gb if k in vb]
    print("GT baselines (m):", {f"{a}-{b}": round(d, 2) for (a, b), d in gb.items()})
    print("GT-optimal per-pair scale:", np.round(sorted(opt), 2),
          " median=%.3f" % np.median(opt))
    print("stored pipeline scale: median=%.3f  frame[%d]=%.3f"
          % (np.median(stored), args.frame, stored[args.frame]))

    # MA images-only
    img_paths = {}
    for cn in vc:
        fdir = Path(args.img_root) / args.scene / cn
        fs = sorted(fdir.glob("*.jp*g"))
        img_paths[cn] = fs[args.frame]
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained(HF_REPO).to(dev).eval()
    order = sorted(vc)
    views = [{"img": load_img(img_paths[c], Hm, Wm).to(dev),
              "data_norm_type": ["dinov2"]} for c in order]
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                            use_amp=True, amp_dtype="bf16", apply_mask=False, mask_edges=False)
    mc = {c: p["camera_poses"][0].cpu().double().numpy()[:3, 3] for c, p in zip(order, preds)}
    mb = baselines(mc)
    rat_gt = [mb[k] / gb[k] for k in mb if k in gb]
    print("[MA images-only] baseline/GT per pair:", np.round(sorted(rat_gt), 2),
          " median=%.2f" % np.median(rat_gt))
    rat = [mb[k] / vb[k] for k in mb if k in vb]
    print("[MA images-only] BASELINE-RATIO scale per pair:", np.round(sorted(rat), 2))
    print("  -> median=%.3f   (GT-optimal %.3f, stored %.3f)"
          % (np.median(rat), np.median(opt), stored[args.frame]))


if __name__ == "__main__":
    main()
