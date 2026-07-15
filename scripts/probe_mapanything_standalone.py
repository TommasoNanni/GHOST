#!/usr/bin/env python
"""Probe: is MapAnything's metric scale wrong, or is our integration wrong?

Runs MapAnything STANDALONE on one frame of an EgoHumans scene (badminton or
tennis) in three modes and compares the implied metric camera baselines to GT:

  A. images only                       (MA predicts poses+depth+metric scale)
  B. images + TRUE intrinsics          (from exo/<cam>/calibration.json)
  C. our pipeline mode                 (images + vggt intrinsics + vggt depth
                                        + vggt poses, scale inputs ignored)

For each mode prints predicted cam-cam baselines (m) vs GT baselines, and for
mode C the implied vggt scale = median(MA depth / vggt depth) vs the optimal
scale from GT cameras.

Usage (GPU node):
  pixi run python scripts/probe_mapanything_standalone.py \
      --scene 031_badminton --activity 06_badminton \
      --gt_root <camera_ready/06_badminton> --frame 400
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.transform import Rotation as SciR

sys.path.insert(0, "/users/tnanni/ghost")
from preprocessing.run_mapanything import DINOV2_MEAN, DINOV2_STD, HF_REPO, PATCH  # noqa

GHOST = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
TEMP  = Path("/iopsstor/scratch/cscs/tnanni/temp_egohumans")
INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"


def load_img(path, H, W):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(img), DINOV2_MEAN, DINOV2_STD).unsqueeze(0)


def gt_centers(gt_scene: Path):
    d = pickle.load(open(gt_scene / "colmap/workplace/colmap_from_aria_transforms.pkl", "rb"))
    Tc2a = np.linalg.inv(np.asarray(d["aria01"]))
    out = {}
    for line in open(gt_scene / "colmap/workplace/images.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 10:
            continue
        cam = p[9].split("/")[0]
        if not cam.startswith("cam") or cam in out:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
        out[cam] = (Tc2a[:3, :3] @ (-R.T @ np.array([tx, ty, tz]))) + Tc2a[:3, 3]
    return out


def baselines(centers: dict):
    cams = sorted(centers)
    return {(a, b): float(np.linalg.norm(centers[a] - centers[b]))
            for i, a in enumerate(cams) for b in cams[i + 1:]}


def centers_from_preds(preds, cams):
    out = {}
    for cam, pred in zip(cams, preds):
        c2w = pred["camera_poses"][0].cpu().double().numpy() if isinstance(pred["camera_poses"], torch.Tensor) else np.asarray(pred["camera_poses"][0])
        out[cam] = c2w[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--activity", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--frame", type=int, default=400)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = GHOST / args.activity / args.scene
    v = np.load(S / "vggt_cameras_centered.npz", allow_pickle=False)
    cams = [n.decode() if isinstance(n, bytes) else n for n in v["camera_names"]]
    ext = v["extrinsics"][0].astype(np.float64)
    intr = v["intrinsics"][0].astype(np.float64)
    dz = np.load(S / "vggt_depth_centered.npz")
    dep = dz["depth"]
    T, K, Hv, Wv = dep.shape
    Hm, Wm = (Hv // PATCH) * PATCH, (Wv // PATCH) * PATCH
    fr = args.frame

    img_root = TEMP / args.activity / INNER / args.activity / args.scene / "exo"
    img_paths, true_K = {}, {}
    for cam in cams:
        fdir = img_root / cam / "images_undistorted" / "frames"
        fs = sorted(fdir.glob("*.jpg"))
        img_paths[cam] = fs[fr]
        cal = json.load(open(img_root / cam / "calibration.json"))
        Kt = np.asarray(cal["K"], np.float64)
        sx, sy = Wm / cal["width"], Hm / cal["height"]
        Kt2 = Kt.copy(); Kt2[0] *= sx; Kt2[1] *= sy
        true_K[cam] = Kt2

    gt = gt_centers(Path(args.gt_root) / args.scene)
    gt_b = baselines({c: gt[c] for c in cams if c in gt})
    print("GT baselines (m):", {f"{a}-{b}": round(d, 2) for (a, b), d in gt_b.items()})

    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained(HF_REPO).to(dev).eval()

    def run(views, tag):
        with torch.no_grad():
            preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                use_amp=True, amp_dtype="bf16", apply_mask=False,
                                mask_edges=False,
                                ignore_depth_scale_inputs=True, ignore_pose_scale_inputs=True)
        ctr = centers_from_preds(preds, cams)
        pb = baselines(ctr)
        rat = [pb[k] / gt_b[k] for k in pb if k in gt_b]
        print(f"[{tag}] pred baselines:", {f"{a}-{b}": round(d, 2) for (a, b), d in pb.items()})
        print(f"[{tag}] pred/GT ratio per pair:", np.round(rat, 2), " median=%.2f" % np.median(rat))
        return preds

    # ── A: images only ─────────────────────────────────────────────────────
    views_a = [{"img": load_img(img_paths[c], Hm, Wm).to(dev),
                "data_norm_type": ["dinov2"]} for c in cams]
    preds_a = run(views_a, "A images-only")
    fx_pred = [float(p["intrinsics"][0][0, 0].cpu()) for p in preds_a]
    msf = [float(p["metric_scaling_factor"][0].cpu()) if "metric_scaling_factor" in p else None
           for p in preds_a]
    print("[A] MA self-estimated fx per cam:", np.round(fx_pred, 1),
          "  true fx=%.1f (calibration.json scaled)" % true_K[cams[0]][0, 0])
    print("[A] metric_scaling_factor:", msf)

    # ── B: images + TRUE intrinsics ────────────────────────────────────────
    views_b = [{"img": load_img(img_paths[c], Hm, Wm).to(dev),
                "data_norm_type": ["dinov2"],
                "intrinsics": torch.from_numpy(true_K[c]).float().unsqueeze(0).to(dev)}
               for c in cams]
    run(views_b, "B images+trueK")

    # ── C: pipeline mode (vggt K + vggt depth + vggt poses, scale ignored) ─
    def sK(Kk):
        Kk = Kk.copy(); Kk[0] *= Wm / Wv; Kk[1] *= Hm / Hv
        return Kk
    def c2w(e):
        m = np.eye(4); m[:3, :3] = e[:3, :3].T; m[:3, 3] = -e[:3, :3].T @ e[:3, 3]
        return m
    ref = np.linalg.inv(c2w(ext[0]))
    views_c = []
    for k, c in enumerate(cams):
        d_m = dep[fr, k].astype(np.float32) / 1000.0
        d_t = F.interpolate(torch.from_numpy(d_m)[None, None], size=(Hm, Wm), mode="nearest")[0, 0]
        views_c.append({"img": load_img(img_paths[c], Hm, Wm).to(dev),
                        "data_norm_type": ["dinov2"],
                        "intrinsics": torch.from_numpy(sK(intr[k])).float().unsqueeze(0).to(dev),
                        "depth_z": d_t.unsqueeze(0).to(dev),
                        "camera_poses": torch.from_numpy(ref @ c2w(ext[k])).float().unsqueeze(0).to(dev),
                        "is_metric_scale": torch.zeros(1, dtype=torch.bool, device=dev)})
    preds_c = run(views_c, "C pipeline-mode")
    scales = []
    for k in range(K):
        pd = preds_c[k]["depth_z"][0].squeeze(-1).cpu().numpy()
        vd = F.interpolate(torch.from_numpy(dep[fr, k].astype(np.float32) / 1000.0)[None, None],
                           size=(Hm, Wm), mode="nearest")[0, 0].numpy()
        m = (pd > 0) & (vd > 0)
        scales.append(float(np.median(pd[m] / vd[m])))
    print("[C] implied vggt scale per cam:", np.round(scales, 2),
          " median=%.2f  (pipeline stored=%.2f)" % (np.median(scales),
          float(np.median(np.load(S / "mapanything_scale_centered.npy")))))


if __name__ == "__main__":
    main()
