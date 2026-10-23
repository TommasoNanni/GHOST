"""
Pipeline-figure panels for one EgoHumans frame.

Renders the stages of GHOST as separate images for a method figure:

  raw_<cam>.png        the input frame, untouched
  perview_<cam>.png    that camera's OWN SAM 3D Body estimate for one person,
                       seen from that camera, one colour per view
  aggregated.png       the geodesic-median pose for the same person, in the
                       colour that person carries in the scene renders
  scene_empty.png      the reconstructed scene from one camera, no bodies
  scene_people.png     the same scene with the placed bodies, no GT overlay

The per-view panels show pose only: each body is re-centred at a fixed distance
in front of a synthetic pinhole, keeping the orientation that camera estimated
but dropping SAM 3D Body's own translation, which lives at that model's assumed
focal length and is not comparable across views.  Aggregation happens on
rotations alone, so this is exactly the quantity the median acts on.

Helpers come from visualize.paper_qualitative_egohumans — same figure family,
same conventions; nothing here is imported by the evaluation code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import visualize.paper_qualitative_egohumans as Q

# 00B0F0, D97706, FFC000 — one per input view, in RGB.
VIEW_RGB = [(0, 176, 240), (217, 119, 6), (255, 192, 0)]

# Synthetic pinhole used for every body panel, so the three per-view estimates
# and the aggregate are framed identically and can be compared side by side.
PANEL_W, PANEL_H, PANEL_F, PANEL_Z = 700, 900, 900.0, 3.0


def _panel_K():
    return np.array([[PANEL_F, 0, PANEL_W / 2], [0, PANEL_F, PANEL_H / 2], [0, 0, 1]])


def _fk_axis_angle(betas, global_orient, body_pose, smplx_model):
    """SMPL-X forward from axis-angle params (what body_data_clean stores)."""
    import torch, smplx as smplx_lib
    p = Path(smplx_model)
    model = smplx_lib.create(model_type="smplx", model_path=str(p),
                             ext=p.suffix.lstrip("."), gender="neutral",
                             use_pca=False, num_betas=10, flat_hand_mean=True,
                             batch_size=1)
    model.eval()
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32).reshape(1, -1))
    with torch.no_grad():
        out = model(betas=t(betas[:10]), global_orient=t(global_orient),
                    body_pose=t(body_pose), return_verts=True)
    return out.vertices[0].numpy(), model.faces.copy()


def _draw_body(verts, faces, rgb, R=np.eye(3)):
    """Shade a mesh onto a white panel, re-centred in front of the synthetic cam."""
    v = np.asarray(verts, np.float64) @ R.T
    v = v - v.mean(0)
    v[:, 2] += PANEL_Z
    K = _panel_K()
    # Auto-fit: a fencing lunge is far wider than it is tall, so a fixed depth
    # either crops it or leaves it tiny.  Project once, then push the body back
    # until its bounding box fills `fill` of the panel and re-centre on that box.
    uv0, _ = Q.project(v, np.eye(3), np.zeros(3), K)
    ok = np.isfinite(uv0).all(1)
    lo, hi = uv0[ok].min(0), uv0[ok].max(0)
    fill = 0.82
    scale = min(fill * PANEL_W / max(hi[0] - lo[0], 1e-6),
                fill * PANEL_H / max(hi[1] - lo[1], 1e-6))
    v[:, 2] = (v[:, 2] - PANEL_Z) + PANEL_Z / scale
    uv1, _ = Q.project(v, np.eye(3), np.zeros(3), K)
    ok = np.isfinite(uv1).all(1)
    ctr = (uv1[ok].min(0) + uv1[ok].max(0)) / 2.0
    zc_mean = float(np.nanmean(v[:, 2]))
    v[:, 0] -= (ctr[0] - PANEL_W / 2) * zc_mean / PANEL_F
    v[:, 1] -= (ctr[1] - PANEL_H / 2) * zc_mean / PANEL_F
    img = np.full((PANEL_H, PANEL_W, 3), 255, np.uint8)
    uv, z = Q.project(v, np.eye(3), np.zeros(3), K)
    keep = (np.isfinite(uv).all(1) & (z > 1e-6))[faces].all(1)
    if not keep.any():
        return img
    fc = Q.face_shading(v, faces[keep], np.eye(3), rgb)
    zc = z[faces[keep]].max(1)
    poly = Q.poly_fx(uv[faces[keep]])
    Q.paint(img, [("t", float(zc[i]), poly[i], fc[i]) for i in range(len(zc))])
    return img



def _draw_frustums(img, scene, frame, render_cam, cams, R_cam, t_cam, K, W, H,
                   depth=1.2, thickness=3, border=3):
    """Draw each input camera's frustum, coloured to match its per-view panel.

    Apex at the camera centre, four edges out to the image corners back-projected
    at `depth` metres, plus the rectangle joining them.  Drawn after the cloud, so
    the wireframe stays legible instead of being eaten by foreground points.  The
    rendering camera is skipped: its own frustum is degenerate from inside it.
    """
    poses = Q.load_colmap_poses(scene)
    T_c2a = Q.colmap_to_aria(scene)
    for cam, rgb in zip(cams, VIEW_RGB):
        if cam == render_cam:
            continue
        Rk, tk = Q.gt_camera_aria(poses, T_c2a, cam, frame)
        Kk, Wk, Hk = Q.undist_K(scene, cam)
        C = -Rk.T @ tk                                  # apex, aria world
        corners = []
        for u, v in [(0, 0), (Wk, 0), (Wk, Hk), (0, Hk)]:
            d = np.linalg.inv(Kk) @ np.array([u, v, 1.0])
            d = Rk.T @ (d / np.linalg.norm(d))           # ray direction in world
            corners.append(C + d * depth)
        pts = np.stack([C] + corners)
        uv, z = Q.project(pts, R_cam, t_cam, K)
        ok = np.isfinite(uv).all(1) & (z > 1e-6)
        P2 = np.round(uv).astype(np.int32)
        col = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        # Black underlay first, colour on top: a thin wireframe over a busy point
        # cloud reads as noise without it, and the outline keeps the hue legible
        # at the small sizes these frustums are drawn at.
        for width, c in ((thickness + border, (0, 0, 0)), (thickness, col)):
            for a, b in edges:
                if ok[a] and ok[b]:
                    cv2.line(img, tuple(P2[a]), tuple(P2[b]), c, width,
                             lineType=cv2.LINE_AA)
    return img

def _scene(scene, frame, cam, pred, args, with_people, frustums=False):
    """Point cloud from every camera, optionally with the placed bodies. No GT."""
    d = np.load(pred, allow_pickle=False)
    fmin, pids = Q.ghost_frame_axis(scene)
    ti = frame - fmin
    names = [str(n) for n in d["camera_names"]]
    poses = Q.load_colmap_poses(scene); T_c2a = Q.colmap_to_aria(scene)
    gc = np.stack([Q.gt_camera_centre_aria(poses, T_c2a, c, frame) for c in names])
    R_a, t_a = Q.se3_align(Q.ghost_cam_centres(d["camera"][ti]), gc)

    pv, pf = Q.smplx_vertices(d["pose"][ti:ti + 1], d["shape"],
                              np.nan_to_num(d["body_transl_world"][ti:ti + 1]),
                              args.smplx_model)
    finite = (np.isfinite(d["body_transl_world"][ti]).all(1)
              & np.isfinite(d["pose"][ti]).all((1, 2)))
    pv_a = pv[0] @ R_a.T + t_a

    R_cam, t_cam = Q.gt_camera_aria(poses, T_c2a, cam, frame)
    K, W, H = Q.undist_K(scene, cam)
    img = np.full((H, W, 3), 255, np.uint8)

    union = np.zeros((H, W), np.uint8)
    if with_people:
        for p in range(pv_a.shape[0]):
            if finite[p]:
                union |= Q.mesh_mask(pv_a[p], pf, R_cam, t_cam, K, H, W, close=False)

    ctx = Q.depth_context(scene)
    prims, pts_all, cols_all = [], [], []
    for cname in ctx["names"]:
        k = ctx["names"].index(cname)
        bgr = cv2.imread(str(Q._frames_dir(scene, cname) / f"{frame:05d}.jpg"))
        got = Q.depth_cloud(ctx, ti, k, bgr, args.depth_stride, 0.0)
        if got is None:
            continue
        pts_all.append(got[0] @ R_a.T + t_a); cols_all.append(got[1])
    pts, cols = np.concatenate(pts_all), np.concatenate(cols_all)
    uv, z = Q.project(pts, R_cam, t_cam, K)
    vis = (np.isfinite(uv).all(1) & (z > 1e-6) & (uv[:, 0] > -10)
           & (uv[:, 0] < W + 10) & (uv[:, 1] > -10) & (uv[:, 1] < H + 10))
    if with_people:
        iu = np.clip(np.round(np.nan_to_num(uv[:, 0])).astype(np.int32), 0, W - 1)
        iv = np.clip(np.round(np.nan_to_num(uv[:, 1])).astype(np.int32), 0, H - 1)
        vis &= union[iv, iu] == 0
    uv, z, cols = uv[vis], z[vis], cols[vis]
    uvi = np.round(uv).astype(np.int32)
    for i in range(len(z)):
        prims.append(("p", float(z[i]),
                      ((int(uvi[i, 0]), int(uvi[i, 1])), args.ball_radius), cols[i]))

    n_bodies = 0
    if with_people:
        for p in range(pv_a.shape[0]):
            if not finite[p]:
                continue
            v = pv_a[p]
            uvb, zb = Q.project(v, R_cam, t_cam, K)
            keep = (np.isfinite(uvb).all(1) & (zb > 1e-6))[pf].all(1)
            if not keep.any():
                continue
            fc = Q.face_shading(v, pf[keep], R_cam, Q._person_rgb(pids[p]))
            zc = zb[pf[keep]].max(1)
            poly = Q.poly_fx(uvb[pf[keep]])
            for i in range(len(zc)):
                prims.append(("t", float(zc[i]), poly[i], fc[i]))
            n_bodies += 1
    print(f"    {len(z)} balls, {n_bodies} bodies")
    Q.paint(img, prims)
    if frustums:
        _draw_frustums(img, scene, frame, cam, args.cams, R_cam, t_cam, K, W, H,
                       args.frustum_depth)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="03_fencing/006_fencing")
    ap.add_argument("--frame", type=int, default=333)
    ap.add_argument("--cams", nargs=3, default=["cam04", "cam05", "cam10"])
    ap.add_argument("--render-cam", default="cam04")
    ap.add_argument("--person", type=int, default=1)
    ap.add_argument("--pred", type=Path,
                    default=Path("fusion_outputs/006_fencing_f333w20.npz"))
    ap.add_argument("--out", type=Path, default=Path("figures/pipeline_draw"))
    ap.add_argument("--smplx-model", type=Path,
                    default=Path("body_models/SMPLX_NEUTRAL.pkl"))
    ap.add_argument("--depth-stride", type=int, default=1)
    ap.add_argument("--ball-radius", type=int, default=1)
    ap.add_argument("--frustum-depth", type=float, default=1.2,
                    help="metres from each camera centre to its drawn image plane")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. raw input frames
    for c in args.cams:
        src = Q._frames_dir(args.scene, c) / f"{args.frame:05d}.jpg"
        img = cv2.imread(str(src))
        cv2.imwrite(str(args.out / f"raw_{c}.png"), img)
        print(f"  raw_{c}.png  {img.shape[1]}x{img.shape[0]}")

    # 2. per-view estimates, one colour per view
    for c, rgb in zip(args.cams, VIEW_RGB):
        f = (Q.GHOST_ROOT / args.scene / c / "body_data_clean"
             / f"person_{args.person}.npz")
        d = np.load(f, allow_pickle=False)
        i = int(np.where(d["frame_indices"].astype(int) == args.frame)[0][0])
        verts, faces = _fk_axis_angle(d["smplx_betas"][i],
                                      d["smplx_global_orient"][i].reshape(3),
                                      d["smplx_body_pose"][i].reshape(63),
                                      args.smplx_model)
        cv2.imwrite(str(args.out / f"perview_{c}.png"), _draw_body(verts, faces, rgb))
        print(f"  perview_{c}.png")

    # 3. aggregated pose, in the person's scene colour, seen from the render cam
    dp = np.load(args.pred, allow_pickle=False)
    fmin, pids = Q.ghost_frame_axis(args.scene)
    p = pids.index(args.person)
    pose = dp["pose"][args.frame - fmin:args.frame - fmin + 1, p:p + 1]
    pv, pf = Q.smplx_vertices(pose, dp["shape"][p:p + 1],
                              np.zeros((1, 1, 3), np.float32), args.smplx_model)
    # The fused root orient is in the GHOST world, while the camera rotation is
    # aria-world -> camera.  Composing them directly skips the ghost->aria SE(3)
    # and tumbles the body; the full chain is R_cam @ R_a.
    poses = Q.load_colmap_poses(args.scene)
    T_c2a = Q.colmap_to_aria(args.scene)
    names = [str(n) for n in dp["camera_names"]]
    gc = np.stack([Q.gt_camera_centre_aria(poses, T_c2a, c, args.frame) for c in names])
    R_a, _ = Q.se3_align(Q.ghost_cam_centres(dp["camera"][args.frame - fmin]), gc)
    R_cam, _ = Q.gt_camera_aria(poses, T_c2a, args.render_cam, args.frame)
    cv2.imwrite(str(args.out / "aggregated.png"),
                _draw_body(pv[0, 0], pf, Q._person_rgb(args.person), R_cam @ R_a))
    print("  aggregated.png")

    # 4/5. scene without and with the placed bodies
    for name, wp, fr in [("scene_empty", False, True), ("scene_people", True, True)]:
        img = _scene(args.scene, args.frame, args.render_cam, args.pred, args, wp, fr)
        cv2.imwrite(str(args.out / f"{name}.png"), img)
        print(f"  {name}.png")


if __name__ == "__main__":
    main()
