"""
TransReID cross-view appearance extractor.

View-invariant person ReID features for cross-view matching (Phase 2 of cross_view_v5).
DINOv3 backbone features are generic and barely separate the same person across wide
camera-angle changes (~0.02 cosine gap on BBQ_001_juggle bystanders); TransReID, trained
for cross-camera person ReID, gives ~0.32 gap on the same crops.

Net: TransReID (He et al., "TransReID: Transformer-based Object Re-Identification", ICCV 2021),
ViT-Base, MSMT17 checkpoint. We use the plain global branch (cls token after the full ViT,
768-d). SIE (camera side-info embedding) is disabled — its lookup table is indexed by the
training dataset's cameras, which don't correspond to ours, and on/off made no difference
in testing. Pre-processing: 256x128, RGB, mean/std = 0.5 (TransReID convention).

Vendored backbone: third_party/transreid/vit_pytorch.py.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from third_party.transreid.vit_pytorch import vit_base_patch16_224_TransReID

_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_IMG_HW = (256, 128)          # (H, W)
_CAMERA_NUM = 15              # MSMT17 training cameras (must match ckpt sie_embed shape)


class TransReIDExtractor:
    """Lazy-loaded TransReID ViT-Base feed-forward feature extractor.

    person_feature() reads a person's per-frame bboxes from json_data, crops the centered
    frames, batch-forwards through the ViT, and returns one L2-normalized 768-d descriptor
    (confidence-free mean over sampled frames).
    """

    def __init__(self, ckpt_path: str | Path, device: str | None = None,
                 batch_size: int = 32, step: int = 15, max_frames: int = 30):
        self.ckpt_path  = Path(ckpt_path)
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.step       = step
        self.max_frames = max_frames
        self._model = None  # built on first use

    # ── Model ───────────────────────────────────────────────────────────────
    def _build(self) -> None:
        if self._model is not None:
            return
        vit = vit_base_patch16_224_TransReID(
            img_size=_IMG_HW, stride_size=16, camera=_CAMERA_NUM, view=0,
            sie_xishu=3.0, local_feature=False, drop_path_rate=0.0,
        )
        sd = torch.load(str(self.ckpt_path), map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        base_sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
        res = vit.load_state_dict(base_sd, strict=False)
        if res.missing_keys:
            logging.info(f"[TransReID] {len(res.missing_keys)} missing key(s) (expected: classifier/bottleneck not in base)")
        vit.sie_xishu = 0.0          # disable SIE (no usable cross-domain cam index)
        vit.eval().to(self.device)
        self._model = vit
        logging.info(f"[TransReID] loaded {self.ckpt_path.name} on {self.device}")

    @torch.no_grad()
    def embed_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """[N,768] L2-normalized features for a list of BGR HxWx3 crops."""
        import cv2
        self._build()
        feats = []
        for i in range(0, len(crops_bgr), self.batch_size):
            chunk = crops_bgr[i:i + self.batch_size]
            batch = []
            for c in chunk:
                img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (_IMG_HW[1], _IMG_HW[0]))  # (W,H)
                batch.append(torch.from_numpy(img).float().permute(2, 0, 1) / 255.0)
            t = torch.stack(batch)
            t = ((t - _MEAN) / _STD).to(self.device)
            g = self._model(t, cam_label=0, view_label=0)        # [B,768] cls token
            g = torch.nn.functional.normalize(g, dim=1)
            feats.append(g.cpu().numpy())
        return np.concatenate(feats, 0) if feats else np.empty((0, 768), np.float32)

    # ── Person-level descriptor ─────────────────────────────────────────────
    @staticmethod
    def _bbox_overlap(b1: tuple, b2: tuple) -> float:
        """Intersection area / b1 area. b = (x1,y1,x2,y2)."""
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area  = (b1[2] - b1[0]) * (b1[3] - b1[1])
        return inter / area if area > 0 else 0.0

    def person_feature(self, frames_dir: Path, json_dir: Path,
                        cam_suffix: str, pid: int,
                        overlap_thr: float = 1.1,
                        min_good_frames: int = 0,
                        mask_npz: Path | None = None) -> np.ndarray | None:
        """Mean L2-normalized descriptor for one person, or None if insufficient clean frames.

        If mask_npz is provided (mask_data.npz path), each crop is masked: pixels not
        belonging to this person (mask != pid) are set to black before passing to TransReID.

        frames_dir : centered frames, named '<fidx>_<cam_suffix>.jpg'
        json_dir   : per-frame bbox json, named 'mask_<fidx>_<cam_suffix>.json'
        mask_npz   : mask_data.npz, keys 'mask_<5digit-fidx>_<cam_suffix>', uint16 pid map
        """
        import cv2
        frames_dir, json_dir = Path(frames_dir), Path(json_dir)
        _masks = np.load(str(mask_npz)) if mask_npz is not None else None
        all_jfiles = sorted(json_dir.glob(f"mask_*_{cam_suffix}.json"))
        tag = f"[TransReID] {cam_suffix}:P{pid}"
        logging.info(f"{tag} — {len(all_jfiles)} total frames")

        # ── Pass 1: collect good frame indices ──────────────────────────────
        good_jfiles = []
        n_missing = n_occluded = 0
        for jf in all_jfiles:
            try:
                d = json.loads(jf.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            labels = d.get("labels", {})
            lab = labels.get(str(pid))
            if lab is None:
                n_missing += 1
                continue
            b = (int(lab["x1"]), int(lab["y1"]), int(lab["x2"]), int(lab["y2"]))
            max_ovlp = 0.0
            for k, v in labels.items():
                if k == str(pid):
                    continue
                ov = self._bbox_overlap(b, (int(v["x1"]), int(v["y1"]),
                                            int(v["x2"]), int(v["y2"])))
                if ov > max_ovlp:
                    max_ovlp = ov
            if max_ovlp > overlap_thr:
                n_occluded += 1
                logging.debug(f"{tag} frame {jf.stem}: occluded by other bbox (overlap={max_ovlp:.2f})")
            else:
                good_jfiles.append(jf)

        logging.info(f"{tag} — {len(good_jfiles)} good / {len(all_jfiles)} total "
                     f"(missing={n_missing}, occluded={n_occluded}, thr={overlap_thr})")

        if len(good_jfiles) < min_good_frames:
            logging.info(f"{tag} — SKIP appearance: only {len(good_jfiles)} good frames "
                         f"< min={min_good_frames} → geometry will resolve")
            return None

        # ── Pass 2: sample up to max_frames and extract crops ───────────────
        step = max(1, len(good_jfiles) // self.max_frames)
        sampled = good_jfiles[::step][:self.max_frames]
        logging.info(f"{tag} — extracting {len(sampled)} crops (step={step})")
        crops = []
        for jf in sampled:
            try:
                d = json.loads(jf.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            lab = d.get("labels", {}).get(str(pid))
            if lab is None:
                continue
            fidx = jf.stem.split("_")[1]
            fp = frames_dir / f"{fidx}_{cam_suffix}.jpg"
            im = cv2.imread(str(fp))
            if im is None:
                logging.warning(f"{tag} frame {fidx}: image not found at {fp}")
                continue
            x1, y1, x2, y2 = int(lab["x1"]), int(lab["y1"]), int(lab["x2"]), int(lab["y2"])
            c = im[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)].copy()
            if c.size == 0:
                logging.warning(f"{tag} frame {fidx}: empty crop bbox=({x1},{y1},{x2},{y2})")
                continue
            if _masks is not None:
                mkey = f"mask_{fidx}_{cam_suffix}"
                if mkey in _masks:
                    mf = _masks[mkey]
                    mc = mf[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                    c[mc != pid] = 0
            crops.append(c)
        if not crops:
            logging.warning(f"{tag} — no valid crops after sampling, returning None")
            return None
        embs = self.embed_crops(crops)
        m = embs.mean(0)
        n = float(np.linalg.norm(m))
        logging.info(f"{tag} — descriptor ready from {len(crops)} crops")
        return (m / n).astype(np.float32) if n > 0 else m.astype(np.float32)
