import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import json
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from preprocessing.parameters_extraction import BodyParameterEstimator

FEAT_DIM = 64   # arbitrary small backbone output channels


def _random_unit(dim: int, seed: int | None = None) -> np.ndarray:
    """Return a random L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_frame_dir(tmpdir: Path, frame_indices: list[int]) -> None:
    """Write tiny blank JPEG files for each requested frame index."""
    frame_dir = tmpdir / "frames"
    frame_dir.mkdir(exist_ok=True)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for fi in frame_indices:
        cv2.imwrite(str(frame_dir / f"{fi:06d}.jpg"), blank)


def _write_json(json_dir: Path, frame_idx: int, labels: dict) -> None:
    """Write a per-frame JSON metadata file (same format as PersonSegmenter).

    labels: {person_id: (x1, y1, x2, y2)}
    """
    json_dir.mkdir(exist_ok=True)
    payload = {
        "labels": {
            str(pid): {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
            for pid, box in labels.items()
        }
    }
    with open(json_dir / f"mask_{frame_idx:06d}.json", "w") as f:
        json.dump(payload, f)


def _fake_body_params() -> dict:
    """Minimal body-parameter dict (enough for _save_body_data_static)."""
    return {
        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
        "pred_cam_t":        np.zeros(3,        dtype=np.float32),
        "shape_params":      np.zeros(10,       dtype=np.float32),
        "body_pose_params":  np.zeros(63,       dtype=np.float32),
        "focal_length":      np.array([500.0],  dtype=np.float32),
    }


class _MockBackbone(nn.Module):
    """Thin nn.Module whose forward returns a pre-set tensor.

    Being a proper nn.Module means register_forward_hook works exactly
    as it does on the real backbone.
    """

    def __init__(self):
        super().__init__()
        self.next_output: torch.Tensor | None = None

    def forward(self, *_, **__):
        return self.next_output


class _MockEstimator:
    """Mimics SAM3DBodyEstimator enough for _process_video_core.

    Parameters
    ----------
    frame_feats : dict[int, torch.Tensor]
        Call index (0, 1, ...) → backbone output of shape (N_persons, C, 1, 1).
        The hook global-avg-pools this to (N_persons, C) for comparison.
    frame_outputs : dict[int, list[dict]]
        Call index → list of body-param dicts (one per person in that frame).
    """

    def __init__(
        self,
        frame_feats: dict[int, torch.Tensor],
        frame_outputs: dict[int, list[dict]],
    ):
        self._backbone = _MockBackbone()
        self.model = type("_M", (), {"backbone": self._backbone})()
        self._frame_feats = frame_feats
        self._frame_outputs = frame_outputs
        self._call_idx = 0

    def process_one_image(self, _frame_rgb, bboxes=None):
        del bboxes   # not needed in mock; real impl uses it to build the batch
        ci = self._call_idx
        self._call_idx += 1
        # Load the right feature and trigger the backbone so the hook fires.
        self._backbone.next_output = self._frame_feats[ci]
        self._backbone(torch.zeros(1))          # hook fires here
        return self._frame_outputs[ci]


def _run_core(tmpdir: Path, estimator: _MockEstimator) -> Path:
    """Run _process_video_core and return the body_data dir."""
    BodyParameterEstimator._process_video_core(
        estimator=estimator,
        video_id="unit_test",
        video_dir=str(tmpdir),
        sam3d_step=1,
        bbox_padding=0.0,
        param_keys=BodyParameterEstimator._PARAM_KEYS,
    )
    return tmpdir / "body_data"


# ===========================================================================
# Unit test 1: same person re-appears under a new SAM2 ID → must be merged
# ===========================================================================

def test_reid_merge():
    """SAM2 assigns a new ID to a person who re-enters the frame.

    Two frames, two different person_ids, but visually nearly identical
    features.  _process_video_core must merge them into one track.
    """
    print("\n=== Unit test: re-ID merge (same person, different SAM2 IDs) ===")

    feat_A = _random_unit(FEAT_DIM, seed=0)
    # feat_B is almost identical to feat_A → cosine sim ≈ 1.0 >> threshold.
    feat_B = feat_A + 0.001 * _random_unit(FEAT_DIM, seed=1)
    feat_B /= np.linalg.norm(feat_B)
    sim = float(np.dot(feat_A, feat_B))

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        json_dir = tmpdir / "json_data"
        _make_frame_dir(tmpdir, [0, 1])
        _write_json(json_dir, 0, {1: (100, 100, 200, 300)})   # person_id=1
        _write_json(json_dir, 1, {2: (105, 102, 205, 302)})   # person_id=2 (same person)

        estimator = _MockEstimator(
            frame_feats={
                0: torch.tensor(feat_A).view(1, FEAT_DIM, 1, 1),
                1: torch.tensor(feat_B).view(1, FEAT_DIM, 1, 1),
            },
            frame_outputs={
                0: [_fake_body_params()],
                1: [_fake_body_params()],
            },
        )

        body_dir = _run_core(tmpdir, estimator)
        npz_files = sorted(body_dir.glob("person_*.npz"))

        print(f"  cosine_sim(feat_A, feat_B) = {sim:.4f}  "
              f"(threshold = {BodyParameterEstimator._REID_THRESHOLD})")
        print(f"  Person files found: {[f.name for f in npz_files]}")

        assert len(npz_files) == 1, (
            f"Expected 1 merged person file, got {len(npz_files)}: "
            f"{[f.name for f in npz_files]}"
        )
        data = dict(np.load(str(npz_files[0])))
        assert len(data["frame_indices"]) == 2, (
            f"Expected 2 frames in merged track, got {len(data['frame_indices'])}"
        )
        print(f"  Frames in merged track: {data['frame_indices'].tolist()}")
        print("  PASSED")


# ===========================================================================
# Unit test 2: two genuinely different people → must NOT be merged
# ===========================================================================

def test_reid_no_false_merge():
    """Two different people with dissimilar visual features.

    _process_video_core must keep them as separate tracks.
    """
    print("\n=== Unit test: re-ID no false merge (two different people) ===")

    feat_A = _random_unit(FEAT_DIM, seed=42)
    feat_B = _random_unit(FEAT_DIM, seed=99)   # independent random → low sim
    sim = float(np.dot(feat_A, feat_B))

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        json_dir = tmpdir / "json_data"
        _make_frame_dir(tmpdir, [0, 1])
        _write_json(json_dir, 0, {1: (100, 100, 200, 300)})   # person_id=1
        _write_json(json_dir, 1, {2: (400, 100, 500, 300)})   # person_id=2 (different person)

        estimator = _MockEstimator(
            frame_feats={
                0: torch.tensor(feat_A).view(1, FEAT_DIM, 1, 1),
                1: torch.tensor(feat_B).view(1, FEAT_DIM, 1, 1),
            },
            frame_outputs={
                0: [_fake_body_params()],
                1: [_fake_body_params()],
            },
        )

        body_dir = _run_core(tmpdir, estimator)
        npz_files = sorted(body_dir.glob("person_*.npz"))

        print(f"  cosine_sim(feat_A, feat_B) = {sim:.4f}  "
              f"(threshold = {BodyParameterEstimator._REID_THRESHOLD})")
        print(f"  Person files found: {[f.name for f in npz_files]}")

        assert len(npz_files) == 2, (
            f"Expected 2 separate person files, got {len(npz_files)}: "
            f"{[f.name for f in npz_files]}"
        )
        print("  PASSED")

def main():
    # --- Unit tests (no GPU required, run first) ---
    test_reid_merge()
    test_reid_no_false_merge()

if __name__ == "__main__":
    main()
