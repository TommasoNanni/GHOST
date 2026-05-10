"""Within-video person re-identification across SAM2 track interruptions.

Maintains an appearance gallery (EMA), handles gap severing, covisibility
blocking, and retry windows.  One instance is created per video inside
:class:`ParametersExtractor`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np


class InVideoReidentifier:
    """Tracks and resolves within-video person identity across SAM2 track interruptions.

    Maintains an appearance gallery (EMA), handles gap severing, covisibility
    blocking, and retry windows.  One instance per video.

    Parameters
    ----------
    reid_threshold : float
        Minimum cosine similarity to accept a ReID merge.
    gallery_max_size : int
        Maximum number of frames kept in each person's gallery (top-K by confidence).
    reid_match_window : int
        Frames to retry ReID before finalising a new person.
    fps : float
        Video frame rate — sets the gap-sever threshold (1 second).
    gpu_label : str
        Prefix for log messages (e.g. "[GPU 0] ").
    video_id : str
        Video identifier for log messages.
    """

    _COVISIBILITY_THRESHOLD: int = 3
    # Weight of appearance vs betas in the combined similarity score.
    # When betas are unavailable the score falls back to appearance only.
    _APPEARANCE_WEIGHT: float = 0.7
    _BETAS_WEIGHT: float = 0.3

    def __init__(
        self,
        reid_threshold: float = 0.65,
        reid_match_window: int = 5,
        fps: float = 15.0,
        gpu_label: str = "",
        video_id: str = "",
        gallery_max_size: int = 30,
        drift_threshold: float = 0.35,
        min_gallery_frames: int = 10,
    ):
        self.reid_threshold = reid_threshold
        self.reid_match_window = reid_match_window
        self.gap_threshold: int = max(1, round(fps))
        self.gpu_label = gpu_label
        self.video_id = video_id
        self.gallery_max_size = gallery_max_size
        self.drift_threshold = drift_threshold
        self.min_gallery_frames = min_gallery_frames

        # Appearance gallery: pid → list of (feat, confidence) pairs, top-K by confidence.
        self._person_gallery_buffer: dict[int, list[tuple[np.ndarray, float]]] = {}
        self._person_feat_buffer: dict[int, list[tuple[int, np.ndarray]]] = {}
        # Betas gallery: pid → running sum and count for online mean.
        self._betas_sum: dict[int, np.ndarray] = {}
        self._betas_count: dict[int, int] = {}
        self._id_remap: dict[int, int] = {}
        self._pending_reid: dict[int, int] = {}
        self._track_last_seen: dict[int, int] = {}
        self._severed_ids: set[int] = set()
        # Tracks severed specifically due to appearance drift (ID steal detection).
        # These bypass covisibility checks in ReID since their recorded covisibility
        # belonged to the track's old identity, not the stolen one.
        self._drift_severed_ids: set[int] = set()
        self._covisible_ids: set[frozenset] = set()

    def build_covisibility(self, json_files: list) -> None:
        """Pre-scan JSON frames to find SAM3 IDs that are ever co-visible."""
        counter: Counter = Counter()
        for jp in json_files:
            with open(jp) as _f:
                meta = json.load(_f)
            ids = sorted(int(s) for s in meta.get("labels", {}).keys())
            for ai, a in enumerate(ids):
                for b in ids[ai + 1:]:
                    counter[frozenset({a, b})] += 1
        self._covisible_ids = {
            pair for pair, count in counter.items()
            if count >= self._COVISIBILITY_THRESHOLD
        }

    def _gallery_descriptor(self, pid: int) -> np.ndarray | None:
        """Return the confidence-weighted mean descriptor for a gallery entry."""
        buf = self._person_gallery_buffer.get(pid)
        if not buf:
            return None
        feats = np.stack([f for f, _ in buf])
        confs = np.array([c for _, c in buf], dtype=np.float32)
        total = confs.sum()
        weights = confs / total if total > 0 else np.ones(len(confs), dtype=np.float32) / len(confs)
        desc = (feats * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(desc)
        return desc / norm if norm > 0 else desc

    def _betas_descriptor(self, pid: int) -> np.ndarray | None:
        """Return the L2-normalised mean betas for a gallery entry."""
        count = self._betas_count.get(pid, 0)
        if count == 0:
            return None
        mean = self._betas_sum[pid] / count
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    def _update_gallery(self, pid: int, feat: np.ndarray, confidence: float) -> None:
        """Append (feat, confidence) to the gallery buffer, keeping top-K by confidence."""
        buf = self._person_gallery_buffer.setdefault(pid, [])
        buf.append((feat.copy(), confidence))
        if len(buf) > self.gallery_max_size:
            min_idx = min(range(len(buf)), key=lambda i: buf[i][1])
            buf.pop(min_idx)

    def _update_betas(self, pid: int, betas: np.ndarray) -> None:
        """Accumulate betas into the running mean for a person."""
        if pid not in self._betas_sum:
            self._betas_sum[pid] = betas.copy()
            self._betas_count[pid] = 1
        else:
            self._betas_sum[pid] += betas
            self._betas_count[pid] += 1

    def _combined_sim(
        self,
        feat: np.ndarray,
        betas: np.ndarray | None,
        pid: int,
    ) -> float:
        """Appearance + betas cosine similarity, blended by weight constants."""
        app_desc = self._gallery_descriptor(pid)
        if app_desc is None:
            return -1.0
        app_sim = float(np.dot(feat, app_desc))

        if betas is not None:
            beta_desc = self._betas_descriptor(pid)
            if beta_desc is not None:
                beta_sim = float(np.dot(betas, beta_desc))
                return self._APPEARANCE_WEIGHT * app_sim + self._BETAS_WEIGHT * beta_sim

        return app_sim

    def process_detection(
        self,
        person_id: int,
        feat: np.ndarray | None,
        frame_idx: int,
        valid_persons: list,
        confidence: float | None = None,
        betas: np.ndarray | None = None,
    ) -> int:
        """Process one detection and return its canonical person ID.

        Parameters
        ----------
        person_id : int
            Raw SAM3 track ID.
        feat : np.ndarray or None
            L2-normalised DINOv3 descriptor, or None if unavailable.
        frame_idx : int
            Current frame index within this video.
        valid_persons : list
            All (person_id, ...) tuples in this frame — for co-visibility exclusion.
        confidence : float or None
            Mean per-joint confidence from SAM3D for this detection.
        betas : np.ndarray or None
            L2-normalised MHR shape parameters, or None if unavailable.
        """
        canonical_id: int = self._id_remap.get(person_id, person_id)
        conf = confidence if confidence is not None else 0.0

        # Gap severing
        if person_id in self._track_last_seen:
            gap = frame_idx - self._track_last_seen[person_id]
            if gap > self.gap_threshold:
                logging.info(
                    f"{self.gpu_label}Gap-sever: SAM3 id {person_id} absent "
                    f"{gap} frames (>{self.gap_threshold}) in {self.video_id} "
                    f"@ frame {frame_idx} — re-routing through ReID"
                )
                self._severed_ids.add(person_id)
                self._pending_reid[person_id] = self.reid_match_window

        if feat is not None:
            # Drift detection: for a continuously active, non-remapped track,
            # check if the current appearance has diverged from its own gallery.
            # A sharp drop signals an ID steal (SAM3 ghost memory assigned this
            # track slot to a different returning person with no gap).
            if (
                person_id not in self._pending_reid
                and person_id not in self._severed_ids
                and person_id not in self._id_remap
                and person_id in self._person_gallery_buffer
                and len(self._person_gallery_buffer[person_id]) >= self.min_gallery_frames
            ):
                gallery_desc = self._gallery_descriptor(person_id)
                if gallery_desc is not None:
                    self_sim = float(np.dot(feat, gallery_desc))
                    if self_sim < self.drift_threshold:
                        self._drift_severed_ids.add(person_id)
                        self._severed_ids.add(person_id)
                        self._pending_reid[person_id] = self.reid_match_window
                        logging.info(
                            f"{self.gpu_label}Drift-sever: SAM3 id {person_id} "
                            f"self-sim={self_sim:.3f} < {self.drift_threshold} "
                            f"in {self.video_id} @ frame {frame_idx} — likely ID steal"
                        )

            should_try_reid = False
            if person_id not in self._person_gallery_buffer and person_id not in self._id_remap:
                should_try_reid = True
                if person_id not in self._pending_reid:
                    self._pending_reid[person_id] = self.reid_match_window
            elif person_id in self._pending_reid:
                should_try_reid = True

            if should_try_reid and self._person_gallery_buffer:
                frame_canonical_ids = {
                    self._id_remap.get(p[0], p[0]) for p in valid_persons
                }
                if person_id in self._severed_ids:
                    frame_canonical_ids.discard(canonical_id)

                sims = {}
                for pid, _ in self._person_gallery_buffer.items():
                    if pid in frame_canonical_ids:
                        continue
                    s = self._combined_sim(feat, betas, pid)
                    if s > -1.0:
                        sims[pid] = s

                # Severed tracks (gap OR drift) bypass covisibility: after an
                # absence the ID may have been reassigned to a different physical
                # person, making pre-gap covisibility stale. The sim threshold
                # (0.55) is the primary guard against false merges. Frame-level
                # exclusion (frame_canonical_ids) already blocks same-frame dupes.
                is_severed = person_id in self._severed_ids
                best_id = max(
                    (
                        pid for pid in sims
                        if is_severed
                        or frozenset({person_id, pid}) not in self._covisible_ids
                    ),
                    key=lambda pid: sims[pid],
                    default=None,
                )

                if (
                    best_id is not None
                    and sims[best_id] >= self.reid_threshold
                ):
                    self._id_remap[person_id] = best_id
                    canonical_id = best_id
                    self._pending_reid.pop(person_id, None)
                    # Clean stale covisibility before discarding from severed sets,
                    # so the condition still holds.
                    if is_severed or person_id in self._drift_severed_ids:
                        self._covisible_ids = {
                            p for p in self._covisible_ids if person_id not in p
                        }
                    self._severed_ids.discard(person_id)
                    self._drift_severed_ids.discard(person_id)
                    # Merge gallery buffers into the canonical person's buffer.
                    if person_id != best_id and person_id in self._person_gallery_buffer:
                        self._person_gallery_buffer.setdefault(best_id, []).extend(
                            self._person_gallery_buffer.pop(person_id)
                        )
                        buf = self._person_gallery_buffer[best_id]
                        if len(buf) > self.gallery_max_size:
                            buf.sort(key=lambda x: x[1], reverse=True)
                            self._person_gallery_buffer[best_id] = buf[:self.gallery_max_size]
                    if person_id != best_id and person_id in self._person_feat_buffer:
                        self._person_feat_buffer.setdefault(best_id, []).extend(
                            self._person_feat_buffer.pop(person_id)
                        )
                    # Merge betas galleries.
                    if person_id != best_id and person_id in self._betas_sum:
                        self._betas_sum[best_id] = (
                            self._betas_sum.get(best_id, np.zeros_like(self._betas_sum[person_id]))
                            + self._betas_sum.pop(person_id)
                        )
                        self._betas_count[best_id] = (
                            self._betas_count.get(best_id, 0)
                            + self._betas_count.pop(person_id)
                        )
                    logging.info(
                        f"{self.gpu_label}Re-ID: SAM3 id {person_id} → "
                        f"person {best_id} (sim={sims[best_id]:.3f}) "
                        f"in {self.video_id} frame {frame_idx}"
                    )
                else:
                    if sims:
                        all_sims = "  ".join(
                            f"P{pid}={sims[pid]:.3f}"
                            + ("[CV]" if frozenset({person_id, pid}) in self._covisible_ids else "")
                            for pid in sorted(sims, key=lambda p: sims[p], reverse=True)
                        )
                        outcome = "no-non-cv-candidate" if best_id is None else f"rejected(thr={self.reid_threshold:.2f})"
                        logging.info(
                            f"[within-video reid] {self.gpu_label}{self.video_id} "
                            f"frame {frame_idx}  SAM3 id {person_id}  [{outcome}]  "
                            f"all_sims: {all_sims}"
                        )
                    if person_id in self._pending_reid:
                        self._pending_reid[person_id] -= 1
                        if self._pending_reid[person_id] <= 0:
                            self._update_gallery(person_id, feat, conf)
                            self._person_feat_buffer.setdefault(person_id, []).append((frame_idx, feat.copy()))
                            del self._pending_reid[person_id]
                            self._severed_ids.discard(person_id)
                            self._drift_severed_ids.discard(person_id)
                    else:
                        self._update_gallery(person_id, feat, conf)
                        self._person_feat_buffer.setdefault(person_id, []).append((frame_idx, feat.copy()))
                        self._severed_ids.discard(person_id)
                        self._drift_severed_ids.discard(person_id)

            elif should_try_reid and not self._person_gallery_buffer:
                self._update_gallery(person_id, feat, conf)
                self._person_feat_buffer.setdefault(person_id, []).append((frame_idx, feat.copy()))
                self._pending_reid.pop(person_id, None)

            # Update the canonical person's appearance gallery with this frame.
            if canonical_id in self._person_gallery_buffer:
                self._update_gallery(canonical_id, feat, conf)
                self._person_feat_buffer.setdefault(canonical_id, []).append((frame_idx, feat.copy()))

        # Update betas gallery for the canonical person whenever betas are available.
        if betas is not None:
            self._update_betas(canonical_id, betas)

        self._track_last_seen[person_id] = frame_idx
        return canonical_id

    @property
    def id_remap(self) -> dict[int, int]:
        return self._id_remap

    @property
    def feature_buffer(self) -> dict[int, list[tuple[int, np.ndarray]]]:
        return self._person_feat_buffer
