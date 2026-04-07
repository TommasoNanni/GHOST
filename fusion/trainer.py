"""Training loop for fusion models."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from fusion.metric import MetricCollection
from utilities.smplx_utilities import get_smplx_joints, clear_smplx_cache

logger = logging.getLogger(__name__)


class LossAccumulator:
    """Buffers per-step scalar losses and computes summary statistics."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def update(self, losses: dict[str, float]) -> None:
        """Updates with a dict of new values for losses"""
        for k, v in losses.items():
            self._values[k].append(float(v))

    def compute(self) -> dict[str, dict[str, float]]:
        """Return {loss_name: {mean, median, std, min, max}}."""
        stats: dict[str, dict[str, float]] = {}
        for k, vals in self._values.items():
            arr = np.asarray(vals, dtype=np.float64)
            stats[k] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        return stats

    def reset(self) -> None:
        self._values.clear()


class Trainer:
    """Simple PyTorch trainer.

    Parameters
    ----------
    model          : the nn.Module to train.
    optimizer      : optimizer instance.
    train_loader   : training DataLoader.
    losses         : {name: (fn, weight)} where fn(preds, targets) -> scalar.
    val_loader     : optional validation DataLoader.
    max_epochs     : number of epochs.
    grad_clip      : max gradient norm (None to disable).
    use_amp        : enable mixed-precision on CUDA.
    scheduler      : optional LR scheduler (stepped per epoch).
    checkpoint_dir : directory for last.pt / best.pt (None to disable).
    device         : defaults to CUDA if available.
    metrics        : optional MetricCollection; computed at the end of every epoch.
    metric_fn      : callable(preds, targets, metrics) -> None that calls
                     .update() on the relevant metrics. Required when metrics
                     is provided.
    use_wandb      : if True, losses and metrics are logged to wandb each epoch.

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        losses: dict[str, tuple[Callable, float]],
        val_loader: DataLoader | None = None,
        max_epochs: int = 100,
        grad_clip: float | None = None,
        use_amp: bool = False,
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | None = None,
        early_stopping_patience: int | None = None,
        device: str | torch.device | None = None,
        metrics: MetricCollection | None = None,
        metric_fn: Callable[[Any, Any, Any], None] | None = None,
        use_wandb: bool = False,
        dtype: torch.dtype | None = None,
        prediction_save_path: str | None = None,
        smplx_chunk_size: int = 32,
        curriculum_schedule: dict[int, list[str]] | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.model = model.to(self.device)
        if dtype is not None:
            self.model = self.model.to(dtype)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.losses = losses
        self.curriculum_schedule = curriculum_schedule or {}
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and self.device.type == "cuda"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.metrics = metrics
        self.metric_fn = metric_fn
        self.use_wandb = use_wandb
        self.prediction_save_path = Path(prediction_save_path) if prediction_save_path else None
        self.smplx_chunk_size = smplx_chunk_size

        if self.metrics is not None and self.metric_fn is None:
            raise ValueError("metrics provided but metric_fn is None. "
                             "Provide a metric_fn(preds, targets, metrics) callable.")

        # torch.autocast on CUDA defaults to bfloat16, which has the same dynamic
        # range as float32 and never needs gradient scaling. GradScaler is only
        # beneficial for float16. Using it with torch.compile + flex_attention
        # causes CUDA illegal memory access, so we never create it.
        self._scaler = None
        self._epoch = 0
        self._step = 0
        self._best_val_loss: float | None = None
        self.early_stopping_patience = early_stopping_patience
        self._no_improve = 0

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_pt = self.checkpoint_dir / "best.pt"
            if best_pt.exists():
                raise FileExistsError(
                    f"{best_pt} already exists. Use a clean checkpoint directory "
                    f"or rename/remove the existing best.pt to avoid overwriting it."
                )


    def train(self) -> None:
        """
        Full training loop.
        """
        logger.info(f"Training on {self.device} | losses: {list(self.losses)}")

        _prev_active: set[str] = set()
        for epoch in range(self._epoch, self.max_epochs):
            self._epoch = epoch

            # Log curriculum stage changes
            if self.curriculum_schedule:
                cur_active = set(self._active_losses())
                new_losses = cur_active - _prev_active
                if new_losses:
                    logger.info(f"[CURRICULUM] Epoch {epoch}: activating losses {sorted(new_losses)}")
                _prev_active = cur_active

            train_stats, train_metrics = self._run_epoch(train=True)
            val_stats, val_metrics = self._run_epoch(train=False) if self.val_loader else ({}, {})

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = (val_stats or train_stats).get("total", {}).get("mean")
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            self._log_epoch(epoch, train_stats, val_stats, train_metrics, val_metrics)

            if self.use_wandb:
                import wandb
                self.log_losses_to_wandb(train_stats, phase="train", epoch=epoch)
                if val_stats:
                    self.log_losses_to_wandb(val_stats, phase="val", epoch=epoch)
                if train_metrics:
                    self.log_metrics_to_wandb(train_metrics, phase="train", epoch=epoch)
                if val_metrics:
                    self.log_metrics_to_wandb(val_metrics, phase="val", epoch=epoch)
                wandb.log({"lr": self._lr()}, step=epoch)

            improved = self._checkpoint(val_stats or train_stats)

            # Early stopping
            if self.early_stopping_patience is not None:
                if improved:
                    self._no_improve = 0
                else:
                    self._no_improve += 1
                    if self._no_improve >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch} "
                                    f"(no improvement for {self._no_improve} epochs).")
                        break

        if self.prediction_save_path is not None:
            self.save_predictions(self.train_loader, self.prediction_save_path)

    @torch.no_grad()
    def validate(self) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """
        Run one validation pass.

        Returns
        -------
        (loss_stats, metric_results) — metric_results is empty if no metrics configured.
        """
        if self.val_loader is None:
            raise RuntimeError("No val_loader provided.")
        return self._run_epoch(train=False)

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Restore model / optimizer / scheduler state.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._epoch = state.get("epoch", 0) + 1
        self._step = state.get("step", 0)
        self._best_val_loss = state.get("best_val_loss")
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self._scaler and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])
        logger.info(f"Resumed from {path} (epoch {state.get('epoch', '?')})")

    @torch.no_grad()
    def save_predictions(self, loader: DataLoader, path: str | Path) -> Path:
        """Run the model in eval mode and save final predictions to a .npz file.

        Saved arrays (all float32, batch dim squeezed when B=1):
          pose                  (T, P, J, 6)  – predicted 6-D body pose in cam-0 world frame
          shape                 (T, P, 10)    – predicted SMPL-X betas
          camera                (T, K, 8)     – predicted [quat(4), trans(3), focal_raw(1)] per camera
          body_transl_world     (T, P, 3)     – predicted body root translation in world (cam-0) frame
          gt_body_pose          (T, P, J, 6)  – ground-truth body pose       (when available in targets)
          gt_body_shape         (T, P, 10)    – ground-truth body shape      (when available in targets)
          gt_camera             (T, K, 8)     – ground-truth camera          (when available in targets)
          gt_body_transl_world  (T, P, 3)     – ground-truth body root translation in world frame (when available)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            scene_name = targets.get("scene_name", ["unknown"])[0] \
                         if isinstance(targets, dict) else "unknown"
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                preds = self._forward(inputs)
            pose_aggr, shape_aggr, camera_pred, body_transl_world = preds[:4]

            def _sq(t): return t.squeeze(0).float().cpu().numpy().astype(np.float32)

            arrays: dict[str, np.ndarray] = {
                "pose":              _sq(pose_aggr),
                "shape":             _sq(shape_aggr),
                "camera":            _sq(camera_pred),
                "body_transl_world": _sq(body_transl_world),
            }
            if isinstance(targets, dict):
                if "pose"   in targets: arrays["gt_body_pose"]         = _sq(targets["pose"])
                if "shape"  in targets: arrays["gt_body_shape"]        = _sq(targets["shape"])
                if "camera" in targets: arrays["gt_camera"]            = _sq(targets["camera"])
                if "trans"  in targets: arrays["gt_body_transl_world"] = _sq(targets["trans"])

            out_file = path / f"{scene_name}.npz"
            np.savez_compressed(str(out_file), **arrays)
            logger.info(f"Predictions saved → {out_file}  (keys: {list(arrays)})")

        return path

    def log_losses_to_wandb(
        self,
        stats: dict[str, dict[str, float]],
        phase: str,
        epoch: int,
    ) -> None:
        """Log epoch loss statistics to wandb.

        Parameters
        ----------
        stats  : loss stats as returned by LossAccumulator.compute(),
                 {loss_name: {stat: value}}.
        phase  : "train" or "val".
        epoch  : current epoch index (used as wandb step).
        """
        import wandb
        wandb.log({f"{phase}_losses/{name}": s["mean"] for name, s in stats.items()}, step=epoch)

    def log_metrics_to_wandb(
        self,
        metric_results: dict[str, dict[str, float]],
        phase: str,
        epoch: int,
    ) -> None:
        """Log epoch metric results to wandb.

        Parameters
        ----------
        metric_results : as returned by MetricCollection.compute(),
                         {metric_name: {stat: value}}.
        phase          : "train" or "val".
        epoch          : current epoch index (used as wandb step).
        """
        import wandb
        wandb.log(
            {f"{phase}_metrics/{name}": s["mean"] for name, s in metric_results.items()},
            step=epoch,
        )

    def _active_losses(self) -> dict[str, tuple[Callable, float]]:
        """Return the subset of losses active at the current epoch per curriculum_schedule.

        curriculum_schedule maps epoch thresholds to the list of loss names that become
        active at that epoch (cumulative — each stage adds to the previous set).
        If no schedule is provided, all losses are always active.
        """
        if not self.curriculum_schedule:
            return self.losses
        active: set[str] = set()
        for epoch_threshold in sorted(self.curriculum_schedule):
            if self._epoch >= epoch_threshold:
                active.update(self.curriculum_schedule[epoch_threshold])
        return {k: v for k, v in self.losses.items() if k in active}

    def _run_epoch(
        self, train: bool
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """
        Runs a full training epoch or validation pass and returns aggregated stats.

        Returns
        -------
        (loss_stats, metric_results) — metric_results is empty dict if no metrics configured.
        """
        loader = self.train_loader if train else self.val_loader
        tag = "train" if train else "val"
        self.model.train(train)
        acc = LossAccumulator()
        active_losses = self._active_losses() if train else self.losses

        pbar = tqdm(loader, desc=f"Epoch {self._epoch:04d} [{tag}]",
                    dynamic_ncols=True, leave=False)

        for batch in pbar:
            inputs, targets = self._unpack_batch(batch)

            # ── per-batch diagnostics ─────────────────────────────────────────
            scene_names = targets.get("scene_name", ["?"])
            scene = scene_names[0] if isinstance(scene_names, list) else scene_names
            pmask = inputs.get("person_mask")   # (B, T, K, P) bool after collation
            if pmask is not None:
                pm = pmask[0]                   # (T, K, P)
                T_dim, K_dim, _ = pm.shape
                valid_cams   = int(pm.any(0).any(-1).sum().item())
                per_cam_ppl  = [int(pm[:, k, :].any(0).sum().item()) for k in range(K_dim)]
                missing_per_cam = [int((~pm[:, k, :].any(-1)).sum().item()) for k in range(K_dim)]
                logger.info(
                    f"[batch] {scene}  T={T_dim}  valid_cams={valid_cams}/{K_dim}  "
                    f"people_per_cam={per_cam_ppl}  missing_frames={missing_per_cam}"
                )
            # ─────────────────────────────────────────────────────────────────

            if train:
                # Training step — wrap step 0 in detect_anomaly (forward+backward)
                # so that DivBackward0 / other NaN ops are traced to their source.
                _detect = self._step == 0
                with torch.autograd.detect_anomaly(check_nan=_detect), \
                     torch.amp.autocast('cuda', enabled=self.use_amp):
                    preds = self._forward(inputs)
                    preds = self._append_smplx_joints(preds)

                    # ── pred tensor sanity check ──────────────────────────────
                    _pred_labels = ["pose_aggr", "shape_aggr", "camera_pred",
                                    "body_transl_world", "orient_per_cam",
                                    "transl_per_cam", "person_visible", "joints_world"]
                    for _pi, _pt in enumerate(preds):
                        if isinstance(_pt, torch.Tensor) and not _pt.isfinite().all():
                            _lbl = _pred_labels[_pi] if _pi < len(_pred_labels) else f"preds[{_pi}]"
                            logger.warning(
                                f"[PRED NaN] {scene}  step={self._step}  "
                                f"{_lbl}  shape={tuple(_pt.shape)}  "
                                f"n_bad={(~_pt.isfinite()).sum().item()}"
                            )
                    # body_transl_world range (always log for first step to confirm init)
                    if self._step == 0 and len(preds) > 2 and isinstance(preds[2], torch.Tensor):
                        _cp = preds[2]   # (B, T, K, 8)
                        _qnorm = _cp[..., :4].norm(dim=-1)
                        _tmag  = _cp[..., 4:7].norm(dim=-1)
                        logger.info(
                            f"[INIT] camera_pred  quat_norm min={_qnorm.min().item():.3f} "
                            f"max={_qnorm.max().item():.3f}  "
                            f"transl_mag min={_tmag.min().item():.3f} "
                            f"max={_tmag.max().item():.3f}"
                        )
                    if self._step == 0 and len(preds) > 3 and isinstance(preds[3], torch.Tensor):
                        _tw = preds[3]
                        logger.info(
                            f"[INIT] body_transl_world  min={_tw.min().item():.2f}  "
                            f"max={_tw.max().item():.2f}  mean={_tw.mean().item():.2f}"
                        )
                    if self._step == 0 and len(preds) > 7 and isinstance(preds[7], torch.Tensor):
                        _jw = preds[7]
                        logger.info(
                            f"[INIT] joints_world  min={_jw.min().item():.2f}  "
                            f"max={_jw.max().item():.2f}  mean={_jw.mean().item():.2f}"
                        )
                    # ─────────────────────────────────────────────────────────

                    step_losses = {name: fn(preds, targets) for name, (fn, _) in active_losses.items()}

                # ── forward NaN/inf check ─────────────────────────────────────
                for _lname, _lval in step_losses.items():
                    if not _lval.isfinite():
                        logger.warning(
                            f"[FORWARD NaN] {scene}  step={self._step}  "
                            f"loss={_lname}  value={_lval.item()}"
                        )
                # ─────────────────────────────────────────────────────────────

                total_loss: torch.Tensor = torch.stack(
                    [w * step_losses[n] for n, (_, w) in active_losses.items()]
                ).sum()

                self.optimizer.zero_grad()
                total_loss.backward()

                # ── backward NaN/inf check + gradient norm log ───────────────
                nan_modules: set[str] = set()
                module_grad_norms: dict[str, float] = {}
                for _pname, _p in self.model.named_parameters():
                    top = _pname.split(".")[0]
                    if _p.grad is not None:
                        gnorm = float(_p.grad.float().norm().item())
                        prev = module_grad_norms.get(top, 0.0)
                        module_grad_norms[top] = max(prev, gnorm)
                        if not _p.grad.isfinite().all():
                            nan_modules.add(top)
                if nan_modules:
                    logger.warning(
                        f"[BACKWARD NaN] {scene}  step={self._step}  "
                        f"modules={sorted(nan_modules)}"
                    )
                # Always log grad norms for the first 3 steps for diagnostics
                if self._step < 3:
                    norm_str = "  ".join(
                        f"{m}={v:.3e}" for m, v in sorted(module_grad_norms.items())
                    )
                    logger.info(f"[GRAD NORMS] step={self._step}  {norm_str}")
                # ─────────────────────────────────────────────────────────────

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                clear_smplx_cache()
                self._step += 1
            else:
                # Validation step
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                    preds = self._forward(inputs)
                    preds = self._append_smplx_joints(preds)
                    step_losses = {name: fn(preds, targets) for name, (fn, _) in active_losses.items()}
                    total_loss = torch.stack(
                        [w * step_losses[n] for n, (_, w) in active_losses.items()]
                    ).sum()

            if self.metric_fn is not None and self.metrics is not None:
                with torch.no_grad():
                    self.metric_fn(preds, targets, self.metrics)

            detached = {k: v.item() for k, v in step_losses.items()}
            detached["total"] = total_loss.item()
            acc.update(detached)

            del preds, step_losses, total_loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # log every step
            pbar.set_postfix({k: f"{v:.4f}" for k, v in detached.items()}
                             | ({"lr": f"{self._lr():.2e}"} if train else {}))
            if train:
                logger.debug(f"[step {self._step}] " +
                             "  ".join(f"{k}={v:.5f}" for k, v in detached.items()))

        metric_results: dict[str, dict[str, float]] = {}
        if self.metrics is not None and self.metric_fn is not None:
            metric_results = self.metrics.compute()
            self.metrics.reset()

        return acc.compute(), metric_results

    def _forward(self, inputs: Any) -> Any:
        """
        Forward pass in the model
        """
        if isinstance(inputs, dict):
            return self.model(**inputs)
        if isinstance(inputs, (list, tuple)):
            return self.model(*inputs)
        return self.model(inputs)

    def _append_smplx_joints(self, preds: Any) -> Any:
        """Compute SMPL-X joints once and append as preds[8].

        Both EpipolarLoss and BoneLengthLoss need world-frame joint positions.
        Computing them here (once per step) instead of once per loss × camera-pair
        reduces SMPL-X calls from K*(K-1)/2 * T/chunk to T/chunk.

        joints_world: (B, T, P, Jsmplx, 3) — pelvis-centred + body_transl_world.
        """
        pose_aggr, shape_aggr, _, body_transl_world = preds[:4]
        B, T, P = pose_aggr.shape[:3]
        chunks = []
        for t0 in range(0, T, self.smplx_chunk_size):
            t1 = min(t0 + self.smplx_chunk_size, T)
            t_chunk = t1 - t0
            shape_chunk = shape_aggr.unsqueeze(1).expand(B, t_chunk, P, 10)
            j = get_smplx_joints(pose_aggr[:, t0:t1], shape_chunk)          # (B, t_chunk, P, Jsmplx, 3)
            j = j + body_transl_world[:, t0:t1].unsqueeze(-2)
            chunks.append(j)
        joints_world = torch.cat(chunks, dim=1)                              # (B, T, P, Jsmplx, 3)
        return (*preds, joints_world)

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        """
        Move batch to device and split into (inputs, targets).
        """
        batch = self._to_device(batch)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError("Expected batch to be (inputs, targets).")

    def _to_device(self, data: Any) -> Any:
        """
        Move to device the data
        """
        if isinstance(data, str):
            return data
        if isinstance(data, torch.Tensor):
            t = data.to(self.device, non_blocking=True)
            if self.dtype is not None and t.is_floating_point():
                t = t.to(self.dtype)
            return t
        if isinstance(data, (list, tuple)):
            return type(data)(self._to_device(x) for x in data)
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        return data

    def _lr(self) -> float:
        """
        Get the learning rate
        """
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_epoch(
        self,
        epoch: int,
        train_stats: dict[str, dict[str, float]],
        val_stats: dict[str, dict[str, float]],
        train_metric_results: dict[str, dict[str, float]] | None = None,
        val_metric_results: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Log the epoch, used by the training step
        """
        parts = [f"[Epoch {epoch:04d}]"]
        for name, s in train_stats.items():
            parts.append(f"train/{name}={s['mean']:.5f}")
        for name, s in val_stats.items():
            parts.append(f"val/{name}={s['mean']:.5f}")
        if train_metric_results:
            for name, s in train_metric_results.items():
                parts.append(f"train/{name}={s['mean']:.5f}")
        if val_metric_results:
            for name, s in val_metric_results.items():
                parts.append(f"val/{name}={s['mean']:.5f}")
        parts.append(f"lr={self._lr():.2e}")
        logger.info("  ".join(parts))

    def _checkpoint(self, stats: dict[str, dict[str, float]]) -> bool:
        """Save checkpoints. Returns True if this epoch improved the best metric."""
        val_mean = stats.get("total", {}).get("mean")
        improved = (
            val_mean is not None
            and (self._best_val_loss is None or val_mean < self._best_val_loss)
        )
        if improved:
            self._best_val_loss = val_mean

        if not self.checkpoint_dir:
            return improved

        state = {
            "epoch": self._epoch,
            "step": self._step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self._best_val_loss,
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        if self._scaler:
            state["scaler"] = self._scaler.state_dict()

        torch.save(state, self.checkpoint_dir / "last.pt")

        if improved:
            torch.save(state, self.checkpoint_dir / "best.pt")
            logger.info(f"  New best checkpoint (total/mean={val_mean:.5f})")

        return improved
