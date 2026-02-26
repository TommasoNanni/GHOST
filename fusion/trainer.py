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
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.losses = losses
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and self.device.type == "cuda"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.metrics = metrics
        self.metric_fn = metric_fn
        self.use_wandb = use_wandb

        if self.metrics is not None and self.metric_fn is None:
            raise ValueError("metrics provided but metric_fn is None. "
                             "Provide a metric_fn(preds, targets, metrics) callable.")

        self._scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
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

        for epoch in range(self._epoch, self.max_epochs):
            self._epoch = epoch

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
                self.log_losses_to_wandb(train_stats, phase="train", epoch=epoch)
                if val_stats:
                    self.log_losses_to_wandb(val_stats, phase="val", epoch=epoch)
                if train_metrics:
                    self.log_metrics_to_wandb(train_metrics, phase="train", epoch=epoch)
                if val_metrics:
                    self.log_metrics_to_wandb(val_metrics, phase="val", epoch=epoch)

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
        wandb.log({f"{phase}/loss/{name}": s["mean"] for name, s in stats.items()}, step=epoch)

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
            {f"{phase}/metric/{name}": s["mean"] for name, s in metric_results.items()},
            step=epoch,
        )

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

        pbar = tqdm(loader, desc=f"Epoch {self._epoch:04d} [{tag}]",
                    dynamic_ncols=True, leave=False)

        for batch in pbar:
            inputs, targets = self._unpack_batch(batch)

            if train:
                # Training step
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    preds = self._forward(inputs)
                    step_losses = {name: fn(preds, targets) for name, (fn, _) in self.losses.items()}
                total_loss: torch.Tensor = torch.stack(
                    [w * step_losses[n] for n, (_, w) in self.losses.items()]
                ).sum()

                self.optimizer.zero_grad()
                if self._scaler:
                    self._scaler.scale(total_loss).backward()
                    if self.grad_clip:
                        self._scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    total_loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self._step += 1
            else:
                # Validation step
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                    preds = self._forward(inputs)
                    step_losses = {name: fn(preds, targets) for name, (fn, _) in self.losses.items()}
                    total_loss = torch.stack(
                        [w * step_losses[n] for n, (_, w) in self.losses.items()]
                    ).sum()

            if self.metric_fn is not None and self.metrics is not None:
                with torch.no_grad():
                    self.metric_fn(preds, targets, self.metrics)

            detached = {k: v.item() for k, v in step_losses.items()}
            detached["total"] = total_loss.item()
            acc.update(detached)

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
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
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
