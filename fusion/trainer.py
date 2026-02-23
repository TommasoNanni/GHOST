"""
Modular Trainer for PyTorch models.

Features
--------
- Pluggable named loss functions with per-loss weights
- Per-loss statistics: mean, median, std, min, max, configurable percentiles
- Multiple logging backends: console (tqdm + Python logging), CSV, JSON Lines, W&B
- Step-level and epoch-level W&B logging
- Gradient accumulation and gradient clipping
- Mixed-precision (AMP) training via torch.cuda.amp
- LR scheduling in "epoch" or "step" mode (ReduceLROnPlateau-aware)
- Checkpointing: best, last, every-N epochs, with full resume support
- Early stopping with configurable patience, metric, mode, and min-delta
- Callback hooks: on_train_start/end, on_epoch_start/end, on_step_end, on_val_end
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)

@dataclass
class LossEntry:
    """A named, weighted loss function."""
    name: str
    fn: Callable[[Any, Any], torch.Tensor]  # fn(predictions, targets) → scalar tensor
    weight: float = 1.0
    log: bool = True


@dataclass
class TrainerConfig:
    """
    All hyper-parameters that govern the training loop.

    Every field has a sensible default and can be overridden individually:

        cfg = TrainerConfig(max_epochs=50, use_amp=True, use_wandb=True,
                            wandb_project="ghost", checkpoint_dir="ckpts/")
    """

    # ── Optimization ──────────────────────────────────────────────────────────
    max_epochs: int = 100
    grad_clip: Optional[float] = None          # None = no clipping
    grad_accumulation_steps: int = 1
    use_amp: bool = False                      # mixed-precision via torch.cuda.amp

    # ── LR scheduler step mode ────────────────────────────────────────────────
    scheduler_step_mode: str = "epoch"         # "epoch" | "step"

    # ── Console / file logging ────────────────────────────────────────────────
    log_every_n_steps: int = 10               # W&B step-level log frequency
    log_to_console: bool = True               # tqdm progress bars + logger.info summaries
    log_to_csv: bool = False                  # write metrics.csv to log_dir
    log_to_jsonl: bool = False                # write metrics.jsonl to log_dir
    log_dir: Optional[str] = None            # directory for CSV / JSONL files

    # ── Statistics ────────────────────────────────────────────────────────────
    stat_percentiles: Tuple[int, ...] = (5, 25, 75, 95)  # which percentiles to compute

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_config: Optional[Dict[str, Any]] = None  # extra config to log on W&B
    wandb_watch_model: bool = True             # call wandb.watch on the model
    wandb_watch_log_freq: int = 100            # gradient logging frequency

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: Optional[str] = None
    save_best: bool = True                    # save checkpoint with best monitored metric
    save_last: bool = True                    # always overwrite last.pt
    save_every_n_epochs: Optional[int] = None # additionally save epoch_XXXX.pt every N

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val/total/mean"  # key in the flat metrics dict
    early_stopping_mode: str = "min"          # "min" | "max"
    early_stopping_min_delta: float = 0.0



class LossAccumulator:
    """
    Buffers per-step scalar loss values and computes summary statistics
    (mean, median, std, min, max, configurable percentiles) over the buffered
    values on demand.
    """

    def __init__(self, percentiles: Sequence[int] = (5, 25, 75, 95)) -> None:
        self._percentiles = list(percentiles)
        self._values: Dict[str, List[float]] = defaultdict(list)

    def update(self, losses: Dict[str, float]) -> None:
        """Append one step's scalar losses."""
        for k, v in losses.items():
            self._values[k].append(float(v))

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Return a nested dict  {loss_name: {stat_name: value, ...}, ...}.

        Computed statistics
        -------------------
        mean, median, std, min, max, count
        p{N} for each N in percentiles (e.g. p5, p25, p75, p95)
        """
        stats: Dict[str, Dict[str, float]] = {}
        for k, vals in self._values.items():
            arr = np.asarray(vals, dtype=np.float64)
            entry: Dict[str, float] = {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std()),
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "count":  float(len(vals)),
            }
            for p in self._percentiles:
                entry[f"p{p}"] = float(np.percentile(arr, p))
            stats[k] = entry
        return stats

    def reset(self) -> None:
        self._values.clear()

    def is_empty(self) -> bool:
        return not bool(self._values)


class Trainer:
    """
    Generic, modular PyTorch trainer.

    Quick-start
    -----------
    >>> trainer = Trainer(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     config=TrainerConfig(
    ...         max_epochs=50,
    ...         use_amp=True,
    ...         use_wandb=True,
    ...         wandb_project="ghost",
    ...         checkpoint_dir="checkpoints/",
    ...         early_stopping=True,
    ...         early_stopping_patience=5,
    ...     ),
    ... )
    >>> trainer.add_loss("pose",   pose_loss_fn,   weight=1.0)
    >>> trainer.add_loss("shape",  shape_loss_fn,  weight=0.5)
    >>> trainer.add_loss("camera", camera_loss_fn, weight=0.3)
    >>> trainer.train()

    Custom batch format
    -------------------
    Override ``_unpack_batch`` to split your batch into ``(inputs, targets)``
    and/or override ``_model_forward`` for non-standard calling conventions.

    Callbacks
    ---------
    >>> trainer.add_callback("on_epoch_end", lambda trainer, epoch, metrics: print(metrics))
    """


    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config or TrainerConfig()
        self.scheduler    = scheduler
        self.device       = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model.to(self.device)

        # Loss registry
        self._losses: List[LossEntry] = []

        # AMP scaler (only meaningful on CUDA)
        self._scaler = (
            torch.cuda.amp.GradScaler()
            if self.config.use_amp and self.device.type == "cuda"
            else None
        )

        # Training state
        self._current_epoch: int    = 0
        self._global_step: int      = 0
        self._best_metric: Optional[float] = None  # best value of the monitored metric
        self._no_improve_count: int = 0            # epochs without improvement (early stopping)

        # File logging handles (opened lazily)
        self._csv_writer = None
        self._csv_file   = None
        self._jsonl_file = None

        # Callbacks registry
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # W&B run handle
        self._wandb_run = None

        # Create output directories
        if self.config.checkpoint_dir:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.config.log_dir:
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    # ── Public API – loss registry ────────────────────────────────────────────

    def add_loss(
        self,
        name: str,
        fn: Callable[[Any, Any], torch.Tensor],
        weight: float = 1.0,
        log: bool = True,
    ) -> "Trainer":
        """
        Register a named loss function.

        Parameters
        ----------
        name   : unique identifier, used as the key in logged metrics
        fn     : callable(predictions, targets) → scalar tensor
        weight : multiplier applied to this loss when computing total_loss
        log    : whether to include this loss in statistics and logging

        Returns self for method chaining.
        """
        if any(e.name == name for e in self._losses):
            raise ValueError(f"A loss named '{name}' is already registered.")
        self._losses.append(LossEntry(name=name, fn=fn, weight=weight, log=log))
        return self

    def remove_loss(self, name: str) -> "Trainer":
        """Remove a registered loss by name."""
        before = len(self._losses)
        self._losses = [e for e in self._losses if e.name != name]
        if len(self._losses) == before:
            raise KeyError(f"Loss '{name}' not found.")
        return self

    def set_loss_weight(self, name: str, weight: float) -> "Trainer":
        """Update the weight of a registered loss in-place."""
        for entry in self._losses:
            if entry.name == name:
                entry.weight = weight
                return self
        raise KeyError(f"Loss '{name}' not found.")

    def get_loss_names(self) -> List[str]:
        """Return the names of all registered losses."""
        return [e.name for e in self._losses]

    # ── Public API – callbacks ────────────────────────────────────────────────

    def add_callback(self, event: str, fn: Callable) -> "Trainer":
        """
        Register a callback for a lifecycle event.

        Supported events and their keyword arguments
        ---------------------------------------------
        "on_train_start"  – (trainer,)
        "on_train_end"    – (trainer,)
        "on_epoch_start"  – (trainer, epoch)
        "on_epoch_end"    – (trainer, epoch, metrics)
        "on_step_end"     – (trainer, step, losses)
        "on_val_end"      – (trainer, stats)

        Returns self for method chaining.
        """
        valid_events = {
            "on_train_start", "on_train_end",
            "on_epoch_start", "on_epoch_end",
            "on_step_end", "on_val_end",
        }
        if event not in valid_events:
            raise ValueError(f"Unknown event '{event}'. Valid events: {valid_events}")
        self._callbacks[event].append(fn)
        return self

    # ── Public API – run ──────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop from _current_epoch to config.max_epochs."""
        self._init_wandb()
        self._fire("on_train_start")
        logger.info(f"Starting training on {self.device}  |  losses: {self.get_loss_names()}")

        try:
            for epoch in range(self._current_epoch, self.config.max_epochs):
                self._current_epoch = epoch
                self._fire("on_epoch_start", epoch=epoch)

                train_stats = self._train_epoch()
                val_stats   = self._val_epoch() if self.val_loader is not None else {}

                # Epoch-level scheduler step
                if self.scheduler and self.config.scheduler_step_mode == "epoch":
                    self._scheduler_step(val_stats or train_stats)

                # Build flat metrics dict and log it
                metrics = self._build_epoch_metrics(train_stats, val_stats, epoch)
                self._log_metrics(metrics, step=epoch)

                # Determine whether the current epoch is the new best
                monitor_val   = metrics.get(self.config.early_stopping_metric)
                is_best       = self._is_better(monitor_val, self._best_metric)
                if is_best and monitor_val is not None:
                    self._best_metric = monitor_val

                # Checkpoint
                self._maybe_checkpoint(metrics, epoch, is_best)

                self._fire("on_epoch_end", epoch=epoch, metrics=metrics)

                # Early stopping
                if self._check_early_stopping(is_best):
                    logger.info(f"Early stopping at epoch {epoch} (no improvement for "
                                f"{self._no_improve_count} epochs).")
                    break

        finally:
            self._cleanup()
            self._fire("on_train_end")

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict[str, float]]:
        """Run a single validation pass and return the statistics dict."""
        if self.val_loader is None:
            raise RuntimeError("No val_loader provided.")
        return self._val_epoch()

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, float]:
        """
        Restore model, optimizer, scheduler, and AMP scaler from a checkpoint.
        Sets _current_epoch so training resumes from where it left off.
        Returns the metrics dict saved with the checkpoint.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._current_epoch   = state["epoch"] + 1
        self._global_step     = state.get("global_step", 0)
        self._best_metric     = state.get("best_metric")
        self._no_improve_count = state.get("no_improve_count", 0)
        if self.scheduler and "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if self._scaler and "scaler_state" in state:
            self._scaler.load_state_dict(state["scaler_state"])
        logger.info(f"Resumed from {path}  (epoch {state['epoch']}, "
                    f"step {self._global_step})")
        return state.get("metrics", {})


    def _train_epoch(self) -> Dict[str, Dict[str, float]]:
        self.model.train()
        accumulator = LossAccumulator(percentiles=self.config.stat_percentiles)
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self._current_epoch:04d} [train]",
            disable=not self.config.log_to_console,
            dynamic_ncols=True,
            leave=False,
        )

        for step_in_epoch, batch in enumerate(pbar):
            step_losses = self._forward_and_compute_losses(batch)

            total_loss: torch.Tensor = sum(  # type: ignore[assignment]
                e.weight * step_losses[e.name]
                for e in self._losses
                if e.name in step_losses
            )
            scaled_total = total_loss / self.config.grad_accumulation_steps

            if self._scaler:
                self._scaler.scale(scaled_total).backward()
            else:
                scaled_total.backward()

            if (step_in_epoch + 1) % self.config.grad_accumulation_steps == 0:
                if self.config.grad_clip is not None:
                    if self._scaler:
                        self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                if self._scaler:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler and self.config.scheduler_step_mode == "step":
                    self._scheduler_step()

            detached = {k: v.item() for k, v in step_losses.items()}
            detached["total"] = total_loss.item()
            accumulator.update(detached)

            self._global_step += 1

            if self._global_step % self.config.log_every_n_steps == 0:
                if self._wandb_run:
                    step_log = {f"train/step/{k}": v for k, v in detached.items()}
                    step_log["train/step/lr"] = self._current_lr()
                    self._wandb_run.log(step_log, step=self._global_step)

            if self.config.log_to_console:
                pbar.set_postfix(
                    {k: f"{v:.4f}" for k, v in detached.items()}
                    | {"lr": f"{self._current_lr():.2e}"}
                )

            self._fire("on_step_end", step=self._global_step, losses=detached)

        return accumulator.compute()


    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        accumulator = LossAccumulator(percentiles=self.config.stat_percentiles)

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self._current_epoch:04d} [val]  ",
            disable=not self.config.log_to_console,
            dynamic_ncols=True,
            leave=False,
        )

        for batch in pbar:
            step_losses = self._forward_and_compute_losses(batch)
            total_loss: torch.Tensor = sum(  # type: ignore[assignment]
                e.weight * step_losses[e.name]
                for e in self._losses
                if e.name in step_losses
            )
            detached = {k: v.item() for k, v in step_losses.items()}
            detached["total"] = total_loss.item()
            accumulator.update(detached)

            if self.config.log_to_console:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in detached.items()})

        stats = accumulator.compute()
        self._fire("on_val_end", stats=stats)
        return stats


    def _forward_and_compute_losses(
        self, batch: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Unpack batch → move to device → forward pass → compute all losses.

        Override this method for completely custom logic (e.g. multi-output
        models where different losses act on different outputs).
        """
        if not self._losses:
            raise RuntimeError(
                "No loss functions registered. Call add_loss() before training."
            )

        inputs, targets = self._unpack_batch(batch)
        inputs  = self._to_device(inputs)
        targets = self._to_device(targets)

        amp_ctx: Any = (
            torch.cuda.amp.autocast()
            if self.config.use_amp and self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp_ctx:
            predictions = self._model_forward(inputs)
            losses: Dict[str, torch.Tensor] = {
                entry.name: entry.fn(predictions, targets)
                for entry in self._losses
            }

        return losses

    def _model_forward(self, inputs: Any) -> Any:
        """
        Call the model. Override for non-standard calling conventions.

        Default behaviour
        -----------------
        - tuple  → model(*inputs)
        - dict   → model(**inputs)
        - other  → model(inputs)
        """
        if isinstance(inputs, tuple):
            return self.model(*inputs)
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    def _unpack_batch(self, batch: Any) -> Tuple[Any, Any]:
        """
        Split a batch into (inputs, targets).

        Default: expects batch = (inputs, targets).
        Override for custom formats (e.g. dict batches).
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(
            "_unpack_batch expects a length-2 sequence (inputs, targets). "
            "Override _unpack_batch or _forward_and_compute_losses for custom formats."
        )

    def _build_epoch_metrics(
        self,
        train_stats: Dict[str, Dict[str, float]],
        val_stats: Dict[str, Dict[str, float]],
        epoch: int,
    ) -> Dict[str, float]:
        """
        Flatten the nested per-split statistics into a single-level dict.

        Key format:  {split}/{loss_name}/{stat_name}
        Convenience: {split}/{loss_name}  →  mean value
        """
        metrics: Dict[str, float] = {"epoch": float(epoch)}

        for split, stats in [("train", train_stats), ("val", val_stats)]:
            for loss_name, stat_dict in stats.items():
                for stat_name, val in stat_dict.items():
                    metrics[f"{split}/{loss_name}/{stat_name}"] = val
                # expose mean at the top level for convenience
                if "mean" in stat_dict:
                    metrics[f"{split}/{loss_name}"] = stat_dict["mean"]

        metrics["train/lr"] = self._current_lr()
        return metrics

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.config.log_to_console:
            self._console_summary(metrics)
        if self.config.log_to_csv:
            self._write_csv(metrics)
        if self.config.log_to_jsonl:
            self._write_jsonl(metrics)
        if self._wandb_run:
            self._wandb_run.log(metrics, step=step)

    def _console_summary(self, metrics: Dict[str, float]) -> None:
        """Log a compact one-line epoch summary via the Python logger."""
        epoch = int(metrics.get("epoch", -1))
        # show mean values and lr only
        parts = [f"[Epoch {epoch:04d}]"]
        for key in sorted(metrics):
            if key.endswith("/mean") or key == "train/lr":
                label = key.replace("/mean", "")
                parts.append(f"{label}={metrics[key]:.5f}")
        logger.info("  ".join(parts))

    def _write_csv(self, metrics: Dict[str, float]) -> None:
        if self._csv_writer is None:
            log_path = Path(self.config.log_dir or ".") / "metrics.csv"
            self._csv_file   = open(log_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=sorted(metrics.keys()),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()  # type: ignore[union-attr]

    def _write_jsonl(self, metrics: Dict[str, float]) -> None:
        if self._jsonl_file is None:
            log_path = Path(self.config.log_dir or ".") / "metrics.jsonl"
            self._jsonl_file = open(log_path, "a")
        self._jsonl_file.write(json.dumps(metrics) + "\n")
        self._jsonl_file.flush()


    def _init_wandb(self) -> None:
        if not self.config.use_wandb:
            return
        if not WANDB_AVAILABLE:
            logger.warning("wandb is not installed; W&B logging disabled.")
            return
        cfg = self.config
        init_config = {
            **(cfg.wandb_config or {}),
            "max_epochs":              cfg.max_epochs,
            "grad_clip":               cfg.grad_clip,
            "grad_accumulation_steps": cfg.grad_accumulation_steps,
            "use_amp":                 cfg.use_amp,
            "scheduler_step_mode":     cfg.scheduler_step_mode,
            "early_stopping":          cfg.early_stopping,
            "early_stopping_patience": cfg.early_stopping_patience,
            "losses": {e.name: {"weight": e.weight} for e in self._losses},
        }
        self._wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=cfg.wandb_tags or [],
            notes=cfg.wandb_notes,
            config=init_config,
            resume="allow",
        )
        if cfg.wandb_watch_model:
            wandb.watch(
                self.model,
                log="gradients",
                log_freq=cfg.wandb_watch_log_freq,
            )

    def _maybe_checkpoint(
        self,
        metrics: Dict[str, float],
        epoch: int,
        is_best: bool,
    ) -> None:
        if not self.config.checkpoint_dir:
            return
        if self.config.save_last:
            self._save_checkpoint("last.pt", metrics)
        if self.config.save_best and is_best:
            self._save_checkpoint("best.pt", metrics)
            logger.info(f"  ↳ New best checkpoint  "
                        f"({self.config.early_stopping_metric}="
                        f"{self._best_metric:.5f})")
        if (
            self.config.save_every_n_epochs is not None
            and (epoch + 1) % self.config.save_every_n_epochs == 0
        ):
            self._save_checkpoint(f"epoch_{epoch:04d}.pt", metrics)

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        path = Path(self.config.checkpoint_dir) / filename  # type: ignore[arg-type]
        state = {
            "epoch":            self._current_epoch,
            "global_step":      self._global_step,
            "model_state":      self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "metrics":          metrics,
            "best_metric":      self._best_metric,
            "no_improve_count": self._no_improve_count,
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
        if self._scaler:
            state["scaler_state"] = self._scaler.state_dict()
        torch.save(state, path)
        logger.debug(f"Checkpoint → {path}")

    def _check_early_stopping(self, is_best: bool) -> bool:
        """
        Update the no-improvement counter and return True if training should stop.
        """
        if not self.config.early_stopping:
            return False
        if is_best:
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        return self._no_improve_count >= self.config.early_stopping_patience

    def _is_better(self, current: Optional[float], best: Optional[float]) -> bool:
        """Return True if *current* is an improvement over *best*."""
        if current is None:
            return False
        if best is None:
            return True
        delta = self.config.early_stopping_min_delta
        if self.config.early_stopping_mode == "min":
            return current < best - delta
        return current > best + delta

    def _scheduler_step(
        self, stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # needs a metric value to decide whether to reduce
            monitor_key = self.config.early_stopping_metric.split("/")[-2:]
            # try to extract the mean of the monitored loss from stats
            val = None
            if stats:
                # early_stopping_metric example: "val/total/mean"
                parts = self.config.early_stopping_metric.split("/")
                if len(parts) >= 2:
                    loss_key = parts[1]
                    val = stats.get(loss_key, {}).get("mean")
            if val is not None:
                self.scheduler.step(val)
            else:
                logger.warning(
                    "ReduceLROnPlateau: could not extract monitored metric from stats; "
                    "scheduler.step() not called."
                )
        else:
            self.scheduler.step()


    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _to_device(self, data: Any) -> Any:
        """Recursively move tensors (or containers of tensors) to self.device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        if isinstance(data, (list, tuple)):
            moved = [self._to_device(x) for x in data]
            return type(data)(moved)
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        return data

    def _fire(self, event: str, **kwargs: Any) -> None:
        for fn in self._callbacks.get(event, []):
            fn(trainer=self, **kwargs)

    def _cleanup(self) -> None:
        """Close open file handles and finish the W&B run."""
        if self._csv_file is not None:
            self._csv_file.close()
        if self._jsonl_file is not None:
            self._jsonl_file.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()