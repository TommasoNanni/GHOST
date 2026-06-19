"""Training loop for PoseFusionModule (fusion_module_v2.py).

Differences from trainer.py
----------------------------
* _forward()             : filters inputs to the three keys PoseFusionModule expects
                           (pose, person_mask, joint_mask) and wraps the output tensor
                           in a 1-tuple so preds[0] always gives pose_aggr.
* _append_smplx_joints() : no-op — PoseFusionModule outputs pose only; no shape or
                           body translation to pass to the SMPL-X forward pass.
* _compute_diagnostics() : camera error block removed; pose block unchanged.
* save_predictions()     : saves pose_aggr only (no shape / camera / transl).
Everything else is identical to trainer.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from fusion.metric import MetricCollection


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_barrier() -> None:
    if _is_distributed():
        dist.barrier()

logger = logging.getLogger(__name__)


class LossAccumulator:
    """Buffers per-step scalar losses and computes summary statistics."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def update(self, losses: dict[str, float]) -> None:
        for k, v in losses.items():
            self._values[k].append(float(v))

    def compute(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for k, vals in self._values.items():
            arr = np.asarray(vals, dtype=np.float64)
            stats[k] = {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std()),
                "min":    float(arr.min()),
                "max":    float(arr.max()),
            }
        return stats

    def reset(self) -> None:
        self._values.clear()


class TrainerV2:
    """Simple PyTorch trainer for PoseFusionModule.

    Parameters
    ----------
    model          : the nn.Module to train (PoseFusionModule).
    optimizer      : optimizer instance.
    train_loader   : training DataLoader.
    losses         : {name: (fn, weight)} where fn(preds, targets) -> scalar.
                     preds[0] is pose_aggr (B, T, P, J, 6).
    val_loader     : optional validation DataLoader.
    max_epochs     : number of epochs.
    grad_clip      : max gradient norm (None to disable).
    use_amp        : enable mixed-precision on CUDA.
    scheduler      : optional LR scheduler (stepped per epoch).
    checkpoint_dir : directory for last.pt / best.pt (None to disable).
    device         : defaults to CUDA if available.
    metrics        : optional MetricCollection; computed at the end of every epoch.
    metric_fn      : callable(preds, targets, metrics) -> None. Required when metrics
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
        curriculum_schedule: dict[int, list[str]] | None = None,
        is_main_process: bool = True,
        train_sampler: DistributedSampler | None = None,
        resume_checkpoint: str | Path | None = None,
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
        self.is_main_process = is_main_process
        self.train_sampler = train_sampler

        if self.metrics is not None and self.metric_fn is None:
            raise ValueError("metrics provided but metric_fn is None. "
                             "Provide a metric_fn(preds, targets, metrics) callable.")

        self._scaler = None
        self._epoch = 0
        self._step = 0
        self._best_val_loss: float | None = None
        self.early_stopping_patience = early_stopping_patience
        self._no_improve = 0

        if self.checkpoint_dir and self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_pt = self.checkpoint_dir / "best.pt"
            if best_pt.exists() and resume_checkpoint is None:
                raise FileExistsError(
                    f"{best_pt} already exists. Use a clean checkpoint directory, "
                    f"rename/remove the existing best.pt, or pass --resume to continue training."
                )

        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint)

    def train(self) -> None:
        if self.is_main_process:
            logger.info(f"Training on {self.device} | losses: {list(self.losses)}")

        _prev_active: set[str] = set()
        for epoch in range(self._epoch, self.max_epochs):
            self._epoch = epoch

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.curriculum_schedule and self.is_main_process:
                cur_active = set(self._active_losses())
                new_losses = cur_active - _prev_active
                if new_losses:
                    logger.info(f"[CURRICULUM] Epoch {epoch}: activating losses {sorted(new_losses)}")
                _prev_active = cur_active

            train_stats, train_metrics = self._run_epoch(train=True)

            _ddp_barrier()

            val_stats, val_metrics = ({}, {})
            improved = False
            should_stop = False

            if self.is_main_process:
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

                if self.early_stopping_patience is not None:
                    if improved:
                        self._no_improve = 0
                    else:
                        self._no_improve += 1
                        if self._no_improve >= self.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch} "
                                        f"(no improvement for {self._no_improve} epochs).")
                            should_stop = True
            else:
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

            if _is_distributed():
                stop_tensor = torch.tensor(int(should_stop), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            _ddp_barrier()

            if should_stop:
                break

        if self.prediction_save_path is not None and self.is_main_process:
            self.save_predictions(self.train_loader, self.prediction_save_path)

    @torch.no_grad()
    def validate(self) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        if self.val_loader is None:
            raise RuntimeError("No val_loader provided.")
        return self._run_epoch(train=False)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = getattr(self.model, "module", self.model)
        raw_model.load_state_dict(state["model"])
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
        """Run the model in eval mode and save predictions to .npz files.

        Saved arrays:
          pose       (T, P, J, 6)  — predicted 6-D body pose
          betas      (P, 10)       — predicted betas (FusionWithBetas only)
          gt_pose    (T, P, J, 6)  — ground-truth body pose (when available)
          gt_valid   (T, P)        — ground-truth validity mask (when available)
          gt_betas   (T, P, 10)    — ground-truth betas (when available)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            scene_name = targets.get("scene_name", ["unknown"])[0] \
                         if isinstance(targets, dict) else "unknown"
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                preds = self._forward(inputs)
            pose_aggr = preds[0]

            def _sq(t): return t.squeeze(0).float().cpu().numpy().astype(np.float32)

            arrays: dict[str, np.ndarray] = {"pose": _sq(pose_aggr)}
            if len(preds) > 1 and isinstance(preds[1], torch.Tensor):
                arrays["betas"] = _sq(preds[1])
            if isinstance(targets, dict):
                if "pose"     in targets: arrays["gt_pose"]  = _sq(targets["pose"])
                if "gt_valid" in targets: arrays["gt_valid"] = _sq(targets["gt_valid"])
                if "shape"    in targets: arrays["gt_betas"] = _sq(targets["shape"])

            out_file = path / f"{scene_name}.npz"
            np.savez_compressed(str(out_file), **arrays)
            logger.info(f"Predictions saved → {out_file}  (keys: {list(arrays)})")

        return path

    def log_losses_to_wandb(self, stats, phase, epoch) -> None:
        import wandb
        wandb.log({f"{phase}_losses/{name}": s["mean"] for name, s in stats.items()}, step=epoch)

    def log_metrics_to_wandb(self, metric_results, phase, epoch) -> None:
        import wandb
        wandb.log(
            {f"{phase}_metrics/{name}": s["mean"] for name, s in metric_results.items()},
            step=epoch,
        )

    def _active_losses(self) -> dict[str, tuple[Callable, float]]:
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
        loader = self.train_loader if train else self.val_loader
        tag = "train" if train else "val"
        self.model.train(train)
        acc = LossAccumulator()
        active_losses = self._active_losses() if train else self.losses

        pbar = tqdm(loader, desc=f"Epoch {self._epoch:04d} [{tag}]",
                    dynamic_ncols=True, leave=False)

        for batch in pbar:
            inputs, targets = self._unpack_batch(batch)

            scene_names = targets.get("scene_name", ["?"])
            scene = scene_names[0] if isinstance(scene_names, list) else scene_names
            pmask = inputs.get("person_mask")
            if pmask is not None:
                pm = pmask[0]
                T_dim, K_dim, _ = pm.shape
                valid_cams  = int(pm.any(0).any(-1).sum().item())
                per_cam_ppl = [int(pm[:, k, :].any(0).sum().item()) for k in range(K_dim)]
                missing_per_cam = [int((~pm[:, k, :].any(-1)).sum().item()) for k in range(K_dim)]
                logger.info(
                    f"[batch] {scene}  T={T_dim}  valid_cams={valid_cams}/{K_dim}  "
                    f"people_per_cam={per_cam_ppl}  missing_frames={missing_per_cam}"
                )

            if train:
                _detect = self._step == 0
                with torch.autograd.detect_anomaly(check_nan=_detect), \
                     torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                    preds = self._forward(inputs)
                    preds = self._append_smplx_joints(preds)

                    # pred NaN check — preds[0] is pose_aggr (B, T, P, J, 6)
                    for _pi, _pt in enumerate(preds):
                        if isinstance(_pt, torch.Tensor) and not _pt.isfinite().all():
                            logger.warning(
                                f"[PRED NaN] {scene}  step={self._step}  "
                                f"preds[{_pi}]  shape={tuple(_pt.shape)}  "
                                f"n_bad={(~_pt.isfinite()).sum().item()}"
                            )

                    step_losses = {name: fn(preds, targets) for name, (fn, _) in active_losses.items()}

                for _lname, _lval in step_losses.items():
                    if not _lval.isfinite():
                        logger.warning(
                            f"[FORWARD NaN] {scene}  step={self._step}  "
                            f"loss={_lname}  value={_lval.item()}"
                        )

                total_loss: torch.Tensor = torch.stack([
                    w * step_losses[n]
                    for n, (_, w) in active_losses.items()
                ]).sum()

                self.optimizer.zero_grad()
                total_loss.backward()

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
                        f"modules={sorted(nan_modules)}  — skipping optimizer step"
                    )
                if self.use_wandb and self.is_main_process and module_grad_norms:
                    import wandb
                    wandb.log(
                        {f"grad_norms/{m}": v for m, v in module_grad_norms.items()},
                        step=self._epoch,
                    )

                if not nan_modules:
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self._step += 1
            else:
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
                    preds = self._forward(inputs)
                    preds = self._append_smplx_joints(preds)
                    step_losses = {name: fn(preds, targets) for name, (fn, _) in active_losses.items()}
                    total_loss = torch.stack([
                        w * step_losses[n]
                        for n, (_, w) in active_losses.items()
                    ]).sum()

            if self.metric_fn is not None and self.metrics is not None:
                with torch.no_grad():
                    self.metric_fn(preds, targets, self.metrics)

            detached = {k: v.item() for k, v in step_losses.items()}
            detached["total"] = total_loss.item()
            detached.update(self._compute_diagnostics(preds, inputs, targets))
            acc.update(detached)

            del preds, step_losses, total_loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

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

    @torch.no_grad()
    def _compute_diagnostics(self, preds, inputs, targets) -> dict[str, float]:
        """Compute diagnostic quantities for logging.

        Keys produced
        -------------
        diag/delta_kin_deg  : mean geodesic angle (°) between pred and input-mean
                              for joints 1-54. Measures how far the model moves from
                              the naive per-camera mean.
        diag/root_error_deg : geodesic angle (°) between predicted root orient and GT.
        """
        import math
        from pytorch3d.transforms import rotation_6d_to_matrix

        diag: dict[str, float] = {}

        if (
            isinstance(inputs, dict)
            and "pose" in inputs
            and "person_mask" in inputs
            and len(preds) > 0
            and isinstance(preds[0], torch.Tensor)
        ):
            pose_in   = inputs["pose"].float()         # (B, T, K, P, J, 6)
            pmask     = inputs["person_mask"].float()  # (B, T, K, P)
            pose_aggr = preds[0].float()               # (B, T, P, J, 6)
            B, T, P, J = pose_aggr.shape[:4]

            # delta_kin_deg: how far pred deviates from naive visibility-weighted mean.
            vis      = pmask.unsqueeze(-1).unsqueeze(-1)          # (B, T, K, P, 1, 1)
            vis_sum  = vis.sum(dim=2).clamp(min=1e-8)
            pose_in_mean = (pose_in * vis).sum(dim=2) / vis_sum   # (B, T, P, J, 6)

            # pose_aggr has J=54 (root stripped by PoseFusionModule).
            # pose_in_mean has J=55 (dataset provides full joints); slice 1: to match.
            R_pred  = rotation_6d_to_matrix(pose_aggr.reshape(-1, 6)).reshape(B, T, P, J, 3, 3)
            R_input = rotation_6d_to_matrix(pose_in_mean[:, :, :, 1:].reshape(-1, 6)).reshape(B, T, P, J, 3, 3)
            R_rel   = torch.matmul(R_pred, R_input.transpose(-2, -1))
            cos_a   = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
            diag["diag/delta_kin_deg"] = float((torch.acos(cos_a) * (180.0 / math.pi)).mean().item())

            # root orient is not predicted by PoseFusionModule (handled by BodyPlacer),
            # so root_error_deg is not computed here.

        return diag

    def _forward(self, inputs: Any) -> tuple:
        """Run the model and return a flat tuple of output tensors.

        For PoseFusionModule (single output): returns (pose_aggr,).
        For FusionWithBetas (two outputs):    returns (pose_aggr, betas_out).

        Filters the inputs dict to the keys the model accepts:
        pose, person_mask, joint_mask, shape. All other keys (e.g. camera,
        body_transl_cam_in) are ignored.
        """
        m = self.model.module if (
            not self.model.training
            and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        ) else self.model

        _MODEL_KEYS = {"pose", "person_mask", "joint_mask"}

        if isinstance(inputs, dict):
            model_inputs = {k: v for k, v in inputs.items() if k in _MODEL_KEYS}
            result = m(**model_inputs)
        elif isinstance(inputs, (list, tuple)):
            result = m(*inputs)
        else:
            result = m(inputs)

        # Normalise: if the model returns a tuple (e.g. FusionWithBetas), use it
        # directly; otherwise wrap the single tensor in a 1-tuple.
        if isinstance(result, tuple):
            return result
        return (result,)

    def _append_smplx_joints(self, preds: Any) -> Any:
        """No-op for PoseFusionModule — no shape or body translation is predicted."""
        return preds

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        batch = self._to_device(batch)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError("Expected batch to be (inputs, targets).")

    def _to_device(self, data: Any) -> Any:
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
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_epoch(self, epoch, train_stats, val_stats, train_metric_results=None, val_metric_results=None) -> None:
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
        val_mean = stats.get("total", {}).get("mean")
        improved = (
            val_mean is not None
            and (self._best_val_loss is None or val_mean < self._best_val_loss)
        )
        if improved:
            self._best_val_loss = val_mean

        if not self.checkpoint_dir:
            return improved

        raw_model = getattr(self.model, "module", self.model)
        state = {
            "epoch":          self._epoch,
            "step":           self._step,
            "model":          raw_model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "best_val_loss":  self._best_val_loss,
        }
        if self.use_wandb:
            try:
                import wandb
                if wandb.run:
                    state["wandb_run_id"] = wandb.run.id
            except Exception:
                pass
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        if self._scaler:
            state["scaler"] = self._scaler.state_dict()

        torch.save(state, self.checkpoint_dir / "last.pt")

        if improved:
            torch.save(state, self.checkpoint_dir / "best.pt")
            logger.info(f"  New best checkpoint (total/mean={val_mean:.5f})")

        return improved
