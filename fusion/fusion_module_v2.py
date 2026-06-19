"""
PoseFusionModule — lightweight multi-view pose fusion.

Architecture
------------
  1. Pose encoder      : 6-D rotation → D-dimensional token per joint
  2. PoseStreamLayers  : joint self-attn → cross-view attn → windowed temporal attn → FFN
  3. Camera mean pool  : visibility-weighted mean over K → (B, T, P, J, D)
  4. Decoder           : D → 6-D rotation (residual on visibility-weighted input mean)

Notation
--------
B = batch size
T = sequence length
K = number of cameras
P = maximum number of people
J = number of joints (default 55)
D = embedding dimension
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (self-contained copies, independent of fusion_module.py)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to the time dimension."""

    def __init__(self, max_len: int, model_dim: int):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


def _safe_mha(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    num_heads: int,
) -> torch.Tensor:
    """Run MHA with NaN-safe masking for fully-masked query positions."""
    if attn_mask is None:
        out, _ = module(query, key, value)
        return out
    N, S_q = query.shape[0], query.shape[1]
    dead_NH = attn_mask.max(dim=-1).values == float('-inf')
    if dead_NH.any():
        safe_mask = attn_mask.clone()
        safe_mask[dead_NH] = 0.0
        out, _ = module(query, key, value, attn_mask=safe_mask)
        dead = dead_NH.view(N, num_heads, S_q).any(dim=1)
        out = out.masked_fill(dead.unsqueeze(-1), 0.0)
    else:
        out, _ = module(query, key, value, attn_mask=attn_mask)
    return out


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(expansion * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = self.norm(x).reshape(-1, shape[-1])
        return x + self.ff(h).reshape(shape)


class JointSelfAttention(nn.Module):
    """Self-attention across J joints within the same (batch, time, camera, person) slot."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, J, D)"""
        h = self.norm(x)
        h = _safe_mha(self.attn, h, h, h, attn_mask, self.attn.num_heads)
        return x + h


class CrossViewAttention(nn.Module):
    """Self-attention across K camera views for the same token."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, K, D)"""
        h = self.norm(x)
        h = _safe_mha(self.attn, h, h, h, attn_mask, self.attn.num_heads)
        return x + h


class WindowedTemporalAttentionSDPA(nn.Module):
    """Windowed temporal self-attention via explicit query chunking.

    Avoids flex_attention and SDPA with float mask (both problematic for large T).
    Loops over chunks of size `chunk_size`; for each chunk only loads the K/V slice
    inside ±W, so the largest intermediate tensor is (N, H, chunk_size, 2W+chunk).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        temporal_window: int = 128,
        dropout: float = 0.0,
        chunk_size: int = 64,
        name: str = "?",
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.temporal_window = temporal_window
        self.chunk_size = chunk_size
        logger.info(
            f"WindowedTemporalAttentionSDPA[{name}]: gradient checkpointing ENABLED "
            f"(chunk_size={chunk_size}, window={temporal_window})"
        )
        self.q_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, T, D)
        confidence : (N, T) per-frame confidence in [0, 1], or None.

        Returns
        -------
        (N, T, D)
        """
        N, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim
        W       = self.temporal_window
        scale   = Dh ** -0.5

        def _proj(linear):
            return linear(x).reshape(N, T, H, Dh).transpose(1, 2)

        q, k, v = _proj(self.q_proj), _proj(self.k_proj), _proj(self.v_proj)

        if confidence is not None:
            dead      = confidence == 0
            safe_conf = confidence.clone()
            safe_conf[dead] = 1.0
            log_conf  = torch.log(safe_conf + 1e-7)
        else:
            dead     = None
            log_conf = None

        chunks = []
        for t_start in range(0, T, self.chunk_size):
            t_end   = min(t_start + self.chunk_size, T)
            k_start = max(0, t_start - W)
            k_end   = min(T, t_end   + W)

            q_c = q[:, :, t_start:t_end, :]
            k_c = k[:, :, k_start:k_end, :]
            v_c = v[:, :, k_start:k_end, :]

            _q_idx = torch.arange(t_start, t_end, device=x.device).unsqueeze(1)
            _k_idx = torch.arange(k_start, k_end, device=x.device).unsqueeze(0)
            _oow   = (_q_idx - _k_idx).abs() > W
            _lq    = log_conf[:, t_start:t_end] if log_conf is not None else None
            _lk    = log_conf[:, k_start:k_end] if log_conf is not None else None
            _dk    = dead[:, k_start:k_end]     if dead    is not None else None

            def _chunk(q_c, k_c, v_c, _oow=_oow, _lq=_lq, _lk=_lk, _dk=_dk):
                attn = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
                attn = attn.masked_fill(_oow.unsqueeze(0).unsqueeze(0), float('-inf'))
                if _lq is not None:
                    cb = _lq.unsqueeze(2) + _lk.unsqueeze(1)
                    cb = cb.masked_fill(_dk.unsqueeze(1).expand_as(cb), float('-inf'))
                    attn = attn + cb.unsqueeze(1)
                all_masked = attn.isinf().all(dim=-1, keepdim=True)
                if all_masked.any():
                    attn = attn.masked_fill(all_masked, 0.0)
                attn  = attn.softmax(dim=-1)
                out_c = torch.matmul(attn, v_c)
                if all_masked.any():
                    out_c = out_c.masked_fill(all_masked, 0.0)
                return out_c

            chunks.append(
                torch.utils.checkpoint.checkpoint(
                    _chunk, q_c, k_c, v_c, use_reentrant=False,
                )
            )

        out = torch.cat(chunks, dim=2)

        if dead is not None:
            out = out.masked_fill(dead.unsqueeze(1).unsqueeze(-1), 0.0)

        out = out.transpose(1, 2).reshape(N, T, D)
        return self.drop(self.out_proj(out))


class PoseStreamLayer(nn.Module):
    """One layer: joint self-attn → cross-view attn → windowed temporal attn → FFN."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        temporal_window: int,
        dropout: float,
        name: str = "pose",
    ):
        super().__init__()
        self.joint_attn    = JointSelfAttention(embedding_dim, num_heads, dropout)
        self.view_attn     = CrossViewAttention(embedding_dim, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttentionSDPA(
            embedding_dim, num_heads, temporal_window, dropout, name=name,
        )
        self.ff = FeedForward(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        B: int, T: int, K: int, P: int, J: int, D: int, H: int,
        joint_mask: torch.Tensor,
        view_mask: torch.Tensor,
        temporal_conf: torch.Tensor | None,
        pe: PositionalEncoding | None,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, K, P, J, D)
        joint_mask   : (B*T*K*P*H, J, J)   additive mask
        view_mask    : (B*T*P*J*H, K, K)   additive mask
        temporal_conf: (B*K*P*J, T)         per-frame confidence, or None

        Returns
        -------
        (B, T, K, P, J, D)
        """
        # Joint self-attention: treat every (b,t,k,p) as an independent sequence of J tokens.
        x = self.joint_attn(x.reshape(B * T * K * P, J, D), attn_mask=joint_mask)
        x = x.reshape(B, T, K, P, J, D)

        # Cross-view attention: treat every (b,t,p,j) as a sequence of K tokens.
        h = self.view_attn(
            x.permute(0, 1, 3, 4, 2, 5).contiguous().reshape(B * T * P * J, K, D),
            attn_mask=view_mask,
        )
        x = h.reshape(B, T, P, J, K, D).permute(0, 1, 4, 2, 3, 5).contiguous()

        # Windowed temporal attention: treat every (b,k,p,j) as a sequence of T tokens.
        h = self.temporal_norm(x)
        h = h.permute(0, 2, 3, 4, 1, 5).contiguous().reshape(B * K * P * J, T, D)
        if pe is not None:
            h = pe(h)
        h = self.temporal_attn(h, confidence=temporal_conf)
        h = h.reshape(B, K, P, J, T, D).permute(0, 4, 1, 2, 3, 5).contiguous()
        x = x + dropout(h)

        return self.ff(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class PoseFusionModule(nn.Module):
    """Fuse per-camera SMPL-X pose estimates into a single world-frame prediction.

    Parameters
    ----------
    embedding_dim    : token dimension D.
    num_heads        : attention heads (must divide embedding_dim).
    num_layers       : number of PoseStreamLayer blocks.
    max_temporal_len : maximum sequence length T for sinusoidal PE.
    dropout          : dropout probability.
    temporal_window  : half-width W of the local temporal attention window.
    num_joints       : J, number of SMPL-X joints (default 54, root excluded).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        max_temporal_len: int = 4096,
        dropout: float = 0.1,
        temporal_window: int = 128,
        num_joints: int = 54,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_joints = num_joints

        self.dropout = nn.Dropout(dropout)
        self.temporal_pe = PositionalEncoding(max_temporal_len, embedding_dim)

        # Per-joint learnable identity embedding, injected once at the input.
        # Joint indices are semantically consistent across all SMPL-X bodies
        # (joint 5 = right knee everywhere), so this is a meaningful signal.
        # No camera ID embedding: cameras have no consistent identity across scenes.
        self.joint_id_embedding = nn.Embedding(num_joints, embedding_dim)

        # Input encoder: project each joint's 6-D rotation into the token space.
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

        self.layers = nn.ModuleList([
            PoseStreamLayer(
                embedding_dim, num_heads, temporal_window, dropout,
                name=f"pose_L{i}",
            )
            for i in range(num_layers)
        ])

        # Output head: LayerNorm → MLP → direct 6-D rotation prediction.
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )

    @staticmethod
    def _build_confidence_mask(flat: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Build (N*H, S, S) additive soft mask from (N, S) confidence values.

        Pairs where either token has zero confidence become -inf (hard exclusion).
        Rows where every token is absent are reset to 0 so softmax stays finite.
        """
        outer = torch.einsum("bi,bj->bij", flat, flat)
        mask = torch.log(outer + 1e-7)
        all_dead = flat.sum(dim=-1) == 0
        if all_dead.any():
            mask[all_dead] = 0.0
        return mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(
            flat.shape[0] * num_heads, flat.shape[1], flat.shape[1]
        )

    def forward(
        self,
        pose: torch.Tensor,
        person_mask: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pose        : (B, T, K, P, J, 6) — per-camera 6-D rotation estimates.
        person_mask : (B, T, K, P) bool or float — 1 where a person was detected.
        joint_mask  : (B, T, K, P, J) float in [0, 1] — per-joint confidence.
                      If None, person_mask is broadcast across all J joints.

        Returns
        -------
        pose_aggr   : (B, T, P, J, 6) — fused 6-D rotation estimate.
        """
        if pose.dim() == 5:
            pose = pose.unsqueeze(0)
            person_mask = person_mask.unsqueeze(0)
            if joint_mask is not None:
                joint_mask = joint_mask.unsqueeze(0)

        # Drop root joint if caller passes all 55 SMPL-X joints; root orientation
        # is handled separately by BodyPlacer.
        if pose.shape[-2] == 55:
            pose = pose[..., 1:, :]
            if joint_mask is not None:
                joint_mask = joint_mask[..., 1:]

        B, T, K, P, J, _ = pose.shape
        D = self.embedding_dim
        H = self.num_heads

        person_visible = person_mask.to(pose.dtype)    # (B, T, K, P) float

        # Per-joint confidence: multiply by binary presence so absent cameras are
        # always exactly zero, regardless of what joint_mask contains there.
        if joint_mask is None:
            conf = person_visible.unsqueeze(-1).expand(B, T, K, P, J)
        else:
            conf = joint_mask * person_visible.unsqueeze(-1)       # (B, T, K, P, J)

        # ── Encode ───────────────────────────────────────────────────────────
        x = self.pose_encoder(pose)                                # (B, T, K, P, J, D)

        # ── Build attention masks (built once, shared across all layers) ──────
        # Joint self-attention: each slot (b,t,k,p) sees J×J interactions.
        joint_attn_mask = self._build_confidence_mask(
            conf.reshape(B * T * K * P, J), H,
        )  # (B*T*K*P*H, J, J)

        # Cross-view attention: each slot (b,t,p,j) sees K×K interactions.
        view_attn_mask = self._build_confidence_mask(
            conf.permute(0, 1, 3, 4, 2).reshape(B * T * P * J, K), H,
        )  # (B*T*P*J*H, K, K)

        # Temporal confidence: each slot (b,k,p,j) gets a T-length confidence vector.
        temporal_conf = conf.permute(0, 2, 3, 4, 1).reshape(B * K * P * J, T)
        # (B*K*P*J, T)

        # ── Inject joint identity once at the input ──────────────────────────
        joint_ids = self.joint_id_embedding.weight[:J].view(1, 1, 1, 1, J, D)
        x = x + joint_ids

        # ── PoseStreamLayer blocks ────────────────────────────────────────────
        for layer_idx, layer in enumerate(self.layers):
            # Temporal PE injected before the first layer only.
            pe = self.temporal_pe if layer_idx == 0 else None

            x = layer(
                x, B, T, K, P, J, D, H,
                joint_attn_mask,
                view_attn_mask,
                temporal_conf,
                pe,
                self.dropout,
            )

        # ── Visibility-weighted mean pool over cameras ────────────────────────
        vis     = person_visible.unsqueeze(-1).unsqueeze(-1)       # (B, T, K, P, 1, 1)
        vis_sum = vis.sum(dim=2).clamp(min=1e-8)                   # (B, T,    P, 1, 1)
        x_pooled = (x * vis).sum(dim=2) / vis_sum                  # (B, T, P, J, D)

        # ── Decode (direct prediction) ────────────────────────────────────────
        pose_aggr = self.decoder(
            self.output_norm(x_pooled).reshape(B * T * P * J, D)
        ).reshape(B, T, P, J, 6)

        return pose_aggr

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def summary(self, indent: int = 2) -> str:
        lines = [type(self).__name__]
        name_width = max((len(n) for n, _ in self.named_children()), default=0)
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            lines.append(f"{' ' * indent}{name:<{name_width}} : {n:>12,} params")
        counts = self.count_parameters()
        sep = "-" * (indent + name_width + 22)
        lines += [
            sep,
            f"Total      : {counts['total']:>12,} params",
            f"Trainable  : {counts['trainable']:>12,} params",
        ]
        return "\n".join(lines)


