"""
Spatio-Spatio-Temporal (SST) fusion module.

Notation
--------
B = batch size
T = length of the sequence
K = number of cameras
P = maximum number of people in the scene
J = number of joints
D = embedding dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding """

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


class SSTEncoder(nn.Module):
    """Project raw pose / shape / camera parameters into a shared embedding space."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(10, 2 * embedding_dim),
            nn.Tanh(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.camera_encoder = nn.Sequential(
            nn.Linear(8, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        camera: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pose_emb   : (B, T, K, P, J, D)
        shape_emb  : (B, T, K, P, D)
        camera_emb : (B, T, K, D)
        """
        return self.pose_encoder(pose), self.shape_encoder(shape), self.camera_encoder(camera)


class WindowedTemporalAttention(nn.Module):
    """Multi-head attention restricted to a local temporal window via FlashAttention.

    Uses torch.nn.attention.flex_attention so that:
      - Q, K, V are never copied into per-window buffers (no unfold).
      - The attention matrix is never materialised (FlashAttention tiles it).
      - The local-window sparsity is expressed as a block_mask compiled once
        per (T, device) and reused across all forward passes with the same T.
      - The per-frame confidence bias is applied as a score_mod closure,
        computed lazily per FA tile — no O(T²) tensor built upfront.

    Parameters
    ----------
    embedding_dim   : total model dimension (must be divisible by num_heads).
    num_heads       : number of attention heads.
    temporal_window : half-width W; frame t attends to frames [t-W … t+W].
    dropout         : dropout applied to the output projection.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        temporal_window: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, \
            "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.temporal_window = temporal_window

        # Separate projections replace nn.MultiheadAttention.
        # bias=False on Q/K follows the common practice for attention projections.
        self.q_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.drop     = nn.Dropout(dropout)

        # block_mask cache: keyed by (T, device_str) so create_block_mask
        # (which triggers a torch.compile) runs only once per sequence length.
        self._block_mask_cache: dict = {}

    def _get_block_mask(self, T: int, device: torch.device):
        """Return (cached) local-window block_mask for length T on device."""
        from torch.nn.attention.flex_attention import create_block_mask
        key = (T, str(device))
        if key not in self._block_mask_cache:
            W = self.temporal_window
            # This closure is compiled by flex_attention into a sparse block pattern.
            # It returns True for every (query, key) pair that should be computed.
            def local_window(b, h, q_idx, kv_idx):
                return (q_idx - kv_idx).abs() <= W
            # B=None, H=None → mask is broadcast over all batch and head dims.
            self._block_mask_cache[key] = create_block_mask(
                local_window, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device,
            )
        return self._block_mask_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, T, D)
        confidence : (N, T) per-frame confidence scores in [0, 1], or None.
                     When provided, each attention logit is additively biased
                     by log(conf[b, q_idx] * conf[b, kv_idx] + 1e-6), which
                     is the same quantity that _build_confidence_mask produced
                     but now computed lazily inside the FA kernel — no T×T
                     matrix is ever allocated.

        Returns
        -------
        (N, T, D)
        """
        from torch.nn.attention.flex_attention import flex_attention

        N, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        # Project and reshape to (N, H, T, Dh) — the format flex_attention expects.
        # All three tensors are views of x (one matmul each), no window copies.
        def _proj(linear):
            return linear(x).reshape(N, T, H, Dh).transpose(1, 2)  # (N, H, T, Dh)

        q, k, v = _proj(self.q_proj), _proj(self.k_proj), _proj(self.v_proj)

        # block_mask encodes "frame t only attends to [t-W … t+W]".
        # It is compiled once and then reused — subsequent calls with the same T
        # hit the cache and pay zero overhead.
        block_mask = self._get_block_mask(T, x.device)

        # score_mod biases each logit by log(conf_q * conf_k + eps).
        # flex_attention calls this closure per FA tile, not per element,
        # so no T×T tensor is ever created.
        score_mod = None
        if confidence is not None:
            conf = confidence  # captured by closure; shape (N, T)
            def _score_mod(score, b, h, q_idx, kv_idx):  # noqa: E306
                return score + torch.log(conf[b, q_idx] * conf[b, kv_idx])
            score_mod = _score_mod

        # flex_attention runs FlashAttention with the sparse block_mask.
        # Output shape: (N, H, T, Dh).
        out: torch.Tensor = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)  # type: ignore[assignment]

        # Merge heads back and project to D.
        out = out.transpose(1, 2).reshape(N, T, D)
        return self.drop(self.out_proj(out))



def _safe_mha(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    num_heads: int,
) -> torch.Tensor:
    """Run MHA with NaN-safe masking for fully-masked query positions.

    When every key position is masked out for a query token (entire row is -inf),
    softmax(-inf, -inf, ...) = NaN.  This detects such rows, temporarily fills them
    with 0.0 so softmax stays finite, then zeros the output so invalid tokens
    contribute exactly 0.0 to the residual stream — no noise leakage.
    """
    if attn_mask is None:
        out, _ = module(query, key, value)
        return out
    N, S_q = query.shape[0], query.shape[1]
    dead_NH = attn_mask.max(dim=-1).values == float('-inf')  # (N*H, S_q)
    if dead_NH.any():
        safe_mask = attn_mask.clone()
        safe_mask[dead_NH] = 0.0  # uniform attention → no NaN
        out, _ = module(query, key, value, attn_mask=safe_mask)
        # Zero out output for dead query positions (any head flagging dead is enough)
        dead = dead_NH.view(N, num_heads, S_q).any(dim=1)  # (N, S_q)
        out = out.masked_fill(dead.unsqueeze(-1), 0.0)
    else:
        out, _ = module(query, key, value, attn_mask=attn_mask)
    return out


class JointSelfAttention(nn.Module):
    """Self-attention across joints within the same person / camera / frame.

    Operates on the J dimension: (B*T*K*P, J, D) → (B*T*K*P, J, D).
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (N, J, D)
        attn_mask : (N*H, J, J) — additive soft mask from joint confidence.
        """
        h = self.norm(x)
        h = _safe_mha(self.attn, h, h, h, attn_mask, self.attn.num_heads)
        return x + h


class CrossViewAttention(nn.Module):
    """Self-attention across camera views for the same token.

    Operates on the K dimension:
    (B*T*P*J, K, D) → (B*T*P*J, K, D)   (pose)
    (B*T*P,   K, D) → (B*T*P,   K, D)   (shape)
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (N, K, D)
        attn_mask : (N*H, K, K) or None.
        """
        h = self.norm(x)
        h = _safe_mha(self.attn, h, h, h, attn_mask, self.attn.num_heads)
        return x + h

class PoseCameraCrossAttention(nn.Module):
    """Bidirectional cross-attention between pose tokens and camera tokens.

    * Pose → Camera: each joint token queries all K camera tokens.
    * Camera → Pose: each camera token queries all P*J pose tokens
      from its own view.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.pose_to_cam_norm = nn.LayerNorm(embedding_dim)
        self.pose_to_cam_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cam_to_pose_norm = nn.LayerNorm(embedding_dim)
        self.cam_to_pose_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        pose_stream: torch.Tensor,
        camera_stream: torch.Tensor,
        pose_cam_kv: torch.Tensor,
        B: int, T: int, K: int, P: int, J: int, D: int,
        dropout: nn.Dropout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pose_stream   : (B, T, K, P, J, D)  — world-frame pose, updated state
        camera_stream : (B, T, K, D)
        pose_cam_kv   : (B, T, K, P, J, D)  — camera-frame pose, frozen input
                        used as KV in cam→pose direction so each camera attends
                        to its own view of the body (distinct per camera).

        Returns
        -------
        pose_stream, camera_stream (same shapes, updated)
        """
        # Pose → Camera: each joint queries all K camera tokens
        x = self.pose_to_cam_norm(pose_stream)
        x = x.permute(0, 1, 3, 4, 2, 5).contiguous().reshape(B * T * P * J, K, D)
        cam_kv = (
            camera_stream
            .unsqueeze(2).unsqueeze(2)
            .expand(B, T, P, J, K, D)
            .reshape(B * T * P * J, K, D)
        )
        x, _ = self.pose_to_cam_attn(x, cam_kv, cam_kv)
        x = x.reshape(B, T, P, J, K, D).permute(0, 1, 4, 2, 3, 5).contiguous()
        pose_stream = pose_stream + dropout(x)

        # Camera → Pose: each camera queries its own camera-frame pose tokens.
        # Using camera-frame KV (distinct per camera) instead of world-frame pose
        # (identical across cameras) so each camera gets a unique update.
        x = self.cam_to_pose_norm(camera_stream)
        x = x.reshape(B * T * K, 1, D)
        pose_cam_kv_flat = pose_cam_kv.reshape(B * T * K, P * J, D)
        x, _ = self.cam_to_pose_attn(x, pose_cam_kv_flat, pose_cam_kv_flat)
        camera_stream = camera_stream + dropout(x.reshape(B, T, K, D))

        return pose_stream, camera_stream


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
        """x can be any shape (*, D).  Returns same shape."""
        shape = x.shape
        h = self.norm(x).reshape(-1, shape[-1])
        return x + self.ff(h).reshape(shape)

class PoseStreamLayer(nn.Module):
    """One layer of the pose stream: joint attn → cross-view attn → windowed temporal attn → FFN."""

    def __init__(self, embedding_dim: int, num_heads: int, temporal_window: int, dropout: float):
        super().__init__()
        self.joint_attn = JointSelfAttention(embedding_dim, num_heads, dropout)
        self.view_attn = CrossViewAttention(embedding_dim, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttention(embedding_dim, num_heads, temporal_window, dropout)
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
        temporal_conf: (B*K*P*J, T) raw per-frame confidence scores, or None.
                       Passed directly to WindowedTemporalAttention which turns
                       them into a per-logit score_mod inside the FA kernel.

        Returns
        -------
        (B, T, K, P, J, D)
        """
        # Joint attention  (B*T*K*P, J, D)
        x = self.joint_attn(x.reshape(B * T * K * P, J, D), attn_mask=joint_mask)
        x = x.reshape(B, T, K, P, J, D)

        # Cross-view attention  (B*T*P*J, K, D)
        h = self.view_attn(
            x.permute(0, 1, 3, 4, 2, 5).contiguous().reshape(B * T * P * J, K, D),
            attn_mask=view_mask,
        )
        x = h.reshape(B, T, P, J, K, D).permute(0, 1, 4, 2, 3, 5).contiguous()

        # Windowed temporal attention  (B*K*P*J, T, D)
        h = self.temporal_norm(x)
        h = h.permute(0, 2, 3, 4, 1, 5).contiguous().reshape(B * K * P * J, T, D)
        if pe is not None:
            h = pe(h)
        h = self.temporal_attn(h, confidence=temporal_conf)
        h = h.reshape(B, K, P, J, T, D).permute(0, 4, 1, 2, 3, 5).contiguous()
        x = x + dropout(h)

        # FFN
        x = self.ff(x)
        return x


class ShapeStreamLayer(nn.Module):
    """One layer of the shape stream: cross-view attn → windowed temporal attn → FFN."""

    def __init__(self, embedding_dim: int, num_heads: int, temporal_window: int, dropout: float):
        super().__init__()
        self.view_attn = CrossViewAttention(embedding_dim, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttention(embedding_dim, num_heads, temporal_window, dropout)
        self.ff = FeedForward(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        B: int, T: int, K: int, P: int, D: int,
        pe: PositionalEncoding | None,
        dropout: nn.Dropout,
        view_mask: torch.Tensor | None = None,
        temporal_conf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, K, P, D)
        view_mask    : (B*T*P*H, K, K) additive mask from person_cross_view_mask, or None.
        temporal_conf: (B*K*P, T) binary presence scores, or None.
        """
        # Cross-view  (B*T*P, K, D)
        h = self.view_attn(
            x.permute(0, 1, 3, 2, 4).contiguous().reshape(B * T * P, K, D),
            attn_mask=view_mask,
        )
        x = h.reshape(B, T, P, K, D).permute(0, 1, 3, 2, 4).contiguous()

        # Windowed temporal  (B*K*P, T, D)
        h = self.temporal_norm(x)
        h = h.permute(0, 2, 3, 1, 4).contiguous().reshape(B * K * P, T, D)
        if pe is not None:
            h = pe(h)
        h = self.temporal_attn(h, confidence=temporal_conf)
        h = h.reshape(B, K, P, T, D).permute(0, 3, 1, 2, 4).contiguous()
        x = x + dropout(h)

        # FFN
        x = self.ff(x)
        return x


class CameraStreamLayer(nn.Module):
    """One layer of the camera stream: windowed temporal attn → FFN."""

    def __init__(self, embedding_dim: int, num_heads: int, temporal_window: int, dropout: float):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttention(embedding_dim, num_heads, temporal_window, dropout)
        self.ff = FeedForward(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        B: int, T: int, K: int, D: int,
        pe: PositionalEncoding | None,
        dropout: nn.Dropout,
        temporal_conf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, K, D)
        temporal_conf: (B*K, T) binary presence scores, or None.
        """
        h = self.temporal_norm(x)
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * K, T, D)
        if pe is not None:
            h = pe(h)
        h = self.temporal_attn(h, confidence=temporal_conf)
        h = h.reshape(B, K, T, D).permute(0, 2, 1, 3).contiguous()
        x = x + dropout(h)

        x = self.ff(x)
        return x


class CameraWeightedPooling(nn.Module):
    def __init__(self, d_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(d_dim, d_dim // 2),
            nn.ReLU(),
            nn.Linear(d_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, k_dim: int = 2) -> torch.Tensor:
        # x: [B, T, K, P, (J), D]
        scores = self.score_net(x) # [B, T, K, P, (J), 1]
        weights = F.softmax(scores, dim=k_dim)
        return torch.sum(x * weights, dim=k_dim)


class SSTOutputHeads(nn.Module):
    """Final norm + linear decoders for pose, shape, camera."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.pose_pool = CameraWeightedPooling(embedding_dim)
        self.shape_pool = CameraWeightedPooling(embedding_dim)
        self.pose_norm = nn.LayerNorm(embedding_dim)
        # Two independent heads: per-camera features and camera-pooled features
        # have different statistical distributions, so sharing weights creates
        # conflicting gradients.
        self.pose_head_per_cam = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )
        self.pose_head_aggr = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )
        self.shape_norm = nn.LayerNorm(embedding_dim)
        self.shape_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 10),
        )
        self.camera_norm = nn.LayerNorm(embedding_dim)
        # Split into two independent heads so that fov gradients cannot flow
        # into the rotation/translation path through a shared hidden layer.
        self.camera_rot_trans_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 7),   # [quat(4), trans(3)]
        )
        self.camera_focal_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),   # [focal_raw(1)]
        )
        # Small-random init for all residual last layers: delta ≈ 0 at start
        # (small weights × activations ≈ 0), but gradients flow to inner layers
        # from epoch 0 — unlike zero-init which freezes W1/b1 and causes
        # step-wise loss curves.
        with torch.no_grad():
            nn.init.normal_(self.pose_head_per_cam[-1].weight, std=1e-3)
            nn.init.zeros_(self.pose_head_per_cam[-1].bias)
            nn.init.normal_(self.pose_head_aggr[-1].weight, std=1e-3)
            nn.init.zeros_(self.pose_head_aggr[-1].bias)
            nn.init.normal_(self.camera_rot_trans_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.camera_rot_trans_head[-1].bias)
            nn.init.normal_(self.camera_focal_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.camera_focal_head[-1].bias)

    def forward(
        self,
        pose_stream: torch.Tensor,
        shape_stream: torch.Tensor,
        camera_stream: torch.Tensor,
        camera_input: torch.Tensor,
        pose_input: torch.Tensor,
        B: int, T: int, K: int, P: int, J: int, D: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose = self.pose_norm(pose_stream)

        # Residual: predict delta on top of SAM3D input so the model starts at
        # SAM3D quality (delta≈0 at init) and only needs to learn corrections.
        pose_delta_per_cam = self.pose_head_per_cam(pose.reshape(B * T * K * P * J, D)).reshape(B, T, K, P, J, 6)
        pose_per_cam = pose_input + pose_delta_per_cam

        pose_aggr_feat = self.pose_pool(pose, k_dim=2)  # (B, T, P, J, D)
        pose_delta_aggr = self.pose_head_aggr(pose_aggr_feat.reshape(B * T * P * J, D)).reshape(B, T, P, J, 6)
        pose_aggr = pose_input.mean(dim=2) + pose_delta_aggr  # mean across K as baseline

        shape = self.shape_norm(shape_stream)

        shape_per_cam = self.shape_head(shape.reshape(B * T * K * P, D)).reshape(B, T, K, P, 10)
        shape_aggr_feat = self.shape_pool(shape, k_dim=2) # [B, T, P, D]
        shape_aggr = self.shape_head(shape_aggr_feat.reshape(B * T * P, D)).reshape(B, T, P, 10)

        camera = self.camera_norm(camera_stream)
        camera_flat = camera.reshape(B * T * K, D)
        rot_trans_delta = self.camera_rot_trans_head(camera_flat).reshape(B, T, K, 7)
        # Multiplicative residual in log space: predict log(f_pred / f_input).
        # At init (zero-init last layer) log_ratio=0 → f_pred=f_input (SAM3D value).
        # Avoids bfloat16 precision loss: adding a small delta to f≈1155 rounds
        # back to f (step size is 8px in bf16), but multiplying by exp(log_ratio)
        # works at any scale since log_ratio itself is near zero.
        log_focal_ratio = self.camera_focal_head(camera_flat).reshape(B, T, K, 1)
        focal_pred = camera_input[..., 7:8] * torch.exp(log_focal_ratio)
        # Additive residual for rot/trans (values in [-1,1], bf16 precision is fine).
        camera = torch.cat([
            camera_input[..., :7] + rot_trans_delta,
            focal_pred,
        ], dim=-1)

        return pose_aggr, shape_aggr, camera, pose_per_cam, shape_per_cam

class SSTNetwork(nn.Module):
    """Spatio-Spatio-Temporal attention module that fuses parameters across
    views and time.

    Composed of:
        * :class:`SSTEncoder`               — input projection
        * :class:`PoseStreamLayer` ×L        — joint + view + temporal + FFN
        * :class:`ShapeStreamLayer` ×L       — view + temporal + FFN
        * :class:`CameraStreamLayer` ×L      — temporal + FFN
        * :class:`PoseCameraCrossAttention` ×L      — bidirectional cross-attn (every layer)
        * :class:`SSTOutputHeads`            — final decoders
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        max_temporal_len: int = 4096,
        dropout: float = 0.1,
        temporal_window: int = 128,
        num_joints: int = 55,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temporal_window = temporal_window

        self.dropout = nn.Dropout(dropout)
        self.temporal_pe = PositionalEncoding(max_temporal_len, embedding_dim)

        self.encoder = SSTEncoder(embedding_dim)
        # Learnable joint identity embedding: gives each of the J joints a unique
        # fingerprint that persists through all attention layers via residual
        # connections.  Without this, JointSelfAttention averages all joint tokens
        # together (over-smoothing) so the pose head sees J identical embeddings
        # and outputs J identical rotations.
        self.joint_id_embedding = nn.Embedding(num_joints, embedding_dim)

        self.pose_layers = nn.ModuleList([
            PoseStreamLayer(embedding_dim, num_heads, temporal_window, dropout)
            for _ in range(num_layers)
        ])
        self.shape_layers = nn.ModuleList([
            ShapeStreamLayer(embedding_dim, num_heads, temporal_window, dropout)
            for _ in range(num_layers)
        ])
        self.camera_layers = nn.ModuleList([
            CameraStreamLayer(embedding_dim, num_heads, temporal_window, dropout)
            for _ in range(num_layers)
        ])
        self.cross_attns = nn.ModuleList([
            PoseCameraCrossAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_heads = SSTOutputHeads(embedding_dim)

    @staticmethod
    def _build_confidence_mask(
        flat: torch.Tensor, num_heads: int,
    ) -> torch.Tensor:
        """Build (N*H, S, S) additive soft mask from (N, S) confidence values.

        Entries where either token has zero confidence (absent) become -inf,
        which hard-excludes them from softmax.
        """
        outer = torch.einsum("bi, bj -> bij", flat, flat)
        mask = torch.log(outer)
        return mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(
            flat.shape[0] * num_heads, flat.shape[1], flat.shape[1]
        )

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        camera: torch.Tensor,
        joint_mask: torch.Tensor,
        person_mask: torch.Tensor,
        pose_cam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pose        : [B, T, K, P, J, 6] or [T, K, P, J, 6]  — world-frame (cam-0)
        shape       : [B, T, K, P, betas] or [T, K, P, betas]
        camera      : [B, T, K, 8] or [T, K, 8]  — [quat(4), trans(3), focal_raw(1)]
        joint_mask  : [B, T, K, P, J] or [T, K, P, J]  (confidence ∈ [0,1])
        person_mask : [B, T, K, P] or [T, K, P]  bool — True when person p is
                      detected in camera k at frame t.  Used to mask the shape
                      and camera streams where no observation exists.
        pose_cam    : [B, T, K, P, J, 6] or [T, K, P, J, 6]  — camera-frame pose,
                      used as frozen KV in cam→pose cross-attention.  Falls back
                      to ``pose`` (world-frame) when None.

        Returns
        -------
        pose   : [B, T, P, J, 6]
        shape  : [B, T, P, 10]
        camera : [B, T, K, 8]  — [quat(4), trans(3), focal_raw(1)]
        """
        # ensure batch dim
        if pose.dim() == 5:
            pose = pose.unsqueeze(0)
        if shape.dim() == 4:
            shape = shape.unsqueeze(0)
        if camera.dim() == 3:
            camera = camera.unsqueeze(0)
        if joint_mask.dim() == 4:
            joint_mask = joint_mask.unsqueeze(0)
        if person_mask.dim() == 3:
            person_mask = person_mask.unsqueeze(0)
        if pose_cam is None:
            pose_cam = pose  # fallback: no distinction (world == camera frame)
        elif pose_cam.dim() == 5:
            pose_cam = pose_cam.unsqueeze(0)

        assert pose.shape[:4] == shape.shape[:4]
        assert pose.shape[:3] == camera.shape[:3]
        assert pose.shape[:5] == joint_mask.shape
        assert pose.shape[:4] == person_mask.shape

        # encode
        pose_emb, shape_emb, camera_emb = self.encoder(pose, shape, camera)
        # Encode camera-frame poses with the same pose encoder (shared weights).
        # Result is used as frozen KV in cam→pose cross-attention.
        pose_cam_emb = self.encoder.pose_encoder(pose_cam)
        B, T, K, P, J, D = pose_emb.shape

        joint_ids = self.joint_id_embedding.weight  # (J, D)
        # Inject once into the frozen cam-frame KV (it never changes across layers).
        pose_cam_emb = pose_cam_emb + joint_ids
        H = self.num_heads

        # person_visible: (B, T, K, P) float — 1 present, 0 absent.
        person_visible = person_mask.to(pose_emb.dtype)

        # --- Pose-stream masks ---
        # Multiply joint confidence by binary presence so that absent slots are
        # guaranteed zero even if joint_mask was somehow non-zero there.
        joint_mask_masked = joint_mask * person_visible.unsqueeze(-1)  # (B, T, K, P, J)

        joint_mask_flat = joint_mask_masked.reshape(B * T * K * P, J)
        pose_joint_mask = self._build_confidence_mask(joint_mask_flat, H)

        view_mask_flat = joint_mask_masked.permute(0, 1, 3, 4, 2).reshape(B * T * P * J, K)
        pose_view_mask = self._build_confidence_mask(view_mask_flat, H)

        # For pose temporal attention: pass raw (B*K*P*J, T) confidence scores.
        # WindowedTemporalAttention converts these to a per-logit score_mod
        # inside the FlashAttention kernel — no T×T tensor is ever allocated.
        pose_temporal_conf = joint_mask_masked.permute(0, 2, 3, 4, 1).reshape(B * K * P * J, T)

        # Cross-view mask for the shape stream: for each (batch, time, person),
        # which cameras can attend to each other.
        # Flat shape (B*T*P, K) → additive (B*T*P*H, K, K) via log outer-product;
        # pairs where either camera is missing the person get -inf.
        person_cross_view_mask_flat = person_visible.permute(0, 1, 3, 2).reshape(B * T * P, K)
        person_cross_view_mask = self._build_confidence_mask(person_cross_view_mask_flat, H)

        # Temporal conf for the shape stream: for each (batch, camera, person),
        # which frames are present.  Passed as (B*K*P, T) to WindowedTemporalAttention,
        # which lazily computes log(conf_q * conf_k + eps) per logit inside FA.
        person_temporal_conf = person_visible.permute(0, 2, 3, 1).reshape(B * K * P, T)

        # Camera-level presence: camera k is "active" at frame t iff at least one
        # person was detected in it.  Drives temporal masking in the camera stream.
        camera_visible = person_visible.any(dim=-1).to(pose_emb.dtype)   # (B, T, K)
        camera_temporal_conf = camera_visible.permute(0, 2, 1).reshape(B * K, T)

        # layer loop
        pose_stream = pose_emb
        shape_stream = shape_emb
        camera_stream = camera_emb

        for layer_idx in range(self.num_layers):
            # Re-inject joint identity before every JointSelfAttention.
            pose_stream = pose_stream + joint_ids

            pe = self.temporal_pe if layer_idx == 0 else None

            pose_stream = self.pose_layers[layer_idx](
                pose_stream, B, T, K, P, J, D, H,
                joint_mask=pose_joint_mask,
                view_mask=pose_view_mask,
                temporal_conf=pose_temporal_conf,
                pe=pe,
                dropout=self.dropout,
            )

            shape_stream = self.shape_layers[layer_idx](
                shape_stream, B, T, K, P, D,
                pe=pe,
                dropout=self.dropout,
                view_mask=person_cross_view_mask,
                temporal_conf=person_temporal_conf,
            )

            camera_stream = self.camera_layers[layer_idx](
                camera_stream, B, T, K, D,
                pe=pe,
                dropout=self.dropout,
                temporal_conf=camera_temporal_conf,
            )

            # Pose - Camera cross-attention every layer
            pose_stream, camera_stream = self.cross_attns[layer_idx](
                pose_stream, camera_stream, pose_cam_emb,
                B, T, K, P, J, D, self.dropout,
            )


        # decode
        return self.output_heads(
            pose_stream, shape_stream, camera_stream, camera, pose,
            B, T, K, P, J, D,
        )


    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
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
            f"Frozen     : {counts['frozen']:>12,} params",
        ]
        return "\n".join(lines)
