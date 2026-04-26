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

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
            nn.Linear(6, 2 * embedding_dim), nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(10, 2 * embedding_dim), nn.Tanh(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.camera_encoder = nn.Sequential(
            nn.Linear(8, 2 * embedding_dim), nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.translation_encoder = nn.Sequential(
            nn.Linear(3, 2 * embedding_dim), nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        camera: torch.Tensor,
        body_transl_cam_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pose_emb   : (B, T, K, P, J, D)   all 55 joints — split by caller
        shape_emb  : (B, T, K, P, D)
        camera_emb : (B, T, K, D)
        trans_emb  : (B, T, K, P, D)
        """
        return (
            self.pose_encoder(pose),
            self.shape_encoder(shape),
            self.camera_encoder(camera),
            self.translation_encoder(body_transl_cam_in),
        )


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
        name: str = "?",
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

        # Compile flex_attention with dynamic=False so that T-specific static
        # Triton kernels are generated.  The default dynamic=True kernel has a
        # backward NaN bug for large T (≥ 904) in bfloat16 (job 61715548).
        # First call for each distinct T triggers a ~30 s recompilation; after
        # that the kernel is cached for the rest of the run.
        from torch.nn.attention.flex_attention import flex_attention as _raw_flex
        self._flex_attention = torch.compile(_raw_flex, dynamic=False)

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
        flex_attention = self._flex_attention

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
                return score + torch.log(conf[b, q_idx] * conf[b, kv_idx] + 1e-7)
            score_mod = _score_mod

        # flex_attention runs FlashAttention with the sparse block_mask.
        # Output shape: (N, H, T, Dh).
        # Use _safe_flex_attention when confidence is provided: dead frames
        # (conf == 0) would make all logits -inf → softmax NaN.
        if confidence is not None:
            out: torch.Tensor = _safe_flex_attention(flex_attention, q, k, v, score_mod, block_mask, confidence)  # type: ignore[assignment]
        else:
            out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)  # type: ignore[assignment]

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


def _safe_flex_attention(
    flex_attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod,
    block_mask,
    confidence: torch.Tensor,  # (N, T)
) -> torch.Tensor:
    """Run flex_attention with NaN-safe confidence score_mod.

    When confidence[n, t] == 0 for a query frame, every logit in that row is
    log(0 * ...) = -inf, and softmax(-inf, ...) = NaN.  The fix mirrors
    _safe_mha: temporarily replace dead-query confidences with 1.0 so softmax
    stays finite, run attention, then zero the output for those rows so dead
    frames contribute nothing to the residual stream.

    Parameters
    ----------
    flex_attention_fn : the flex_attention callable (imported inside forward).
    q, k, v          : (N, H, T, Dh) query/key/value tensors.
    score_mod        : callable built from the safe confidence (passed in).
    block_mask       : local-window block mask.
    confidence       : (N, T) original confidence values in [0, 1].

    Returns
    -------
    (N, H, T, Dh)
    """
    dead = confidence == 0  # (N, T) — frames with no detection
    if dead.any():
        safe_conf = confidence.clone()
        safe_conf[dead] = 1.0  # non-zero → log(1*...) finite → softmax well-defined

        def _safe_score_mod(score, b, h, q_idx, kv_idx):
            return score + torch.log(safe_conf[b, q_idx] * safe_conf[b, kv_idx] + 1e-7)

        out = flex_attention_fn(q, k, v, score_mod=_safe_score_mod, block_mask=block_mask)
        # Zero out dead query positions: (N, T) → (N, 1, T, 1) for broadcast over H, Dh
        out = out.masked_fill(dead.unsqueeze(1).unsqueeze(-1), 0.0)
    else:
        out = flex_attention_fn(q, k, v, score_mod=score_mod, block_mask=block_mask)
    return out


class WindowedTemporalAttentionSDPA(nn.Module):
    """Windowed temporal self-attention via explicit query chunking.

    Avoids flex_attention (Triton backward NaN/crash for large T, jobs
    61712668/61716218) and SDPA with float mask (materialises T×T in float32,
    OOM for T≥1166, job 61718955).

    Instead, we loop over query chunks of size `chunk_size`.  For each chunk
    we only load the key/value slice that falls inside the ±W window, so the
    largest intermediate tensor is (N, H, chunk_size, 2W+chunk_size) — a few
    hundred KB regardless of T.  Only standard torch.matmul / softmax / masked_fill
    are used; no Triton, no xformers.

    Memory per chunk: (1, 8, 64, 320) × 2 bytes ≈ 327 KB at W=128, chunk=64.
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
        self.attn_name = name  # stream/layer identity for logging
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
            return linear(x).reshape(N, T, H, Dh).transpose(1, 2)  # (N, H, T, Dh)

        q, k, v = _proj(self.q_proj), _proj(self.k_proj), _proj(self.v_proj)

        # Pre-compute log-confidence once before the loop.
        if confidence is not None:
            dead      = confidence == 0          # (N, T)
            safe_conf = confidence.clone()
            safe_conf[dead] = 1.0                # avoid log(0) for dead query rows
            log_conf  = torch.log(safe_conf + 1e-7)  # (N, T)
        else:
            dead     = None
            log_conf = None

        chunks = []

        for t_start in range(0, T, self.chunk_size):
            t_end   = min(t_start + self.chunk_size, T)
            k_start = max(0, t_start - W)
            k_end   = min(T, t_end   + W)

            q_c = q[:, :, t_start:t_end, :]   # (N, H, C, Dh)
            k_c = k[:, :, k_start:k_end, :]   # (N, H, kW, Dh)
            v_c = v[:, :, k_start:k_end, :]   # (N, H, kW, Dh)

            # Pre-compute masks outside the checkpointed function (no grad needed).
            _q_idx = torch.arange(t_start, t_end, device=x.device).unsqueeze(1)
            _k_idx = torch.arange(k_start, k_end, device=x.device).unsqueeze(0)
            _oow   = (_q_idx - _k_idx).abs() > W                       # (C, kW) bool
            _lq    = log_conf[:, t_start:t_end] if log_conf is not None else None
            _lk    = log_conf[:, k_start:k_end] if log_conf is not None else None
            _dk    = dead[:, k_start:k_end]     if dead    is not None else None

            # Default-arg capture binds the current iteration's values at definition
            # time, avoiding the Python loop-closure bug where a late-binding closure
            # would see the last iteration's values during the backward recompute.
            def _chunk(q_c, k_c, v_c,
                       _oow=_oow, _lq=_lq, _lk=_lk, _dk=_dk):
                attn = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
                attn = attn.masked_fill(_oow.unsqueeze(0).unsqueeze(0), float('-inf'))
                if _lq is not None:
                    cb = _lq.unsqueeze(2) + _lk.unsqueeze(1)
                    cb = cb.masked_fill(_dk.unsqueeze(1).expand_as(cb), float('-inf'))
                    attn = attn + cb.unsqueeze(1)
                all_masked = attn.isinf().all(dim=-1, keepdim=True)    # (N, H, C, 1)
                if all_masked.any():
                    attn = attn.masked_fill(all_masked, 0.0)
                attn = attn.softmax(dim=-1)
                out_c = torch.matmul(attn, v_c)
                if all_masked.any():
                    out_c = out_c.masked_fill(all_masked, 0.0)
                return out_c

            # Gradient checkpointing: recompute attn during backward instead of
            # storing all chunk attn tensors simultaneously.  For T=1166, W=128,
            # chunk=64 this drops ~22 GB → ~1 GB for kin_layer attention.
            chunks.append(
                torch.utils.checkpoint.checkpoint(
                    _chunk, q_c, k_c, v_c, use_reentrant=False,
                )
            )

        # torch.cat preserves requires_grad so gradients flow to q/k/v projections.
        out = torch.cat(chunks, dim=2)                                  # (N, H, T, Dh)

        # Zero dead query rows so they add nothing to the residual stream.
        if dead is not None:
            out = out.masked_fill(dead.unsqueeze(1).unsqueeze(-1), 0.0)

        out = out.transpose(1, 2).reshape(N, T, D)
        return self.drop(self.out_proj(out))


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
        cam_vis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pose_stream   : (B, T, K, P, J, D)  — world-frame pose, updated state
        camera_stream : (B, T, K, D)
        pose_cam_kv   : (B, T, K, P, J, D)  — camera-frame pose, frozen input
                        used as KV in cam→pose direction so each camera attends
                        to its own view of the body (distinct per camera).
        cam_vis       : (B, T, K) float — 1 if camera active, 0 if absent.
                        When provided, absent cameras are hard-masked (-inf) as
                        KV keys in the pose→camera direction so body tokens do
                        not absorb signal from cameras that detected nothing.

        Returns
        -------
        pose_stream, camera_stream (same shapes, updated)
        """
        H = self.pose_to_cam_attn.num_heads

        # Pose → Camera: each joint queries all K camera tokens
        x = self.pose_to_cam_norm(pose_stream)
        x = x.permute(0, 1, 3, 4, 2, 5).contiguous().reshape(B * T * P * J, K, D)
        cam_kv = (
            camera_stream
            .unsqueeze(2).unsqueeze(2)
            .expand(B, T, P, J, K, D)
            .reshape(B * T * P * J, K, D)
        )

        if cam_vis is not None:
            # Build (B*T*P*J, K) hard key mask: 0 for present, -inf for absent.
            # Broadcast camera presence over (P, J) — every body token sees the
            # same set of active cameras.
            flat = cam_vis[:, :, None, None, :].expand(B, T, P, J, K)  # (B,T,P,J,K)
            key_mask = flat.reshape(B * T * P * J, K).new_zeros(B * T * P * J, K)
            key_mask = key_mask.masked_fill(
                flat.reshape(B * T * P * J, K) == 0, float('-inf')
            )
            # Expand to (N*H, K_q, K_kv): same key mask for every query position.
            attn_mask = (
                key_mask.unsqueeze(1).expand(-1, K, -1)   # (N, K_q, K_kv)
                .unsqueeze(1).expand(-1, H, -1, -1)        # (N, H, K_q, K_kv)
                .reshape(B * T * P * J * H, K, K)
            )
            x = _safe_mha(self.pose_to_cam_attn, x, cam_kv, cam_kv, attn_mask, H)
        else:
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


class CameraPoseEncoding(nn.Module):
    """Encodes a camera extrinsic (R, t) into a D-dimensional embedding.

    Uses a minimal 6D geometric descriptor — axis-angle (3D) + translation (3D)
    — then projects through a small MLP.  Axis-angle is the Lie-algebra
    representation of SO(3): direction = rotation axis, magnitude = angle.
    Together with t this covers the full SE(3) pose in an unconstrained,
    minimal parameterisation (no orthogonality constraints to learn).

    Injected into spatial stream tokens before every attention layer so the
    model knows which camera frame each (root_orient, translation) token
    lives in, making cross-view attention geometrically meaningful.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(7, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

    def forward(self, camera: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        camera : (B, T, K, 8)  — [quat(4), trans(3), focal(1)]
                 Quaternion uses pytorch3d convention [w, x, y, z].

        Returns
        -------
        (B, T, K, D)
        """
        B, T, K, _ = camera.shape
        quat  = camera[..., :4]   # (B, T, K, 4)  [w, x, y, z]
        trans = camera[..., 4:7]  # (B, T, K, 3)

        # Feed quaternion + translation directly — no singularity anywhere.
        # Axis-angle conversion was singular at identity (reference camera always
        # has quat=[1,0,0,0]), producing NaN/infinite gradients.
        geom = torch.cat([quat, trans], dim=-1).reshape(B * T * K, 7)  # (BTK, 7)

        return self.mlp(geom).reshape(B, T, K, self.embedding_dim)   # (B, T, K, D)


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

    def __init__(self, embedding_dim: int, num_heads: int, temporal_window: int, dropout: float, name: str = "pose"):
        super().__init__()
        self.joint_attn = JointSelfAttention(embedding_dim, num_heads, dropout)
        self.view_attn = CrossViewAttention(embedding_dim, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttentionSDPA(embedding_dim, num_heads, temporal_window, dropout, name=name)
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

    def __init__(self, embedding_dim: int, num_heads: int, temporal_window: int, dropout: float, name: str = "shape"):
        super().__init__()
        self.view_attn = CrossViewAttention(embedding_dim, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttentionSDPA(embedding_dim, num_heads, temporal_window, dropout, name=name)
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
    """One layer of the camera stream: cross-camera attn → windowed temporal attn → FFN."""

    def __init__(self, embedding_dim: int, num_heads: int, temporal_window: int, dropout: float, name: str = "camera"):
        super().__init__()
        self.view_attn = CrossViewAttention(embedding_dim, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attn = WindowedTemporalAttentionSDPA(embedding_dim, num_heads, temporal_window, dropout, name=name)
        self.ff = FeedForward(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        B: int, T: int, K: int, D: int,
        pe: PositionalEncoding | None,
        dropout: nn.Dropout,
        temporal_conf: torch.Tensor | None = None,
        cam_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, K, D)
        temporal_conf: (B*K, T) binary presence scores, or None.
        cam_attn_mask: (B*T*num_heads, K, K) additive mask — -inf for absent cameras,
                       0 for present ones.  Built by SSTNetwork using
                       _build_confidence_mask(camera_visible.reshape(B*T, K)).
                       All-absent rows are pre-set to 0.0 by _build_confidence_mask
                       and further guarded by _safe_mha inside CrossViewAttention,
                       so no NaN can enter the backward pass.
        """
        # Cross-camera attention: at each frame, all K cameras attend to each other.
        # Reshape (B, T, K, D) → (B*T, K, D) so K is the sequence dimension.
        x_flat = x.reshape(B * T, K, D)
        x_flat = self.view_attn(x_flat, attn_mask=cam_attn_mask)  # residual included
        x = x_flat.reshape(B, T, K, D)

        # Camera temporal attention disabled: cameras are static (fixed tripods),
        # so the camera token is identical at every timestep and temporal attention
        # learns nothing. Re-enable if moving cameras are supported in future.
        # h = self.temporal_norm(x)
        # h = h.permute(0, 2, 1, 3).contiguous().reshape(B * K, T, D)
        # if pe is not None:
        #     h = pe(h)
        # h = self.temporal_attn(h, confidence=temporal_conf)
        # h = h.reshape(B, K, T, D).permute(0, 2, 1, 3).contiguous()
        # x = x + dropout(h)

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

    def forward(self, x: torch.Tensor, person_visible: torch.Tensor, k_dim: int = 2) -> torch.Tensor:
        # x:              (B, T, K, P, ..., D)
        # person_visible: (B, T, K, P) — 1 present, 0 absent
        scores = self.score_net(x)  # (B, T, K, P, ..., 1)

        # Broadcast person_visible to match extra dims (e.g. J=54) and the trailing score dim.
        mask = person_visible
        for _ in range(scores.dim() - mask.dim()):
            mask = mask.unsqueeze(-1)  # (B, T, K, P, ..., 1)

        scores = scores.masked_fill(mask == 0, float('-inf'))
        all_absent = (mask == 0).all(dim=k_dim, keepdim=True)

        # Guard: fill all-absent rows with 0.0 before softmax so NaN never enters the graph.
        # Same pattern as _safe_mha / _safe_flex_attention.
        safe_scores = scores.masked_fill(all_absent, 0.0)
        weights = F.softmax(safe_scores, dim=k_dim)
        weights = weights.masked_fill(all_absent, 0.0)

        return torch.sum(x * weights, dim=k_dim)


class KinToSpatialAttention(nn.Module):
    """One-directional cross-attention: spatial tokens (Q) read from kinematic tokens (KV).

    Kinematic stream is not modified — joints influence root/translation, not the reverse.
    Operates per (camera, person, frame): (B*T*K*P, 2, D) × (B*T*K*P, 54, D).
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        spatial: torch.Tensor,                   # (B, T, K, P, 2, D)
        kinematic: torch.Tensor,                 # (B, T, K, P, 54, D)
        kin_conf: torch.Tensor | None = None,    # (B, T, K, P, 54) confidence
    ) -> torch.Tensor:
        B, T, K, P, J_sp, D = spatial.shape
        J_kin = kinematic.shape[4]
        N = B * T * K * P
        q  = self.norm_q(spatial).reshape(N, J_sp, D)
        kv = self.norm_kv(kinematic).reshape(N, J_kin, D)

        attn_mask = None
        if kin_conf is not None:
            # Soft mask: log(conf_k + 1e-7) broadcast over J_sp query positions
            # shape: (N, 1, J_kin) → (N*H, J_sp, J_kin)
            H = self.attn.num_heads
            log_conf = torch.log(kin_conf.reshape(N, J_kin) + 1e-7)      # (N, J_kin)
            attn_mask = log_conf.unsqueeze(1).expand(-1, J_sp, -1)        # (N, J_sp, J_kin)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, H, -1, -1).reshape(N * H, J_sp, J_kin)

        out = _safe_mha(self.attn, q, kv, kv, attn_mask, self.attn.num_heads)
        return spatial + out.reshape(B, T, K, P, J_sp, D)


class SSTOutputHeads(nn.Module):
    """Final norm + linear decoders for pose, shape, camera.

    Pose decoding is split into two geometrically distinct paths:

    * Kinematic joints 1-54: camera-independent local rotations.
      Camera-weighted pooling across K is valid; residual on the K-mean is valid.

    * Root orient (joint 0) + Translation: camera-dependent quantities,
      now provided in each camera's own frame.
      Decoded per-camera (no pooling), then analytically back-projected to
      world frame using the camera extrinsics, then confidence-weighted mean.
      This is geometrically correct triangulation rather than naive averaging.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        # When True, the camera head delta is zeroed out so GT cameras pass through
        # unchanged. Useful for diagnosing whether root-orientation errors come from
        # the camera head or from the root head itself.
        self.force_gt_cameras: bool = False
        self.pose_pool = CameraWeightedPooling(embedding_dim)
        self.pose_norm = nn.LayerNorm(embedding_dim)
        # Decodes kinematic joints 1-54 (camera-independent, after K-pooling).
        self.pose_head_aggr = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )
        # Decodes root orient delta per-camera (before back-projection to world).
        self.root_head = nn.Sequential(
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
        self.root_norm = nn.LayerNorm(embedding_dim)
        self.trans_norm = nn.LayerNorm(embedding_dim)
        self.camera_rot_trans_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 7),   # [quat(4), trans(3)]
        )
        # Decodes translation delta per-camera (before back-projection to world).
        self.trans_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 3),
        )
        # Near-zero init only the last linear layer of each residual head.
        # First layers keep Kaiming default so the Jacobian through the head is
        # O(1e-2) rather than O((1e-2)^2), giving ~10× more gradient to the transformer.
        # The last layer at std=1e-2 still ensures delta ≈ 0 at initialisation.
        with torch.no_grad():
            for head in [self.pose_head_aggr, self.camera_rot_trans_head]:
                linears = [m for m in head if isinstance(m, nn.Linear)]
                nn.init.normal_(linears[-1].weight, std=1e-2)
                nn.init.zeros_(linears[-1].bias)

    def forward(
        self,
        spatial_stream: torch.Tensor,     # (B, T, K, P, 2, D)
        kin_stream: torch.Tensor,         # (B, T, K, P, 54, D)
        shape_stream: torch.Tensor,       # (B, T, K, P, D)
        camera_stream: torch.Tensor,      # (B, T, K, D)
        pose_input: torch.Tensor,         # (B, T, K, P, 55, 6) — joint 0 in camera frame
        body_transl_cam_in: torch.Tensor, # (B, T, K, P, 3)     — in camera frame
        camera_input: torch.Tensor,       # (B, T, K, 8)
        shape_input: torch.Tensor,        # (B, T, K, P, 10) raw input betas
        person_visible: torch.Tensor,     # (B, T, K, P) float: 1 = visible, 0 = absent
        B: int, T: int, K: int, P: int, J: int, D: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix

        BTK = B * T * K

        # Confidence weights for the mean: (B, T, K, P) → (B, T, P, 1) denominator.
        w_sum = person_visible.sum(dim=2, keepdim=True).clamp(min=1e-8)  # (B, T, 1, P)
        w_sum = w_sum.squeeze(2).unsqueeze(-1)                           # (B, T, P, 1)

        def _nan_check(name, t):
            if not torch.isfinite(t).all():
                print(f"[NaN] {name}: {(~torch.isfinite(t)).sum().item()} non-finite "
                      f"out of {t.numel()} (shape={tuple(t.shape)})")

        # STEP 1: Camera — decoded first so the predicted extrinsics are used for back-projection.
        camera = self.camera_norm(camera_stream)
        camera_flat = camera.reshape(BTK, D)
        rot_trans_delta = self.camera_rot_trans_head(camera_flat).reshape(B, T, K, 7)
        # Camera 0 is the world origin — anchor it to identity rotation + zero translation.
        # Out-of-place to preserve gradient flow through camera_stream.
        cam0_zero = rot_trans_delta.new_zeros(B, T, 1, 7)
        rot_trans_delta = torch.cat([cam0_zero, rot_trans_delta[:, :, 1:, :]], dim=2)
        if self.force_gt_cameras:
            rot_trans_delta = torch.zeros_like(rot_trans_delta)
        # Focal length (position 7) is the GT focal from calibration — pass through unchanged.
        # Normalize the quaternion after adding the delta: input_quat + delta is not unit,
        # and a non-unit quaternion in quaternion_to_matrix produces a non-orthogonal matrix
        # that scales z-coordinates, potentially putting joints behind cameras.
        quat_out  = F.normalize(camera_input[..., :4] + rot_trans_delta[..., :4], dim=-1)
        trans_out = camera_input[..., 4:7] + rot_trans_delta[..., 4:7]
        camera_out = torch.cat([quat_out, trans_out, camera_input[..., 7:8]], dim=-1)
        _nan_check("camera_stream", camera_stream)
        _nan_check("camera_out", camera_out)

        # Extract rotation and translation from the predicted camera.
        # Convention: v_cam = cam_rot_w2c @ v_world + cam_transl_w2c
        # Guard: replace near-zero quaternions (unfilled camera slots) with identity
        # so quaternion_to_matrix never divides by ~0. These slots are already
        # zeroed out in person_visible (see SSTNetwork.forward), so their
        # back-projected translations never enter any loss.
        q = camera_out[..., :4].reshape(BTK, 4)
        q_safe = torch.where(
            (q.pow(2).sum(dim=-1, keepdim=True) > 0.01),
            q,
            q.new_tensor([1., 0., 0., 0.]).expand_as(q),
        )
        cam_rot_w2c    = quaternion_to_matrix(q_safe)  # (BTK, 3, 3)
        cam_transl_w2c = camera_out[..., 4:7]                                       # (B, T, K, 3)
        _nan_check("cam_rot_w2c", cam_rot_w2c)

        # STEP 2: Kinematic joints 1-54
        # Camera-independent local rotations: pool across K then decode delta.
        kin = self.pose_norm(kin_stream)                          # (B, T, K, P, 54, D)
        kin_pooled = self.pose_pool(kin, person_visible, k_dim=2)  # (B, T, P, 54, D)
        kin_delta = self.pose_head_aggr(
            kin_pooled.reshape(B * T * P * 54, D)
        ).reshape(B, T, P, 54, 6)
        # Visibility-weighted mean across K (excludes absent cameras).
        vis = person_visible.unsqueeze(-1).unsqueeze(-1)          # (B, T, K, P, 1, 1)
        vis_sum = vis.sum(dim=2).clamp(min=1e-8)                  # (B, T, P, 1, 1)
        kin_input_mean = (pose_input[:, :, :, :, 1:, :] * vis).sum(dim=2) / vis_sum  # (B, T, P, 54, 6)
        kin_aggr = kin_input_mean + kin_delta
        _nan_check("kin_stream", kin_stream)
        _nan_check("kin_aggr", kin_aggr)

        # STEP 3: Root orient (joint 0): per-camera → back-project → mean
        # Decode per-camera delta in camera frame (no pooling).
        print(f"[HEAD ID] spatial_stream id={id(spatial_stream)} data_ptr={spatial_stream.data_ptr()}")
        if spatial_stream.requires_grad:
            spatial_stream.register_hook(lambda g: print(f"[HEAD GRAD] spatial_stream(inside_heads) max={g.abs().max().item():.3e}"))
        _root_x = spatial_stream[:, :, :, :, 0, :]
        root_feat = _root_x + (self.root_norm(_root_x) - _root_x).detach()  # (B, T, K, P, D)
        if root_feat.requires_grad:
            root_feat.register_hook(lambda g: print(f"[HEAD GRAD] root_feat max={g.abs().max().item():.3e}"))
        root_cam_delta = self.root_head(
            root_feat.reshape(BTK * P, D)
        ).reshape(B, T, K, P, 6)
        root_cam_6d = pose_input[:, :, :, :, 0, :] + root_cam_delta  # (B, T, K, P, 6)
        _nan_check("spatial_stream", spatial_stream)
        _nan_check("root_cam_6d", root_cam_6d)

        # Guard: absent cameras have zero 6D input → rotation_6d_to_matrix would NaN.
        # Replace with identity 6D [1,0,0, 0,1,0] for those slots; they are masked
        # out by person_visible in the weighted mean below so output is unaffected.
        identity_6d = root_cam_6d.new_tensor([1., 0., 0., 0., 1., 0.])
        absent = (person_visible == 0).unsqueeze(-1).expand_as(root_cam_6d)  # (B,T,K,P,6)
        root_cam_6d = torch.where(absent, identity_6d.expand_as(root_cam_6d), root_cam_6d)

        # 6D → rotation matrix, then back-project: R_world = cam_rot_w2c^T @ R_body_cam.
        R_body_cam      = rotation_6d_to_matrix(root_cam_6d.reshape(BTK * P, 6))          # (BTK*P, 3, 3)
        cam_rot_w2c_exp = cam_rot_w2c.unsqueeze(1).expand(BTK, P, 3, 3).reshape(BTK * P, 3, 3)
        R_body_world    = cam_rot_w2c_exp.transpose(-1, -2) @ R_body_cam                  # (BTK*P, 3, 3)
        _nan_check("R_body_cam", R_body_cam)
        _nan_check("R_body_world", R_body_world)

        body_orient_world_per_cam = torch.cat(
            [R_body_world[:, 0, :], R_body_world[:, 1, :]], dim=-1
        ).reshape(B, T, K, P, 6)

        # SO(3) mean via SVD: find the rotation minimising the sum of squared
        # geodesic distances to all K per-camera world-frame estimates.
        # Averaging 6D vectors in R^6 is wrong: the mean of two rotations that
        # are far apart (e.g. due to camera extrinsic error) produces a
        # geometrically arbitrary result after Gram-Schmidt normalisation.
        # SVD gives the exact Fréchet mean on SO(3) and is differentiable.
        from pytorch3d.transforms import matrix_to_rotation_6d
        R_world  = R_body_world.reshape(B, T, K, P, 3, 3)
        vis_mat  = person_visible.unsqueeze(-1).unsqueeze(-1)  # (B, T, K, P, 1, 1)
        M        = (R_world * vis_mat).sum(dim=2)              # (B, T, P, 3, 3)
        # Guard: all-absent (b,t,p) → M=0, SVD undefined. Add identity so SVD
        # yields I for those positions (they are masked out in later losses anyway).
        all_absent = (person_visible.sum(dim=2) == 0)          # (B, T, P)
        if all_absent.any():
            M = M + (all_absent.to(M.dtype).unsqueeze(-1).unsqueeze(-1)
                     * torch.eye(3, device=M.device, dtype=M.dtype))
        U, _, Vt = torch.linalg.svd(M)
        # Ensure det = +1 (rotation, not reflection).
        d         = torch.linalg.det(U) * torch.linalg.det(Vt)   # (B, T, P)
        # Out-of-place: assigning d into a leaf ones-tensor would detach d from the graph.
        ones2     = torch.ones(B, T, P, 2, device=M.device, dtype=M.dtype)
        sign_diag = torch.cat([ones2, d.unsqueeze(-1)], dim=-1)
        R_mean    = U @ torch.diag_embed(sign_diag) @ Vt          # (B, T, P, 3, 3)
        root_aggr = matrix_to_rotation_6d(R_mean)                 # (B, T, P, 6)
        if root_aggr.requires_grad:
            root_aggr.register_hook(lambda g: print(f"[HEAD GRAD] root_aggr max={g.abs().max().item():.3e}"))
        # Diversity check: if all root predictions are identical (collapsed), std≈0.
        # Expected: nonzero std across T frames; near-zero std = mode collapse or bad init.
        with torch.no_grad():
            ra = root_aggr.detach()  # (B, T, P, 6)
            print(f"[ROOT DIV] std={ra.std().item():.4e}  max={ra.max().item():.4e}  min={ra.min().item():.4e}")
        _nan_check("root_aggr", root_aggr)

        # STEP 4: Translation: per-camera → back-project → mean
        _trans_x = spatial_stream[:, :, :, :, 1, :]
        trans_feat = _trans_x + (self.trans_norm(_trans_x) - _trans_x).detach()  # (B, T, K, P, D)
        if trans_feat.requires_grad:
            trans_feat.register_hook(lambda g: print(f"[HEAD GRAD] trans_feat max={g.abs().max().item():.3e}"))
        body_transl_cam_delta = self.trans_head(
            trans_feat.reshape(BTK * P, D)
        ).reshape(B, T, K, P, 3)
        body_transl_cam = body_transl_cam_in + body_transl_cam_delta   # (B, T, K, P, 3) camera frame

        # Back-project: v_world = cam_rot_w2c^T @ (v_cam - cam_transl_w2c).
        # Row-vector convention: v_world = (v_cam - cam_transl_w2c) @ cam_rot_w2c.
        body_transl_cam_flat    = body_transl_cam.reshape(BTK, P, 3)
        cam_transl_w2c_flat     = cam_transl_w2c.reshape(BTK, 1, 3)
        body_transl_world_per_cam = torch.bmm(
            body_transl_cam_flat - cam_transl_w2c_flat, cam_rot_w2c
        ).reshape(B, T, K, P, 3)
        if body_transl_world_per_cam.requires_grad:
            body_transl_world_per_cam.register_hook(lambda g: print(f"[HEAD GRAD] body_transl_world_per_cam max={g.abs().max().item():.3e}"))
        _nan_check("body_transl_cam", body_transl_cam)
        _nan_check("body_transl_world_per_cam", body_transl_world_per_cam)

        # Confidence-weighted mean across cameras.
        body_transl_world = (
            (body_transl_world_per_cam * person_visible.unsqueeze(-1)).sum(dim=2) / w_sum
        )  # (B, T, P, 3) final translation in world frame

        # Concatenate root + kinematic → full world-frame pose
        pose_aggr = torch.cat([root_aggr.unsqueeze(-2), kin_aggr], dim=-2)  # (B, T, P, 55, 6)
        _nan_check("pose_aggr", pose_aggr)
        _nan_check("body_transl_world", body_transl_world)

        # STEP 5: Shape 
        visible_flat = person_visible.reshape(B, T * K, P)
        shape = self.shape_norm(shape_stream)
        shape_all = self.shape_head(
            shape.reshape(B * T * K * P, D)
        ).reshape(B, T * K, P, 10)
        shape_all[visible_flat.unsqueeze(-1).expand_as(shape_all) == 0] = float('nan')
        shape_aggr = torch.nanmean(shape_all, dim=1)                             # (B, P, 10)
        input_flat = shape_input.reshape(B, T * K, P, 10).clone()
        input_flat[visible_flat.unsqueeze(-1).expand_as(input_flat) == 0] = float('nan')
        input_mean = torch.nanmean(input_flat, dim=1).nan_to_num(0.0)
        shape_aggr = torch.where(shape_aggr.isnan(), input_mean, shape_aggr)

        # body_orient_world_per_cam and body_transl_world_per_cam are the per-camera world-frame estimates
        # before aggregation — exposed for the triangulation loss.
        # person_visible is returned so the triangulation loss can mask absent persons.
        # body_transl_cam (preds[7]) is the per-camera camera-frame translation before back-projection,
        # used by TranslationMSELoss to supervise translation without going through predicted cameras.
        # joints_world is appended by Trainer._append_smplx_joints as preds[8].
        return pose_aggr, shape_aggr, camera_out, body_transl_world, body_orient_world_per_cam, body_transl_world_per_cam, person_visible, body_transl_cam

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
        max_cameras: int = 16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temporal_window = temporal_window

        self.dropout = nn.Dropout(dropout)
        self.temporal_pe = PositionalEncoding(max_temporal_len, embedding_dim)

        self.encoder = SSTEncoder(embedding_dim)
        # Joint IDs for the 54 kinematic joints only.
        self.joint_id_embedding = nn.Embedding(num_joints - 1, embedding_dim)
        # Camera IDs: give each camera a unique learnable identity (analogous to joint IDs).
        self.camera_id_embedding = nn.Embedding(max_cameras, embedding_dim)
        # Geometric PE for the spatial stream: encodes the actual camera extrinsic
        # (axis-angle + translation → MLP → D) so cross-view attention on the
        # spatial stream knows which camera frame each token is expressed in.
        self.camera_pose_encoding = CameraPoseEncoding(embedding_dim)

        # Kinematic stream: joint self-attn + cross-view + temporal (no camera cross-attn).
        self.kin_layers = nn.ModuleList([
            PoseStreamLayer(embedding_dim, num_heads, temporal_window, dropout, name=f"kin_L{i}")
            for i in range(num_layers)
        ])
        # Spatial stream: root orient + translation tokens, same layer type (J=2).
        self.spatial_layers = nn.ModuleList([
            PoseStreamLayer(embedding_dim, num_heads, temporal_window, dropout, name=f"spatial_L{i}")
            for i in range(num_layers)
        ])
        # Camera cross-attention only for the spatial stream.
        self.spatial_cam_attns = nn.ModuleList([
            PoseCameraCrossAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # One-directional: kinematic joints → spatial tokens (root + trans).
        self.kin_to_spatial_attns = nn.ModuleList([
            KinToSpatialAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.shape_layers = nn.ModuleList([
            ShapeStreamLayer(embedding_dim, num_heads, temporal_window, dropout, name=f"shape_L{i}")
            for i in range(num_layers)
        ])
        self.camera_layers = nn.ModuleList([
            CameraStreamLayer(embedding_dim, num_heads, temporal_window, dropout, name=f"camera_L{i}")
            for i in range(num_layers)
        ])
        self.output_heads = SSTOutputHeads(embedding_dim)

    @staticmethod
    def _build_confidence_mask(
        flat: torch.Tensor, num_heads: int,
    ) -> torch.Tensor:
        """Build (N*H, S, S) additive soft mask from (N, S) confidence values.

        Entries where either token has zero confidence (absent) become -inf,
        which hard-excludes them from softmax.

        Edge case: when ALL tokens in a row are absent (e.g. a joint occluded
        in every camera at a frame), every entry is -inf and softmax produces
        NaN.  Those rows are set to 0 instead — the attention output is
        meaningless but will be zeroed out downstream by _safe_flex_attention
        (kin_temporal_conf is also 0 for the same positions).
        """
        outer = torch.einsum("bi, bj -> bij", flat, flat)
        mask = torch.log(outer + 1e-7)
        all_dead = flat.sum(dim=-1) == 0  # (N,) rows with no visible tokens
        if all_dead.any():
            mask[all_dead] = 0.0
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
        body_transl_cam_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pose               : [B, T, K, P, J, 6] or [T, K, P, J, 6]  — world-frame (cam-0)
        shape              : [B, T, K, P, betas] or [T, K, P, betas]
        camera             : [B, T, K, 8] or [T, K, 8]  — [quat(4), trans(3), focal_raw(1)]
        joint_mask         : [B, T, K, P, J] or [T, K, P, J]  (confidence ∈ [0,1])
        person_mask        : [B, T, K, P] or [T, K, P]  bool
        body_transl_cam_in : [B, T, K, P, 3] or [T, K, P, 3]  — body root translation in camera frame

        Returns
        -------
        pose_aggr  : [B, T, P, J, 6]
        shape_aggr : [B, P, 10]
        camera     : [B, T, K, 8]
        body_transl_world : [B, T, P, 3]
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
        if body_transl_cam_in.dim() == 4:
            body_transl_cam_in = body_transl_cam_in.unsqueeze(0)

        assert pose.shape[:4] == shape.shape[:4]
        assert pose.shape[:3] == camera.shape[:3]
        assert pose.shape[:5] == joint_mask.shape
        assert pose.shape[:4] == person_mask.shape
        assert pose.shape[:4] == body_transl_cam_in.shape[:4]

        def _nc(name, t):
            if not torch.isfinite(t).all():
                print(f"[NaN] {name}: {(~torch.isfinite(t)).sum().item()}/{t.numel()} "
                      f"non-finite (shape={tuple(t.shape)})")

        _nc("input/pose", pose)
        _nc("input/shape", shape)
        _nc("input/camera", camera)
        _nc("input/joint_mask", joint_mask)
        _nc("input/body_transl_cam_in", body_transl_cam_in)

        pose_emb, shape_emb, camera_emb, trans_emb = self.encoder(
            pose, shape, camera, body_transl_cam_in
        )
        B, T, K, P, J, D = pose_emb.shape

        _nc("emb/pose_emb", pose_emb)
        _nc("emb/shape_emb", shape_emb)
        _nc("emb/camera_emb", camera_emb)
        _nc("emb/trans_emb", trans_emb)

        # Spatial stream: [root_orient(0), translation(1)] — attend cameras.
        # Camera pose encoding injected once here to give each spatial token its geometric
        # camera identity before the first attention layer. Previously this was re-added
        # at every layer iteration (cumulative), which caused the same fixed PE to
        # accumulate L times and overwhelm the learned body features.
        cam_pe = self.camera_pose_encoding(camera).unsqueeze(3).unsqueeze(4)  # (B,T,K,1,1,D)
        spatial_stream = torch.cat([
            pose_emb[:, :, :, :, :1, :],   # root orient  (B,T,K,P,1,D)
            trans_emb.unsqueeze(4),          # translation  (B,T,K,P,1,D)
        ], dim=4) + cam_pe                   # → (B,T,K,P,2,D)

        _spatial_grad_step = [0]
        def _hook_spatial(tag):
            def _h(g):
                print(f"[SPATIAL GRAD] step={_spatial_grad_step[0]}  {tag}  "
                      f"norm={g.float().norm().item():.4e}  "
                      f"max={g.float().abs().max().item():.4e}")
            return _h
        def _reg(t, tag):
            if t.requires_grad:
                t.register_hook(_hook_spatial(tag))
            return t

        # Kinematic stream: 54 body joints — no camera cross-attn.
        kin_stream = pose_emb[:, :, :, :, 1:, :]  # (B,T,K,P,54,D)

        joint_ids = self.joint_id_embedding.weight  # (54, D)
        H = self.num_heads

        # person_visible: (B, T, K, P) float — 1 present, 0 absent.
        # Also mask out cameras whose input quaternion is zero (no observation for
        # that frame): those slots produce garbage back-projections and must be
        # excluded from aggregation and all losses.
        person_visible = person_mask.to(pose_emb.dtype)

        # --- Pose-stream masks ---
        # Multiply joint confidence by binary presence so that absent slots are
        # guaranteed zero even if joint_mask was somehow non-zero there.
        joint_mask_masked = joint_mask * person_visible.unsqueeze(-1)  # (B, T, K, P, J)

        # Spatial mask (J=2): root confidence + 1.0 for translation token.
        ones = torch.ones(B, T, K, P, 1, dtype=joint_mask_masked.dtype, device=joint_mask_masked.device)
        spatial_mask = torch.cat([joint_mask_masked[:, :, :, :, :1], ones], dim=4)  # (B,T,K,P,2)
        sp_joint_mask   = self._build_confidence_mask(spatial_mask.reshape(B * T * K * P, 2), H)
        sp_view_mask    = self._build_confidence_mask(spatial_mask.permute(0, 1, 3, 4, 2).reshape(B * T * P * 2, K), H)
        sp_temporal_conf = spatial_mask.permute(0, 2, 3, 4, 1).reshape(B * K * P * 2, T)

        # Kinematic mask (J=54).
        kin_mask = joint_mask_masked[:, :, :, :, 1:]  # (B,T,K,P,54)
        kin_joint_mask   = self._build_confidence_mask(kin_mask.reshape(B * T * K * P, 54), H)
        kin_view_mask    = self._build_confidence_mask(kin_mask.permute(0, 1, 3, 4, 2).reshape(B * T * P * 54, K), H)
        kin_temporal_conf = kin_mask.permute(0, 2, 3, 4, 1).reshape(B * K * P * 54, T)

        # Cross-view mask for the shape stream.
        person_cross_view_mask_flat = person_visible.permute(0, 1, 3, 2).reshape(B * T * P, K)
        person_cross_view_mask = self._build_confidence_mask(person_cross_view_mask_flat, H)
        person_temporal_conf = person_visible.permute(0, 2, 3, 1).reshape(B * K * P, T)

        # Camera-level presence: camera k is "active" at frame t iff at least one
        # person was detected in it.  Drives temporal masking in the camera stream.
        camera_visible = person_visible.any(dim=-1).to(pose_emb.dtype)   # (B, T, K)
        camera_temporal_conf = camera_visible.permute(0, 2, 1).reshape(B * K, T)

        # Cross-camera attention mask: (B*T*H, K, K) — absent cameras are -inf keys.
        # _build_confidence_mask sets all-absent rows to 0.0 (uniform attn) to prevent
        # NaN softmax; _safe_mha inside CrossViewAttention zeros those outputs afterwards.
        camera_cross_view_mask = self._build_confidence_mask(
            camera_visible.reshape(B * T, K), H
        )

        shape_stream = shape_emb
        camera_stream = camera_emb

        for layer_idx in range(self.num_layers):
            pe = self.temporal_pe if layer_idx == 0 else None

            # Re-inject IDs before every self-attn so each token retains its identity.
            kin_stream    = kin_stream    + joint_ids
            camera_stream = camera_stream + self.camera_id_embedding.weight[:K]
            kin_stream = self.kin_layers[layer_idx](
                kin_stream, B, T, K, P, 54, D, H,
                joint_mask=kin_joint_mask,
                view_mask=kin_view_mask,
                temporal_conf=kin_temporal_conf,
                pe=pe,
                dropout=self.dropout,
            )
            _nc(f"layer{layer_idx}/kin_stream", kin_stream)

            # Spatial stream: root + translation attend each other, cameras, time.
            _reg(spatial_stream, f"L{layer_idx}/0_after_cam_pe")
            spatial_stream = self.spatial_layers[layer_idx](
                spatial_stream, B, T, K, P, 2, D, H,
                joint_mask=sp_joint_mask,
                view_mask=sp_view_mask,
                temporal_conf=sp_temporal_conf,
                pe=pe,
                dropout=self.dropout,
            )
            _reg(spatial_stream, f"L{layer_idx}/1_after_spatial_layer")
            _nc(f"layer{layer_idx}/spatial_stream", spatial_stream)

            camera_stream = self.camera_layers[layer_idx](
                camera_stream, B, T, K, D,
                pe=pe,
                dropout=self.dropout,
                temporal_conf=camera_temporal_conf,
                cam_attn_mask=camera_cross_view_mask,
            )
            _nc(f"layer{layer_idx}/camera_stream", camera_stream)

            spatial_stream, camera_stream = self.spatial_cam_attns[layer_idx](
                spatial_stream, camera_stream, spatial_stream,
                B, T, K, P, 2, D, self.dropout,
                cam_vis=camera_visible,
            )
            _reg(spatial_stream, f"L{layer_idx}/2_after_cam_attn")
            _nc(f"layer{layer_idx}/spatial_stream_post_cam_attn", spatial_stream)
            _nc(f"layer{layer_idx}/camera_stream_post_cam_attn", camera_stream)

            # One-directional: kinematic joints inform spatial tokens (not reverse).
            spatial_stream = self.kin_to_spatial_attns[layer_idx](spatial_stream, kin_stream, kin_conf=kin_mask)
            _reg(spatial_stream, f"L{layer_idx}/3_after_kin2sp")
            print(f"[LOOP ID] L{layer_idx}/3 spatial_stream id={id(spatial_stream)} data_ptr={spatial_stream.data_ptr()}")
            _nc(f"layer{layer_idx}/spatial_stream_post_kin2sp", spatial_stream)

            shape_stream = self.shape_layers[layer_idx](
                shape_stream, B, T, K, P, D,
                pe=pe,
                dropout=self.dropout,
                view_mask=person_cross_view_mask,
                temporal_conf=person_temporal_conf,
            )
            _nc(f"layer{layer_idx}/shape_stream", shape_stream)


        # decode
        return self.output_heads(
            spatial_stream, kin_stream, shape_stream, camera_stream,
            pose, body_transl_cam_in, camera, shape,
            person_visible,
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
