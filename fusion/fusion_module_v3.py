"""
PoseFusionModuleV3 — residual multi-view pose fusion.

Self-contained copy of fusion_module_v2.py. The attention stack is IDENTICAL;
only the way the module relates to the multi-view mean changed.

Why v3 exists
-------------
Measured on RICH test (52 scenes, one protocol, median scale smoothing):

    chordal mean   WA-100 47.6  W-100 67.7  PA 26.5  RTE 0.98
    v2 module      WA-100 50.4  W-100 70.4  PA 30.4  RTE 1.00

The trained v2 module is beaten by a parameter-free chordal mean of the
per-camera estimates, on every metric and in 48 of 52 scenes. The
explainability study explains why: v2 behaves like a per-joint linear filter
on that same mean (rank-1 model explains 88% of its 54x54 influence matrix,
per-joint R^2 ~ 0.9) — but it has to REBUILD the mean from scratch through a
decoder on every forward pass, and pays for every reconstruction error. It also
learned per-joint gains > 1 on the hips, which restores SAM3D's attenuated hip
rotations on average while multiplying their view-correlated bias.

v3 removes reconstruction from the job entirely:

    the mean is computed in closed form, SUBTRACTED from the inputs,
    and ADDED BACK to the outputs — the network only ever sees, and only ever
    emits, CORRECTIONS.

Two changes, both flag-gated so the v2 behaviour remains reachable:

  CHANGE A — residual head (`residual_head=True`)
      The decoder's 6-D output is read as a CORRECTION rotation rather than as
      the pose itself, applied in the joint's own body frame:

          R_out = R_mean @ R_delta,      R_delta = gram_schmidt(decoder(...))

      The final decoder layer is initialised to ZERO weights with the IDENTITY
      6-D vector [1,0,0, 0,1,0] as bias, so at step 0 R_delta == I and the module
      reproduces the chordal mean EXACTLY. Training therefore starts at the
      baseline's accuracy and every gradient step is an attempt to improve on it
      — the failure mode above becomes structurally hard.

      Zero weights make the gradient w.r.t. everything UPSTREAM of this layer
      vanish at step 0 (it is W^T dL/dout). This is not a dead network: the
      layer's OWN gradient is dL/dout h^T with h != 0, so W leaves zero on the
      first update and the body receives gradient from step 1 onward — the
      standard ReZero / zero-conv construction. Body gradients are scaled by
      ||W|| for the first epochs; a higher LR on this one layer removes the ramp.

  CHANGE B — centred input (`centered_input=True`)
      Each camera token carries ONLY its DEVIATION from the mean, as a 6-D
      rotation exactly like the absolute input it replaces:

          D_k = R_mean^T @ R_k                   (== I when view k agrees)

      Residual out deserves residual in: with centred tokens, "weighted average
      of the views" is a nearly LINEAR function of the input, whereas from
      absolute rotations the network must first infer the mean internally before
      it can correct it. The absolute rotation is NOT fed — it is redundant
      (R_k = R_mean @ exp(delta_k), so the mean plus the deviations determine it)
      and near-identical across the K camera tokens, since all views observe the
      same pose.

      KNOWN LIMITATION, recorded deliberately. A deviation-only model is blind
      to VIEW-CORRELATED bias: SAM3D attenuates hip rotation in every view at
      once, so the cameras agree and are all wrong together, the deviations are
      ~0, and no correction can be inferred from them. v2's hip damage came from
      exactly that bias being AMPLIFIED (learned gain > 1); v3 cannot amplify it
      either, so the hips should land at mean level rather than below it. If the
      hips turn out to need active correction, the fix is to also feed the
      operating point R_mean as one shared context embedding per (t, p, j) —
      deliberately NOT done here, to keep the design to a single idea.

EVERYTHING IS 6-D, exactly as in v2: the encoder still reads 6 numbers per
(camera, joint) token and the decoder still writes 6. Parameter shapes and
parameter count are IDENTICAL to v2, so a v3-vs-v2 comparison isolates the
residual formulation and nothing else. 6-D is also the better-behaved choice
for both roles: Gram-Schmidt maps essentially all of R^6 onto SO(3), so any
decoder output is a valid rotation, with no wrap at pi and no 2*pi ambiguity.

Architecture
------------
  1. Pose encoder      : 6-D deviation from the mean -> D-dim token per joint
  2. PoseStreamLayers  : joint self-attn -> cross-view attn -> windowed temporal attn -> FFN
  3. Camera mean pool  : visibility-weighted mean over K -> (B, T, P, J, D)
  4. Residual head     : D -> 6-D correction rotation, composed onto the mean

Numerics
--------
`R_mean` and the input deviations depend only on the INPUTS, never on the
parameters, so both are computed under `no_grad` — SVD backward (unstable for
degenerate singular values) is never invoked. Those parts run in float32 even
under bfloat16 autocast, since `linalg.svd` is not usable at bf16.

Notation
--------
B = batch size
T = sequence length
K = number of cameras
P = maximum number of people
J = number of joints (54, root excluded)
D = embedding dimension
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers (torch-native — this module has no pytorch3d dependency)
# ─────────────────────────────────────────────────────────────────────────────

def sixd_to_matrix(sixd: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3). The 6 values are the FIRST TWO ROWS of R.

    Same convention as fusion/placer.py:_6d_to_aa_batch and the training data in
    data/fusion_dataset.py: Gram-Schmidt on rows, third row from the cross
    product, so the result is a proper rotation.
    """
    r0, r1 = sixd[..., :3], sixd[..., 3:]
    b1 = r0 / (r0.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_sixd(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): rows 0 and 1, matching `sixd_to_matrix`."""
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)


def chordal_mean(
    pose_sixd: torch.Tensor,
    person_visible: torch.Tensor,
) -> torch.Tensor:
    """Visibility-weighted chordal mean of the per-camera rotations.

    R_bar = argmin_R sum_k w_k ||R - R_k||_F^2, solved in closed form as the SVD
    projection of the arithmetic mean matrix onto SO(3). This is the SAME
    estimator as the published baseline (evaluation/evaluate_rich_mean.py:
    mean_fuse), reproduced here so the module carries no dependency on an
    evaluation script.

    Parameters
    ----------
    pose_sixd      : (B, T, K, P, J, 6)
    person_visible : (B, T, K, P) float — 1 where camera k detected person p.

    Returns
    -------
    (B, T, P, J, 3, 3) in float32.
    """
    R = sixd_to_matrix(pose_sixd.float())                       # (B,T,K,P,J,3,3)
    w = person_visible.float()[..., None, None, None]           # (B,T,K,P,1,1,1)

    num = (R * w).sum(dim=2)                                    # (B,T,P,J,3,3)
    den = w.sum(dim=2).clamp(min=1e-8)
    M = num / den

    # No camera saw this (t, p): M is all-zero and its SVD is arbitrary. Seed the
    # identity so the decomposition stays well-conditioned. These slots are
    # dropped by the visibility mask before any loss or metric sees them.
    empty = (person_visible.float().sum(dim=2) == 0)            # (B,T,P)
    if bool(empty.any()):
        eye = torch.eye(3, dtype=M.dtype, device=M.device)
        M = torch.where(empty[..., None, None, None], eye.expand_as(M), M)

    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)                                # (B,T,P,J) +-1
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (self-contained copies, independent of fusion_module_v2.py)
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
# Kinematic tree, for the joint-attention hop mask
# ─────────────────────────────────────────────────────────────────────────────
#
# LAYOUT WARNING — two different joint orderings are in play (verified
# empirically by scripts/verify_joint_layout.py):
#
#   canonical (SMPL-X FK OUTPUT) : pelvis | body 1..21 | jaw,eyes 22..24 | lhand 25..39 | rhand 40..54
#   packed    (the POSE TENSOR)  : pelvis | body 1..21 | lhand 22..36     | rhand 37..51 | jaw,eyes 52..54
#
# `parents` below is CANONICAL (it is model.parents from the SMPL-X model,
# checked against it in verify_joint_layout.py). The pose tensor this module
# consumes is PACKED, and by the time the mask is applied the root has been
# stripped, so index i of the attention axis == packed slot i+1. The permutation
# is applied in ONE place, `_packed_hop_matrix`, and nowhere else.
_SMPLX_PARENTS_CANONICAL = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21,
    40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]
_N_SMPLX_JOINTS = 55


def _canonical_to_packed() -> list[int]:
    """canonical index -> packed index. See the layout warning above."""
    mapping = list(range(22))                       # pelvis + 21 body joints align
    mapping += [52, 53, 54]                         # canonical 22,23,24 = jaw, leye, reye
    mapping += list(range(22, 37))                  # canonical 25..39 = left hand
    mapping += list(range(37, 52))                  # canonical 40..54 = right hand
    assert sorted(mapping) == list(range(_N_SMPLX_JOINTS))
    return mapping


def _packed_hop_matrix(num_joints: int) -> torch.Tensor:
    """(num_joints, num_joints) hop distance in ROOT-STRIPPED PACKED order.

    Undirected edge count along the kinematic tree. Row/col i corresponds to
    packed slot i+1, i.e. exactly the axis the joint attention runs over.
    """
    c2p = _canonical_to_packed()
    n = _N_SMPLX_JOINTS
    INF = 10 ** 6
    hop = torch.full((n, n), INF, dtype=torch.long)
    hop.fill_diagonal_(0)
    for c in range(1, n):
        pc = _SMPLX_PARENTS_CANONICAL[c]
        if pc >= 0:
            i, j = c2p[c], c2p[pc]
            hop[i, j] = 1
            hop[j, i] = 1
    for k in range(n):                                   # Floyd-Warshall
        hop = torch.minimum(hop, hop[:, k, None] + hop[None, k, :])
    hop = hop[1:, 1:]                                    # drop root -> pose_aggr space
    if num_joints > hop.shape[0]:
        raise ValueError(f"num_joints={num_joints} exceeds the SMPL-X tree ({hop.shape[0]})")
    return hop[:num_joints, :num_joints].contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class PoseFusionModuleV3(nn.Module):
    """Fuse per-camera SMPL-X pose estimates into a single world-frame prediction.

    Residual formulation: the visibility-weighted chordal mean of the inputs is
    the operating point. The network sees per-camera DEVIATIONS from it and
    emits a CORRECTION to it; with `residual_head=True` the head is
    zero-initialised, so an untrained module reproduces the mean exactly.

    Parameters
    ----------
    embedding_dim    : token dimension D.
    num_heads        : attention heads (must divide embedding_dim).
    num_layers       : number of PoseStreamLayer blocks.
    max_temporal_len : maximum sequence length T for sinusoidal PE.
    dropout          : dropout probability.
    temporal_window  : half-width W of the local temporal attention window.
    num_joints       : J, number of SMPL-X joints (default 54, root excluded).
    kintree_mask_k   : hop radius of the kinematic-tree attention mask, or None.
    residual_head    : CHANGE A — predict a correction to the chordal mean.
    centered_input   : CHANGE B — feed per-camera deviations from the mean.
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
        kintree_mask_k: int | None = None,
        residual_head: bool = True,
        centered_input: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_joints = num_joints
        self.kintree_mask_k = kintree_mask_k
        self.residual_head = residual_head
        self.centered_input = centered_input

        # Kinematic-tree hard mask on the joint attention axis (v2 CHANGE 2).
        # A joint may attend only to joints within `kintree_mask_k` edges of it in
        # the SMPL-X tree; the diagonal is always allowed. Stored as an ADDITIVE
        # bias (0 = allowed, -inf = blocked) because the joint attention already
        # receives an additive log-confidence mask, and -inf + finite = -inf.
        #
        # NOTE this buffer is non-persistent: it is NOT in state_dict and
        # load_state_dict(strict=True) cannot detect its absence, so
        # `kintree_mask_k` MUST be carried in the checkpoint's model_config.
        # `arch_config()` below exists so no caller has to remember that.
        if kintree_mask_k is not None:
            hop = _packed_hop_matrix(num_joints)
            allowed = hop <= int(kintree_mask_k)
            allowed.fill_diagonal_(True)               # never let a row go empty
            assert allowed.any(dim=1).all(), "kintree mask has an all-False row"
            bias = torch.zeros(num_joints, num_joints)
            bias.masked_fill_(~allowed, float("-inf"))
            self.register_buffer("kintree_bias", bias, persistent=False)
        else:
            self.kintree_bias = None

        self.dropout = nn.Dropout(dropout)
        self.temporal_pe = PositionalEncoding(max_temporal_len, embedding_dim)

        # Per-joint learnable identity embedding, injected once at the input.
        # Joint indices are semantically consistent across all SMPL-X bodies
        # (joint 5 = right knee everywhere), so this is a meaningful signal.
        # No camera ID embedding: cameras have no consistent identity across scenes.
        self.joint_id_embedding = nn.Embedding(num_joints, embedding_dim)

        # Input encoder — SAME SHAPE AS v2. CHANGE B only changes what the 6
        # numbers mean: the per-camera deviation R_mean^T @ R_k instead of the
        # absolute rotation R_k.
        self.input_dim = 6
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

        # Output head — SAME SHAPE AS v2. CHANGE A only changes what the 6
        # numbers mean: a correction rotation instead of the pose itself.
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )

        if residual_head:
            # Zero weights + IDENTITY 6-D bias => R_delta == I at step 0, so the
            # module reproduces the chordal mean exactly. See CHANGE A in the
            # module docstring for why this does not freeze the network.
            nn.init.zeros_(self.decoder[-1].weight)
            with torch.no_grad():
                self.decoder[-1].bias.copy_(
                    torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                )

    def arch_config(self) -> dict:
        """Everything needed to rebuild this module from a checkpoint.

        NONE of these change a parameter shape — v3 has exactly v2's parameter
        set — so `load_state_dict(strict=True)` cannot catch a mismatch on any
        of them. `residual_head` and `centered_input` in particular alter what
        the SAME weights mean; loading a v3 checkpoint with them wrong produces
        silent garbage, not an error. Persist this dict under "model_config" and
        pass it back to the constructor.
        """
        return {
            "embedding_dim":    self.embedding_dim,
            "num_heads":        self.num_heads,
            "num_layers":       self.num_layers,
            "max_temporal_len": self.temporal_pe.pe.shape[0],
            "temporal_window":  self.layers[0].temporal_attn.temporal_window,
            "num_joints":       self.num_joints,
            "kintree_mask_k":   self.kintree_mask_k,
            "residual_head":    self.residual_head,
            "centered_input":   self.centered_input,
        }

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

        # ── Operating point: chordal mean of the inputs ──────────────────────
        # A function of the INPUTS only, never of the parameters, so no_grad is
        # exact (not an approximation) and SVD backward is never invoked. Forced
        # to float32: linalg.svd is not usable at bfloat16 under autocast.
        R_mean = None
        if self.residual_head or self.centered_input:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                R_mean = chordal_mean(pose, person_visible)        # (B,T,P,J,3,3) fp32

        # ── Encode ───────────────────────────────────────────────────────────
        if self.centered_input:
            # D_k = R_mean^T @ R_k — the group "subtraction": camera k's
            # deviation from the operating point, == I when it agrees. Same 6-D
            # form as the absolute rotation it replaces, so the encoder is
            # unchanged.
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                R_k = sixd_to_matrix(pose.float())                 # (B,T,K,P,J,3,3)
                R_mean_k = R_mean.unsqueeze(2).expand_as(R_k)      # broadcast over K
                D_k = R_mean_k.transpose(-1, -2) @ R_k             # (B,T,K,P,J,3,3)
                # Absent cameras carry no information, and their stored pose may
                # be arbitrary. Force them to the identity deviation so they
                # cannot inject a spurious signal; their tokens are masked out of
                # every attention and excluded from the pooling anyway.
                eye = torch.eye(3, dtype=D_k.dtype, device=D_k.device)
                seen = person_visible.bool()[..., None, None, None]
                D_k = torch.where(seen, D_k, eye.expand_as(D_k))
                tokens = matrix_to_sixd(D_k).to(pose.dtype)        # (B,T,K,P,J,6)
        else:
            tokens = pose                                           # (B,T,K,P,J,6)

        x = self.pose_encoder(tokens)                               # (B,T,K,P,J,D)

        # ── Build attention masks (built once, shared across all layers) ──────
        # Joint self-attention: each slot (b,t,k,p) sees J×J interactions.
        joint_attn_mask = self._build_confidence_mask(
            conf.reshape(B * T * K * P, J), H,
        )  # (B*T*K*P*H, J, J)

        # Kinematic-tree hard mask on top of the confidence bias. Both are
        # additive, so -inf on a blocked pair survives whatever finite confidence
        # bias sits there. The diagonal is always 0, so every row keeps at least
        # one finite entry and softmax cannot produce NaN — including rows that
        # _build_confidence_mask reset to 0 for absent people.
        if self.kintree_bias is not None:
            if self.kintree_bias.shape[0] != J:
                raise RuntimeError(
                    f"kintree mask built for {self.kintree_bias.shape[0]} joints "
                    f"but attention runs over J={J}")
            joint_attn_mask = joint_attn_mask + self.kintree_bias.to(
                joint_attn_mask.dtype).unsqueeze(0)

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

        # ── Decode ────────────────────────────────────────────────────────────
        head_out = self.decoder(
            self.output_norm(x_pooled).reshape(B * T * P * J, D)
        )

        if not self.residual_head:
            # v2 behaviour: direct 6-D prediction.
            return head_out.reshape(B, T, P, J, 6)

        # CHANGE A — the group "addition": compose the predicted correction onto
        # the operating point, in the joint's own body frame. float32 throughout,
        # since a few-degree correction is below bfloat16's resolution near 1.0.
        R_delta = sixd_to_matrix(head_out.reshape(B, T, P, J, 6).float())
        R_out = R_mean @ R_delta                                   # (B,T,P,J,3,3)
        return matrix_to_sixd(R_out).to(pose.dtype)

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
