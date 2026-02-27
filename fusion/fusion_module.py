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
            nn.Linear(7, 2 * embedding_dim),
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
    """Multi-head attention restricted to a local temporal window.

    Each query at position *t* attends only to KV positions in
    [t − W, t + W] where W = temporal_window.  The attention
    matrix per query is (1, 2W+1) → O(T*W) instead of O(T^2).

    When T <= 2W + 1 (window covers the full sequence), it falls back
    transparently to standard full attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        temporal_window: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temporal_window = temporal_window
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        confidence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, T, D)
        confidence_mask : (N*H, T, T) or None
            Additive soft mask (e.g. from joint confidence scores).  Only the
            relevant (T, 2W+1) slice is gathered — the full matrix is
            never passed to the MHA kernel.

        Returns
        -------
        (N, T, D)
        """
        N, T, D = x.shape
        W = min(self.temporal_window, T - 1)
        win_size = 2 * W + 1
        H = self.num_heads

        if win_size >= T:
            out, _ = self.attn(x, x, x, attn_mask=confidence_mask)
            return out

        # build KV windows via pad + unfold
        x_pad = F.pad(x, (0, 0, W, W))                       # (N, T+2W, D)
        kv = x_pad.unfold(1, win_size, 1).permute(0, 1, 3, 2).contiguous()

        q  = x.reshape(N * T, 1, D)
        kv = kv.reshape(N * T, win_size, D)

        # boundary mask (padding positions → −inf)
        t_idx = torch.arange(T, device=x.device)
        r_idx = torch.arange(win_size, device=x.device)
        orig_pos = t_idx.unsqueeze(1) + (r_idx - W).unsqueeze(0)   # (T, win)
        boundary = torch.where(
            (orig_pos >= 0) & (orig_pos < T), 0.0, float("-inf"),
        )

        mask = (
            boundary
            .unsqueeze(0).expand(N, -1, -1)
            .reshape(N * T, 1, win_size)
            .unsqueeze(1).expand(-1, H, -1, -1)
            .reshape(N * T * H, 1, win_size)
        )

        # optional confidence mask slicing
        if confidence_mask is not None:
            cm = confidence_mask.reshape(N, H, T, T)
            col_idx = orig_pos.clamp(0, T - 1)
            col_idx = col_idx.unsqueeze(0).unsqueeze(0).expand(N, H, -1, -1)
            cm_win = cm.gather(3, col_idx)
            cm_win = cm_win.permute(0, 2, 1, 3).reshape(N * T * H, 1, win_size)
            mask = mask + cm_win

        out, _ = self.attn(q, kv, kv, attn_mask=mask)
        return out.reshape(N, T, D)



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
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
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
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
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
        B: int, T: int, K: int, P: int, J: int, D: int,
        dropout: nn.Dropout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pose_stream   : (B, T, K, P, J, D)
        camera_stream : (B, T, K, D)

        Returns
        -------
        pose_stream, camera_stream (same shapes, updated)
        """
        # Pose - Camera
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

        # Camera - Pose
        x = self.cam_to_pose_norm(camera_stream)
        x = x.reshape(B * T * K, 1, D)
        pose_kv = pose_stream.reshape(B * T * K, P * J, D)
        x, _ = self.cam_to_pose_attn(x, pose_kv, pose_kv)
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
        temporal_mask: torch.Tensor | None,
        pe: PositionalEncoding | None,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, K, P, J, D)

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
        h = self.temporal_attn(h, confidence_mask=temporal_mask)
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
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, K, P, D)
        """
        # Cross-view  (B*T*P, K, D)
        h = self.view_attn(
            x.permute(0, 1, 3, 2, 4).contiguous().reshape(B * T * P, K, D),
        )
        x = h.reshape(B, T, P, K, D).permute(0, 1, 3, 2, 4).contiguous()

        # Windowed temporal  (B*K*P, T, D)
        h = self.temporal_norm(x)
        h = h.permute(0, 2, 3, 1, 4).contiguous().reshape(B * K * P, T, D)
        if pe is not None:
            h = pe(h)
        h = self.temporal_attn(h)
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
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, K, D)
        """
        h = self.temporal_norm(x)
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * K, T, D)
        if pe is not None:
            h = pe(h)
        h = self.temporal_attn(h)
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
        self.pose_head = nn.Sequential(
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
        self.camera_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 7),
        )

    def forward(
        self,
        pose_stream: torch.Tensor,
        shape_stream: torch.Tensor,
        camera_stream: torch.Tensor,
        B: int, T: int, K: int, P: int, J: int, D: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose = self.pose_norm(pose_stream)

        pose_per_cam = self.pose_head(pose.reshape(B * T * P * K * J, D)).reshape(B, T, K, P, J, 6)
        pose_aggr_feat = self.pose_pool(pose, k_dim=2) # [B, T, P, J, D]
        pose_aggr = self.pose_head(pose_aggr_feat.reshape(B * T * P * J, D)).reshape(B, T, P, J, 6)

        shape = self.shape_norm(shape_stream)

        shape_per_cam = self.shape_head(shape.reshape(B * T * K * P, D)).reshape(B, T, K, P, 10)
        shape_aggr_feat = self.shape_pool(shape, k_dim=2) # [B, T, P, D]
        shape_aggr = self.shape_head(shape_aggr_feat.reshape(B * T * P, D)).reshape(B, T, P, 10)

        camera = self.camera_norm(camera_stream)
        camera = self.camera_head(camera.reshape(B * T * K, D)).reshape(B, T, K, 7)

        return pose_aggr, shape_aggr, camera, pose_per_cam, shape_per_cam

class SSTNetwork(nn.Module):
    """Spatio-Spatio-Temporal attention module that fuses parameters across
    views and time.

    Composed of:
        * :class:`SSTEncoder`               — input projection
        * :class:`PoseStreamLayer` ×L        — joint + view + temporal + FFN
        * :class:`ShapeStreamLayer` ×L       — view + temporal + FFN
        * :class:`CameraStreamLayer` ×L      — temporal + FFN
        * :class:`PoseCameraCrossAttention` ×(L//2) — bidirectional cross-attn (every 2 layers)
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
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temporal_window = temporal_window

        self.dropout = nn.Dropout(dropout)
        self.temporal_pe = PositionalEncoding(max_temporal_len, embedding_dim)

        self.encoder = SSTEncoder(embedding_dim)

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
            for _ in range(num_layers // 2)
        ])
        self.output_heads = SSTOutputHeads(embedding_dim)

    @staticmethod
    def _build_confidence_mask(
        flat: torch.Tensor, num_heads: int,
    ) -> torch.Tensor:
        """Build (N*H, S, S) additive soft mask from (N, S) confidence values."""
        outer = torch.einsum("bi, bj -> bij", flat, flat)
        mask = torch.log(outer + 1e-6)
        return mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(
            flat.shape[0] * num_heads, flat.shape[1], flat.shape[1]
        )

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        camera: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pose       : [B, T, K, P, J, 6] or [T, K, P, J, 6]
        shape      : [B, T, K, P, betas] or [T, K, P, betas]
        camera     : [B, T, K, 7] or [T, K, 7]
        joint_mask : [B, T, K, P, J] or [T, K, P, J]  (confidence ∈ [0,1])

        Returns
        -------
        pose   : [B, T, P, J, 6]
        shape  : [B, T, P, 10]
        camera : [B, T, K, 7]
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

        assert pose.shape[:4] == shape.shape[:4]
        assert pose.shape[:3] == camera.shape[:3]
        assert pose.shape[:5] == joint_mask.shape

        # encode
        pose_emb, shape_emb, camera_emb = self.encoder(pose, shape, camera)
        B, T, K, P, J, D = pose_emb.shape
        H = self.num_heads

        # build soft confidence masks
        joint_mask_flat = joint_mask.reshape(B * T * K * P, J)
        pose_joint_mask = self._build_confidence_mask(joint_mask_flat, H)

        view_mask_flat = joint_mask.permute(0, 1, 3, 4, 2).reshape(B * T * P * J, K)
        pose_view_mask = self._build_confidence_mask(view_mask_flat, H)

        # For pose temporal attention — full (T,T) built once, sliced per window inside WTA
        temp_mask_flat = joint_mask.permute(0, 2, 3, 4, 1).reshape(B * K * P * J, T)
        pose_temporal_mask = self._build_confidence_mask(temp_mask_flat, H)

        # layer loop
        pose_stream = pose_emb
        shape_stream = shape_emb
        camera_stream = camera_emb

        for layer_idx in range(self.num_layers):
            pe = self.temporal_pe if layer_idx == 0 else None

            pose_stream = self.pose_layers[layer_idx](
                pose_stream, B, T, K, P, J, D, H,
                joint_mask=pose_joint_mask,
                view_mask=pose_view_mask,
                temporal_mask=pose_temporal_mask,
                pe=pe,
                dropout=self.dropout,
            )

            shape_stream = self.shape_layers[layer_idx](
                shape_stream, B, T, K, P, D,
                pe=pe,
                dropout=self.dropout,
            )

            camera_stream = self.camera_layers[layer_idx](
                camera_stream, B, T, K, D,
                pe=pe,
                dropout=self.dropout,
            )

            # Pose - Camera cross-attention every 2 layers
            if (layer_idx + 1) % 2 == 0:
                cross_idx = (layer_idx + 1) // 2 - 1
                pose_stream, camera_stream = self.cross_attns[cross_idx](
                    pose_stream, camera_stream,
                    B, T, K, P, J, D, self.dropout,
                )

        # decode
        return self.output_heads(
            pose_stream, shape_stream, camera_stream,
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
