"""
Notation:
B = batch size
T = length of the sequence
K = number of cameras
P = maximum number of people in the scene
J = number of joints
betas = number of betas
"""
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, model_dim):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.model_dim = model_dim
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]



class SSTNetwork(nn.Module):
    """
    Spatio-Spatio-Temporal attention module that fuses parameters across views and time
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        max_temporal_len: int = 4096,
    ):
        super(SSTNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.temporal_pe = PositionalEncoding(max_temporal_len, embedding_dim)
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim),
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(10, 2*embedding_dim),
            nn.Tanh(),
            nn.Linear(2*embedding_dim, embedding_dim),
        )
        self.camera_encoder = nn.Sequential(
            nn.Linear(7, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim),
        )
        # Pose stream: joint attn + cross-view attn + temporal attn 
        self.pose_joint_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.pose_joint_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.pose_view_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.pose_view_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.pose_temporal_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.pose_temporal_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.pose_ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 2*embedding_dim),
                nn.ReLU(),
                nn.Linear(2*embedding_dim, embedding_dim),
            ) for _ in range(self.num_layers)
        ])
        self.pose_ff_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])

        # Shape stream: cross-view attn + temporal attn
        self.shape_view_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.shape_view_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.shape_temporal_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.shape_temporal_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.shape_ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 2*embedding_dim),
                nn.ReLU(),
                nn.Linear(2*embedding_dim, embedding_dim),
            ) for _ in range(self.num_layers)
        ])
        self.shape_ff_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])

        # Pose - Camera cross-attention (every 2 layers)
        self.num_cross_layers = num_layers // 2
        self.pose_to_camera_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_cross_layers)
        ])
        self.pose_to_camera_cross_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_cross_layers)
        ])
        self.camera_to_pose_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_cross_layers)
        ])
        self.camera_to_pose_cross_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_cross_layers)
        ])

        # Camera stream: temporal attn only
        self.camera_temporal_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.camera_temporal_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.camera_ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 2*embedding_dim),
                nn.ReLU(),
                nn.Linear(2*embedding_dim, embedding_dim),
            ) for _ in range(self.num_layers)
        ])
        self.camera_ff_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(self.num_layers)
        ])
        self.output_pose = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )
        self.output_shape = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 10),
        )
        self.output_camera = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 7),
        )
    def encode_inputs(self, pose, shape, camera):
        """
        Encode the input parameters into a common embedding space
        """
        pose_emb = self.pose_encoder(pose)          # B, T, K, P, J, D
        shape_emb = self.shape_encoder(shape)       # B, T, K, P, D
        camera_emb = self.camera_encoder(camera)    # B, T, K, D
        return pose_emb, shape_emb, camera_emb

    def forward(self, pose, shape, camera, joint_mask):
        """
        Forward pass of the model, requires:
        pose: tensor with poses at every frame in time for every camera and every person, shape [B, T, K, P, J, 6] or [T, K, P, J, 6]
        shape: tensor with shapes at every frame in time for every camera and every person, shape [B, T, K, P, betas] or [T, K, P, betas]
        camera: tensor with cameras at every frame in time, shape [B, T, K, 7] or [T, K, 7]. Cameras represented as quaternions
        joint_mask: tensor with joint confidence in [0, 1] for every frame in time for every camera and every person, shape [B, T, K, P, J] or [T, K, P, J]
        """

        if len(pose.shape) == 5:
            pose = pose.unsqueeze(0)
        if len(shape.shape) == 4:
            shape = shape.unsqueeze(0)
        if len(camera.shape) == 3:
            camera = camera.unsqueeze(0)
        if len(joint_mask.shape) == 4:
            joint_mask = joint_mask.unsqueeze(0)

        assert pose.shape[:4] == shape.shape[:4], f"shape mismatch for pose and shape, expected [B,T,K,P,J,6] and [B,T,K,P,betas], got {pose.shape} and {shape.shape}"
        assert pose.shape[:3] == camera.shape[:3], f"shape mismatch for pose and camera, expected [B,T,K,P,J,6] and [B,T,K,7], got {pose.shape} and {camera.shape}"
        assert pose.shape[:5] == joint_mask.shape, f"shape mismatch for pose and joint mask, expected [B,T,K,P,J,6] and [B,T,K,P, J], got {pose.shape} and {joint_mask.shape}"

        pose_emb, shape_emb, camera_emb = self.encode_inputs(pose, shape, camera)

        B, T, K, P, J, D = pose_emb.shape

        # soft attention masks for pose 
        # Pose joint mask: joint-to-joint within same person/camera/frame
        joint_mask_flat = joint_mask.reshape(B*T*K*P, J)
        pose_joint_mask = torch.log(torch.einsum("bi, bj -> bij", joint_mask_flat, joint_mask_flat) + 1e-6)

        # Pose view mask: same joint across cameras
        view_mask_flat = joint_mask.permute(0,1,3,4,2).reshape(B*T*P*J, K)
        pose_view_mask = torch.log(torch.einsum("bi, bj -> bij", view_mask_flat, view_mask_flat) + 1e-6)

        # Pose temporal mask: same joint across time
        temp_mask_flat = joint_mask.permute(0,2,3,4,1).reshape(B*K*P*J, T)
        pose_temporal_mask = torch.log(torch.einsum("bi, bj -> bij", temp_mask_flat, temp_mask_flat) + 1e-6)

        # Attention loop: three independent streams
        pose_stream = pose_emb      # B, T, K, P, J, D
        shape_stream = shape_emb    # B, T, K, P, D
        camera_stream = camera_emb  # B, T, K, D

        for layer in range(self.num_layers):
            # POSE STREAM: joint attn -> cross-view attn -> temporal attn -> FFN

            # Joint attention
            x = self.pose_joint_norms[layer](pose_stream)
            x = x.reshape(B*T*K*P, J, D)
            x, _ = self.pose_joint_attentions[layer](x, x, x, attn_mask=pose_joint_mask)
            pose_stream = pose_stream + x.reshape(B, T, K, P, J, D)

            # Cross-view attention
            x = self.pose_view_norms[layer](pose_stream)
            x = x.permute(0,1,3,4,2,5).contiguous().reshape(B*T*P*J, K, D)
            x, _ = self.pose_view_attentions[layer](x, x, x, attn_mask=pose_view_mask)
            x = x.reshape(B, T, P, J, K, D).permute(0, 1, 4, 2, 3, 5).contiguous()
            pose_stream = pose_stream + x

            # Temporal attention
            x = self.pose_temporal_norms[layer](pose_stream)
            x = x.permute(0,2,3,4,1,5).contiguous().reshape(B*K*P*J, T, D)
            x = self.temporal_pe(x)
            x, _ = self.pose_temporal_attentions[layer](x, x, x, attn_mask=pose_temporal_mask)
            x = x.reshape(B, K, P, J, T, D).permute(0, 4, 1, 2, 3, 5).contiguous()
            pose_stream = pose_stream + x

            # FFN
            x = self.pose_ff_norms[layer](pose_stream)
            x = x.reshape(B*T*K*P*J, D)
            pose_stream = pose_stream + self.pose_ff[layer](x).reshape(B, T, K, P, J, D)

            # SHAPE STREAM: cross-view attn -> temporal attn -> FFN

            # Cross-view attention
            x = self.shape_view_norms[layer](shape_stream)
            x = x.permute(0,1,3,2,4).contiguous().reshape(B*T*P, K, D)
            x, _ = self.shape_view_attentions[layer](x, x, x)
            x = x.reshape(B, T, P, K, D).permute(0, 1, 3, 2, 4).contiguous()
            shape_stream = shape_stream + x

            # Temporal attention
            x = self.shape_temporal_norms[layer](shape_stream)
            x = x.permute(0,2,3,1,4).contiguous().reshape(B*K*P, T, D)
            x = self.temporal_pe(x)
            x, _ = self.shape_temporal_attentions[layer](x, x, x)
            x = x.reshape(B, K, P, T, D).permute(0, 3, 1, 2, 4).contiguous()
            shape_stream = shape_stream + x

            # FFN
            x = self.shape_ff_norms[layer](shape_stream)
            x = x.reshape(B*T*K*P, D)
            shape_stream = shape_stream + self.shape_ff[layer](x).reshape(B, T, K, P, D)

            # CAMERA STREAM: temporal attn -> FFN

            # Temporal attention (same camera across time)
            x = self.camera_temporal_norms[layer](camera_stream)
            x = x.permute(0,2,1,3).contiguous().reshape(B*K, T, D)
            x = self.temporal_pe(x)
            x, _ = self.camera_temporal_attentions[layer](x, x, x)
            x = x.reshape(B, K, T, D).permute(0, 2, 1, 3).contiguous()
            camera_stream = camera_stream + x

            # FFN
            x = self.camera_ff_norms[layer](camera_stream)
            x = x.reshape(B*T*K, D)
            camera_stream = camera_stream + self.camera_ff[layer](x).reshape(B, T, K, D)

            # Pose - Camera cross-attention 
            if (layer + 1) % 2 == 0:
                cross_idx = (layer + 1) // 2 - 1

                # Each joint token queries all K camera tokens to learn viewpoint-dependent weighting
                x = self.pose_to_camera_cross_norms[cross_idx](pose_stream)
                x = x.permute(0,1,3,4,2,5).contiguous().reshape(B*T*P*J, K, D)
                cam_kv = camera_stream.unsqueeze(2).unsqueeze(2).expand(B, T, P, J, K, D)
                cam_kv = cam_kv.reshape(B*T*P*J, K, D)
                x, _ = self.pose_to_camera_cross_attn[cross_idx](x, cam_kv, cam_kv)
                x = x.reshape(B, T, P, J, K, D).permute(0, 1, 4, 2, 3, 5).contiguous()
                pose_stream = pose_stream + x

                # Each camera token queries all P*J pose tokens from that view
                x = self.camera_to_pose_cross_norms[cross_idx](camera_stream)
                x = x.reshape(B*T*K, 1, D)
                pose_kv = pose_stream.reshape(B*T*K, P*J, D)
                x, _ = self.camera_to_pose_cross_attn[cross_idx](x, pose_kv, pose_kv)
                x = x.reshape(B, T, K, D)
                camera_stream = camera_stream + x

        # Pose: aggregate over cameras, decode per joint
        pose = self.output_pose(pose_stream.mean(dim=2).reshape(B*T*P*J, D)).reshape(B, T, P, J, 6)

        # Shape: aggregate over cameras, decode per person
        shape = self.output_shape(shape_stream.mean(dim=2).reshape(B*T*P, D)).reshape(B, T, P, 10)

        # Camera: decode per camera
        camera = self.output_camera(camera_stream.reshape(B*T*K, D)).reshape(B, T, K, 7)

        return pose, shape, camera

    def count_parameters(self) -> dict:
        """
        Count total, trainable, and frozen parameters.

        Returns a dict with:
            total       - all parameters
            trainable   - parameters with requires_grad=True
            frozen      - parameters with requires_grad=False
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    total - trainable,
        }

    def summary(self, indent: int = 2) -> str:
        """
        Return a human-readable summary of the model's sub-modules and their
        parameter counts. Useful for quick sanity-checks at the start of a run.

        Example output:
            SSTNetwork
              pose_encoder         :    66,048 params
              shape_encoder        :    66,560 params
              ...
            ------------------------------------------
            Total      :  12,582,912 params
            Trainable  :  12,582,912 params
            Frozen     :           0 params
        """
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
