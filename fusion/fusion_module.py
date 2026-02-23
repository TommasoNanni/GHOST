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

class SSTNetwork(nn.Module):
    """
    Spatio-Spatio-Temporal attention module that fuses parameters across views and time
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
    ):
        super(SSTNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
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
        self.spatial_1_attentions = nn.ModuleList([
            nn.MultiheadAttention(3*embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.spatial_1_norms = nn.ModuleList([
            nn.LayerNorm(3*embedding_dim) for _ in range(self.num_layers)
        ])
        self.spatial_2_attentions = nn.ModuleList([
            nn.MultiheadAttention(3*embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.spatial_2_norms = nn.ModuleList([
            nn.LayerNorm(3*embedding_dim) for _ in range(self.num_layers)
        ])
        self.temporal_attentions = nn.ModuleList([
            nn.MultiheadAttention(3*embedding_dim, num_heads, batch_first=True) for _ in range(self.num_layers)
        ])
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(3*embedding_dim) for _ in range(self.num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3*embedding_dim, 6*embedding_dim),
                nn.ReLU(),
                nn.Linear(6*embedding_dim, 3*embedding_dim),
            ) for _ in range(self.num_layers)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(3*embedding_dim) for _ in range(self.num_layers)
        ])
        self.output_pose = nn.Sequential(
            nn.Linear(3*embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 6),
        )
        self.output_shape = nn.Sequential(
            nn.Linear(3*embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 10),
        )
        self.output_camera = nn.Sequential(
            nn.Linear(3*embedding_dim, embedding_dim),
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

        shape_expanded = shape_emb.unsqueeze(-2).expand(B, T, K, P, J, D)
        camera_expanded = camera_emb.unsqueeze(-2).unsqueeze(-2).expand(B, T, K, P, J, D)


        inputs = torch.cat([
            pose_emb,
            shape_expanded,
            camera_expanded,
        ], dim = -1)          # B, T, K, P, J, 3*D

        spatial_1_mask = torch.einsum("btkpm, btkpn -> btkpmn", joint_mask, joint_mask).reshape((B*T*K*P, J, J))
        soft_spatial_1_mask = torch.log(spatial_1_mask + 1e-6)
        spatial_2_mask = torch.einsum("btmpj, btnpj -> btpjmn", joint_mask, joint_mask).reshape((B*T*P*J, K, K))
        soft_spatial_2_mask = torch.log(spatial_2_mask + 1e-6)
        temporal_mask  = torch.einsum("bmkpj, bnkpj -> bkpjmn", joint_mask, joint_mask).reshape((B*K*P*J, T, T))
        soft_temporal_mask = torch.log(temporal_mask + 1e-6)

        for layer in range(self.num_layers):
            # first attention module, every joint with other joints in the same camera at the same time
            spatial_1_inputs = inputs.reshape(B*T*K*P, J, 3*D)
            spatial_1_outputs, _ = self.spatial_1_attentions[layer](
                spatial_1_inputs,
                spatial_1_inputs,
                spatial_1_inputs,
                attn_mask = soft_spatial_1_mask,
            )
            spatial_1_outputs = spatial_1_outputs.reshape(B*T*K*P*J, 3*D)
            spatial_1_outputs = self.spatial_1_norms[layer](spatial_1_outputs)
            spatial_1_outputs = spatial_1_outputs.reshape(B,T,K,P,J,3*D)

            # Second attention module, every joint with the same joint across cameras at the same time
            spatial_2_inputs = inputs.permute(0,1,3,4,2,5).contiguous().reshape(B*T*P*J, K, 3*D)
            spatial_2_outputs, _ = self.spatial_2_attentions[layer](
                spatial_2_inputs,
                spatial_2_inputs,
                spatial_2_inputs,
                attn_mask = soft_spatial_2_mask,
            )
            spatial_2_outputs = spatial_2_outputs.reshape(B*T*K*P*J, 3*D)
            spatial_2_outputs = self.spatial_2_norms[layer](spatial_2_outputs)
            spatial_2_outputs = spatial_2_outputs.reshape(B,T,P,J,K,3*D)
            spatial_2_outputs = spatial_2_outputs.permute(0, 1, 4, 2, 3, 5).contiguous()

            # Third attention module, every joint with the same joint across frames in the same camera
            temporal_inputs = inputs.permute(0,2,3,4,1,5).contiguous().reshape(B*K*P*J, T, 3*D)
            temporal_outputs, _ = self.temporal_attentions[layer](
                temporal_inputs,
                temporal_inputs,
                temporal_inputs,
                attn_mask = soft_temporal_mask
            )
            temporal_outputs = temporal_outputs.reshape(B*T*K*P*J, 3*D)
            temporal_outputs = self.temporal_norms[layer](temporal_outputs)
            temporal_outputs = temporal_outputs.reshape(B, K, P, J, T, 3*D)
            temporal_outputs = temporal_outputs.permute(0, 4, 1, 2, 3, 5).contiguous()

            # Single residual at the merge, then feed forward
            attn_output = inputs + spatial_1_outputs + spatial_2_outputs + temporal_outputs
            attn_output = attn_output.reshape(B*T*K*P*J, 3*D)
            ff_output = self.ff_layers[layer](attn_output)
            ff_output = ff_output + attn_output
            ff_output = self.ff_norms[layer](ff_output)
            inputs = ff_output.reshape(B, T, K, P, J, 3*D)

        # Pose: per joint prediction from full 3*D token
        delta_pose = self.output_pose(inputs.reshape(B*T*K*P*J, 3*D)).reshape(B, T, K, P, J, 6)

        # Shape: aggregate over joints, decode from full 3*D token
        delta_shape = self.output_shape(inputs.mean(dim=4).reshape(B*T*K*P, 3*D)).reshape(B, T, K, P, 10)

        # Camera: aggregate over people and joints, decode from full 3*D token
        delta_camera = self.output_camera(inputs.mean(dim=[3, 4]).reshape(B*T*K, 3*D)).reshape(B, T, K, 7)

        return delta_pose, delta_shape, delta_camera

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
