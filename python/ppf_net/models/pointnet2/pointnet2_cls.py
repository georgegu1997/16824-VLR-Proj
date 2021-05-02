import torch.nn as nn
import torch.nn.functional as F
from .pointnet2 import PointNetSetAbstraction

from ppf_net.models.base import BaseModel

class PointNet2Cls(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config.model

        in_channel = 3 + self.input_channel if self.config.ppf_mode is not None else self.input_channel

        if self.config.ppf_first:
            # Whether to compute PPF at different layers
            ppf1, ppf2, ppf3 = True, False, False
            
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False, ppf=ppf1)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False, ppf=ppf2)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True, ppf=ppf3)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, self.config.k)

    def forward(self, data):
        points = data['point_original'] if self.config.ppf_mode else data['point']  # (B, n_points, n_feats)
        # points = data['point']  # (B, n_points, n_feats)
        target = data['label']  # (B, 1)

        points = points.transpose(2, 1)  # (B, n_feats, n_points)
        target = target.squeeze()  # (B, )

        # XYZ position will be used for sampling and grouping
        xyz = points[:, :3, :]

        if self.config.ppf_first:
            # PPF will be computed in the first abstration layer
            norm = points[:, 3:, :]
        elif self.config.ppf_mode is not None:
            # PPF is computed before input and will be set as the point feature
            norm = data['point'].transpose(2, 1)
        elif self.config.normal_channel:
            # Not using PPF at all, but will include the normal vectors
            norm = points[:, 3:, :]
        else:
            # Only using XYZ information, no normal vectors
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(-1, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        pred_logits = self.fc3(x)
        pred_prob = pred_logits.softmax(dim=1)

        loss = self.get_loss(pred_logits, target)

        out = {
            "pred_logits": pred_logits,
            "pred_prob": pred_prob,
            "loss": loss,
        }
        return out

    def get_loss(self, pred, target):
        total_loss = F.cross_entropy(pred, target)
        return total_loss