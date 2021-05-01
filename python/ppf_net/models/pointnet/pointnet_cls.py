from lmdb.cffi import preload
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet import PointNetEncoder, feature_transform_reguliarzer

import pytorch_lightning as pl

from ppf_net.models.base import BaseModel

class PointNetCls(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config.model

        if self.config.normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.config.k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, data):
        points = data['point'] # (B, n_points, n_feats)
        target = data['label'] # (B, 1)

        points = points.transpose(2, 1) # (B, n_feats, n_points)
        target = target.squeeze() # (B, )

        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        pred_logits = self.fc3(x)
        pred_prob = pred_logits.softmax(dim=1)

        loss = self.getLoss(
            pred_logits, 
            target, trans_feat
        )

        out = {
            "pred_logits": pred_logits,
            "pred_prob": pred_prob,
            "loss": loss,
        }

        return out

    def getLoss(self, pred, target, trans_feat):
        loss = F.cross_entropy(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.config.mat_diff_loss_scale
        return total_loss
