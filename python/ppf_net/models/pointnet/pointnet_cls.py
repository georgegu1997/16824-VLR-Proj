import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet import PointNetEncoder, feature_transform_reguliarzer

import pytorch_lightning as pl

class PointNetCls(pl.LightningModule):
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

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

    def getLoss(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.config.mat_diff_loss_scale
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr = self.config.learning_rate, 
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay = self.config.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step, gamma=self.config.lr_decay_rate),
            'interval': 'step'
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        '''
        @param batch: a tuple of points and target
            points: (B, n_points, n_feats)
            target: (B, 1), ground truth class
        '''
        points, target = batch
        points = points.transpose(2, 1) # (B, n_feats, n_points)
        target = target.squeeze() # (B, )

        # pred: (B, n_class), log_softmax of the predicted logits
        # trans_feat: (B, 64, 64), the feature transformation matrix
        pred, trans_feat = self(points) 

        loss = self.getLoss(pred, target.long(), trans_feat)

        pred_class = pred.detach().max(1)[1]
        correct = (pred_class.long() == target.long()).cpu().sum()

        train_acc = correct / float(len(points))

        self.log("train_inst_acc", train_acc, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        points, target = batch
        points = points.transpose(2, 1) # (B, n_feats, n_points)
        target = target.squeeze() # (B, )

        # pred: (B, n_class), log_softmax of the predicted logits
        # trans_feat: (B, 64, 64), the feature transformation matrix
        pred, trans_feat = self(points) 

        loss = self.getLoss(pred, target.long(), trans_feat)

        pred_class = pred.detach().max(1)[1]

        correct = (pred_class.long() == target.long()).cpu().sum()
        valid_acc = correct / float(len(points))

        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True)
        self.log("valid_inst_acc", valid_acc, logger=True, on_step=False, on_epoch=True)

        out = {
            "gt_class": target,
            "pred_class": pred_class
        }

        return out

    def validation_epoch_end(self, outputs):
        gt_class = torch.cat([o['gt_class'] for o in outputs])
        pred_class = torch.cat([o['pred_class'] for o in outputs])

        class_acc = []
        for c in range(self.config.k):
            pred_this = pred_class[gt_class == c]
            if len(pred_this) == 0:
                continue
            else:
                acc_this = (pred_this == c).sum() / len(pred_this)
                class_acc.append(acc_this.item())

        valid_cls_acc = np.mean(class_acc)

        self.log("valid_cls_acc", valid_cls_acc, logger=True)
