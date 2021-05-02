import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import pytorch_lightning as pl

from ppf_net.utils.ppf import generate_ppf

class BaseModel(pl.LightningModule):
    '''
    Abstract Base Class for point cloud classification models
    '''
    def __init__(self, config):
        '''
        Handle PPF related stuff
        '''
        super().__init__()

        self.config = config.model
        self.ppf_mode = self.config.ppf_mode
        if self.ppf_mode is not None:
            self.input_channel = 4
        else:
            if self.config.normal_channel:
                self.input_channel = 6
            else:
                self.input_channel = 3

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr = self.config.learning_rate, 
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay = self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        '''
        @param batch: a dict of point and label
            point: (B, n_points, n_feats)
            label: (B, 1), ground truth class
        '''
        target = batch['label']
        target = target.squeeze() # (B, )

        # Convert the input point+normal to PPF here
        if self.ppf_mode is not None:
            batch['point_original'] = batch['point']
            batch['point'] = generate_ppf(batch['point'], self.ppf_mode)
    
        out = self(batch) 

        loss = out['loss']
        pred_prob = out['pred_prob']

        # Compute the classification accuracy
        pred_class = pred_prob.detach().max(1)[1]
        correct = (pred_class.long() == target.long()).cpu().sum()

        train_acc = correct / float(len(target))

        self.log("train_inst_acc", train_acc, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        target = batch['label']
        target = target.squeeze() # (B, )

        # Convert the input point+normal to PPF here
        if self.ppf_mode is not None:
            batch['point_original'] = batch['point']
            batch['point'] = generate_ppf(batch['point'], self.ppf_mode)
        
        out = self(batch) 

        loss = out['loss']
        pred_prob = out['pred_prob']

        pred_class = pred_prob.detach().max(1)[1]

        correct = (pred_class.long() == target.long()).cpu().sum()
        valid_acc = correct / float(len(target))

        self.log("valid_loss", loss, logger=True, on_step=False, on_epoch=True)
        self.log("valid_inst_acc", valid_acc, logger=True, on_step=False, on_epoch=True)

        out = {
            "gt_class": target.cpu(),
            "pred_class": pred_class.cpu()
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
