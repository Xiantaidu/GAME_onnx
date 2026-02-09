
import torch
import torch.nn.functional as F
import torch.nn as nn
def dice_loss(pred, target, smooth=1.): #target01110序列
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=1)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth))
    return loss.mean()


def quality_focal_loss(pred, target, beta=2.0):#sigmoid pred



    scale = torch.abs(target - pred) ** beta
    loss = -scale * (target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))

    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha # 平衡正负样本权重
        self.gamma = gamma # 聚焦困难样本

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # 惩罚 FP
        self.beta = beta  # 惩罚 FN
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, L], target: [B, L] 00010001
        pred = pred.view(-1)
        target = target.view(-1)
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - Tversky