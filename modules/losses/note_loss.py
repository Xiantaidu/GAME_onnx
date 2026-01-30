import torch
from torch import nn, Tensor


class GaussianBlurredBinsLoss(nn.Module):
    """
    This loss map ground truth scores to a set of Gaussian-blurred bins, and computes the BCEWithLogitsLoss
    between the predicted logits and the blurred targets.
    Arguments:
        min_val: float, minimum value of the score range.
        max_val: float, maximum value of the score range.
        num_bins: int, number of bins (N) to quantize the score range.
        deviation: float, standard deviation of the Gaussian blur in the original score scale.
    Inputs:
        - logits: Tensor of shape [..., T, N], predicted logits for each bin.
        - scores: Tensor of shape [..., T], target scores.
        - presence: Tensor of shape [..., T], target presence indicators, 0 means no score.
        - mask: Optional Tensor of shape [..., T], mask to apply on the loss.
    Outputs:
        Scalar tensor representing the Gaussian blurred bins loss.
    """

    def __init__(self, min_val: float, max_val: float, num_bins: int, deviation: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.std = deviation / (max_val - min_val) * (num_bins - 1)
        centers = torch.linspace(min_val, max_val, num_bins)
        self.register_buffer("centers", centers, persistent=False)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, scores: Tensor, presence: Tensor, mask=None) -> Tensor:
        B = (1,) * (logits.ndim - 2)
        if mask is not None:
            mask = mask.unsqueeze(-1).float().expand_as(logits)
        centers = self.centers.reshape(*B, 1, -1)  # [..., 1, N]
        diffs = scores.unsqueeze(-1) - centers  # [..., T, N]
        gaussians = torch.exp(-0.5 * (diffs / self.std) ** 2)  # [..., T, N]
        gaussians = gaussians / (gaussians.sum(dim=-1, keepdim=True) + 1e-6)  # normalize
        targets = gaussians * presence.unsqueeze(-1)  # zero out where no presence
        loss = self.criterion(logits, targets)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss
