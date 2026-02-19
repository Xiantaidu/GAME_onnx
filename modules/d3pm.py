import torch
from torch.nn import functional as F


def d3pm_time_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine time schedule for D3PM.
    :param t: 0~1, 0 = full noise, 1 = no noise
    :return: p for the probability to remove boundaries
    """
    p = (1 + torch.cos(t * torch.pi)) / 2
    return p


def remove_boundaries(boundaries: torch.Tensor, p: torch.Tensor):
    """
    Remove random boundaries with a uniform probability p.
    :param boundaries: [..., T], bool, 1 = boundary, 0 = non-boundary
    :param p: [...], probability to remove each boundary
    :return: [..., T], bool, remaining boundaries
    """
    p = p.unsqueeze(-1)
    q = 1 - p
    rnd = torch.rand_like(boundaries, dtype=torch.float32)
    boundaries_remain = (rnd <= q) & boundaries
    return boundaries_remain


def remove_mutable_boundaries(boundaries: torch.Tensor, immutable: torch.Tensor, p: torch.Tensor):
    """
    Remove random mutable boundaries with adjusted probabilities based on the ratio of mutable boundaries.
    This function will try to keep the expected number of remaining boundaries the same as using uniform
    probability p on all boundaries. If this cannot be achieved, all mutable boundaries will be removed.
    :param boundaries: [..., T], bool, 1 = boundary, 0 = non-boundary
    :param immutable: [..., T], bool, 1 = immutable, 0 = mutable
    :param p: [...], base probability to remove each boundary as in uniform case
    :return: [..., T], bool, remaining boundaries
    """
    boundaries_mutable = boundaries & ~immutable
    n = boundaries.float().sum(dim=-1)
    m = boundaries_mutable.float().sum(dim=-1)
    P = torch.clamp(n * p / (m + 1e-8), max=1.0)
    boundaries_mutable_remain = remove_boundaries(boundaries_mutable, p=P)
    boundaries_remain = boundaries_mutable_remain | immutable
    return boundaries_remain


def remove_boundaries_with_confidence(boundaries: torch.Tensor, confidences: torch.Tensor, p: torch.Tensor):
    """
    Remove random boundaries with adjusted probabilities based on confidence,
    while keeping the expected number of remaining boundaries the same as
    using uniform probability p.
    :param boundaries: [..., T], bool, 1 = boundary, 0 = non-boundary
    :param confidences: [..., T], confidence scores for each boundary, 0~1
    :param p: [...], base probability to remove each boundary as in uniform case
    :return: [..., T], bool, remaining boundaries
    """
    p = p.unsqueeze(-1)
    q = 1 - p
    confidences = torch.where(boundaries, confidences, 0.0)
    c_avg = confidences.sum(dim=-1, keepdim=True) / (boundaries.float().sum(dim=-1, keepdim=True) + 1e-8)
    alpha = torch.minimum(q / c_avg, p / (1 - c_avg))
    Q = q + alpha * (confidences - c_avg)
    rnd = torch.rand_like(boundaries, dtype=torch.float32)
    boundaries_remain = (rnd <= Q) & boundaries
    return boundaries_remain


def remove_mutable_boundaries_with_confidence(
        boundaries: torch.Tensor,
        immutable: torch.Tensor,
        confidences: torch.Tensor,
        p: torch.Tensor
):
    """
    Remove random mutable boundaries with adjusted probabilities based on confidence and the ratio of
    mutable boundaries. This function will try to keep the expected number of remaining boundaries
    the same as using uniform probability p on all boundaries. If this cannot be achieved, all mutable
    boundaries will be removed.
    :param boundaries: [..., T], bool, 1 = boundary, 0 = non-boundary
    :param immutable: [..., T], bool, 1 = immutable, 0 = mutable
    :param confidences: [..., T], confidence scores for each boundary, 0~1
    :param p: [...], base probability to remove each boundary as in uniform case
    :return: [..., T], bool, remaining boundaries
    """
    boundaries_mutable = boundaries & ~immutable
    n = boundaries.float().sum(dim=-1)
    m = boundaries_mutable.float().sum(dim=-1)
    P = torch.clamp(n * p / (m + 1e-8), max=1.0)
    boundaries_mutable_remain = remove_boundaries_with_confidence(
        boundaries_mutable, confidences, p=P
    )
    boundaries_remain = boundaries_mutable_remain | immutable
    return boundaries_remain


def d3pm_region_noise(regions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    p = (1 + torch.cos(t * torch.pi)) / 2
    regions = merge_random_regions(regions, p=p)
    return regions


def merge_random_regions(regions: torch.Tensor, p: torch.Tensor):
    """
    :param regions: [..., T]
    :param p: [...]
    :return: [..., T]
    """
    N = regions.max()
    if N <= 1:
        return regions.clone()
    *B, _ = regions.shape
    drops = torch.rand((*B, N - 1), device=regions.device) < p.unsqueeze(-1)  # [..., N-1]
    shifts = F.pad(drops.long().cumsum(dim=-1), (2, 0), mode="constant", value=0)  # [..., N+1]
    regions_merged = regions - shifts.gather(dim=-1, index=regions)  # [..., T]
    return regions_merged


def split_random_regions(regions: torch.Tensor, p: torch.Tensor):
    """
    :param regions: [..., T]
    :param p: [...]
    :return: [..., T]
    """
    p = p.clamp(max=0.99).unsqueeze(-1)  # [..., 1]
    N = regions.amax(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    L = (regions != 0).sum(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    pi_hat = (N.float() / L.float() / (1 - p)).clamp(max=1.0)  # [..., 1]
    P = p * pi_hat / (p * pi_hat + (1 - pi_hat))  # [..., 1]
    boundaries = F.pad(torch.diff(regions, dim=-1) > 0, (1, 0), mode="constant", value=0)  # [..., T]
    inserts = (torch.rand_like(regions, dtype=torch.float32) < P) & ~boundaries  # [..., T]
    shifts = inserts.long().cumsum(dim=-1)  # [..., T]
    regions_split = regions + shifts * (regions != 0).long()  # [..., T]
    return regions_split
