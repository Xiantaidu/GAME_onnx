import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.plot import (
    similarity_to_figure,
    boundary_to_figure,
)
from modules.d3pm import d3pm_region_noise, merge_random_regions
from modules.decoding import (
    decode_soft_boundaries,
)
from modules.losses import (
    GaussianSoftBoundaryLoss,
    RegionalCosineSimilarityLoss,
)
from modules.losses.boundary_loss import gaussian_soften_boundaries
from modules.losses.region_loss import self_cosine_similarity
from modules.metrics import (
    AverageChamferDistance,
    QuantityMetricCollection,
)
from modules.metrics.quantity import match_nearest_boundaries
from modules.midi_extraction import SegmentationModel
from .data import BaseDataset
from .pl_module_base import BaseLightningModule


class SegmentationDataset(BaseDataset):
    pass


class SegmentationLightningModule(BaseLightningModule):
    __dataset__ = SegmentationDataset

    def build_model(self) -> nn.Module:
        return SegmentationModel(self.model_config)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("region_loss", RegionalCosineSimilarityLoss(
            neighborhood_size=self.training_config.loss.region_loss.neighborhood_size,
            exponential_decay=self.training_config.loss.region_loss.exponential_decay,
        ))
        self.register_loss("boundary_loss", GaussianSoftBoundaryLoss(
            std=self.training_config.loss.boundary_loss.std,
        ))
        # noinspection PyAttributeOutsideInit
        self._register_metrics()
        if self.use_parallel_dirty_metrics:
            self._register_metrics(postfix="_dirty")

    def _register_metrics(self, postfix: str = "") -> None:
        self.register_metric(f"average_chamfer_distance{postfix}", AverageChamferDistance())
        self.register_metric(f"quantity_metric_collection{postfix}", QuantityMetricCollection(
            tolerance=self.training_config.validation.boundary_matching_tolerance,
            postfix=postfix,
        ))

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        spectrogram = sample["spectrogram"]
        if self.model_config.use_languages:
            language_ids = sample["language_id"]
            if not infer:
                language_ids = torch.where(
                    torch.rand(language_ids.shape, device=language_ids.device) < 0.5,
                    language_ids,
                    torch.zeros_like(language_ids)
                )
        else:
            language_ids = None
        regions = sample["regions"]
        boundaries = sample["boundaries"]
        mask = regions != 0

        if infer:
            soft_boundaries, boundaries_pred, latent = self._forward_infer(
                spectrogram, language_ids=language_ids, mask=mask
            )
            similarities = self_cosine_similarity(latent)  # [B, T, T]
            self._update_metrics(
                boundaries_pred=boundaries_pred,
                boundaries_gt=boundaries
            )

            if self.use_parallel_dirty_metrics:
                _, boundaries_pred_dirty, _ = self._forward_infer(
                    sample["spectrogram_dirty"], language_ids=language_ids, mask=mask
                )
                self._update_metrics(
                    boundaries_pred=boundaries_pred_dirty,
                    boundaries_gt=boundaries,
                    postfix="_dirty"
                )

            return {
                "similarities": similarities,
                "soft_boundaries": soft_boundaries,
                "boundaries": boundaries_pred,
            }
        else:
            logits, latent = self._forward_train(
                spectrogram, language_ids=language_ids, regions=regions, mask=mask
            )
            region_loss = self.losses["region_loss"](latent, regions)
            boundary_loss = self.losses["boundary_loss"](logits, boundaries, mask=mask)
            return {
                "region_loss": region_loss,
                "boundary_loss": boundary_loss,
            }

    def _forward_train(self, spectrogram, language_ids, regions, mask):
        B = spectrogram.shape[0]
        if self.model_config.mode == "d3pm":
            # Choose random t, merge regions by p(t)
            t = torch.rand(B, device=spectrogram.device)
            noise = d3pm_region_noise(regions, t=t)  # [B, T]
        elif self.model_config.mode == "completion":
            # Choose random p, merge regions by p
            t = None
            p = torch.rand(B, device=spectrogram.device)
            noise = merge_random_regions(regions, p=p)
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}.")

        logits, latent = self.model(
            spectrogram, regions=noise, t=t,
            language=language_ids, mask=mask,
        )  # [B, T]

        return logits, latent

    def _forward_infer(self, spectrogram, language_ids, mask):
        B = spectrogram.shape[0]
        if self.model_config.mode == "d3pm":
            # 1. Initialize with a whole region (no boundaries).
            # 2. Merge regions by p(t) before each step.
            # 3. Predict full boundaries.
            latent = None
            soft_boundaries = None
            boundaries_pred = None
            num_steps = self.training_config.validation.d3pm_sample_steps
            timestep = torch.full(
                (B,), fill_value=1 / num_steps,
                dtype=torch.float32, device=spectrogram.device
            )
            regions_pred = mask.long()
            for i in range(num_steps):
                t = i * timestep
                noise = d3pm_region_noise(regions_pred, t=t)  # [B, T]
                logits, latent_ = self.model(
                    spectrogram, regions=noise, t=t,
                    language=language_ids, mask=mask,
                )  # [B, T]
                if i == 0:
                    latent = latent_
                soft_boundaries, boundaries_pred = self._decode_boundaries(logits, mask)
                regions_pred = (boundaries_pred.long().cumsum(dim=-1) + 1) * mask.long()
        elif self.model_config.mode == "completion":
            # One-step prediction from a whole region (no boundaries).
            logits, latent = self.model(
                spectrogram, regions=mask.long(),
                language=language_ids, mask=mask,
            )  # [B, T]
            soft_boundaries, boundaries_pred = self._decode_boundaries(logits, mask)
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}.")

        return soft_boundaries, boundaries_pred, latent

    def _decode_boundaries(self, logits, mask):
        soft_boundaries = logits.sigmoid()
        boundaries = decode_soft_boundaries(
            boundaries=soft_boundaries, mask=mask,
            threshold=self.training_config.validation.boundary_decoding_threshold,
            radius=self.training_config.validation.boundary_decoding_radius,
        )
        return soft_boundaries, boundaries

    def _update_metrics(self, boundaries_pred: torch.Tensor, boundaries_gt: torch.Tensor, postfix: str = ""):
        self.metrics[f"average_chamfer_distance{postfix}"].update(boundaries_pred, boundaries_gt)
        self.metrics[f"quantity_metric_collection{postfix}"].update(boundaries_pred, boundaries_gt)

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            if data_idx >= self.training_config.validation.max_plots:
                continue
            T = self.valid_dataset.info["lengths"][data_idx]
            N = self.valid_dataset.info["durations"][data_idx]
            durations = sample["durations"][i, :N]  # [N]
            boundaries = sample["boundaries"][i, :T]  # [T]
            similarities = outputs["similarities"][i, :T, :T]  # [T, T]
            soft_boundaries_pred = outputs["soft_boundaries"][i, :T]  # [T]
            boundaries_pred = outputs["boundaries"][i, :T]  # [T]

            match_pred_to_target, match_target_to_pred = match_nearest_boundaries(
                boundaries_pred, boundaries, tolerance=self.training_config.validation.boundary_matching_tolerance
            )
            boundaries_tp = match_pred_to_target
            boundaries_fp = boundaries_pred & ~match_pred_to_target
            boundaries_fn = boundaries & ~match_target_to_pred

            soft_boundaries_gt = gaussian_soften_boundaries(
                boundaries, std=self.training_config.loss.boundary_loss.std
            )
            threshold = self.training_config.validation.boundary_decoding_threshold

            self.plot_regions(
                data_idx, similarities, durations,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )
            self.plot_boundaries(
                data_idx, soft_boundaries_gt, soft_boundaries_pred,
                threshold=threshold,
                boundaries_tp=boundaries_tp,
                boundaries_fp=boundaries_fp,
                boundaries_fn=boundaries_fn,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )

    def plot_regions(
            self, idx: int,
            similarities: torch.Tensor, durations: torch.Tensor,
            title=None
    ):
        similarities = similarities.cpu().numpy()
        durations = durations.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"regions/regions_{idx}", similarity_to_figure(
            similarities, durations, title=title
        ), global_step=self.global_step)

    def plot_boundaries(
            self, idx: int,
            boundaries_gt: torch.Tensor, boundaries_pred: torch.Tensor,
            threshold: float = None,
            boundaries_tp: torch.Tensor = None,
            boundaries_fp: torch.Tensor = None,
            boundaries_fn: torch.Tensor = None,
            title=None
    ):
        boundaries_gt = boundaries_gt.cpu().numpy()
        boundaries_pred = boundaries_pred.cpu().numpy()
        if boundaries_tp is not None:
            boundaries_tp = boundaries_tp.cpu().numpy()
        if boundaries_fp is not None:
            boundaries_fp = boundaries_fp.cpu().numpy()
        if boundaries_fn is not None:
            boundaries_fn = boundaries_fn.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"boundaries/boundaries_{idx}", boundary_to_figure(
            boundaries_gt, boundaries_pred,
            threshold=threshold,
            boundaries_tp=boundaries_tp,
            boundaries_fp=boundaries_fp,
            boundaries_fn=boundaries_fn,
            title=title
        ), global_step=self.global_step)
