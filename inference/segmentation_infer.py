import torch
from torch import nn, Tensor

from inference.functional import format_regions
from lib.config.schema import ModelConfig, InferenceConfig
from lib.feature.mel import StretchableMelSpectrogram
from modules.commons.tts_modules import LengthRegulator
from modules.d3pm import d3pm_region_noise
from modules.decoding import decode_soft_boundaries
from modules.midi_extraction import SegmentationModel


class SegmentationInferenceModel(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
            inference_config: InferenceConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.inference_config = inference_config
        self.timestep = self.inference_config.features.timestep
        self.mel_spectrogram = StretchableMelSpectrogram(
            sample_rate=inference_config.features.audio_sample_rate,
            n_mels=inference_config.features.spectrogram.num_bins,
            n_fft=inference_config.features.fft_size,
            win_length=inference_config.features.win_size,
            hop_length=inference_config.features.hop_size,
            fmin=inference_config.features.spectrogram.fmin,
            fmax=inference_config.features.spectrogram.fmax,
            clip_val=1e-5,
        )
        self.lr = LengthRegulator()
        self.model = SegmentationModel(model_config)

    def _forward_and_decode(
            self, spectrogram, regions, mask,
            threshold, radius,
            language=None, t=None,
    ):
        logits, _ = self.model(
            spectrogram, regions=regions, t=t,
            language=language, mask=mask,
        )  # [B, T]
        soft_boundaries = logits.sigmoid()
        boundaries = decode_soft_boundaries(
            boundaries=soft_boundaries, mask=mask,
            threshold=threshold, radius=radius,
        )  # [B, T]
        return boundaries

    def forward(
            self, waveform: Tensor,
            known_durations: Tensor,
            threshold: Tensor,
            radius: Tensor,
            language: Tensor = None,
            t: Tensor = None,
    ) -> Tensor:
        """
        :param waveform: float32 [batch_size, num_samples]
        :param known_durations: float32 [batch_size, num_regions]
        :param threshold: float32 scalar
        :param radius: float32 scalar
        :param language: int64 [batch_size]
        :param t: float32 [num_steps]
        :return: durations: float32 [batch_size, num_predicted_regions]
        """
        spectrogram = self.mel_spectrogram(waveform).mT  # [B, T, C]
        B = waveform.size(0)
        T = spectrogram.size(1)
        Nt = t.size(0)
        regions = format_regions(
            durations=known_durations, length=T, timestep=self.timestep, lr=self.lr
        )  # [B, T]
        mask = regions != 0
        mask_long = mask.long()
        radius = (radius / self.timestep).round().long().clamp(min=1)
        if self.model_config.mode == "d3pm":
            for i in range(Nt):
                ti = t[i]
                if i > 0:
                    regions = d3pm_region_noise(regions, t=ti)
                boundaries_ = self._forward_and_decode(
                    spectrogram, regions=regions, t=ti,
                    language=language, mask=mask,
                    threshold=threshold, radius=radius,
                )  # [B, T]
                regions = ((boundaries_.long()).cumsum(dim=1) + 1) * mask_long
        elif self.model_config.mode == "completion":
            boundaries_ = self._forward_and_decode(
                spectrogram, regions=regions,
                language=language, mask=mask,
                threshold=threshold, radius=radius,
            )  # [B, T]
            regions = ((boundaries_.long()).cumsum(dim=1) + 1) * mask_long
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}.")
        N = regions.max()
        durations = regions.new_zeros((B, N + 1)).scatter_add(
            dim=1, index=regions, src=torch.ones_like(regions)
        )[:, 1:]
        durations = durations * self.timestep
        return durations
