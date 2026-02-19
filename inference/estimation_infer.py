from torch import nn, Tensor

from lib.config.schema import ModelConfig, InferenceConfig
from lib.feature.mel import StretchableMelSpectrogram
from modules.commons.tts_modules import LengthRegulator
from modules.decoding import decode_cascaded_dial_pointers
from modules.functional import format_regions
from modules.midi_extraction import EstimationModel


class EstimationInferenceModel(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
            inference_config: InferenceConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.inference_config = inference_config
        self.timestep = self.inference_config.features.timestep
        self.min_val = self.inference_config.midi_min
        self.max_val = self.inference_config.midi_max
        self.note_dial_periods = self.inference_config.note_dial_periods
        self.num_dials = len(self.inference_config.note_dial_periods)
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
        self.model = EstimationModel(model_config)

    def forward(
            self, waveform: Tensor,
            durations: Tensor,
            threshold: Tensor,
    ):
        spectrogram = self.mel_spectrogram(waveform).mT  # [B, T, C]
        T = spectrogram.size(1)
        N = durations.size(1)
        regions = format_regions(
            durations=durations, length=T, timestep=self.timestep, lr=self.lr
        )  # [B, T]
        t_mask = regions != 0
        n_mask = durations > 0
        n_mask_invert = ~n_mask
        estimations, _ = self.model(
            spectrogram, regions=regions, max_n=N,
            t_mask=t_mask, n_mask=n_mask,
        )  # [B, N, C_out]
        presence_logits = estimations[:, :, 0]  # [B, N]
        beam_norm = estimations[:, :, 1]  # [B, N]
        dials = estimations[:, :, 2:].reshape(-1, N, self.num_dials, 2)  # [B, N, num_dials, 2]
        presence = presence_logits.sigmoid() >= threshold
        presence = presence.masked_fill(n_mask_invert, False)
        beam = beam_norm * (self.max_val - self.min_val) + self.min_val
        scores = decode_cascaded_dial_pointers(
            beam=beam,
            dials=dials,
            periods=self.inference_config.note_dial_periods,
        )  # [B, N]
        scores = scores.masked_fill(n_mask_invert, 0.0)
        return scores, presence
