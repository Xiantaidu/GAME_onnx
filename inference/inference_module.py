import librosa
import lightning.pytorch
import torch
from torch import Tensor

from inference.estimation_infer import EstimationInferenceModel
from inference.segmentation_infer import SegmentationInferenceModel


class InferenceModule(lightning.pytorch.LightningModule):
    def __init__(
            self,
            segmentation_model: SegmentationInferenceModel,
            estimation_model: EstimationInferenceModel,
            segmentation_threshold: float = 0.2,
            segmentation_radius: float = 0.02,
            segmentation_d3pm_ts: list[float] = None,
            estimation_threshold: float = 0.2,
    ):
        super().__init__()
        if segmentation_d3pm_ts is None:
            segmentation_d3pm_ts = [0.0]
        self.segmentation_d3pm_ts = segmentation_d3pm_ts
        self.segmentation_threshold = segmentation_threshold
        self.segmentation_radius = segmentation_radius
        self.estimation_threshold = estimation_threshold
        self.segmentation_model = segmentation_model
        self.estimation_model = estimation_model

    def predict_step(self, batch: dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        waveform = batch["waveform"]
        samplerate = batch["samplerate"]
        if samplerate != (seg_sr := self.segmentation_model.inference_config.features.audio_sample_rate):
            waveforms_seg = torch.stack([
                librosa.resample(w.cpu().numpy(), orig_sr=samplerate, target_sr=seg_sr)
                for w in waveform.unbind(dim=0)
            ], dim=0).to(waveform)
        else:
            waveforms_seg = waveform
        if samplerate != (est_sr := self.estimation_model.inference_config.features.audio_sample_rate):
            waveforms_est = torch.stack([
                librosa.resample(w.cpu().numpy(), orig_sr=samplerate, target_sr=est_sr)
                for w in waveform.unbind(dim=0)
            ], dim=0).to(waveform)
        else:
            waveforms_est = waveform
        known_durations = batch["known_durations"]
        language = batch["language"]
        durations = self.segmentation_model(
            waveform=waveforms_seg,
            known_durations=known_durations,
            language=language,
            t=torch.tensor(self.segmentation_d3pm_ts).to(waveform),
            threshold=torch.tensor(self.segmentation_threshold).to(waveform),
            radius=torch.tensor(self.segmentation_radius).to(waveform),
        )
        scores, presence = self.estimation_model(
            waveform=waveforms_est,
            durations=durations,
            threshold=torch.tensor(self.estimation_threshold).to(waveform),
        )
        return {
            "durations": durations,
            "scores": scores,
            "presence": presence,
        }
