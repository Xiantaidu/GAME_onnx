import librosa
import lightning.pytorch
import torch
from torch import Tensor

from inference.me_infer import SegmentationEstimationInferenceModel


class InferenceModule(lightning.pytorch.LightningModule):
    def __init__(
            self,
            model: SegmentationEstimationInferenceModel,
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
        self.model = model

    def predict_step(self, batch: dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        waveform = batch["waveform"]
        samplerate = batch["samplerate"]
        if samplerate != (model_sr := self.model.inference_config.features.audio_sample_rate):
            waveform = torch.stack([
                librosa.resample(w.cpu().numpy(), orig_sr=samplerate, target_sr=model_sr)
                for w in waveform.unbind(dim=0)
            ], dim=0).to(waveform)
        known_durations = batch["known_durations"]
        language = batch["language"]
        durations, presence, scores = self.model(
            waveform=waveform,
            known_durations=known_durations,
            language=language,
            t=torch.tensor(self.segmentation_d3pm_ts).to(waveform),
            boundary_threshold=torch.tensor(self.segmentation_threshold).to(waveform),
            boundary_radius=torch.tensor(self.segmentation_radius).to(waveform),
            score_threshold=torch.tensor(self.estimation_threshold).to(waveform),
        )
        return {
            "durations": durations,
            "presence": presence,
            "scores": scores,
        }
