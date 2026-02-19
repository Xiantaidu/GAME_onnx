import json
import pathlib

import lightning.pytorch.callbacks
import torch
import yaml
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_info
from torch import Tensor

from lib import logging
from lib.config.core import ConfigBaseModel
from lib.config.formatter import ModelFormatter
from .me_infer import SegmentationEstimationInferenceModel
from .segmentation_infer import SegmentationInferenceModel
from .estimation_infer import EstimationInferenceModel
from .inference_module import InferenceModule
from lib.config.schema import ModelConfig, InferenceConfig, ConfigurationScope

__all__ = [
    "load_config_for_inference",
    "load_state_dict_for_inference",
    "load_inference_model",
    "load_segmentation_inference_model",
    "load_estimation_inference_model",
    "run_inference",
]


@rank_zero_only
def _log_config(cfg: ConfigBaseModel):
    formatter = ModelFormatter()
    print(formatter.format(cfg))


def load_config_for_inference(path: pathlib.Path, scope: int = 0) -> tuple[ModelConfig, InferenceConfig]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    model_config = ModelConfig.model_validate(config["model"], scope=scope)
    inference_config = InferenceConfig.model_validate(config["inference"], scope=scope)
    model_config.check(scope_mask=scope)
    inference_config.check(scope_mask=scope)

    return model_config, inference_config


def load_state_dict_for_inference(path: pathlib.Path, ema=True) -> dict[str, Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict: dict = checkpoint.get("state_dict", {})
    if ema and (ema_state_dict := checkpoint.get("ema_state_dict")) is not None:
        state_dict.update(ema_state_dict)
    if not state_dict:
        raise KeyError(f"No valid state dict found in checkpoint: {path}.")
    return state_dict


def load_inference_model(path: pathlib.Path) -> tuple[SegmentationEstimationInferenceModel, dict[str, int] | None]:
    model_config, inference_config = load_config_for_inference(
        path.parent / "config.yaml",
        scope=ConfigurationScope.SEGMENTATION | ConfigurationScope.ESTIMATION
    )
    model = SegmentationEstimationInferenceModel(model_config=model_config, inference_config=inference_config)
    state_dict = load_state_dict_for_inference(path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model_config.use_languages:
        lang_map_path = path.parent / "lang_map.json"
        if not lang_map_path.exists():
            raise FileNotFoundError(f"Language map file not found for segmentation model: {lang_map_path}")
        with open(lang_map_path, "r") as f:
            lang_map = json.load(f)
    else:
        lang_map = None

    logging.info(f"Loaded model from \'{path}\'.", callback=rank_zero_info)
    _log_config(model_config)
    _log_config(inference_config)

    return model, lang_map


def load_segmentation_inference_model(path: pathlib.Path) -> tuple[SegmentationInferenceModel, dict[str, int] | None]:
    model_config, inference_config = load_config_for_inference(
        path.parent / "config.yaml",
        scope=ConfigurationScope.SEGMENTATION
    )

    model = SegmentationInferenceModel(model_config=model_config, inference_config=inference_config)
    state_dict = load_state_dict_for_inference(path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model_config.use_languages:
        lang_map_path = path.parent / "lang_map.json"
        if not lang_map_path.exists():
            raise FileNotFoundError(f"Language map file not found for segmentation model: {lang_map_path}")
        with open(lang_map_path, "r") as f:
            lang_map = json.load(f)
    else:
        lang_map = None

    logging.info(f"Loaded segmentation model from \'{path}\'.", callback=rank_zero_info)
    _log_config(model_config)
    _log_config(inference_config)

    return model, lang_map


def load_estimation_inference_model(path: pathlib.Path) -> EstimationInferenceModel:
    model_config, inference_config = load_config_for_inference(
        path.parent / "config.yaml",
        scope=ConfigurationScope.ESTIMATION
    )
    model = EstimationInferenceModel(model_config=model_config, inference_config=inference_config)
    state_dict = load_state_dict_for_inference(path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logging.info(f"Loaded estimation model from \'{path}\'.", callback=rank_zero_info)
    _log_config(model_config)
    _log_config(inference_config)

    return model


def run_inference(
        model: SegmentationEstimationInferenceModel,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        num_workers: int,
        callbacks: list[lightning.pytorch.callbacks.Callback],
        segmentation_threshold: float = 0.3,
        segmentation_radius: float = 0.02,
        segmentation_d3pm_ts: list[float] = None,
        estimation_threshold: float = 0.2,
):
    module = InferenceModule(
        model=model,
        segmentation_threshold=segmentation_threshold,
        segmentation_radius=segmentation_radius,
        segmentation_d3pm_ts=segmentation_d3pm_ts,
        estimation_threshold=estimation_threshold,
    )
    trainer = lightning.pytorch.Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        shuffle=False,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate if hasattr(dataset, "collate") else None,
    )
    trainer.predict(module, dataloader)
