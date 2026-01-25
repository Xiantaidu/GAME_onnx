import csv
import pathlib
from dataclasses import dataclass

import dask
import numpy

from lib import logging
from .binarizer_base import BaseBinarizer, MetadataItem, DataSample, find_waveform_file


SYLLABLES_ITEM_ATTRIBUTES = [
    "language_id",  # int64
    "spectrogram",  # float32 [T, C]
    "durations",  # int64 [N,]
    "regions",  # int64 [T,]
    "boundaries",  # bool [T,]
]


@dataclass
class SyllablesMetadataItem(MetadataItem):
    syllable_durations: list[float]


class SyllablesBinarizer(BaseBinarizer):
    __data_attrs__ = SYLLABLES_ITEM_ATTRIBUTES

    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        with open(subset_dir / "index.csv", "r", encoding="utf8") as f:
            items = list(csv.DictReader(f))
        metadata_items = []
        for item in items:
            item_name = item["name"]
            language = item.get("language")
            waveform_fn = find_waveform_file(subset_dir, item_name)
            if waveform_fn is None:
                continue
            syllable_durations = [
                float(dur) for dur in item["syllables"].split()
            ]
            estimated_duration = sum(syllable_durations)
            metadata_items.append(SyllablesMetadataItem(
                item_name=item_name,
                language=language,
                waveform_fn=waveform_fn,
                estimated_duration=estimated_duration,
                syllable_durations=syllable_durations,
            ))
        return metadata_items

    def process_item(self, item: SyllablesMetadataItem) -> DataSample:
        language_id = numpy.array(self.lang_map.get(item.language, 0), dtype=numpy.int64)
        waveform = self.load_waveform(item.waveform_fn)
        spectrogram, length = self.get_mel(waveform)
        syllable_dur_sec = numpy.array(item.syllable_durations, dtype=numpy.float32)
        syllable_dur_frames = self.sec_dur_to_frame_dur(syllable_dur_sec, length)
        frame2syllable = self.length_regulator(syllable_dur_frames)
        boundaries = self.regions_to_boundaries(frame2syllable)
        data = {
            "language_id": language_id,
            "spectrogram": spectrogram,
            "durations": syllable_dur_frames,
            "regions": frame2syllable,
            "boundaries": boundaries,
        }
        data, length = dask.compute(data, length)
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.item_name,
            length=length,
            data=data,
        )
