import csv
import pathlib
from dataclasses import dataclass

import dask
import librosa
import numpy
from scipy import interpolate

from lib import logging
from .binarizer_base import BaseBinarizer, MetadataItem, DataSample, find_waveform_file

NOTES_ITEM_ATTRIBUTES = [
    "spectrogram",  # float32 [T, C]
    "scores",  # float32 [N,] MIDI pitch
    "rest",  # bool [N,] rest flags
    "durations",  # int64 [N,] note durations in frames
    "regions",  # int64 [T,] syllable regions mapping frame -> syllable index
    "subregions",  # int64 [T,] note regions mapping frame -> note index
    "division",  # bool [T,] syllable boundaries
    "subdivision",  # bool [T,] note boundaries
]


@dataclass
class NotesMetadataItem(MetadataItem):
    note_scores: list[str]
    note_durations: list[float]
    note_slurs: list[bool]


class NotesBinarizer(BaseBinarizer):
    __data_attrs__ = NOTES_ITEM_ATTRIBUTES

    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        with open(subset_dir / "index.csv", "r", encoding="utf8") as f:
            items = list(csv.DictReader(f))
        metadata_items = []
        for item in items:
            item_name = item["name"]
            waveform_fn = find_waveform_file(subset_dir, item_name)
            if waveform_fn is None:
                continue
            notes = item["notes"].split()
            durations = [float(dur) for dur in item["durations"].split()]
            slurs = [bool(int(s)) for s in item["slurs"].split()]
            if not (len(notes) == len(durations) == len(slurs)):
                logging.error(
                    f"Length mismatch in raw dataset \'{subset_dir.as_posix()}\': "
                    f"item \'{item_name}\', notes({len(notes)}), durations({len(durations)}), slurs({len(slurs)})."
                )
                continue
            estimated_duration = sum(durations)
            metadata_items.append(NotesMetadataItem(
                item_name=item_name,
                language=None,
                waveform_fn=waveform_fn,
                estimated_duration=estimated_duration,
                note_scores=notes,
                note_durations=durations,
                note_slurs=slurs,
            ))
        return metadata_items

    def process_item(self, item: NotesMetadataItem) -> DataSample:
        waveform = self.load_waveform(item.waveform_fn)
        spectrogram, length = self.get_mel(waveform)
        note_midi = numpy.array(
            [(librosa.note_to_midi(n, round_midi=False) if n != "rest" else -1) for n in item.note_scores],
            dtype=numpy.float32
        )
        note_rest = note_midi < 0
        note_midi_interp = self.interpolate_rest(note_midi, note_rest)
        note_slur = numpy.array(item.note_slurs, dtype=numpy.bool_)
        note_dur_sec = numpy.array(item.note_durations, dtype=numpy.float32)
        note_dur_frames = self.sec_dur_to_frame_dur(note_dur_sec, length)
        syllable_dur_frames = self.note_dur_to_syllable_dur(note_dur_frames, note_slur)
        frame2note = self.length_regulator(note_dur_frames)
        frame2syllable = self.length_regulator(syllable_dur_frames)
        syllable_boundaries = self.regions_to_boundaries(frame2syllable)
        note_boundaries = self.regions_to_boundaries(frame2note)
        data = {
            "spectrogram": spectrogram,
            "scores": note_midi_interp,
            "rest": note_rest,
            "durations": note_dur_frames,
            "regions": frame2syllable,
            "subregions": frame2note,
            "division": syllable_boundaries,
            "subdivision": note_boundaries,
        }
        data, length = dask.compute(data, length)
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.item_name,
            length=length,
            data=data,
        )

    @dask.delayed
    def interpolate_rest(self, note_midi: numpy.ndarray, note_rest: numpy.ndarray) -> numpy.ndarray:
        interp_func = interpolate.interp1d(
            numpy.where(~note_rest)[0], note_midi[~note_rest],
            kind='nearest', fill_value='extrapolate'
        )
        interp_midi = note_midi.copy()
        interp_midi[note_rest] = interp_func(numpy.where(note_rest)[0])
        return interp_midi

    @dask.delayed
    def note_dur_to_syllable_dur(self, note_dur_frames: numpy.ndarray, note_slurs: numpy.ndarray) -> numpy.ndarray:
        note2syllable = numpy.cumsum(~note_slurs)
        syllable_dur = numpy.zeros((note2syllable[-1] + 1,), dtype=numpy.int64)
        numpy.add.at(syllable_dur, note2syllable, note_dur_frames)
        return syllable_dur[1:]
