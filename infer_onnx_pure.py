import glob
import pathlib
import json
import math
from typing import Any
import numpy as np
import click
import librosa
import mido

_OPT_KEY_SEG_THRESHOLD = "seg_threshold"
_OPT_KEY_SEG_RADIUS = "seg_radius"
_OPT_KEY_SEG_D3PM_T0 = "t0"
_OPT_KEY_SEG_D3PM_NSTEPS = "nsteps"
_OPT_KEY_EST_THRESHOLD = "est_threshold"

def _validate_d3pm_ts(ctx, param, value) -> list[float] | None:
    if value is None:
        return None
    try:
        ts = [float(t.strip()) for t in value.split(",")]
        if not ts:
            raise ValueError("At least one T value must be provided.")
        if any(t < 0 or t >= 1 for t in ts):
            raise ValueError("All T values must be in the range (0, 1).")
        return ts
    except Exception as e:
        raise click.BadParameter(f"Invalid T values: {e}")

def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")

def _validate_output_formats(ctx, param, value) -> set[str]:
    try:
        formats = {fmt.strip().lower() for fmt in value.split(",")}
        supported_formats = {"mid"}
        if not formats.issubset(supported_formats):
            raise ValueError(
                f"Unsupported formats: {formats - supported_formats}. "
                f"Supported formats: {supported_formats}"
            )
        return formats
    except Exception as e:
        raise click.BadParameter(f"Invalid output formats: {e}")

def _validate_path_or_glob(ctx, param, value) -> str:
    try:
        paths = []
        for v in value:
            if glob.has_magic(v):
                paths.extend(glob.glob(v, recursive=True))
            else:
                paths.append(v)
        if not paths:
            raise FileNotFoundError(f"No files found for paths: {value}")
        paths = [pathlib.Path(p) for p in paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Path does not exist: {p}")
            if not p.is_file():
                raise FileNotFoundError(f"Path is not a file: {p}")
        return paths
    except Exception as e:
        raise click.BadParameter(f"Invalid path or glob: {e}")

def _t0_nstep_to_ts(t0: float, nsteps: int) -> list[float]:
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]

def _get_language_id(language: str, lang_map: dict[str, int]) -> int:
    if language and lang_map:
        if language not in lang_map:
            raise ValueError(
                f"Language '{language}' not supported by the segmentation model. "
                f"Supported languages: {', '.join(lang_map.keys())}"
            )
        language_id = lang_map[language]
    else:
        language_id = 0
    return language_id

def _parse_filemap(path: pathlib.Path, exts: set[str], glb: str | None) -> dict[str, pathlib.Path]:
    if path.is_file():
        return {path.name: path}
    elif path.is_dir():
        if glb:
            files = [f for f in path.rglob(glb) if f.is_file()]
        else:
            files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in exts]
        filemap = {f.relative_to(path).as_posix(): f for f in files}
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    else:
        raise ValueError(f"Invalid path: {path}")

class ONNXInferenceModelPure:
    def __init__(self, model_dir: pathlib.Path):
        import onnxruntime as ort
        
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            self.onnx_config = json.load(f)
            
        self.samplerate = self.onnx_config["samplerate"]
        self.timestep = self.onnx_config["timestep"]
        self.loop = self.onnx_config.get("loop", True)
        self.languages = self.onnx_config.get("languages", None)
        
        # Using DmlExecutionProvider for DirectML hardware acceleration, fallback to CPU
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.encoder = ort.InferenceSession((model_dir / "encoder.onnx").as_posix(), providers=providers)
        self.segmenter = ort.InferenceSession((model_dir / "segmenter.onnx").as_posix(), providers=providers)
        self.estimator = ort.InferenceSession((model_dir / "estimator.onnx").as_posix(), providers=providers)
        self.dur2bd = ort.InferenceSession((model_dir / "dur2bd.onnx").as_posix(), providers=providers)
        self.bd2dur = ort.InferenceSession((model_dir / "bd2dur.onnx").as_posix(), providers=providers)

        active_provider = self.encoder.get_providers()[0]
        if active_provider == 'DmlExecutionProvider':
            print("🚀 Using DirectML (GPU) for inference.")
        elif active_provider == 'CPUExecutionProvider':
            print("🐢 Using CPU for inference.")
        else:
            print(f"Using {active_provider} for inference.")

    def forward(
            self, waveform_np: np.ndarray,
            known_durations_np: np.ndarray,
            boundary_threshold_np: np.ndarray,
            boundary_radius_np: np.ndarray,
            score_threshold_np: np.ndarray,
            language_np: np.ndarray = None,
            t_np: np.ndarray = None,
    ):
        L_STATIC = 441000  # 10 seconds at 44100 Hz
        T_STATIC = 1000    # 10 seconds / 0.01 timestep
        N_STATIC = 200     # Fixed max notes per chunk
        
        if language_np is None:
            language_np = np.zeros(waveform_np.shape[0], dtype=np.int64)
            
        if t_np is None:
            t_np = np.array([], dtype=np.float32)

        B = waveform_np.shape[0]
        L = waveform_np.shape[1]
        
        all_durations = []
        all_presence = []
        all_scores = []
        
        for b in range(B):
            b_durations = []
            b_presence = []
            b_scores = []
            
            w = waveform_np[b]
            durs = known_durations_np[b]
            
            num_chunks = int(np.ceil(L / L_STATIC))
            if num_chunks == 0:
                num_chunks = 1
                
            for c in range(num_chunks):
                start_idx = c * L_STATIC
                end_idx = min((c + 1) * L_STATIC, L)
                
                chunk_w = w[start_idx:end_idx]
                chunk_len = len(chunk_w)
                
                padded_w = np.zeros((1, L_STATIC), dtype=np.float32)
                padded_w[0, :chunk_len] = chunk_w
                
                chunk_duration_sec = chunk_len / self.samplerate
                
                padded_known_durs = np.zeros((1, N_STATIC), dtype=np.float32)
                padded_known_durs[0, 0] = chunk_duration_sec
                
                enc_out = self.encoder.run(None, {
                    "waveform": padded_w,
                    "duration": np.array([chunk_duration_sec], dtype=np.float32)
                })
                x_seg, x_est, maskT = enc_out[0], enc_out[1], enc_out[2]
                
                # Pass maskT to dur2bd for dynamic models compatibility
                dur2bd_inputs = [i.name for i in self.dur2bd.get_inputs()]
                dur2bd_args = {"durations": padded_known_durs}
                if "maskT" in dur2bd_inputs:
                    dur2bd_args["maskT"] = maskT
                
                dur2bd_out = self.dur2bd.run(None, dur2bd_args)
                known_boundaries = dur2bd_out[0]
                
                boundaries = known_boundaries
                seg_inputs = [i.name for i in self.segmenter.get_inputs()]
                
                if self.loop and len(t_np) > 0:
                    for ti in t_np:
                        seg_args = {
                            "x_seg": x_seg,
                            "maskT": maskT,
                            "threshold": boundary_threshold_np.astype(np.float32),
                            "radius": boundary_radius_np.astype(np.int64)
                        }
                        if "language" in seg_inputs:
                            seg_args["language"] = np.array([language_np[b]], dtype=np.int64)
                        if "known_boundaries" in seg_inputs:
                            seg_args["known_boundaries"] = known_boundaries
                        if "prev_boundaries" in seg_inputs:
                            seg_args["prev_boundaries"] = boundaries
                        if "t" in seg_inputs:
                            seg_args["t"] = np.array(ti, dtype=np.float32)
                        
                        seg_out = self.segmenter.run(None, seg_args)
                        boundaries = seg_out[0]
                else:
                    seg_args = {
                        "x_seg": x_seg,
                        "maskT": maskT,
                        "threshold": boundary_threshold_np.astype(np.float32),
                        "radius": boundary_radius_np.astype(np.int64)
                    }
                    if "language" in seg_inputs:
                        seg_args["language"] = np.array([language_np[b]], dtype=np.int64)
                    if "known_boundaries" in seg_inputs:
                        seg_args["known_boundaries"] = known_boundaries
                    if "prev_boundaries" in seg_inputs:
                        seg_args["prev_boundaries"] = boundaries
                    if "t" in seg_inputs:
                        seg_args["t"] = np.array(0.0, dtype=np.float32)
                        
                    seg_out = self.segmenter.run(None, seg_args)
                    boundaries = seg_out[0]
                    
                bd2dur_out = self.bd2dur.run(None, {
                    "boundaries": boundaries,
                    "maskT": maskT
                })
                durations, maskN = bd2dur_out[0], bd2dur_out[1]
                
                actual_n = maskN.shape[1]
                if actual_n < N_STATIC:
                    pad_maskN = np.zeros((1, N_STATIC), dtype=bool)
                    pad_maskN[0, :actual_n] = maskN[0]
                    padded_maskN = pad_maskN
                else:
                    padded_maskN = maskN[:, :N_STATIC]
                    
                if actual_n < N_STATIC:
                    pad_durs = np.zeros((1, N_STATIC), dtype=np.float32)
                    pad_durs[0, :actual_n] = durations[0]
                    padded_durations = pad_durs
                else:
                    padded_durations = durations[:, :N_STATIC]
                
                est_inputs = [i.name for i in self.estimator.get_inputs()]
                est_args = {
                    "x_est": x_est,
                    "boundaries": boundaries,
                    "maskT": maskT,
                    "maskN": padded_maskN,
                    "threshold": score_threshold_np.astype(np.float32)
                }
                est_args = {k: v for k, v in est_args.items() if k in est_inputs}
                
                est_out = self.estimator.run(None, est_args)
                presence, scores = est_out[0], est_out[1]
                
                valid_notes = padded_maskN[0].astype(bool)
                b_durations.extend(padded_durations[0][valid_notes].tolist())
                b_presence.extend(presence[0][valid_notes].tolist())
                b_scores.extend(scores[0][valid_notes].tolist())
                
            all_durations.append(b_durations)
            all_presence.append(b_presence)
            all_scores.append(b_scores)
            
        max_n = max([len(d) for d in all_durations] + [1])
        
        out_dur = np.zeros((B, max_n), dtype=np.float32)
        out_pres = np.zeros((B, max_n), dtype=bool)
        out_score = np.zeros((B, max_n), dtype=np.float32)
        
        for b in range(B):
            n = len(all_durations[b])
            if n > 0:
                out_dur[b, :n] = all_durations[b]
                out_pres[b, :n] = all_presence[b]
                out_score[b, :n] = all_scores[b]
                
        return out_dur, out_pres, out_score

def load_onnx_inference_model_pure(path: pathlib.Path):
    model = ONNXInferenceModelPure(path)
    print(f"Loaded ONNX model from '{path}'.")
    return model, model.languages

@click.group()
def main():
    pass

def shared_options(func=None, *, defaults: dict[str, Any] = None):
    if defaults is None:
        defaults = {}

    options = [
        click.option(
            "-m", "--model", type=click.Path(
                exists=True, dir_okay=True, file_okay=False, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the ONNX model directory."
        ),
        click.option(
            "-l", "--language", type=str, default=None, show_default=False,
            help="Language code for better segmentation if supported."
        ),
        click.option(
            "--seg-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_THRESHOLD, 0.2),
            help="Boundary decoding threshold for segmentation model."
        ),
        click.option(
            "--seg-radius", type=click.FloatRange(min=0.01), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_RADIUS, 0.02),
            help="Boundary decoding radius for segmentation model."
        ),
        click.option(
            "--t0", "--seg-d3pm-t0", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_D3PM_T0, 0.0),
            help="Starting T value (t0) of D3PM for segmentation model."
        ),
        click.option(
            "--nsteps", "--seg-d3pm-nsteps", type=click.IntRange(min=1), show_default=True,
            default=defaults.get(_OPT_KEY_SEG_D3PM_NSTEPS, 8),
            help="Number of D3PM sampling steps for segmentation model."
        ),
        click.option(
            "--ts", "--seg-d3pm-ts", type=str, default=None, show_default=False,
            callback=_validate_d3pm_ts,
            help=(
                "Custom T values for D3PM sampling in segmentation model, separated by commas. "
                "Overrides --t0 and --nsteps if provided."
            )
        ),
        click.option(
            "--est-threshold", type=click.FloatRange(
                min=0, min_open=False, max=1, max_open=True
            ), show_default=True,
            default=defaults.get(_OPT_KEY_EST_THRESHOLD, 0.2),
            help="Presence detecting threshold for estimation model."
        ),
    ]

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)

@main.command(
    name="extract",
    help="Extract MIDI from single or multiple audio files (Pure Numpy/ONNX implementation)."
)
@click.argument(
    "path", type=click.Path(
        exists=True, dir_okay=True, file_okay=True, readable=True, path_type=pathlib.Path
    ),
)
@shared_options(defaults={
    _OPT_KEY_SEG_D3PM_T0: 0.0,
    _OPT_KEY_SEG_D3PM_NSTEPS: 8,
})
@click.option(
    "--input-formats", type=str, default="wav,flac,mp3,aac,ogg", show_default=True,
    callback=_validate_exts,
    help="List of audio file extensions to process, separated by commas."
)
@click.option(
    "--glob", "glb", type=str, default=None, show_default=False,
    help="Glob pattern to filter audio files (i.e. *.wav)."
)
@click.option(
    "--output-formats", type=str, default="mid", show_default=True,
    callback=_validate_output_formats,
    help="List of output formats to save the extracted results, separated by commas. Supported formats: mid."
)
@click.option(
    "--tempo", type=click.FloatRange(min=0, min_open=True), default=120, show_default=True,
    help="Tempo (in BPM) to save MIDI files with."
)
@click.option(
    "--output-dir", type=click.Path(
        exists=False, dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path
    ),
    default=None, show_default=False,
    help="Directory to save the extracted results."
)
def extract(
        path: pathlib.Path,
        model: pathlib.Path,
        language: str,
        seg_threshold: float,
        seg_radius: float,
        t0: float,
        nsteps: int,
        ts: list[float],
        est_threshold: float,
        input_formats: set[str],
        glb: str,
        output_formats: set[str],
        tempo: float,
        output_dir: pathlib.Path,
):
    if ts is None:
        ts = _t0_nstep_to_ts(t0, nsteps)
    filemap = _parse_filemap(path, input_formats, glb)
    if output_dir is None:
        output_dir = path if path.is_dir() else path.parent

    onnx_model, lang_map = load_onnx_inference_model_pure(model)
    language_id = _get_language_id(language, lang_map)
    sr = onnx_model.samplerate

    import time
    from inference.slicer2 import Slicer
    
    slicer = Slicer(
        sr=sr,
        threshold=-40.,
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    
    for key, filepath in filemap.items():
        print(f"Processing {filepath}...")
        waveform, _ = librosa.load(filepath, sr=sr, mono=True)
        chunks = slicer.slice(waveform)
        
        all_notes = []
        
        for chunk in chunks:
            chunk_wav = chunk["waveform"]
            offset = chunk["offset"]
            length = len(chunk_wav) / sr
            
            b_wav = np.expand_dims(chunk_wav.astype(np.float32), 0)
            b_dur = np.array([[length]], dtype=np.float32)
            
            t_np = np.array(ts, dtype=np.float32)
            seg_thresh = np.array(seg_threshold, dtype=np.float32)
            seg_rad = np.array(round(seg_radius / onnx_model.timestep), dtype=np.int64)
            est_thresh = np.array(est_threshold, dtype=np.float32)
            lang = np.array([language_id], dtype=np.int64)
            
            durations, presence, scores = onnx_model.forward(
                waveform_np=b_wav,
                known_durations_np=b_dur,
                boundary_threshold_np=seg_thresh,
                boundary_radius_np=seg_rad,
                score_threshold_np=est_thresh,
                language_np=lang,
                t_np=t_np
            )
            
            durations = durations[0]
            scores = scores[0]
            presence = presence[0]
            
            # Reconstruct notes (from cumsum logic in callbacks.py)
            note_onset = np.cumsum(np.pad(durations, (1, 0), mode="constant")[:-1])
            note_onset = np.clip(note_onset, a_min=None, a_max=length) + offset
            note_offset = np.cumsum(durations)
            note_offset = np.clip(note_offset, a_min=None, a_max=length) + offset
            
            for onset, offset_t, score, valid in zip(note_onset, note_offset, scores, presence):
                if offset_t - onset <= 0:
                    continue
                if not valid:
                    continue
                all_notes.append({
                    "onset": float(onset),
                    "offset": float(offset_t),
                    "pitch": float(score)
                })
        
        # Sort and merge overlapping notes
        sorted_notes = sorted(all_notes, key=lambda x: (x["onset"], x["offset"], x["pitch"]))
        last_time = 0
        i = 0
        while i < len(sorted_notes):
            note = sorted_notes[i]
            note["onset"] = max(note["onset"], last_time)
            note["offset"] = max(note["offset"], note["onset"])
            if note["offset"] <= note["onset"]:
                sorted_notes.pop(i)
            else:
                last_time = note["offset"]
                i += 1
                
        # Save MIDI
        if "mid" in output_formats:
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
            last_time_ticks = 0
            
            # Use float ticks for intermediate calculation to avoid accumulation of rounding errors
            for note in sorted_notes:
                # To prevent rounding errors, we first compute the exact float target ticks.
                # Note: `tempo * 8` is the conversion factor for seconds -> ticks at 480 PPQ and 120 BPM.
                # Specifically: 1 second = (tempo / 60) * 480 ticks. So seconds * (tempo / 60) * 480 = seconds * tempo * 8.
                onset_abs_ticks_float = note["onset"] * tempo * 8
                offset_abs_ticks_float = note["offset"] * tempo * 8
                midi_pitch = round(note["pitch"])
                
                if offset_abs_ticks_float <= onset_abs_ticks_float:
                    continue
                
                # We need to reach `onset_abs_ticks_float` starting from `last_time_ticks` (which is an exact integer).
                delta_onset = max(0, round(onset_abs_ticks_float - last_time_ticks))
                
                # We append note_on. This consumes `delta_onset` ticks.
                track.append(mido.Message("note_on", note=midi_pitch, time=delta_onset))
                
                # Update our integer tick position.
                last_time_ticks += delta_onset
                
                # Now we need to reach `offset_abs_ticks_float` starting from the new `last_time_ticks`.
                delta_offset = max(0, round(offset_abs_ticks_float - last_time_ticks))
                
                # Append note_off. This consumes `delta_offset` ticks.
                track.append(mido.Message("note_off", note=midi_pitch, time=delta_offset))
                
                # Update our integer tick position.
                last_time_ticks += delta_offset

            out_path = (output_dir / key).with_suffix(".mid")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with mido.MidiFile(charset="utf8") as midi_file:
                midi_file.tracks.append(track)
                midi_file.save(out_path)
            print(f"Saved MIDI file: {out_path}")

    print("Inference completed.")

if __name__ == '__main__':
    main()
