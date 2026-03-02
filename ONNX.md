# ONNX Workflow and Structures

## Overview

<img src="assets/onnx-workflow.jpg" style="zoom: 50%;"  alt=""/>

## Configurations

A `config.json` carries the information needed for ONNX model inference. Structure:

```json5
{
    samplerate: 44100,
    // accepted audio sampling rate
    timestep: 0.01,
    // time step of each frame, in seconds
    languages: {
        // mappings from language codes to language IDs, starting at 1
    },
    embedding_dim: 256
    // embedding dim
}
```

## Modules and functions

| name           | description                                                                   |
|----------------|-------------------------------------------------------------------------------|
| encoder.onnx   | The encoder module. Maps waveforms to latent features.                        |
| segmenter.onnx | The segmenter module. Predicts note boundaries. Supports D3PM sampling loops. |
| estimator.onnx | The estimator module. Estimate note pitches from given boundaries.            |
| dur2bd.onnx    | Function to convert durations in seconds to boundaries in frames.             |
| bd2dur.onnx    | Function to convert boundaries in frame to durations in seconds.              |

## Dimensions

| name | type    | description                       |
|------|---------|-----------------------------------|
| B    | dynamic | Batch size.                       |
| L    | dynamic | Number of waveform samples.       |
| T    | dynamic | Number of feature frames.         |
| N    | dynamic | Number of notes.                  |
| C    | static  | Embedding dim of latent features. |

## Variables

| name             | type    | dtype   | shape     | description                                                                                                                                                                                            |
|------------------|---------|---------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| waveform         | in      | float32 | [B, T]    | Audio waveform.                                                                                                                                                                                        |
| duration         | in      | float32 | [B]       | Waveform duration, in seconds.                                                                                                                                                                         |
| x_seg            | mid     | float32 | [B, T, C] | Latent feature for segmentation.                                                                                                                                                                       |
| x_est            | mid     | float32 | [B, T, C] | Latent feature for estimation.                                                                                                                                                                         |
| maskT            | mid     | bool    | [B, T]    | Mask on frames.                                                                                                                                                                                        |
| known_durations  | in      | bool    | [B, T]    | Known or pre-defined region durations, in seconds. If there are no known regions, use `duration`.                                                                                                      |
| known_boundaries | mid     | bool    | [B, T]    | Known or pre-defined boundaries.                                                                                                                                                                       |
| prev_boundaries  | in/mid  | bool    | [B, T]    | Predicted boundaries of the previous sampling step. For the first sampling step, use `known_boundaries`.                                                                                               |
| language         | in      | int64   | [B]       | Language ID. 0 is unset or universal.                                                                                                                                                                  |
| threshold (1)    | in      | float32 | scalar    | Boundary decoding threshold. A frame with confidence above this value can be decoded as boundary. Recommended value: 0.2.                                                                              |
| radius           | in      | int64   | scalar    | Boundary decoding radius, in number of frames. A frame with confidence that is local maxima within this radius can be decoded as boundary.                                                             |
| t                | in      | float32 | scalar    | The time $t$ of D3PM sampling. Must be within $[0,1)$.                                                                                                                                                 |
| boundaries       | mid     | bool    | [B, T]    | Predicted boundaries. Should be flowed back to `prev_boundaries` during D3PM sampling loops. Can be replaced with `known_boundaries` if all note boundaries are known and the task is estimation-only. |
| maskN            | mid/out | bool    | [B, N]    | Mask on notes. 1 = valid, 0 = padding. Paddings only appear in the ends.                                                                                                                               |
| durations        | out     | float32 | [B, N]    | Predicted note durations, in seconds.                                                                                                                                                                  |
| threshold (2)    | in      | float32 | scalar    | Note presence threshold. A note with confidence above this value is considered as voiced.                                                                                                              |
| presence         | out     | bool    | [B, N]    | Note presence flags. 1 = voiced, 0 = unvoiced.                                                                                                                                                         |
| scores           | out     | float32 | [B, N]    | Note pitches, in semitones of MIDI scale (A4 = 69).                                                                                                                                                    |
