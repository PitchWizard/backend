## 피치 및 에너지 요약 기준 설정
import numpy as np
import librosa
from dataclasses import dataclass

@dataclass
class PitchSummary:
    frames: int
    voiced_ratio: float
    midi_min: float | None
    midi_median: float | None
    midi_max: float | None

@dataclass
class EnergySummary:
    rms_mean: float
    rms_std: float

def summarize_pitch(f0_hz, sr, hop_length=256, min_duration=0.3) -> PitchSummary:
    valid = ~np.isnan(f0_hz)
    frames = int(f0_hz.size)
    voiced_ratio = float(np.mean(valid)) if frames > 0 else 0.0
    if np.sum(valid) == 0:
        return PitchSummary(frames, 0.0, None, None, None)

    midi = librosa.hz_to_midi(f0_hz[valid])

    # 최소 지속시간 필터링
    min_frames = int(min_duration * sr / hop_length)
    midi_filtered = []
    last_val, count = None, 0
    for val in midi:
        if last_val is None or abs(val - last_val) < 0.5:
            count += 1
        else:
            if count >= min_frames:
                midi_filtered.extend([last_val] * count)
            count = 1
        last_val = val
    if last_val is not None and count >= min_frames:
        midi_filtered.extend([last_val] * count)

    midi_arr = np.array(midi_filtered) if len(midi_filtered) else midi

    # 퍼센타일 필터링
    q_low, q_high = np.percentile(midi_arr, [5, 95])
    midi_arr = midi_arr[(midi_arr >= q_low) & (midi_arr <= q_high)]

    return PitchSummary(frames, voiced_ratio,
                        float(np.min(midi_arr)),
                        float(np.median(midi_arr)),
                        float(np.max(midi_arr)))

def summarize_energy(rms: np.ndarray) -> EnergySummary:
    return EnergySummary(float(np.mean(rms)), float(np.std(rms)))
