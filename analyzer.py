# analyzer.py
from audio_io import download_youtube_audio, separate_vocals_demucs, load_audio
from pitch_extract import estimate_pitch_librosa, estimate_pitch_torchcrepe
from features import summarize_pitch, summarize_energy
from utils import midi_to_note

import numpy as np
import librosa
from librosa import display as lrdisplay   # ✅ 추가: 함수 밖에서 별칭으로 임포트
import torch

def compute_rms(y, sr, hop_length=256):
    print("[4/5] RMS 에너지 추출 중…", flush=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times

def _hybrid_replace_extremes_with_crepe(y, sr, hop_length, f0_librosa):
    """상·하위 5% 극단구간을 torchcrepe 값으로 보정"""
    valid = ~np.isnan(f0_librosa)
    if np.sum(valid) == 0:
        print("[하이브리드] 유효한 librosa 피치가 없어 보정을 생략합니다.", flush=True)
        return f0_librosa

    vals = f0_librosa[valid]
    lo, hi = np.percentile(vals, [5, 95])

    # torchcrepe로 전체 f0를 한 번 산출
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[하이브리드] torchcrepe 재분석 시작 (device={device})", flush=True)
    f0_crepe, _ = estimate_pitch_torchcrepe(y, sr, hop_length=hop_length, device=device)

    # 길이 정렬(간혹 1프레임 차이 대비)
    L = min(len(f0_librosa), len(f0_crepe))
    f0_librosa = f0_librosa[:L]
    f0_crepe   = f0_crepe[:L]

    # 극단 구간 마스크
    mask_extreme = (f0_librosa <= lo) | (f0_librosa >= hi)
    # librosa에서 nan이던 프레임은 crepe 값으로 대체하는 것도 유용
    mask_nan = np.isnan(f0_librosa)

    # 교체
    replaced = f0_librosa.copy()
    replaced[mask_extreme | mask_nan] = f0_crepe[mask_extreme | mask_nan]

    print(f"[하이브리드] 교체 프레임 수: {int(np.sum(mask_extreme | mask_nan))} / {L}", flush=True)
    return replaced

def analyze_audio_summary(source: str, engine="pyin", target_sr=22050,
                          hop_length=256, plot=False):
    if source.startswith("http"):
        source = download_youtube_audio(source)
    source = separate_vocals_demucs(source)
    y, sr = load_audio(source, target_sr=target_sr, mono=True)

    if engine in ["pyin", "yin"]:
        f0_hz, times = estimate_pitch_librosa(y, sr, hop_length=hop_length, use_pyin=(engine=="pyin"))
    elif engine == "torchcrepe":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f0_hz, times = estimate_pitch_torchcrepe(y, sr, hop_length=hop_length, device=device)
    elif engine == "hybrid":
        # 1차: librosa 전체
        f0_l, times = estimate_pitch_librosa(y, sr, hop_length=hop_length, use_pyin=True)
        # 2차: 극단구간/결측 프레임 torchcrepe로 보정
        f0_hz = _hybrid_replace_extremes_with_crepe(y, sr, hop_length, f0_l)
    else:
        raise ValueError("engine은 pyin, yin, torchcrepe, hybrid 중 하나여야 합니다.")

    valid_f0 = f0_hz[~np.isnan(f0_hz)]
    if len(valid_f0) > 0:
        raw_max_hz = np.max(valid_f0)
        raw_min_hz = np.min(valid_f0)
        print(f"[디버그] 최저 Hz: {raw_min_hz:.2f}, 최고 Hz: {raw_max_hz:.2f}", flush=True)
        max_midi = librosa.hz_to_midi(raw_max_hz)
        min_midi = librosa.hz_to_midi(raw_min_hz)
        print(f"[디버그] 최저음: {librosa.midi_to_note(min_midi, octave=True)} (MIDI {min_midi:.2f}), "
              f"최고음: {librosa.midi_to_note(max_midi, octave=True)} (MIDI {max_midi:.2f})", flush=True)

    rms, _ = compute_rms(y, sr, hop_length=hop_length)
    p_sum = summarize_pitch(f0_hz, sr, hop_length)
    e_sum = summarize_energy(rms)

    print("[완료] 분석 종료", flush=True)

    if plot:
        import matplotlib.pyplot as plt
    # import librosa.display   # ❌ 이 줄 삭제
        fig, ax = plt.subplots(figsize=(10, 4))
        lrdisplay.specshow(  # ✅ librosa.display → lrdisplay 로 변경
            librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
            sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax
    )
        ax.set_title('Spectrogram + Pitch Track')
        ax.plot(times, f0_hz, label='f0', linewidth=2)
        ax.legend()
        plt.savefig("pitch_track.png")
        print("[시각화] pitch_track.png 저장 완료", flush=True)


    return {
        "engine": engine,
        "sr": int(sr),
        "duration_s": float(len(y) / sr),
        "frames": p_sum.frames,
        "voiced_ratio": p_sum.voiced_ratio,
        "midi_min": p_sum.midi_min,
        "midi_min_note": midi_to_note(p_sum.midi_min),
        "midi_median": p_sum.midi_median,
        "midi_median_note": midi_to_note(p_sum.midi_median),
        "midi_max": p_sum.midi_max,
        "midi_max_note": midi_to_note(p_sum.midi_max),
        "rms_mean": e_sum.rms_mean,
        "rms_std": e_sum.rms_std,
    }
