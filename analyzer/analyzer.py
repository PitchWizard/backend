# analyzer/analyzer.py
from .audio_io import download_youtube_audio, separate_vocals_uvr5_mdx, load_audio
from .features import summarize_pitch, summarize_energy, compute_rms
from .utils import midi_to_note

import os
import numpy as np
import librosa
from librosa import display as lrdisplay
import torch

HOP_LENGTH_DEFAULT = 256
RMVPE_SR = 16000          # RMVPE 요구 샘플레이트
RMVPE_HOP = 160           # RMVPE 내부 hop (10ms @ 16kHz)

# 프로젝트 루트의 models/rmvpe.pt 또는 루트 rmvpe.pt 순으로 탐색
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RMVPE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "rmvpe.pt")
if not os.path.exists(RMVPE_MODEL_PATH):
    RMVPE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "rmvpe.pt")


def analyze_audio_summary(source: str, hop_length=HOP_LENGTH_DEFAULT, plot=False):
    if source.startswith("http"):
        source = download_youtube_audio(source)

    vocals_path, instrumental_path = separate_vocals_uvr5_mdx(source)

    # RMVPE는 16kHz 필요
    y, sr = load_audio(vocals_path, target_sr=RMVPE_SR, mono=True)

    print("[4/5] RMVPE 피치 추출 중…", flush=True)
    from .rmvpe import RMVPE

    if not os.path.exists(RMVPE_MODEL_PATH):
        raise FileNotFoundError(
            f"RMVPE 모델 파일이 없습니다: {RMVPE_MODEL_PATH}\n"
            "다운로드: https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RMVPE] device={device}", flush=True)

    rmvpe_model = RMVPE(RMVPE_MODEL_PATH, is_half=False, device=device)
    f0_hz = rmvpe_model.infer_from_audio(y, thred=0.03)
    f0_hz = np.where(f0_hz > 0, f0_hz, np.nan)  # 0(무성음) → NaN
    times = np.arange(len(f0_hz)) * (RMVPE_HOP / RMVPE_SR)  # 10ms 단위

    # 디버그 로그
    valid_f0 = f0_hz[~np.isnan(f0_hz)]
    if len(valid_f0) > 0:
        raw_max_hz = float(np.max(valid_f0))
        raw_min_hz = float(np.min(valid_f0))
        print(f"[디버그] 최저 Hz: {raw_min_hz:.2f}, 최고 Hz: {raw_max_hz:.2f}", flush=True)
        max_midi = float(librosa.hz_to_midi(raw_max_hz))
        min_midi = float(librosa.hz_to_midi(raw_min_hz))
        print(f"[디버그] 최저음: {librosa.midi_to_note(min_midi, octave=True)} (MIDI {min_midi:.2f}), "
              f"최고음: {librosa.midi_to_note(max_midi, octave=True)} (MIDI {max_midi:.2f})", flush=True)

    rms, _ = compute_rms(y, sr, hop_length=hop_length)
    p_sum = summarize_pitch(f0_hz, sr=RMVPE_SR, hop_length=RMVPE_HOP)
    e_sum = summarize_energy(rms)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        lrdisplay.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram + Pitch Track (RMVPE)')
        ax.plot(times, f0_hz, label='f0 (RMVPE)', linewidth=2)
        ax.legend()
        plt.savefig("pitch_track.png")
        print("[시각화] pitch_track.png 저장 완료", flush=True)

    return {
        "engine": "rmvpe",
        "sr": int(sr),
        "duration_s": float(len(y) / sr),
        "frames": int(p_sum.frames),
        "voiced_ratio": float(p_sum.voiced_ratio),
        "midi_min": float(p_sum.midi_min) if p_sum.midi_min is not None else None,
        "midi_min_note": midi_to_note(p_sum.midi_min),
        "midi_median": float(p_sum.midi_median) if p_sum.midi_median is not None else None,
        "midi_median_note": midi_to_note(p_sum.midi_median),
        "midi_max": float(p_sum.midi_max) if p_sum.midi_max is not None else None,
        "midi_max_note": midi_to_note(p_sum.midi_max),
        "rms_mean": float(e_sum.rms_mean),
        "rms_std": float(e_sum.rms_std),
        "instrumental_path": instrumental_path,
    }
