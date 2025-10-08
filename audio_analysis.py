"""
음역 변환 마법사 — 유튜브/MP3 대응 librosa 기반 오디오 요약/피처 추출 모듈
(v9, Demucs 보컬 분리 + 퍼센타일/지속시간 필터링 + 시각화 옵션)
"""
from __future__ import annotations
import io, os, tempfile, subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import yt_dlp

# =========================
# 유틸
# =========================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note(m: float | None) -> Optional[str]:
    if m is None:
        return None
    m_int = int(round(m))
    name = NOTE_NAMES[m_int % 12]
    octave = (m_int // 12) - 1
    return f"{name}{octave}"

# =========================
# 유튜브 다운로드
# =========================
def download_youtube_audio(url: str) -> str:
    print("[1/5] 유튜브 다운로드 시작…", flush=True)
    tmpdir = tempfile.mkdtemp()
    outfile = os.path.join(tmpdir, "yt_audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outfile,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": False,
        "verbose": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    for f in os.listdir(tmpdir):
        if f.startswith("yt_audio"):
            path = os.path.join(tmpdir, f)
            print(f"[1/5] 다운로드 완료: {path}", flush=True)
            return path
    raise FileNotFoundError("유튜브 다운로드 실패")

# =========================
# Demucs 보컬 분리
# =========================
def separate_vocals_demucs(input_path: str) -> str:
    print("[2/5] Demucs 보컬 분리 시작…", flush=True)
    tmp_out = tempfile.mkdtemp()
    cmd = [
    "python", "-m", "demucs",
    "-n", "htdemucs",            # 최신 모델 지정
    "--two-stems=vocals",        # 보컬 + 반주 분리
    input_path,
    "-o", tmp_out
]
    subprocess.run(cmd, check=True)

    vocals_path = None
    for root, _, files in os.walk(tmp_out):
        for f in files:
            # ✅ 'no_vocals.wav'는 건너뛰고 'vocals.wav'만 선택
            if f.lower() == "vocals.wav":
                vocals_path = os.path.join(root, f)
                break
        if vocals_path:
            break

    if vocals_path is None:
        raise FileNotFoundError("Demucs 출력에서 vocals.wav을 찾을 수 없음")

    print(f"[2/5] 보컬 파일 경로: {vocals_path}", flush=True)
    return vocals_path

# =========================
# 오디오 로드
# =========================
def load_audio(source: str, target_sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    print("[3/5] 오디오 로딩/리샘플링 시작…", flush=True)
    y, sr = librosa.load(source, sr=None, mono=mono)
    y = y.astype(np.float32, copy=False)
    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y, _ = librosa.effects.trim(y, top_db=60)
    print("[3/5] 오디오 로딩 완료", flush=True)
    return y, sr

# =========================
# 피치 / RMS 추정
# =========================
def estimate_pitch(y: np.ndarray, sr: int, hop_length: int = 256, use_pyin: bool = True):
    print("[4/5] 피치 추출 중…", flush=True)
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    if use_pyin:
        f0_hz, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        f0_hz = np.where(voiced_flag, f0_hz, np.nan)
    else:
        f0_hz = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(f0_hz)), sr=sr, hop_length=hop_length)
    return f0_hz, times

def compute_rms(y: np.ndarray, sr: int, hop_length: int = 256):
    print("[4/5] RMS 에너지 추출 중…", flush=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times

# =========================
# 요약
# =========================
@dataclass
class PitchSummary:
    frames: int
    voiced_ratio: float
    midi_min: Optional[float]
    midi_median: Optional[float]
    midi_max: Optional[float]

@dataclass
class EnergySummary:
    rms_mean: float
    rms_std: float

def summarize_pitch(f0_hz: np.ndarray, sr: int, hop_length: int = 256,
                    min_duration: float = 0.3) -> PitchSummary:
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

# =========================
# 상위 API
# =========================
def analyze_audio_summary(source: str, use_pyin: bool = True, target_sr: int = 22050,
                          hop_length: int = 256, plot: bool = False) -> Dict:
    if source.startswith("http"):
        source = download_youtube_audio(source)
    source = separate_vocals_demucs(source)
    y, sr = load_audio(source, target_sr=target_sr, mono=True)
    f0_hz, times = estimate_pitch(y, sr, hop_length=hop_length, use_pyin=use_pyin)
    valid_f0 = f0_hz[~np.isnan(f0_hz)]
    if len(valid_f0) > 0:   
        raw_max_hz = np.max(valid_f0)
        raw_min_hz = np.min(valid_f0)
        print(f"[디버그] 최저 Hz: {raw_min_hz:.2f}, 최고 Hz: {raw_max_hz:.2f}")

    import librosa
    max_midi = librosa.hz_to_midi(raw_max_hz)
    min_midi = librosa.hz_to_midi(raw_min_hz)
    print(f"[디버그] 최저음: {librosa.midi_to_note(min_midi, octave=True)} "
          f"(MIDI {min_midi:.2f}), "
          f"최고음: {librosa.midi_to_note(max_midi, octave=True)} "
          f"(MIDI {max_midi:.2f})")
    rms, _ = compute_rms(y, sr, hop_length=hop_length)
    p_sum = summarize_pitch(f0_hz, sr, hop_length)
    e_sum = summarize_energy(rms)
    print("[완료] 분석 종료", flush=True)

    # (옵션) 시각화 저장
    if plot:
        import matplotlib.pyplot as plt
        import librosa.display
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
                                 sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram + Pitch Track')
        ax.plot(times, f0_hz, label='f0', color='cyan', linewidth=2)
        ax.legend()
        plt.savefig("pitch_track.png")
        print("[시각화] pitch_track.png 저장 완료", flush=True)

    return {
        "engine": "librosa-pyin" if use_pyin else "librosa-yin",
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

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="유튜브/MP3 오디오 분석 (Demucs 보컬 분리 + 필터링 + 시각화)")
    parser.add_argument("input", type=str, help="오디오 파일 경로 또는 유튜브 URL")
    parser.add_argument("--yin", action="store_true", help="pyin 대신 yin 사용")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--plot", action="store_true", help="스펙트로그램 + 피치 트랙 시각화 저장")
    args = parser.parse_args()

    result = analyze_audio_summary(args.input, use_pyin=(not args.yin),
                                   target_sr=args.sr, hop_length=args.hop,
                                   plot=args.plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
