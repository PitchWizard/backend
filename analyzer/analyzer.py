# api/analyzer.py
from .audio_io import download_youtube_audio, separate_vocals_demucs, load_audio
from .pitch_extract import estimate_pitch_librosa, estimate_pitch_torchcrepe
from .features import summarize_pitch, summarize_energy, compute_rms
from .utils import midi_to_note

import numpy as np
import librosa
from librosa import display as lrdisplay
import torch
import torchcrepe

# ===== 튜닝 파라미터 =====
FMIN_BASE = 80.0
FMAX_BASE = 1400.0
FMIN_HIGH = 600.0
FMAX_HIGH = 1900.0
VOICING_THRESHOLD = 0.10
RMS_REL_THRESHOLD = 0.01
MEDIAN_FILTER_WIN = 5
MIN_SEG_DURATION_S = 0.10
HOP_LENGTH_DEFAULT = 256

# --- 내부 유틸: NaN 보간/세그먼트 정리/미디안 필터/옥타브 보정 ---
def _remove_short_segments(f0, sr, hop_length, min_duration_s=MIN_SEG_DURATION_S):
    frame_time = hop_length / sr
    min_len = int(min_duration_s / frame_time)
    if min_len <= 1:
        return f0

    f0_filtered = f0.copy()
    is_voiced = ~np.isnan(f0_filtered)
    start = None
    for i, voiced in enumerate(is_voiced):
        if voiced and start is None:
            start = i
        elif (not voiced or i == len(is_voiced) - 1) and start is not None:
            end = i if not voiced else i + 1
            seg_len = end - start
            if seg_len < min_len:
                f0_filtered[start:end] = np.nan
            start = None
    return f0_filtered

def _interp_fill_nan(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    n = len(y)
    idx = np.arange(n)
    m = ~np.isnan(y)
    if not m.any():
        return x
    y[~m] = np.interp(idx[~m], idx[m], y[m])
    return y

def _median_filter_crepe(f0_hz: np.ndarray, win: int) -> np.ndarray:
    if win is None or win < 3 or (win % 2 == 0):
        return f0_hz
    mask_voiced = ~np.isnan(f0_hz)
    if not mask_voiced.any():
        return f0_hz
    f0_fill = _interp_fill_nan(f0_hz)
    t = torch.from_numpy(f0_fill).float().unsqueeze(0)
    with torch.no_grad():
        t_filt = torchcrepe.filter.median(t, win)
    f0_filt = t_filt.squeeze(0).cpu().numpy()
    f0_filt[~mask_voiced] = np.nan
    return f0_filt

def _fix_octave_jump(f0: np.ndarray, strength: float = 0.6) -> np.ndarray:
    out = f0.copy()
    voiced = ~np.isnan(out)
    if voiced.sum() < 10:
        return out
    med = np.nanmedian(out)
    if med <= 0:
        return out
    cand_hi = out * 2.0
    cand_lo = out / 2.0
    dev = np.abs(out - med)
    far = dev > (med * strength)
    idx = np.arange(len(out))
    win = 7
    local_med = out.copy()
    for i in range(len(out)):
        a = max(0, i - win // 2); b = min(len(out), i + win // 2 + 1)
        local_med[i] = np.nanmedian(out[a:b]) if np.any(voiced[a:b]) else med

    def closer(x, y, target): return np.abs(x - target) < np.abs(y - target)
    choose_hi = closer(cand_hi, out, med) & closer(cand_hi, out, local_med)
    choose_lo = closer(cand_lo, out, med) & closer(cand_lo, out, local_med)
    out[far & choose_hi] = cand_hi[far & choose_hi]
    out[far & choose_lo] = cand_lo[far & choose_lo]
    return out

# --- torchcrepe 1패스/2패스 ---
def _predict_crepe_once(y, sr, hop_length, fmin, fmax, device):
    if y.ndim > 1:
        y = librosa.to_mono(y)
    audio = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        f0, per = torchcrepe.predict(
            audio=audio,
            sample_rate=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            model='full',
            batch_size=512,
            device=device,
            return_periodicity=True
        )
    f0 = f0.squeeze(0).cpu().numpy()
    per = per.squeeze(0).cpu().numpy()

    mask_per = per > VOICING_THRESHOLD
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = float(np.max(rms)) if len(rms) else 0.0
    rms_thr = rms_max * RMS_REL_THRESHOLD if rms_max > 0 else 0.0
    mask_rms = rms > rms_thr

    L = min(len(f0), len(rms))
    f0 = f0[:L]; per = per[:L]
    mask = (mask_per[:L]) & (mask_rms[:L])

    f0_masked = f0.copy()
    f0_masked[~mask] = np.nan
    f0_masked = _median_filter_crepe(f0_masked, MEDIAN_FILTER_WIN)
    f0_masked = _remove_short_segments(f0_masked, sr, hop_length, MIN_SEG_DURATION_S)

    times = librosa.frames_to_time(np.arange(L), sr=sr, hop_length=hop_length)
    return f0_masked, times, per, rms[:L]

def _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True):
    f0_1, times, per_1, rms = _predict_crepe_once(y, sr, hop_length, FMIN_BASE, FMAX_BASE, device)
    if not two_pass:
        return f0_1, times, per_1, rms

    f0_2, _, per_2, _ = _predict_crepe_once(y, sr, hop_length, FMIN_HIGH, FMAX_HIGH, device)
    hi_gate_per = 0.60
    rms_max = float(np.max(rms)) if len(rms) else 0.0
    hi_gate_rms = (rms_max * RMS_REL_THRESHOLD) * 1.2
    need_hi = ((np.isnan(f0_1) | (f0_1 > (0.95 * FMAX_BASE))) & (~np.isnan(f0_2)) &
               (per_2 > hi_gate_per) & (rms > hi_gate_rms))
    f0 = f0_1.copy()
    f0[need_hi] = f0_2[need_hi]
    f0 = _median_filter_crepe(f0, 3)
    per = per_1
    return f0, times, per, rms

def _adaptive_hybrid_fuse(f0_pyin, f0_crepe, per_crepe, rms, widen_pct: int = 15):
    L = min(len(f0_pyin), len(f0_crepe))
    l = f0_pyin[:L]; c = f0_crepe[:L]
    per = per_crepe[:L]; rms = rms[:L]
    out = l.copy()

    valid_l = ~np.isnan(l)
    if valid_l.any():
        lo, hi = np.percentile(l[valid_l], [widen_pct, 100 - widen_pct])
    else:
        lo, hi = -np.inf, np.inf
    mask_ext = (l <= lo) | (l >= hi) | np.isnan(l)

    # 상단 눌림 감지 → 교체 폭 확대
    try:
        l_max = np.nanmax(librosa.hz_to_midi(l))
        c_max = np.nanmax(librosa.hz_to_midi(c))
        if np.isfinite(l_max) and np.isfinite(c_max) and (c_max - l_max) >= 3.0:
            mask_ext = mask_ext | ((~np.isnan(c)) & (per > 0.5))
    except Exception:
        pass

    out[mask_ext & (~np.isnan(c))] = c[mask_ext & (~np.isnan(c))]

    both = (~np.isnan(l)) & (~np.isnan(c))
    if both.any():
        per_n = (per[both] - np.nanmin(per[both])) / (np.nanmax(per[both]) - np.nanmin(per[both]) + 1e-8)
        rms_n = (rms[both] - np.nanmin(rms[both])) / (np.nanmax(rms[both]) - np.nanmin(rms[both]) + 1e-8)
        w_crepe = 0.7 + 0.2 * (0.5 * per_n + 0.5 * rms_n)  # 0.7~0.9
        w_pyin  = 1.0 - w_crepe
        mix = w_crepe * c[both] + w_pyin * l[both]
        out[both] = mix

    out = _fix_octave_jump(out, strength=0.6)
    return out

# --- 메인 엔트리 ---
def analyze_audio_summary(source: str, engine="pyin", target_sr=22050,
                          hop_length=HOP_LENGTH_DEFAULT, plot=False):
    global MEDIAN_FILTER_WIN, FMAX_BASE, FMAX_HIGH

    if source.startswith("http"):
        source = download_youtube_audio(source)

    source = separate_vocals_demucs(source)
    y, sr = load_audio(source, target_sr=target_sr, mono=True)

    if engine in ["pyin", "yin"]:
        f0_hz, times = estimate_pitch_librosa(y, sr, hop_length=hop_length, use_pyin=(engine=="pyin"))
    elif engine == "torchcrepe":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CREPE] device={device}", flush=True)
        f0_hz, times, per, rms = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)

        # 퇴행 가드: 범위가 너무 좁으면 필터 완화해 재시도
        voiced = f0_hz[~np.isnan(f0_hz)]
        if voiced.size > 8:
            midi = librosa.hz_to_midi(voiced)
            if (np.nanmax(midi) - np.nanmin(midi)) < 0.5:
                print("[경고] f0 범위 협소 → 필터 해제 + 상한 확장 재시도", flush=True)
                old_win, old_fmax_base, old_fmax_high = MEDIAN_FILTER_WIN, FMAX_BASE, FMAX_HIGH
                MEDIAN_FILTER_WIN = None
                FMAX_BASE, FMAX_HIGH = 1400.0, 2000.0
                f0_retry, _t2, _p2, _r2 = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)
                MEDIAN_FILTER_WIN = old_win
                FMAX_BASE, FMAX_HIGH = old_fmax_base, old_fmax_high
                f0_hz = f0_retry

        f0_hz = _fix_octave_jump(f0_hz, strength=0.6)

    elif engine == "hybrid":
        f0_l, times = estimate_pitch_librosa(y, sr, hop_length=hop_length, use_pyin=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f0_c, _t2, per, rms = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)
        f0_hz = _adaptive_hybrid_fuse(f0_l, f0_c, per, rms, widen_pct=15)
    else:
        raise ValueError("engine은 pyin, yin, torchcrepe, hybrid 중 하나여야 합니다.")

    # 디버그 로그(원시 최소/최대)
    valid_f0 = f0_hz[~np.isnan(f0_hz)]
    if len(valid_f0) > 0:
        raw_max_hz = float(np.max(valid_f0))
        raw_min_hz = float(np.min(valid_f0))
        print(f"[디버그] 최저 Hz: {raw_min_hz:.2f}, 최고 Hz: {raw_max_hz:.2f}", flush=True)
        max_midi = float(librosa.hz_to_midi(raw_max_hz))
        min_midi = float(librosa.hz_to_midi(raw_min_hz))
        print(f"[디버그] 최저음: {librosa.midi_to_note(min_midi, octave=True)} (MIDI {min_midi:.2f}), "
              f"최고음: {librosa.midi_to_note(max_midi, octave=True)} (MIDI {max_midi:.2f})", flush=True)

    # 에너지 요약치
    rms, _ = compute_rms(y, sr, hop_length=hop_length)
    p_sum = summarize_pitch(f0_hz, sr, hop_length)
    e_sum = summarize_energy(rms)

    # 결과 반환
    result = {
        "engine": engine,
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
    }

    # (옵션) 시각화 저장
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        lrdisplay.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram + Pitch Track')
        ax.plot(times, f0_hz, label='f0 (stabilized)', linewidth=2)
        ax.legend()
        plt.savefig("pitch_track.png")
        print("[시각화] pitch_track.png 저장 완료", flush=True)

    return result
