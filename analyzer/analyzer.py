from .audio_io import download_youtube_audio, separate_vocals_demucs, load_audio
from .pitch_extract import estimate_pitch_librosa, estimate_pitch_torchcrepe
from .features import summarize_pitch, summarize_energy
from .utils import midi_to_note


import numpy as np
import librosa
from librosa import display as lrdisplay
import torch
import torchcrepe

# =========================
# 안정화용 공통 하이퍼파라미터
# =========================
FMIN_BASE = 80.0             # 1패스 하한(E2)
FMAX_BASE = 1400.0           # 1패스 상한(여성 고음+가성 일부, 상향)
FMIN_HIGH = 600.0            # 2패스(고음 구조) 하한
FMAX_HIGH = 1900.0           # 2패스(고음 구조) 상한(고음 곡 대응)
VOICING_THRESHOLD = 0.10     # periodicity(0~1) > 0.10 만 유성
RMS_REL_THRESHOLD = 0.01      # 프레임 RMS > max(RMS)*2% 만 유성
MEDIAN_FILTER_WIN = 5         # CREPE 전용 median 윈도우(홀수 권장: 3/5)
MIN_SEG_DURATION_S = 0.10     # 0.1초 미만 유성 세그먼트 제거
HOP_LENGTH_DEFAULT = 256

# =========================
# 유틸
# =========================
def compute_rms(y, sr, hop_length=HOP_LENGTH_DEFAULT):
    print("[4/5] RMS 에너지 추출 중…", flush=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times

def _remove_short_segments(f0, sr, hop_length, min_duration_s=MIN_SEG_DURATION_S):
    """지속시간이 짧은(예: 0.3s 이하) 유성 구간을 NaN으로 정리"""
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
    """NaN을 양 끝 보간으로 임시 채움(필터 적용을 위한 보조), 이후 원래 NaN은 복구"""
    y = x.copy()
    n = len(y)
    idx = np.arange(n)
    m = ~np.isnan(y)
    if not m.any():
        return x
    y[~m] = np.interp(idx[~m], idx[m], y[m])
    return y

def _median_filter_crepe(f0_hz: np.ndarray, win: int) -> np.ndarray:
    """CREPE 전용 median 필터(토치 연산). NaN은 보간 후 필터→원복."""
    if win is None or win < 3 or (win % 2 == 0):
        return f0_hz
    mask_voiced = ~np.isnan(f0_hz)
    if not mask_voiced.any():
        return f0_hz
    f0_fill = _interp_fill_nan(f0_hz)
    t = torch.from_numpy(f0_fill).float().unsqueeze(0)  # [1, T]
    with torch.no_grad():
        t_filt = torchcrepe.filter.median(t, win)
    f0_filt = t_filt.squeeze(0).cpu().numpy()
    f0_filt[~mask_voiced] = np.nan
    return f0_filt

# =========================
# 옥타브 보정(검산) 필터
# =========================
def _fix_octave_jump(f0: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    옥타브 하향/상향 혼동 보정.
    - 전역 중앙값 기준으로 0.6배 이상 벗어난 프레임에 대해 2배/1/2 후보를 테스트
    - 국부(윈도우) 중앙값으로도 재검산
    """
    out = f0.copy()
    voiced = ~np.isnan(out)
    if voiced.sum() < 10:
        return out

    med = np.nanmedian(out)
    if med <= 0:
        return out

    # 후보 생성
    cand_hi = out * 2.0
    cand_lo = out / 2.0

    # 전역 기준
    dev = np.abs(out - med)
    far = dev > (med * strength)

    # 국부 기준(±0.25s)
    idx = np.arange(len(out))
    # hop_length는 상위에서만 알 수 있으므로, 간단 국부 창 크기(7프레임) 사용
    win = 7
    local_med = out.copy()
    for i in range(len(out)):
        a = max(0, i - win // 2)
        b = min(len(out), i + win // 2 + 1)
        local_med[i] = np.nanmedian(out[a:b]) if np.any(voiced[a:b]) else med

    # 거리 비교(전역/국부 모두에서 더 가까운 후보로 교체)
    def closer(x, y, target):
        return np.abs(x - target) < np.abs(y - target)

    choose_hi = closer(cand_hi, out, med) & closer(cand_hi, out, local_med)
    choose_lo = closer(cand_lo, out, med) & closer(cand_lo, out, local_med)

    out[far & choose_hi] = cand_hi[far & choose_hi]
    out[far & choose_lo] = cand_lo[far & choose_lo]
    return out

# =========================
# CREPE 예측 (단일 패스)
# =========================
def _predict_crepe_once(y, sr, hop_length, fmin, fmax, device):
    """
    단일 패스 CREPE + periodicity/RMS 마스크 + median 필터 + 짧은 세그먼트 제거
    """
    if y.ndim > 1:
        y = librosa.to_mono(y)
    audio = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        f0, per = torchcrepe.predict(
            audio=audio,
            sample_rate=sr,          # ✅ 최신 torchcrepe 인자명
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            model='full',
            batch_size=512,
            device=device,
            return_periodicity=True
        )  # [1, T]

    f0 = f0.squeeze(0).cpu().numpy()
    per = per.squeeze(0).cpu().numpy()

    # periodicity / RMS 마스크
    mask_per = per > VOICING_THRESHOLD
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = float(np.max(rms)) if len(rms) else 0.0
    rms_thr = rms_max * RMS_REL_THRESHOLD if rms_max > 0 else 0.0
    mask_rms = rms > rms_thr

    L = min(len(f0), len(rms))
    f0 = f0[:L]
    per = per[:L]
    mask = (mask_per[:L]) & (mask_rms[:L])

    f0_masked = f0.copy()
    f0_masked[~mask] = np.nan

    # median 필터
    f0_masked = _median_filter_crepe(f0_masked, MEDIAN_FILTER_WIN)

    # 짧은 세그먼트 제거
    f0_masked = _remove_short_segments(f0_masked, sr, hop_length, MIN_SEG_DURATION_S)

    times = librosa.frames_to_time(np.arange(L), sr=sr, hop_length=hop_length)
    return f0_masked, times, per, rms[:L]

# =========================
# CREPE 예측 (2패스 결합)
# =========================
def _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True):
    """
    1패스: 80–1200 Hz (일반, 상한 상향)
    2패스(선택): 600–1800 Hz (고음 구조), 조건 충족 프레임만 교체
    """
    f0_1, times, per_1, rms = _predict_crepe_once(
        y, sr, hop_length, FMIN_BASE, FMAX_BASE, device
    )
    if not two_pass:
        return f0_1, times, per_1, rms

    # 고음 구조 패스
    f0_2, _, per_2, _ = _predict_crepe_once(
        y, sr, hop_length, FMIN_HIGH, FMAX_HIGH, device
    )

    # 교체 조건
    hi_gate_per = 0.60
    rms_max = float(np.max(rms)) if len(rms) else 0.0
    hi_gate_rms = (rms_max * RMS_REL_THRESHOLD) * 1.2

    need_hi = (
        (np.isnan(f0_1) | (f0_1 > (0.95 * FMAX_BASE)))
        & (~np.isnan(f0_2))
        & (per_2 > hi_gate_per)
        & (rms > hi_gate_rms)
    )

    f0 = f0_1.copy()
    f0[need_hi] = f0_2[need_hi]

    # 마지막 안전 median 한 번(작은 윈도우로 과도 평탄화 방지)
    f0 = _median_filter_crepe(f0, 3)
    per = per_1  # 대표 periodicity는 1패스를 사용
    return f0, times, per, rms

# =========================
# HYBRID: pYIN + CREPE 융합/보정
# =========================
def _adaptive_hybrid_fuse(f0_pyin, f0_crepe, per_crepe, rms, widen_pct: int = 15):
    """
    1) pYIN 결과를 기본으로 삼되,
       - 상/하위 widen_pct% + NaN은 CREPE로 교체
       - pYIN 최고음이 낮게 눌린 경우(상단 90% 미만) CREPE 교체 폭을 자동 확대
    2) 겹치는 프레임은 가중 융합 (고음/신뢰 높을수록 CREPE 가중↑)
    3) 옥타브 검산으로 최종 보정
    """
    L = min(len(f0_pyin), len(f0_crepe))
    l = f0_pyin[:L]
    c = f0_crepe[:L]
    per = per_crepe[:L]
    rms = rms[:L]

    out = l.copy()

    # 퍼센타일 경계
    valid_l = ~np.isnan(l)
    if valid_l.any():
        lo, hi = np.percentile(l[valid_l], [widen_pct, 100 - widen_pct])
    else:
        lo, hi = -np.inf, np.inf

    # 기본 교체 마스크
    mask_ext = (l <= lo) | (l >= hi) | np.isnan(l)

    # 상단 눌림 감지(하드 가드): pYIN 최고가 CREPE 최고보다 3세미톤 이상 낮으면 교체 폭 확대
    try:
        l_max = np.nanmax(librosa.hz_to_midi(l))
        c_max = np.nanmax(librosa.hz_to_midi(c))
        if np.isfinite(l_max) and np.isfinite(c_max) and (c_max - l_max) >= 3.0:
            # 눌림 심함 → CREPE와 겹치는 구간도 교체 허용
            mask_ext = mask_ext | ((~np.isnan(c)) & (per > 0.5))
    except Exception:
        pass

    # 교체
    out[mask_ext & (~np.isnan(c))] = c[mask_ext & (~np.isnan(c))]

    # 겹치는 프레임 가중 융합 (CREPE 가중 0.7~0.9; RMS/periodicity에 비례)
    both = (~np.isnan(l)) & (~np.isnan(c))
    if both.any():
        # normalize weights from periodicity + rms
        per_n = (per - np.nanmin(per[both])) / (np.nanmax(per[both]) - np.nanmin(per[both]) + 1e-8)
        rms_n = (rms - np.nanmin(rms[both])) / (np.nanmax(rms[both]) - np.nanmin(rms[both]) + 1e-8)
        w_crepe = 0.7 + 0.2 * (0.5 * per_n + 0.5 * rms_n)  # 0.7~0.9
        w_pyin  = 1.0 - w_crepe
        mix = w_crepe * c[both] + w_pyin * l[both]
        out[both] = mix

    # 옥타브 검산
    out = _fix_octave_jump(out, strength=0.6)
    return out

def _hybrid_replace_extremes_with_crepe(y, sr, hop_length, f0_librosa):
    """
    (레거시) 상/하위 5% + NaN을 CREPE(2패스 안정화)로 보정
    → 본 파일에서는 _adaptive_hybrid_fuse 쪽이 더 정밀하여, hybrid 엔진에서 그 함수를 사용.
    """
    valid = ~np.isnan(f0_librosa)
    if np.sum(valid) == 0:
        print("[하이브리드] 유효한 librosa 피치가 없어 보정 생략", flush=True)
        return f0_librosa

    lo, hi = np.percentile(f0_librosa[valid], [5, 95])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[하이브리드] CREPE 재분석(2패스) device={device}", flush=True)
    f0_crepe, _times, _per, _rms = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)

    L = min(len(f0_librosa), len(f0_crepe))
    f0_l = f0_librosa[:L]
    f0_c = f0_crepe[:L]

    mask_ext = (f0_l <= lo) | (f0_l >= hi) | np.isnan(f0_l)
    out = f0_l.copy()
    out[mask_ext] = f0_c[mask_ext]
    return out

# =========================
# 메인 엔트리
# =========================
def analyze_audio_summary(source: str, engine="pyin", target_sr=22050,
                          hop_length=HOP_LENGTH_DEFAULT, plot=False):
    global MEDIAN_FILTER_WIN, FMAX_BASE, FMAX_HIGH
    # 1) 소스 정규화
    if source.startswith("http"):
        source = download_youtube_audio(source)

    # 2) 보컬 분리 (Demucs)
    #    - 내부 구현에서 htdemucs / --two-stems=vocals 사용 가정
    source = separate_vocals_demucs(source)

    # 3) 로드
    y, sr = load_audio(source, target_sr=target_sr, mono=True)

    # 4) 피치 추정
    if engine in ["pyin", "yin"]:
        f0_hz, times = estimate_pitch_librosa(y, sr, hop_length=hop_length, use_pyin=(engine=="pyin"))

    elif engine == "torchcrepe":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[CREPE] device={device} "
            f"base=({FMIN_BASE}-{FMAX_BASE})Hz hi=({FMIN_HIGH}-{FMAX_HIGH})Hz "
            f"voicing_th={VOICING_THRESHOLD} median={MEDIAN_FILTER_WIN}",
            flush=True
        )
        f0_hz, times, per, rms = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)

        # 🔎 퇴행 감지 가드: 범위가 0.5 semitone 이하이면 재시도(필터 해제 + fmax 확장)
        voiced = f0_hz[~np.isnan(f0_hz)]
        if voiced.size > 8:
            midi = librosa.hz_to_midi(voiced)
            if (np.nanmax(midi) - np.nanmin(midi)) < 0.5:
                print("[경고] f0 범위가 비정상적으로 협소. 필터 해제 + 상한 확장 재시도.", flush=True)
                old_win, old_fmax_base, old_fmax_high = MEDIAN_FILTER_WIN, FMAX_BASE, FMAX_HIGH
                MEDIAN_FILTER_WIN = None
                FMAX_BASE, FMAX_HIGH = 1400.0, 2000.0
                f0_retry, _times2, _per2, _rms2 = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)
                MEDIAN_FILTER_WIN = old_win
                FMAX_BASE, FMAX_HIGH = old_fmax_base, old_fmax_high
                f0_hz = f0_retry

        # 옥타브 검산(최종)
        f0_hz = _fix_octave_jump(f0_hz, strength=0.6)

    elif engine == "hybrid":
        # 1) pYIN 기본 추정
        f0_l, times = estimate_pitch_librosa(y, sr, hop_length=hop_length, use_pyin=True)
        # 2) CREPE(2패스, 고음 대응) 추정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f0_c, _times2, per, rms = _predict_crepe_with_masks(y, sr, hop_length, device, two_pass=True)
        # 3) 적응형 융합/보정 + 옥타브 검산
        f0_hz = _adaptive_hybrid_fuse(f0_l, f0_c, per, rms, widen_pct=15)

    else:
        raise ValueError("engine은 pyin, yin, torchcrepe, hybrid 중 하나여야 합니다.")

    # 5) 디버그: 원시 최솟값/최댓값
    valid_f0 = f0_hz[~np.isnan(f0_hz)]
    if len(valid_f0) > 0:
        raw_max_hz = float(np.max(valid_f0))
        raw_min_hz = float(np.min(valid_f0))
        print(f"[디버그] 최저 Hz: {raw_min_hz:.2f}, 최고 Hz: {raw_max_hz:.2f}", flush=True)
        max_midi = float(librosa.hz_to_midi(raw_max_hz))
        min_midi = float(librosa.hz_to_midi(raw_min_hz))
        print(
            f"[디버그] 최저음: {librosa.midi_to_note(min_midi, octave=True)} (MIDI {min_midi:.2f}), "
            f"최고음: {librosa.midi_to_note(max_midi, octave=True)} (MIDI {max_midi:.2f})",
            flush=True
        )

    # 6) 에너지·요약치
    rms, _ = compute_rms(y, sr, hop_length=hop_length)

    # ✅ summarize_pitch는 NaN을 무성으로 처리한다고 가정
    p_sum = summarize_pitch(f0_hz, sr, hop_length)
    e_sum = summarize_energy(rms)

    print("[완료] 분석 종료", flush=True)

    # 7) 시각화
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

    # 8) 결과
    return {
        "engine": engine,
        "sr": int(sr),
        "duration_s": float(len(y) / sr),
        "frames": int(p_sum.frames),
        "voiced_ratio": float(p_sum.voiced_ratio),
        "midi_min": float(p_sum.midi_min),
        "midi_min_note": midi_to_note(p_sum.midi_min),
        "midi_median": float(p_sum.midi_median),
        "midi_median_note": midi_to_note(p_sum.midi_median),
        "midi_max": float(p_sum.midi_max),
        "midi_max_note": midi_to_note(p_sum.midi_max),
        "rms_mean": float(e_sum.rms_mean),
        "rms_std": float(e_sum.rms_std),
    }
