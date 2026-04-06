# analyzer/pitch_shift.py
import os
import numpy as np
import soundfile as sf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUBBERBAND_EXE = os.path.join(_PROJECT_ROOT, "rubberband.exe")


def shift_accompaniment(input_path: str, semitones: float, output_path: str) -> str:
    """반주 파일을 semitones만큼 피치시프트하여 output_path에 저장.
    Rubber Band Library 사용 — 포먼트 보존으로 먹먹함 없음.

    Args:
        input_path:  원본 반주 파일 경로 (wav/flac)
        semitones:   이동할 반음 수 (-5 ~ +5)
        output_path: 저장할 경로 (.wav)

    Returns:
        output_path
    """
    if not -5 <= semitones <= 5:
        raise ValueError(f"semitones는 -5~+5 범위여야 합니다. (입력값: {semitones})")

    if not os.path.exists(_RUBBERBAND_EXE):
        raise FileNotFoundError(f"rubberband.exe를 찾을 수 없습니다: {_RUBBERBAND_EXE}")

    # rubberband.exe가 있는 프로젝트 루트를 PATH 앞에 추가
    env_path = os.environ.get("PATH", "")
    if _PROJECT_ROOT not in env_path:
        os.environ["PATH"] = _PROJECT_ROOT + os.pathsep + env_path

    import pyrubberband.pyrb as _pyrb
    _pyrb.__dict__["__RUBBERBAND_UTIL"] = _RUBBERBAND_EXE  # 내부 변수 직접 지정

    import pyrubberband as rb

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    y, sr = sf.read(input_path, always_2d=True)  # (frames, channels)

    shifted = rb.pitch_shift(
        y, sr,
        n_steps=semitones,
        rbargs={"--formant": "", "--fine": ""},
    )

    # 음을 내릴수록 먹먹해지는 현상 보정: 고음역 살짝 부스트
    if semitones < 0:
        from pedalboard import Pedalboard, HighShelfFilter
        board = Pedalboard([
            HighShelfFilter(
                cutoff_frequency_hz=4000,
                gain_db=abs(semitones) * 0.8,  # 1키당 0.8dB 부스트
            )
        ])
        shifted_T = shifted.T.astype(np.float32)  # (channels, frames)
        shifted_T = board(shifted_T, sr)
        shifted = shifted_T.T

    sf.write(output_path, shifted, sr)
    print(f"[피치시프트] {semitones:+}키 → {output_path}", flush=True)
    return output_path
