# api/services/analyzer_adapter.py
from analyzer.analyzer import analyze_audio_summary

def run_analyzer(audio_path: str, *, engine: str = "pyin", target_sr: int = 22050, hop_length: int = 256, plot: bool = False) -> dict:
    """
    네 분석기(analyze_audio_summary)를 호출하고,
    DB에 저장하기 좋은 키로 정규화해서 반환.
    """
    out = analyze_audio_summary(
        audio_path,
        engine=engine,
        target_sr=target_sr,
        hop_length=hop_length,
        plot=plot,
    )

    # 중첩 구조 대응
    if "midi" in out and "rms" in out:
        return {
            "midi_min": float(out["midi"]["min"]),
            "midi_median": float(out["midi"]["median"]),
            "midi_max": float(out["midi"]["max"]),
            "rms_mean": float(out["rms"]["mean"]),
            "rms_std": float(out["rms"]["std"]),
        }

    # 플랫 구조 대응
    return {
        "midi_min": float(out["midi_min"]),
        "midi_median": float(out["midi_median"]),
        "midi_max": float(out["midi_max"]),
        "rms_mean": float(out["rms_mean"]),
        "rms_std": float(out["rms_std"]),
    }

def build_song_payload(*, title: str, artist: str, result: dict) -> dict:
    return {
        "title": title,
        "artist": artist,
        "midi_min": result["midi_min"],
        "midi_median": result["midi_median"],
        "midi_max": result["midi_max"],
        "rms_mean": result["rms_mean"],
        "rms_std": result["rms_std"],
    }
