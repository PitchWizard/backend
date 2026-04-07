# api/services/analyze_and_save.py
import os
import json

# 상대 임포트로 고정 (패키지 인식 문제 방지)
from ..database import SessionLocal
from .. import crud

# 너의 분석기 함수 직접 호출
from analyzer.analyzer import analyze_audio_summary

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PITCH_FRAMES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pitch_frames")
os.makedirs(PITCH_FRAMES_DIR, exist_ok=True)


def _extract_7_fields(out: dict) -> dict:
    """
    analyzer 결과에서 DB에 넣을 5개 지표만 뽑아 float으로 정규화.
    - 중첩 구조(out["midi"], out["rms"])와 플랫 구조(out["midi_min"], ...) 모두 대응
    """
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


def analyze_and_save(
    *,
    title: str,
    artist: str,
    audio_path: str,
    hop_length: int = 256,
    plot: bool = False,
) -> tuple[int, str | None]:
    """
    1) analyzer 실행 (RMVPE)
    2) 결과 5개 지표만 추출
    3) title/artist 붙여 DB 저장
    4) pitch_frames를 outputs/pitch_frames/{song_id}.json에 저장
    5) (song_id, instrumental_path) 반환
    """
    # 1) 분석
    out = analyze_audio_summary(
        audio_path,
        hop_length=hop_length,
        plot=plot,
    )

    # 2) 7개 payload 구성
    metrics = _extract_7_fields(out)
    payload = {
        "title": title,
        "artist": artist,
        **metrics,
    }

    # 3) DB 저장
    print("[5/5] DB 저장 중…", flush=True)
    db = SessionLocal()
    try:
        row = crud.add_song(
            db=db,
            title=payload["title"],
            artist=payload["artist"],
            midi_min=payload["midi_min"],
            midi_median=payload["midi_median"],
            midi_max=payload["midi_max"],
            rms_mean=payload["rms_mean"],
            rms_std=payload["rms_std"],
        )
        song_id = getattr(row, "song_id", getattr(row, "id", None))
    finally:
        db.close()

    # 4) pitch_frames 파일 저장
    pitch_frames = out.get("pitch_frames")
    if pitch_frames and song_id:
        frames_path = os.path.join(PITCH_FRAMES_DIR, f"{song_id}.json")
        with open(frames_path, "w") as f:
            json.dump(pitch_frames, f)

    return song_id, out.get("instrumental_path")
