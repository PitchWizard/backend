# main.py
import argparse, json, sys
from analyzer.analyzer import analyze_audio_summary

# ▼ API 패키지 임포트가 안 잡히는 환경 대비(프로젝트 루트에서 실행 가정)
#   필요 없으면 이 3줄은 지워도 됨
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DB/CRUD
from api.database import SessionLocal
from api import crud


def _extract_7_fields(result: dict) -> dict:
    """
    analyzer 결과(result)에서 DB에 넣을 7개만 뽑아서 반환.
    - 중첩 구조: result["midi"]["min"/"median"/"max"], result["rms"]["mean"/"std"]
    - 플랫 구조: result["midi_min"/"midi_median"/"midi_max"], result["rms_mean"/"rms_std"]
    두 케이스를 모두 지원.
    """
    # 1) 중첩 구조 지원
    if "midi" in result and "rms" in result:
        midi = result["midi"]
        rms = result["rms"]
        return {
            "midi_min": float(midi["min"]),
            "midi_median": float(midi["median"]),
            "midi_max": float(midi["max"]),
            "rms_mean": float(rms["mean"]),
            "rms_std": float(rms["std"]),
        }
    # 2) 플랫 구조 지원
    return {
        "midi_min": float(result["midi_min"]),
        "midi_median": float(result["midi_median"]),
        "midi_max": float(result["midi_max"]),
        "rms_mean": float(result["rms_mean"]),
        "rms_std": float(result["rms_std"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="유튜브/MP3 오디오 분석 (Demucs 보컬 분리 + 필터링 + 시각화) + DB 저장"
    )
    parser.add_argument("input", type=str, help="오디오 파일 경로 또는 유튜브 URL")
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--plot", action="store_true", help="스펙트로그램 + 피치 트랙 시각화 저장")

    # ▼ DB 저장을 위한 메타데이터(필수 권장)
    parser.add_argument("--title", required=True, help="곡 제목")
    parser.add_argument("--artist", required=True, help="가수명")

    args = parser.parse_args()

    # 1) 분석 실행
    result = analyze_audio_summary(
        args.input,
        hop_length=args.hop,
        plot=args.plot
    )

    # 2) 콘솔에 기존처럼 JSON 출력(가독 위해 pretty-print)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 3) DB 저장 (7개 필드만)
    try:
        metrics = _extract_7_fields(result)  # midi/rms 5개 + title/artist은 아래서 합침

        db = SessionLocal()
        try:
            row = crud.add_song(
                db=db,
                title=args.title,
                artist=args.artist,
                midi_min=metrics["midi_min"],
                midi_median=metrics["midi_median"],
                midi_max=metrics["midi_max"],
                rms_mean=metrics["rms_mean"],
                rms_std=metrics["rms_std"],
            )
            # PK 이름이 song_id 라면 아래처럼, id 라면 row.id 로 출력
            song_pk = getattr(row, "song_id", getattr(row, "id", None))
            print(f"\n✅ DB 저장 완료: id={song_pk}, title='{row.title}', artist='{row.artist}'")
        finally:
            db.close()

    except KeyError as e:
        print(f"\n[ERROR] 분석 결과에 필요한 키가 없습니다: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] DB 저장 중 오류: {e}", file=sys.stderr)
        sys.exit(2)
