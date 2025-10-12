# main.py
import argparse, json
from analyzer.analyzer import analyze_audio_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="유튜브/MP3 오디오 분석 (Demucs 보컬 분리 + 필터링 + 시각화)")
    parser.add_argument("input", type=str, help="오디오 파일 경로 또는 유튜브 URL")
    parser.add_argument("--engine", choices=["pyin", "yin", "torchcrepe", "hybrid"], default="pyin")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--plot", action="store_true", help="스펙트로그램 + 피치 트랙 시각화 저장")
    args = parser.parse_args()

    result = analyze_audio_summary(
        args.input,
        engine=args.engine,
        target_sr=args.sr,
        hop_length=args.hop,
        plot=args.plot
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
