## 오디오 입출력 관련 유틸 (다운 및 전처리)
import os, tempfile, subprocess
from typing import Tuple
import numpy as np
import librosa
import yt_dlp

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

def separate_vocals_demucs(input_path: str) -> str:
    print("[2/5] Demucs 보컬 분리 시작…", flush=True)
    tmp_out = tempfile.mkdtemp()
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems=vocals",
        input_path,
        "-o", tmp_out,
    ]
    subprocess.run(cmd, check=True)

    vocals_path = None
    for root, _, files in os.walk(tmp_out):
        for f in files:
            if f.lower() == "vocals.wav":
                vocals_path = os.path.join(root, f)
                break
        if vocals_path:
            break

    if vocals_path is None:
        raise FileNotFoundError("Demucs 출력에서 vocals.wav을 찾을 수 없음")

    print(f"[2/5] 보컬 파일 경로: {vocals_path}", flush=True)
    return vocals_path

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
