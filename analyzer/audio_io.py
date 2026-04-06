# api/audio_io.py
import os, tempfile
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

def separate_vocals_uvr5_mdx(input_path: str) -> tuple[str, str]:
    """보컬과 반주를 분리한다.

    Returns:
        (vocals_path, instrumental_path)
    """
    print("[2/5] Kim_Vocal_2 보컬 분리 시작…", flush=True)

    try:
        from audio_separator.separator import Separator
    except ImportError:
        raise ImportError(
            "audio-separator 패키지가 필요합니다.\n"
            "GPU 사용: pip install audio-separator[gpu]\n"
            "CPU 전용: pip install audio-separator"
        )

    tmp_out = tempfile.mkdtemp()
    separator = Separator(output_dir=tmp_out)
    separator.load_model("Kim_Vocal_2.onnx")
    output_files = separator.separate(input_path)

    vocals_path = None
    instrumental_path = None
    for f in output_files:
        basename = os.path.basename(f)
        full = f if os.path.isabs(f) else os.path.join(tmp_out, basename)
        if "(Vocals)" in basename:
            vocals_path = full
        elif "(Instrumental)" in basename:
            instrumental_path = full

    if vocals_path is None:
        raise FileNotFoundError("분리 결과에서 보컬 파일을 찾을 수 없음")
    if instrumental_path is None:
        raise FileNotFoundError("분리 결과에서 반주 파일을 찾을 수 없음")

    print(f"[2/5] 보컬: {vocals_path}", flush=True)
    print(f"[2/5] 반주: {instrumental_path}", flush=True)
    return vocals_path, instrumental_path

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
