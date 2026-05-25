# api/audio_io.py
import os, tempfile
from typing import Tuple
import numpy as np
import librosa
import yt_dlp

def download_youtube_audio(url: str) -> str:
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
        "quiet": True,
        "no_warnings": True,
    }
    print("[1/5] 유튜브 다운로드 중…", flush=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    for f in os.listdir(tmpdir):
        if f.startswith("yt_audio"):
            print("[1/5] 다운로드 완료", flush=True)
            return os.path.join(tmpdir, f)
    raise FileNotFoundError("유튜브 다운로드 실패")

def separate_vocals_uvr5_mdx(input_path: str) -> tuple[str, str]:
    """보컬과 반주를 분리한다.

    Returns:
        (vocals_path, instrumental_path)
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        raise ImportError(
            "audio-separator 패키지가 필요합니다.\n"
            "GPU 사용: pip install audio-separator[gpu]\n"
            "CPU 전용: pip install audio-separator"
        )

    print("[2/5] Kim_Vocal_2 보컬 분리 중…", flush=True)
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

    print("[2/5] 분리 완료", flush=True)
    return vocals_path, instrumental_path


def separate_accompaniment_mdx_main(input_path: str) -> str:
    """UVR-MDX-NET-Inst_Main 모델로 반주를 분리한다.

    Returns:
        instrumental_path
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        raise ImportError(
            "audio-separator 패키지가 필요합니다.\n"
            "GPU 사용: pip install audio-separator[gpu]\n"
            "CPU 전용: pip install audio-separator"
        )

    print("[2b/5] UVR-MDX-NET-Inst_Main 반주 분리 중…", flush=True)
    tmp_out = tempfile.mkdtemp()
    separator = Separator(output_dir=tmp_out)
    separator.load_model("UVR-MDX-NET-Inst_Main.onnx")
    output_files = separator.separate(input_path)

    instrumental_path = None
    for f in output_files:
        basename = os.path.basename(f)
        full = f if os.path.isabs(f) else os.path.join(tmp_out, basename)
        if "(Instrumental)" in basename:
            instrumental_path = full
            break

    # fallback: 보컬 파일이 아닌 것 선택
    if instrumental_path is None:
        for f in output_files:
            basename = os.path.basename(f)
            full = f if os.path.isabs(f) else os.path.join(tmp_out, basename)
            if "(Vocals)" not in basename:
                instrumental_path = full
                break

    if instrumental_path is None:
        raise FileNotFoundError("MDX 분리 결과에서 반주 파일을 찾을 수 없음")

    print("[2b/5] MDX 반주 분리 완료", flush=True)
    return instrumental_path


def load_audio(source: str, target_sr: int = 22050, mono: bool = True, trim: bool = False) -> Tuple[np.ndarray, int]:
    print("[3/5] 오디오 로딩 중…", flush=True)
    y, sr = librosa.load(source, sr=None, mono=mono)
    y = y.astype(np.float32, copy=False)
    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if trim:
        y, _ = librosa.effects.trim(y, top_db=60)
    return y, sr
