import numpy as np
import librosa
import torch
import torchcrepe

def estimate_pitch_librosa(y, sr, hop_length=256, use_pyin=True):
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    if use_pyin:
        f0_hz, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        f0_hz = np.where(voiced_flag, f0_hz, np.nan)
    else:
        f0_hz = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(f0_hz)), sr=sr, hop_length=hop_length)
    return f0_hz, times

def estimate_pitch_torchcrepe(y, sr, hop_length=256, device="cuda"):
    print("[4/5] torchcrepe 피치 추출 중…", flush=True)
    audio = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    f0_hz = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin=50.0,
        fmax=2000.0,
        model="full",
        batch_size=1024,
        device=device,
        return_periodicity=False,
    )
    f0_hz = f0_hz.squeeze(0).cpu().numpy()
    times = librosa.frames_to_time(np.arange(len(f0_hz)), sr=sr, hop_length=hop_length)
    return f0_hz, times
