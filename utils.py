## 헤르츠 -> 음계 변환 유틸리티

from __future__ import annotations
from typing import Optional

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note(m: float | None) -> Optional[str]:
    if m is None:
        return None
    m_int = int(round(m))
    name = NOTE_NAMES[m_int % 12]
    octave = (m_int // 12) - 1
    return f"{name}{octave}"
