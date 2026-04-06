# api/services/recommend_service.py

import math
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from .. import models  # Song, User 모델


def calc_smart_transpose(
    user_min: float,
    user_max: float,
    song_min: float,
    song_max: float,
    low_tolerance: int = 3,      # 저음은 user_min - 3키까지 허용
    low_song_threshold: int = 4, # 곡 최고음이 user_max보다 4키 이상 낮으면 올려줌
    max_abs_shift: int = 8       # 전조 최대 범위 (±8키)
) -> Optional[int]:
    """
    사용자/곡 음역대를 기반으로 전조 값(k) 계산.
    k > 0: 키 올림, k < 0: 키 내림, k = 0: 원키
    """

    # 저음 조건: song_min + k >= user_min - low_tolerance
    # 고음 조건: song_max + k <= user_max
    low_k_real = (user_min - low_tolerance) - song_min    # k >= low_k_real
    high_k_real = user_max - song_max                     # k <= high_k_real

    low_k = math.ceil(low_k_real)
    high_k = math.floor(high_k_real)

    if low_k > high_k:
        return None

    # 전조 절대값 제한
    candidates = [k for k in range(low_k, high_k + 1) if abs(k) <= max_abs_shift]
    if not candidates:
        return None

    # 곡이 너무 낮은지 판단 (user_max보다 4키 이상 낮으면)
    gap = user_max - song_max  # 양수면 곡이 사용자보다 낮음
    if gap >= low_song_threshold:
        # 이상적으로는 song_max + k ≈ user_max 가 되도록
        k_ideal = math.floor(gap)
        best_k = min(
            candidates,
            key=lambda k: (abs(k - k_ideal), abs(k))
        )
    else:
        # 이미 충분히 높으면, 원키에 가깝게
        best_k = min(
            candidates,
            key=lambda k: (abs(k), abs((song_max + k) - user_max))
        )

    return best_k


def describe_transpose(k: Optional[int]) -> str:
    if k is None:
        return "현재 설정 기준으로는 이 곡을 편하게 부르기 어려울 수 있습니다."
    if k == 0:
        return "원키로 불러도 무난한 음역대입니다."
    direction = "올려야" if k > 0 else "내려야"
    steps = abs(k)
    return f"{steps}키만큼 {direction} 합니다."


def get_transpose_for_song(
    db: Session,
    user_id: int,
    song_id: int,
) -> Dict[str, Any]:
    """
    특정 사용자 + 특정 곡에 대해
    몇 키 전조하면 좋은지 계산해서 dict로 반환.
    """

    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    song = db.query(models.Song).filter(models.Song.song_id == song_id).first()

    if not user or not song:
        raise ValueError("user 또는 song 을 찾을 수 없습니다.")

    user_min = user.midi_min
    user_max = user.midi_max
    song_min = song.midi_min
    song_max = song.midi_max

    k = calc_smart_transpose(user_min, user_max, song_min, song_max)

    result: Dict[str, Any] = {
        "user_id": user_id,
        "song_id": song_id,
        "user_range": [user_min, user_max],
        "song_range": [song_min, song_max],
        "recommended_shift": k,
        "message": describe_transpose(k),
    }

    if k is not None:
        result["shifted_song_range"] = [song_min + k, song_max + k]

    return result


def get_recommended_songs_for_user(
    db: Session,
    user_id: int,
    max_abs_shift: int = 8,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    한 사용자 기준으로 DB에 있는 여러 곡에 대해
    '부를 만한 곡 + 추천 키' 리스트 반환.
    """

    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise ValueError("user 를 찾을 수 없습니다.")

    user_min = user.midi_min
    user_max = user.midi_max

    songs = db.query(models.Song).limit(limit).all()
    results: List[Dict[str, Any]] = []

    for song in songs:
        song_min = song.midi_min
        song_max = song.midi_max

        k = calc_smart_transpose(
            user_min=user_min,
            user_max=user_max,
            song_min=song_min,
            song_max=song_max,
            max_abs_shift=max_abs_shift,
        )

        if k is None:
            # 이 곡은 조건 맞추기 힘들면 패스
            continue

        results.append({
            "song_id": song.song_id,
            "title": song.title,
            "artist": song.artist,
            "original_range": [song_min, song_max],
            "recommended_shift": k,
            "shifted_range": [song_min + k, song_max + k],
            "message": describe_transpose(k),
        })

    # 이동량이 적은 순서대로 정렬
    results.sort(key=lambda x: (abs(x["recommended_shift"]), x["title"] or ""))
    return results
