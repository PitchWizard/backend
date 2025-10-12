"""
🎵 음역 기반 키 추천 + 곡 추천 시스템
---------------------------------
사용자의 테시투라(user_min, user_max)와
DB에 저장된 각 곡의 음역대(song_min, song_max)를 비교하여

1) 각 곡의 키 조정 추천
2) 음역 겹침률(overlap ratio) 계산
3) 상위 5곡 추천
"""

import librosa
import pandas as pd

# --------------------------
# 1️⃣ 사용자 / DB 데이터 (임시 자리)
# --------------------------

# TODO: 실제 DB에서 사용자 음역대 가져오기
user_min_midi = None  # 예: 57.4 (A3)
user_max_midi = None  # 예: 76.0 (E5)

# TODO: 실제 DB에서 곡 리스트 불러오기
# 아래는 샘플 데이터프레임 예시
songs = pd.DataFrame([
    {"title": "태연 - I", "midi_min": 60, "midi_max": 79},
    {"title": "뉴진스 - ETA", "midi_min": 55, "midi_max": 74},
    {"title": "아이유 - 좋은 날", "midi_min": 59, "midi_max": 83},
    {"title": "폴킴 - 모든 날 모든 순간", "midi_min": 52, "midi_max": 73},
    {"title": "악뮤 - Love Lee", "midi_min": 58, "midi_max": 78},
    {"title": "10cm - 사랑은 은하수 다방에서", "midi_min": 53, "midi_max": 72},
])


# --------------------------
# 2️⃣ 키 조정 및 겹침 계산 로직
# --------------------------

def recommend_key_shift(user_min, user_max, song_min, song_max):
    """노래방 기준 1키=1세미톤 단위로 조정 추천"""
    if song_min >= user_max:
        shift = user_max - song_min
        direction = "내려야 함"
    elif song_max <= user_min:
        shift = user_min - song_max
        direction = "올려야 함"
    else:
        shift = 0
        direction = "조정 불필요"
    return int(round(shift)), direction


def overlap_ratio(user_min, user_max, song_min, song_max):
    """사용자-곡 음역 겹침률 계산"""
    overlap_min = max(user_min, song_min)
    overlap_max = min(user_max, song_max)
    overlap = max(0, overlap_max - overlap_min)
    user_range = user_max - user_min
    return round(overlap / user_range, 3) if user_range > 0 else 0


# --------------------------
# 3️⃣ 추천 시스템
# --------------------------

def recommend_songs(user_min, user_max, songs_df, top_k=5):
    results = []

    for _, row in songs_df.iterrows():
        song_min, song_max = row["midi_min"], row["midi_max"]
        shift, direction = recommend_key_shift(user_min, user_max, song_min, song_max)
        ratio = overlap_ratio(user_min, user_max, song_min, song_max)

        # 사용자가 곡의 음역을 포함하거나 일정 부분 겹치면 추천 후보로
        if ratio > 0:
            results.append({
                "title": row["title"],
                "overlap_ratio": ratio,
                "recommended_shift": shift,
                "direction": direction
            })

    # 겹침률 높은 순으로 정렬 후 상위 5개
    ranked = sorted(results, key=lambda x: x["overlap_ratio"], reverse=True)
    return ranked[:top_k]


# --------------------------
# 4️⃣ 실행 예시
# --------------------------
if __name__ == "__main__":
    # 🎤 샘플 사용자 음역대 (A3 ~ E5)
    user_min_midi = librosa.note_to_midi("A3")  # 57
    user_max_midi = librosa.note_to_midi("E5")  # 76

    recs = recommend_songs(user_min_midi, user_max_midi, songs)

    print(f"🎙️ 사용자 음역대: {librosa.midi_to_note(user_min_midi)} ~ {librosa.midi_to_note(user_max_midi)}\n")
    print("🎵 추천 결과 (상위 5곡):")
    for r in recs:
        shift_msg = f"{abs(r['recommended_shift'])}키 {r['direction']}" if r['recommended_shift'] != 0 else "그대로 가능"
        print(f"- {r['title']:25s} │ 겹침률: {r['overlap_ratio']:.2f} │ 추천: {shift_msg}")
