# api/models.py
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, text, UniqueConstraint
from .database import Base   # ✅ Base를 database.py에서 불러오기


class Song(Base):
    __tablename__ = "songs"

    song_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(100), nullable=False)
    artist = Column(String(100))
    midi_min = Column(Float)
    midi_median = Column(Float)
    midi_max = Column(Float)
    rms_mean = Column(Float)
    rms_std = Column(Float)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        UniqueConstraint("title", "artist", name="uq_title_artist"),
    )


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    # ✅ 사용자별 최신 MIDI 기준값 (음역대)
    midi_min = Column(Float, default=0.0)
    midi_median = Column(Float, default=0.0)
    midi_max = Column(Float, default=0.0)

    # ✅ 추가: 음역대의 문자열 표기 & RMS 요약값
    low_note = Column(String(10), nullable=True)   # 예: "C3"
    high_note = Column(String(10), nullable=True)  # 예: "A4"
    avg_rms = Column(Float, nullable=True)         # 전체 평균 RMS

    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
    )
