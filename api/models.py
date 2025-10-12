# api/models.py
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, text
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
