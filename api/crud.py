# C:\Users\RC\wizard\api\crud.py
from sqlalchemy.orm import Session
from . import models

def add_song(db: Session, title: str, artist: str, midi_min: float, midi_median: float,
             midi_max: float, rms_mean: float, rms_std: float):
    song = models.Song(
        title=title,
        artist=artist,
        midi_min=midi_min,
        midi_median=midi_median,
        midi_max=midi_max,
        rms_mean=rms_mean,
        rms_std=rms_std,
    )
    db.add(song)
    db.commit()
    db.refresh(song)
    return song

def list_songs(db: Session, limit: int = 10):
    return db.query(models.Song).limit(limit).all()
