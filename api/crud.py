# api/crud.py
from sqlalchemy.orm import Session
from . import models
from .security import hash_password  # ← 비번 해시

# ---- Song ----
def add_song(db: Session, title: str, artist: str | None,
             midi_min: float | None, midi_median: float | None, midi_max: float | None,
             rms_mean: float | None, rms_std: float | None):
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

def list_songs(db: Session, limit: int = 50):
    return db.query(models.Song).limit(limit).all()

# ---- User ----
def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(
    db: Session,
    *,
    username: str,
    email: str,
    password: str,
    midi_min: float = 0.0,
    midi_median: float = 0.0,
    midi_max: float = 0.0,
):
    # 중복 체크
    if get_user_by_username(db, username):
        raise ValueError("username already exists")
    if get_user_by_email(db, email):
        raise ValueError("email already exists")

    user = models.User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        midi_min=midi_min,
        midi_median=midi_median,
        midi_max=midi_max,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def list_users(db: Session, limit: int = 50):
    return db.query(models.User).limit(limit).all()
