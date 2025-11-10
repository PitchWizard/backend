# api/crud.py
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from . import models
from .security import hash_password  # User 생성 시 비밀번호 해시에 사용


# ----------------------------
# SONG
# ----------------------------
def add_song(
    db: Session,
    *,
    title: str,
    artist: str,
    midi_min: float,
    midi_median: float,
    midi_max: float,
    rms_mean: float,
    rms_std: float,
):
    """
    목적: 오직 7개 필드만 DB에 저장.
    - None 불허(필수 값 누락 시 즉시 오류)
    - 예외 발생 시 rollback 보장
    """
    try:
        song = models.Song(
            title=title,
            artist=artist,
            midi_min=float(midi_min),
            midi_median=float(midi_median),
            midi_max=float(midi_max),
            rms_mean=float(rms_mean),
            rms_std=float(rms_std),
        )
        db.add(song)
        db.commit()
        db.refresh(song)
        return song
    except (KeyError, TypeError, ValueError) as e:
        db.rollback()
        # 입력값 문제는 상위에서 400대로 매핑하기 좋도록 그대로 올림
        raise e
    except SQLAlchemyError as e:
        db.rollback()
        # DB 오류는 상위에서 500대로 처리
        raise e


def add_song_from_result(
    db: Session,
    *,
    title: str,
    artist: str,
    result: dict,
):
    """
    목적: analyzer가 돌려준 큰 dict(result)에서 필요한 7개만 뽑아 add_song으로 위임.
    - 키 누락 시 KeyError
    - 비정상 수치(NaN/inf/str 등) → float(...)에서 ValueError/TypeError
    """
    payload = {
        "title": title,
        "artist": artist,
        "midi_min": float(result["midi_min"]),
        "midi_median": float(result["midi_median"]),
        "midi_max": float(result["midi_max"]),
        "rms_mean": float(result["rms_mean"]),
        "rms_std": float(result["rms_std"]),
    }
    return add_song(db, **payload)


def list_songs(db: Session, limit: int = 50, offset: int = 0):
    return (
        db.query(models.Song)
        .order_by(models.Song.song_id.desc())  # ← 여기!
        .limit(limit)
        .offset(offset)
        .all()
    )


# (선택) 제목+가수 기준 upsert가 필요하면 아래 함수 사용.
# 모델에 UniqueConstraint("title","artist")가 설정되어 있으면 더 안전하다.
def upsert_song_by_title_artist(
    db: Session,
    *,
    title: str,
    artist: str,
    midi_min: float,
    midi_median: float,
    midi_max: float,
    rms_mean: float,
    rms_std: float,
):
    """
    (title, artist) 기준으로 존재하면 업데이트, 없으면 생성.
    배치 수집/재분석 반영용.
    """
    try:
        song = (
            db.query(models.Song)
            .filter(models.Song.title == title, models.Song.artist == artist)
            .one_or_none()
        )
        if song is None:
            song = models.Song(
                title=title,
                artist=artist,
                midi_min=float(midi_min),
                midi_median=float(midi_median),
                midi_max=float(midi_max),
                rms_mean=float(rms_mean),
                rms_std=float(rms_std),
            )
            db.add(song)
        else:
            song.midi_min = float(midi_min)
            song.midi_median = float(midi_median)
            song.midi_max = float(midi_max)
            song.rms_mean = float(rms_mean)
            song.rms_std = float(rms_std)

        db.commit()
        db.refresh(song)
        return song
    except (TypeError, ValueError, KeyError) as e:
        db.rollback()
        raise e
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# ----------------------------
# USER
# ----------------------------
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
    """
    User 생성.
    - username/email 중복 검사
    - 비밀번호는 해시 저장
    """
    # 중복 체크
    if get_user_by_username(db, username):
        raise ValueError("username already exists")
    if get_user_by_email(db, email):
        raise ValueError("email already exists")

    try:
        user = models.User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            midi_min=float(midi_min),
            midi_median=float(midi_median),
            midi_max=float(midi_max),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except (TypeError, ValueError) as e:
        db.rollback()
        raise e
    except SQLAlchemyError as e:
        db.rollback()
        raise e


def list_users(db: Session, limit: int = 50):
    return db.query(models.User).limit(limit).all()
