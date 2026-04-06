# api/main.py
import os, shutil
from fastapi import FastAPI, Depends, HTTPException, Query, Body, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy.engine import Row
from fastapi.middleware.cors import CORSMiddleware
from .services.recommend_service import (
    get_transpose_for_song,
    get_recommended_songs_for_user,
)

# 반주 파일 영구 저장 디렉터리
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUMENTAL_DIR = os.path.join(_PROJECT_ROOT, "outputs", "instrumentals")
SHIFTED_DIR = os.path.join(_PROJECT_ROOT, "outputs", "shifted")
os.makedirs(INSTRUMENTAL_DIR, exist_ok=True)
os.makedirs(SHIFTED_DIR, exist_ok=True)


# DB/ORM
from .models import User
from .database import engine, Base, get_db
from . import models, crud

# ✅ 서비스: 분석 + DB 저장 원샷
from .services.analyze_and_save import analyze_and_save

# ---------------------------
# 앱 기동 시 테이블 생성 (1회)
# ---------------------------
Base.metadata.create_all(bind=engine)

# ---------------------------
# FastAPI 앱
# ---------------------------
app = FastAPI(title="Vocal Wizard API")

origins = [
    "http://localhost:5173",  # Vite 기본 포트 일단 안씀
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic Schemas (v2)
# ---------------------------
class SongCreate(BaseModel):
    title: str = Field(min_length=1)
    artist: str = Field(min_length=1)
    midi_min: float
    midi_median: float
    midi_max: float
    rms_mean: float
    rms_std: float

class SongOut(BaseModel):
    # 모델 PK가 song_id든 id든 모두 수용
    song_id: int = Field(validation_alias="id")
    title: str
    artist: str
    midi_min: float
    midi_median: float
    midi_max: float
    rms_mean: float
    rms_std: float
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    midi_min: float = 0.0
    midi_median: float = 0.0
    midi_max: float = 0.0

class UserOut(BaseModel):
    # 모델 PK가 user_id여도 받고, id여도 받는다.
    id: int = Field(validation_alias="user_id")
    username: str
    email: EmailStr
    midi_min: float
    midi_median: float
    midi_max: float
    low_note: str | None = None
    high_note: str | None = None
    avg_rms: float | None = None
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class VocalRangeRequest(BaseModel):
    user_id: int
    midi_min: float
    midi_median: float
    midi_max: float
    low_note: str | None = None
    high_note: str | None = None
    avg_rms: float | None = None


# ---------------------------
# Row/ORM ↔ Pydantic 보정 유틸
# ---------------------------
def as_list_of(items, schema, model_type=None):
    """
    items가 ORM 객체 목록이든, Row 목록이든, 튜플이든 상관없이
    pydantic 스키마 목록으로 안전 변환.
    """
    out = []
    for it in items:
        # 1) 이미 스키마면 그대로
        if isinstance(it, schema):
            out.append(it)
            continue
        # 2) ORM 객체면 바로 검증 (from_attributes=True)
        if model_type and isinstance(it, model_type):
            out.append(schema.model_validate(it))
            continue
        # 3) Row(User=...), RowMapping, dict, tuple 처리
        if isinstance(it, Row):
            mapping = it._mapping
            # Row(User=...) 형태 우선
            if model_type and "User" in mapping and isinstance(mapping["User"], model_type):
                out.append(schema.model_validate(mapping["User"]))
            else:
                out.append(schema.model_validate(mapping))
        elif hasattr(it, "_mapping"):
            out.append(schema.model_validate(it._mapping))
        elif isinstance(it, dict):
            out.append(schema.model_validate(it))
        else:
            # 마지막 안전장치: 객체의 __dict__ 시도
            try:
                out.append(schema.model_validate(it.__dict__))
            except Exception:
                raise HTTPException(status_code=500, detail="Unexpected row shape for response serialization")
    return out

# ---------------------------
# SONGS API
# ---------------------------
@app.get("/songs", response_model=list[SongOut])
def read_songs(limit: int = 50, db: Session = Depends(get_db)):
    items = crud.list_songs(db, limit=limit)
    return as_list_of(items, SongOut, model_type=models.Song)

@app.post("/songs", response_model=SongOut, status_code=201)
def create_song(body: SongCreate, db: Session = Depends(get_db)):
    try:
        song = crud.add_song(
            db=db,
            title=body.title,
            artist=body.artist,
            midi_min=body.midi_min,
            midi_median=body.midi_median,
            midi_max=body.midi_max,
            rms_mean=body.rms_mean,
            rms_std=body.rms_std,
        )
        # 단일 객체도 스키마로 확실히 직렬화
        return SongOut.model_validate(song)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/songs/run", response_model=SongOut, status_code=201)
async def run_analysis_and_create_song(
    title: str = Form(..., description="곡 제목"),
    artist: str = Form(..., description="가수명"),
    url: str = Form(..., description="유튜브 URL"),
    db: Session = Depends(get_db),
):
    """
    유튜브 URL → 다운로드 → analyzer 실행 → DB 저장 → 반주 파일 보존
    """
    import uuid, shutil, tempfile

    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="유효한 URL을 입력하세요.")

    tmpdir = tempfile.mkdtemp(prefix="wizard_")
    try:
        # 1) 서비스 호출: 다운로드 + 분석 + DB 저장
        song_id, instrumental_src = analyze_and_save(
            title=title,
            artist=artist,
            audio_path=url,
        )

        # 2) 반주 파일 영구 저장
        if instrumental_src and os.path.exists(instrumental_src):
            ext = os.path.splitext(instrumental_src)[1] or ".wav"
            dest = os.path.join(INSTRUMENTAL_DIR, f"{song_id}{ext}")
            shutil.move(instrumental_src, dest)

        # 3) 생성된 레코드 읽어서 반환
        row = db.get(models.Song, song_id)
        if not row:
            raise HTTPException(status_code=500, detail="Song saved but not found")
        return SongOut.model_validate(row)

    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"필수 분석 키 누락: {ke}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# ---------------------------
# USERS API
# ---------------------------
@app.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    try:
        user = crud.create_user(
            db=db,
            username=body.username,
            email=body.email,
            password=body.password,
            midi_min=body.midi_min,
            midi_median=body.midi_median,
            midi_max=body.midi_max,
        )
        return UserOut.model_validate(user)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users", response_model=list[UserOut])
def list_users(limit: int = 50, db: Session = Depends(get_db)):
    items = crud.list_users(db, limit)
    return as_list_of(items, UserOut, model_type=models.User)

@app.get("/")
def root():
    return {"status": "FastAPI running properly"}

@app.post("/vocal-range")
def save_vocal_range(data: VocalRangeRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == data.user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.midi_min = data.midi_min
    user.midi_median = data.midi_median
    user.midi_max = data.midi_max
    user.low_note = data.low_note
    user.high_note = data.high_note
    user.avg_rms = data.avg_rms

    db.commit()

    return {"status": "ok"}

@app.get("/songs/{song_id}/transpose")
def api_get_transpose_for_song(
    song_id: int,
    user_id: int,  # /songs/1/transpose?user_id=3 이런 식으로 부름
    db: Session = Depends(get_db),
):
    try:
        result = get_transpose_for_song(db, user_id=user_id, song_id=song_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@app.get("/songs/recommend")
def api_get_recommended_songs(
    user_id: int,
    db: Session = Depends(get_db),
):
    try:
        result = get_recommended_songs_for_user(db, user_id=user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result

@app.delete("/songs/{song_id}")
def delete_song(song_id: int, db: Session = Depends(get_db)):
    song = db.query(models.Song).filter(models.Song.song_id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    db.delete(song)
    db.commit()
    return {"message": "Song deleted"}


# ---------------------------
# ACCOMPANIMENT (반주 피치시프트)
# ---------------------------
@app.get("/songs/{song_id}/accompaniment")
def get_accompaniment(
    song_id: int,
    semitones: float = Query(0, ge=-5, le=5),
    db: Session = Depends(get_db),
):
    """
    저장된 반주 파일을 semitones만큼 피치시프트하여 반환.
    semitones: -5 ~ +5 (기본값 0 = 원본)
    """
    from analyzer.pitch_shift import shift_accompaniment

    # 반주 파일 탐색 (wav/mp3/flac 모두 허용)
    instrumental_path = None
    for ext in (".wav", ".mp3", ".flac"):
        candidate = os.path.join(INSTRUMENTAL_DIR, f"{song_id}{ext}")
        if os.path.exists(candidate):
            instrumental_path = candidate
            break

    if instrumental_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"반주 파일이 없습니다. /songs/run 으로 먼저 분석을 실행하세요. (song_id={song_id})",
        )

    # 원본 요청이면 그대로 반환
    if semitones == 0:
        return FileResponse(instrumental_path, media_type="audio/wav", filename=f"song_{song_id}_original.wav")

    # 캐시된 파일이 있으면 재사용
    sign = "+" if semitones > 0 else ""
    shifted_filename = f"{song_id}_{sign}{semitones}.wav"
    shifted_path = os.path.join(SHIFTED_DIR, shifted_filename)

    if not os.path.exists(shifted_path):
        shift_accompaniment(instrumental_path, semitones=semitones, output_path=shifted_path)

    return FileResponse(shifted_path, media_type="audio/wav", filename=shifted_filename)