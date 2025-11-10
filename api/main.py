# api/main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy.engine import Row

# DB/ORM
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
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

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
    title: str = Form(...),
    artist: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    업로드 파일 → analyzer 실행(서비스 호출) → DB 저장
    """
    import os, uuid, shutil, tempfile

    tmpdir = tempfile.mkdtemp(prefix="wizard_")
    try:
        # 1) 파일 임시 저장
        ext = os.path.splitext(audio.filename)[1] or ".wav"
        tmp_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{ext}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # 2) 서비스 호출: 분석 + DB 저장 원샷
        song_id = analyze_and_save(title=title, artist=artist, audio_path=tmp_path)

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
