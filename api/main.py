# api/main.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from .database import SessionLocal, engine
from . import models, crud

# ✅ 테이블 생성
models.Base.metadata.create_all(bind=engine)

# ✅ FastAPI 인스턴스
app = FastAPI(title="Vocal Wizard API")

# ✅ DB 세션 주입
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ 요청 바디 모델 (Songs)
class SongCreate(BaseModel):
    title: str
    artist: str | None = None
    midi_min: float | None = None
    midi_median: float | None = None
    midi_max: float | None = None
    rms_mean: float | None = None
    rms_std: float | None = None

# ✅ 요청 바디 모델 (Users)
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    midi_min: float | None = 0.0
    midi_median: float | None = 0.0
    midi_max: float | None = 0.0

# ✅ SONGS API
@app.get("/songs")
def read_songs(limit: int = 50, db: Session = Depends(get_db)):
    return crud.list_songs(db, limit)

@app.post("/songs")
def create_song(body: SongCreate, db: Session = Depends(get_db)):
    return crud.add_song(
        db=db,
        title=body.title,
        artist=body.artist,
        midi_min=body.midi_min,
        midi_median=body.midi_median,
        midi_max=body.midi_max,
        rms_mean=body.rms_mean,
        rms_std=body.rms_std
    )

# ✅ USERS API
@app.post("/users")
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(
        db=db,
        username=body.username,
        email=body.email,
        password=body.password,
        midi_min=body.midi_min if body.midi_min is not None else 0.0,
        midi_median=body.midi_median if body.midi_median is not None else 0.0,
        midi_max=body.midi_max if body.midi_max is not None else 0.0
    )

@app.get("/users")
def list_users(limit: int = 50, db: Session = Depends(get_db)):
    return crud.list_users(db, limit)

@app.get("/")
def root():
    return {"status": "FastAPI running properly"}
