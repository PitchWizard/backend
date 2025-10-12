# api/main.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .database import SessionLocal, engine
from . import models, crud
import importlib, api.crud
print(">>> USING CRUD MODULE:", api.crud.__file__)
print(">>> CRUD NAMES:", [n for n in dir(api.crud) if n in ("add_song","get_songs")])
importlib.reload(api.crud)
print(">>> AFTER RELOAD CRUD NAMES:", [n for n in dir(api.crud) if n in ("add_song","get_songs")])


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

# ✅ 요청 바디 모델 정의
class SongCreate(BaseModel):
    title: str
    artist: str | None = None
    midi_min: float | None = None
    midi_median: float | None = None
    midi_max: float | None = None
    rms_mean: float | None = None
    rms_std: float | None = None

# ✅ API 라우트
@app.get("/songs")
def read_songs(limit: int = 50, db: Session = Depends(get_db)):
    return crud.list_songs(db, limit)

@app.get("/")
def root():
    return {"status": "FastAPI running properly"}

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
