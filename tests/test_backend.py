"""
PitchWizard 백엔드 시스템 시험 — D07 시스템 시험 결과서의 자동 시험 케이스(STC-T01~T07).

- 함수 이름의 STC-xxx 가 D07 표의 시험 ID 와 1:1로 대응한다.
- 기대값(assert)은 AI가 코드를 읽고 작성한 것이다. 팀원이 D03 유즈케이스·D04 백로그의
  인수 조건과 대조해 '요구사항상 맞는 값인지' 검토해야 한다. (코드가 틀렸으면 시험도 같이 틀린다)
- 실행 방법은 tests/README.md 참고. 반드시 시험 전용 DB(TEST_DATABASE_URL)로 실행.
"""
import os, json, math, wave, struct, shutil
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api import main as m
from api.database import engine, SessionLocal
from api import models
from api.services.recommend_service import calc_smart_transpose, describe_transpose
from analyzer.features import summarize_pitch
from analyzer.utils import midi_to_note

client = TestClient(m.app)


@pytest.fixture(scope="module", autouse=True)
def clean_db():
    # conftest.py 에서 DB 이름에 'test' 가 들어간 시험 DB인지 이미 확인함
    assert "test" in (engine.url.database or "").lower()
    with engine.begin() as c:
        c.execute(text("DELETE FROM users"))
        c.execute(text("DELETE FROM songs"))
    yield


def _signup(u="alice", e="alice@test.com", p="Passw0rd!"):
    return client.post("/users", json={"username": u, "email": e, "password": p})


# ---------------- T-01 회원 관리 ----------------
def test_STC_T01_001_signup_ok():
    r = _signup()
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "alice" and body["email"] == "alice@test.com"
    db = SessionLocal()
    u = db.query(models.User).filter_by(username="alice").first()
    assert u.hashed_password != "Passw0rd!" and len(u.hashed_password) == 64
    db.close()


def test_STC_T01_002_signup_dup_username():
    r = _signup(e="other@test.com")
    assert r.status_code == 400 and r.json()["detail"] == "username already exists"


def test_STC_T01_003_signup_dup_email():
    r = _signup(u="bob")
    assert r.status_code == 400 and r.json()["detail"] == "email already exists"


def test_STC_T01_004_signup_bad_email():
    r = client.post("/users", json={"username": "carol", "email": "not-an-email", "password": "x"})
    assert r.status_code == 422


def test_STC_T01_005_login_ok():
    r = client.post("/login", data={"username": "alice", "password": "Passw0rd!"})
    assert r.status_code == 200 and r.json()["username"] == "alice"


def test_STC_T01_006_login_wrong_password():
    r = client.post("/login", data={"username": "alice", "password": "wrong"})
    assert r.status_code == 401


def test_STC_T01_007_login_unknown_user():
    r = client.post("/login", data={"username": "nobody", "password": "Passw0rd!"})
    assert r.status_code == 401


# ---------------- T-02 음역대 측정 결과 저장 ----------------
def _uid(name="alice"):
    db = SessionLocal()
    uid = db.query(models.User).filter_by(username=name).first().user_id
    db.close()
    return uid


def test_STC_T02_001_save_vocal_range():
    uid = _uid()
    r = client.post("/vocal-range", json={"user_id": uid, "midi_min": 50, "midi_median": 58.5,
                                          "midi_max": 67, "low_note": "D3", "high_note": "G4"})
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    r2 = client.post("/login", data={"username": "alice", "password": "Passw0rd!"}).json()
    assert (r2["midi_min"], r2["midi_median"], r2["midi_max"]) == (50, 58.5, 67)
    assert (r2["low_note"], r2["high_note"]) == ("D3", "G4")


def test_STC_T02_002_save_vocal_range_unknown_user():
    r = client.post("/vocal-range", json={"user_id": 999999, "midi_min": 50, "midi_median": 55, "midi_max": 60})
    assert r.status_code == 404


def _dave_id():
    r = client.post("/login", data={"username": "dave", "password": "Passw0rd!"})
    if r.status_code != 200:
        _signup("dave", "dave@test.com")
        r = client.post("/login", data={"username": "dave", "password": "Passw0rd!"})
    return r.json()["id"]


def _save_range(uid, chest_max=None, chest_high_note=None):
    # 측정 음역 D3~C5(가성 포함). alice 의 음역은 다른 시험이 쓰므로 dave 로 시험
    return client.post("/vocal-range", json={"user_id": uid, "midi_min": 50, "midi_median": 60,
                                             "midi_max": 72, "low_note": "D3", "high_note": "C5",
                                             "chest_max": chest_max, "chest_high_note": chest_high_note})


def test_STC_T02_007_save_chest_max():
    uid = _dave_id()
    assert _save_range(uid, 65, "F4").status_code == 200
    r = client.post("/login", data={"username": "dave", "password": "Passw0rd!"}).json()
    assert (r["midi_max"], r["chest_max"], r["chest_high_note"]) == (72, 65, "F4")
    # 진성 최고음 없이 다시 저장(새 측정)하면 비워진다
    assert _save_range(uid).status_code == 200
    r = client.post("/login", data={"username": "dave", "password": "Passw0rd!"}).json()
    assert r["chest_max"] is None and r["chest_high_note"] is None


# ---------------- T-04 곡 검색(목록) ----------------
@pytest.fixture(scope="module")
def songs():
    from api import crud
    db = SessionLocal()
    ids = []
    for t, a, lo, md, hi in [("곡A", "가수1", 55, 63, 72), ("곡B", "가수1", 52, 60, 68),
                             ("곡C", "가수2", 45, 60, 75)]:
        s = crud.add_song(db, title=t, artist=a, midi_min=lo, midi_median=md, midi_max=hi,
                          rms_mean=0.1, rms_std=0.02)
        ids.append(s.song_id)
    db.close()
    return ids


def test_STC_T04_001_list_songs(songs):
    r = client.get("/songs")
    assert r.status_code == 200
    data = r.json()
    got = [d["song_id"] for d in data]
    assert got == sorted(got, reverse=True)
    assert {"title", "artist", "midi_min", "midi_median", "midi_max"} <= set(data[0])


# ---------------- T-05 음역대 비교·키 추천 ----------------
def test_STC_T05_001_transpose_api(songs):
    uid = _uid()  # 사용자 50~67
    r = client.get(f"/songs/{songs[0]}/transpose", params={"user_id": uid})  # 곡 55~72
    body = r.json()
    assert r.status_code == 200 and body["recommended_shift"] == -5
    assert body["message"] == "5키만큼 내려야 합니다."
    assert body["shifted_song_range"] == [50, 67]


def test_STC_T05_002_transpose_unknown(songs):
    r = client.get("/songs/999999/transpose", params={"user_id": _uid()})
    assert r.status_code == 404


def test_STC_T05_003_recommend_list(songs):
    r = client.get("/songs/recommend", params={"user_id": _uid()})
    assert r.status_code == 200
    data = r.json()
    titles = [d["title"] for d in data]
    assert "곡C" not in titles  # 30반음 폭 곡은 사용자 음역(17반음)으로 소화 불가 → 제외
    shifts = [abs(d["recommended_shift"]) for d in data]
    assert shifts == sorted(shifts)


def test_STC_T05_004_calc_transpose_rules():
    assert calc_smart_transpose(48, 70, 52, 68) == 0          # 이미 음역 안 → 원키
    assert calc_smart_transpose(48, 70, 50, 65) == 5          # 곡이 4키 이상 낮음 → 올림
    assert calc_smart_transpose(55, 60, 45, 75) is None       # 소화 불가
    assert describe_transpose(0) == "원키로 불러도 무난한 음역대입니다."
    assert describe_transpose(3) == "3키만큼 올려야 합니다."


def test_STC_T05_006_transpose_chest_max(songs):
    # 계산 규칙: 진성 최고음이 있으면 고음 기준을 진성 최고음(+반음 절반 여유)으로 바꾼다
    assert calc_smart_transpose(50, 72, 59.3, 71.1) == 0                          # 가성 포함 최고음 기준 → 원키
    assert calc_smart_transpose(50, 72, 59.3, 71.1, chest_max=66) == -5           # 진성 F#4 기준 → 5키 내림
    assert calc_smart_transpose(50, 72, 59.3, 71.1, chest_max=66, chest_tolerance=0) == -6  # 여유 없으면 -6
    assert calc_smart_transpose(50, 72, 59.3, 71.1, chest_max=80) == 0            # 측정 최고음보다 높으면 무시
    assert calc_smart_transpose(50, 72, 48.9, 62.1) == 8                          # 낮은 곡을 가성 최고음까지 끌어올림
    assert calc_smart_transpose(50, 72, 48.9, 62.1, chest_max=66) == 0            # 진성 기준이면 원키
    # API: 사용자 정보의 진성 최고음이 추천 키와 추천 목록에 반영된다 (곡A 55~72)
    uid = _dave_id()
    _save_range(uid)
    assert client.get(f"/songs/{songs[0]}/transpose", params={"user_id": uid}).json()["recommended_shift"] == 0
    _save_range(uid, 65, "F4")
    assert client.get(f"/songs/{songs[0]}/transpose", params={"user_id": uid}).json()["recommended_shift"] == -7
    rec = {d["song_id"]: d["recommended_shift"] for d in client.get("/songs/recommend", params={"user_id": uid}).json()}
    assert rec.get(songs[0]) == -7


# ---------------- T-06 반주 연습 ----------------
def _write_wav(path, sec=0.5, sr=22050):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr)))
                               for i in range(int(sr * sec))))


def test_STC_T06_001_accompaniment_original(songs):
    sid = songs[0]
    _write_wav(os.path.join(m.INSTRUMENTAL_DIR, f"{sid}.wav"))
    r = client.get(f"/songs/{sid}/accompaniment", params={"semitones": 0, "model": "kim"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/")
    assert r.content[:4] == b"RIFF"


def test_STC_T06_002_accompaniment_cache_reuse(songs):
    sid = songs[0]
    cached = os.path.join(m.SHIFTED_DIR, f"{sid}_kim_+2.0.wav")
    _write_wav(cached, sec=0.2)
    r = client.get(f"/songs/{sid}/accompaniment", params={"semitones": 2, "model": "kim"})
    assert r.status_code == 200 and len(r.content) == os.path.getsize(cached)


def test_STC_T06_003_accompaniment_out_of_range(songs):
    r = client.get(f"/songs/{songs[0]}/accompaniment", params={"semitones": 6})
    assert r.status_code == 422


@pytest.mark.heavy
def test_STC_T06_012_accompaniment_real_shift(songs):
    """캐시가 없을 때 실제 Rubber Band 로 +1키 변환 (rubberband.exe 필요, PW_HEAVY=1)."""
    sid = songs[0]
    _write_wav(os.path.join(m.INSTRUMENTAL_DIR, f"{sid}.wav"), sec=2.0)
    r = client.get(f"/songs/{sid}/accompaniment", params={"semitones": 1, "model": "kim"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert os.path.exists(os.path.join(m.SHIFTED_DIR, f"{sid}_kim_+1.0.wav"))


def test_STC_T06_004_accompaniment_bad_model(songs):
    r = client.get(f"/songs/{songs[0]}/accompaniment", params={"model": "abc"})
    assert r.status_code == 422


def test_STC_T06_005_accompaniment_missing_mdx(songs):
    r = client.get(f"/songs/{songs[1]}/accompaniment", params={"model": "mdx"})
    assert r.status_code == 404 and "MDX Main Inst" in r.json()["detail"]


def test_STC_T06_006_pitch_frames(real_pitch_frames_sample):
    # 실제 분석 결과 파일이 있으면 그 복사본으로, 없으면 1초 분량 합성 데이터로 시험
    dst = os.path.join(m.PITCH_FRAMES_DIR, "424242.json")
    if real_pitch_frames_sample:
        shutil.copy(real_pitch_frames_sample, dst)
    else:
        with open(dst, "w") as f:
            json.dump({"hop_ms": 10, "times": [i / 100 for i in range(200)], "hz": [220.0] * 200}, f)
    r = client.get("/songs/424242/pitch-frames")
    assert r.status_code == 200
    pf = r.json()
    assert pf["hop_ms"] == 10 and len(pf["times"]) == len(pf["hz"])
    assert abs((pf["times"][100] - pf["times"][0]) - 1.0) < 1e-6


def test_STC_T06_007_pitch_frames_missing():
    r = client.get("/songs/999999/pitch-frames")
    assert r.status_code == 404


# ---------------- T-07 음원 관리 ----------------
def test_STC_T07_001_run_invalid_url():
    r = client.post("/songs/run", data={"title": "x", "artist": "y", "url": "ftp://bad"})
    assert r.status_code == 400


def test_STC_T07_002_summarize_pitch():
    f0 = np.full(500, 220.0)
    f0[::10] = np.nan
    s = summarize_pitch(f0, sr=16000, hop_length=160)
    assert round(s.midi_min) == round(s.midi_median) == round(s.midi_max) == 57
    assert abs(s.voiced_ratio - 0.9) < 1e-9
    assert midi_to_note(57) == "A3" and midi_to_note(60) == "C4"


def test_STC_T07_003_delete_song(songs):
    sid = songs[2]
    assert client.delete(f"/songs/{sid}").status_code == 200
    assert client.delete(f"/songs/{sid}").status_code == 404
