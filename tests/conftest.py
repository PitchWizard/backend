"""
PitchWizard 백엔드 시험 공통 설정.

안전장치
- 시험은 users / songs 테이블을 비우므로 반드시 '시험 전용 DB'에서만 돌아가야 한다.
- TEST_DATABASE_URL 환경변수가 없거나, DB 이름에 'test'가 없으면 시험을 시작하지 않는다.
- outputs/ 폴더(실제 반주·피치 파일)는 건드리지 않도록 시험 동안 임시 폴더로 바꿔 끼운다.
"""
import os
import sys
import shutil
import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# 1) 시험 DB 확인 (api 모듈을 import 하기 전에 해야 함)
# ---------------------------------------------------------------------------
TEST_URL = os.environ.get("TEST_DATABASE_URL")
if not TEST_URL:
    raise pytest.UsageError(
        "TEST_DATABASE_URL 환경변수를 설정하세요. 예) "
        "mysql+pymysql://root:비밀번호@localhost/wizard_test?charset=utf8mb4"
    )
_db_name = make_url(TEST_URL).database or ""
if "test" not in _db_name.lower():
    raise pytest.UsageError(
        f"DB 이름 '{_db_name}' 에 'test' 가 없습니다. "
        "시험은 users/songs 테이블을 비우므로 실제 DB(wizard_db)로는 실행할 수 없습니다."
    )

# ---------------------------------------------------------------------------
# 2) 곡 분석 파이프라인(analyzer.analyzer) import
#    - 기본: 실제 모듈 사용 (venv311에 torch 등이 설치돼 있으면 그대로 import)
#    - PW_STUB_ANALYZER=1 이거나 import 실패 시: 분석 함수만 '실행 불가' 대역으로 교체
#      (분석 자체는 이 시험 범위 밖. /songs/run 은 URL 검증만 확인)
# ---------------------------------------------------------------------------
def _install_analyzer_stub(reason):
    stub = types.ModuleType("analyzer.analyzer")

    def _not_available(*a, **k):
        raise RuntimeError(f"analysis pipeline stubbed in tests ({reason})")

    stub.analyze_audio_summary = _not_available
    import analyzer  # noqa: F401  (패키지만 로드)
    sys.modules["analyzer.analyzer"] = stub
    print(f"[conftest] analyzer.analyzer 대역 사용: {reason}")


if os.environ.get("PW_STUB_ANALYZER") == "1":
    _install_analyzer_stub("PW_STUB_ANALYZER=1")
else:
    try:
        import analyzer.analyzer  # noqa: F401
    except Exception as e:  # torch 미설치 등
        _install_analyzer_stub(f"import 실패: {type(e).__name__}")

# ---------------------------------------------------------------------------
# 3) DB 엔진을 시험 DB로 교체 (api.main 이 import 될 때 create_all 이 시험 DB에 실행됨)
# ---------------------------------------------------------------------------
import api.database as _db  # noqa: E402

_db.engine = create_engine(TEST_URL, pool_pre_ping=True)
_db.SessionLocal.configure(bind=_db.engine)
_db.DATABASE_URL = TEST_URL

from api import main as _main  # noqa: E402

assert _main.engine is _db.engine, "api.main 이 시험 DB 엔진을 사용하지 않습니다."
print(f"[conftest] 시험 DB: {make_url(TEST_URL).render_as_string(hide_password=True)}")

# 실제 outputs 폴더 위치 (피치 프레임 샘플을 읽기 전용으로 복사할 때만 사용)
REAL_PITCH_FRAMES_DIR = _main.PITCH_FRAMES_DIR


# ---------------------------------------------------------------------------
# 4) outputs/ 를 임시 폴더로 바꿔 끼우기 → 실제 반주·피치 파일 보호
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def isolated_outputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("outputs")
    saved = {}
    for name in ("INSTRUMENTAL_DIR", "SHIFTED_DIR", "PITCH_FRAMES_DIR", "VOCALS_DIR"):
        if hasattr(_main, name):
            saved[name] = getattr(_main, name)
            d = base / name.lower()
            d.mkdir()
            setattr(_main, name, str(d))
    yield base
    for name, v in saved.items():
        setattr(_main, name, v)


@pytest.fixture(scope="session")
def real_pitch_frames_sample():
    """실제 분석 결과(outputs/pitch_frames/*.json) 하나의 경로. 없으면 None."""
    if not os.path.isdir(REAL_PITCH_FRAMES_DIR):
        return None
    files = sorted(f for f in os.listdir(REAL_PITCH_FRAMES_DIR) if f.endswith(".json"))
    return os.path.join(REAL_PITCH_FRAMES_DIR, files[0]) if files else None


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: 실제 rubberband 등 외부 프로그램이 필요한 시험 (PW_HEAVY=1 일 때만 실행)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PW_HEAVY") == "1":
        return
    skip = pytest.mark.skip(reason="PW_HEAVY=1 일 때만 실행")
    for it in items:
        if "heavy" in it.keywords:
            it.add_marker(skip)
