# api/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ⚙️ MySQL 연결 정보 — 실제 root 비밀번호로 교체하시오
DATABASE_URL = "mysql+pymysql://root:guswns6262@localhost/wizard_db?charset=utf8mb4"

# ✅ 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True  # 연결 확인용 옵션
)

# ✅ 세션 팩토리 생성
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ✅ 베이스 클래스 선언
Base = declarative_base()
