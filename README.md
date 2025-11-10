
---

```markdown
# 🧠 Vocal Wizard Backend

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-success)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> 🎵 **오디오 분석 → 데이터베이스 자동 저장**  
> FastAPI 없이 동작하는 백엔드 분석 및 데이터 관리 시스템입니다.

---

## 📘 프로젝트 개요

**Vocal Wizard Backend**는 오디오 파일 또는 유튜브 링크를 입력받아  
AI 기반 엔진(`torchcrepe`, `pyin`, `yin`, `hybrid`)으로 피치(Pitch)와 음량(RMS)을 분석하고,  
결과를 MySQL 데이터베이스에 자동 저장하는 백엔드 시스템입니다.

- FastAPI 서버 없이 CLI에서 바로 실행 가능  
- 분석 결과는 `Song` 테이블에 자동 반영  
- ORM 기반으로 유지보수와 확장이 용이

---

## 🏗️ 디렉터리 구조

```

wizard/
│
├── analyzer/
│   └── analyzer.py               # 오디오 분석 로직 (analyze_audio_summary)
│
├── api/
│   ├── database.py               # SQLAlchemy 엔진 및 세션
│   ├── models.py                 # DB 테이블 정의 (Song, User 등)
│   ├── crud.py                   # CRUD 함수 (add_song 등)
│   └── services/
│       └── analyze_and_save.py   # 분석 + DB 저장 통합 서비스
│
├── main.py                       # CLI 엔트리포인트 (분석 및 저장 실행)
└── requirements.txt

````

---

## ⚙️ 설치 및 설정

### 1️⃣ 환경 구성

```bash
git clone https://github.com/yourname/vocal-wizard-backend.git
cd vocal-wizard-backend

python -m venv venv
venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
````

### 2️⃣ MySQL 연결 설정

`api/database.py` 수정:

```python
DATABASE_URL = "mysql+pymysql://root:비밀번호@localhost/wizard_db?charset=utf8mb4"
```

MySQL 데이터베이스가 없다면 생성:

```sql
CREATE DATABASE wizard_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

---

## 🚀 실행 방법 (CLI 모드)

단일 오디오 파일 분석 및 DB 저장:

```bash
python main.py "C:\audio\hypeboy.wav" \
  --engine torchcrepe \
  --title "Hype Boy" \
  --artist "NewJeans"
```

### 실행 흐름

1. `main.py` → `analyze_audio_summary()` 호출
2. 분석 결과(JSON) 출력
3. `crud.add_song()`을 통해 MySQL에 저장
4. 성공 시 콘솔 출력 예시:

```
✅ DB 저장 완료: id=3, Hype Boy - NewJeans
```

---

## 🧠 분석 결과 예시

```json
{
  "midi_min": 43.2,
  "midi_median": 60.0,
  "midi_max": 72.8,
  "rms_mean": 0.124,
  "rms_std": 0.031
}
```

이 값들은 모두 `Song` 테이블에 저장됩니다.

---

## 🗄️ 데이터베이스 구조

| 컬럼          | 타입      | 설명          |
| ----------- | ------- | ----------- |
| song_id     | INT     | 기본 키        |
| title       | VARCHAR | 곡 제목        |
| artist      | VARCHAR | 가수명         |
| midi_min    | FLOAT   | 피치 최소값      |
| midi_median | FLOAT   | 피치 중앙값      |
| midi_max    | FLOAT   | 피치 최대값      |
| rms_mean    | FLOAT   | RMS 평균 (음량) |
| rms_std     | FLOAT   | RMS 표준편차    |

> 필요 시 `(title, artist)` 유니크 제약을 추가하여 중복을 방지할 수 있습니다.

---

## 🧩 핵심 구성 요소

| 파일                                 | 역할     | 설명                     |
| ---------------------------------- | ------ | ---------------------- |
| `analyzer/analyzer.py`             | 분석 엔진  | 오디오 피치 및 RMS 계산        |
| `api/database.py`                  | DB 연결  | SQLAlchemy 엔진 및 세션 관리  |
| `api/models.py`                    | ORM 모델 | Song, User 등 정의        |
| `api/crud.py`                      | DB 연산  | add_song(), upsert() 등 |
| `api/services/analyze_and_save.py` | 통합 서비스 | 분석 → DB 저장 처리          |
| `main.py`                          | 실행 엔트리 | CLI 기반 분석/저장 수행        |

---

## 🧰 개발 및 유지보수 가이드

* **FastAPI 미사용 시**
  `api/main.py`, `schemas.py` 등 REST 관련 파일은 제거해도 무방합니다.

* **CRUD 단순화**
  `add_song()`과 `upsert_song_by_title_artist()`만 유지해도 충분합니다.

* **분석 교체**
  `analyzer/analyzer.py`의 `analyze_audio_summary()`를 수정하면
  전체 파이프라인에 자동 반영됩니다.

* **환경 변수 관리**
  `.env` 파일에 DB 정보를 분리:

  ```bash
  DATABASE_URL=mysql+pymysql://user:pass@localhost/wizard_db?charset=utf8mb4
  ```

  코드에서 불러오기:

  ```python
  import os
  DATABASE_URL = os.getenv("DATABASE_URL")
  ```

---

## 🧪 테스트

### 단일 파일 테스트

```bash
python main.py "C:\audio\test.wav" --engine pyin --title "Test Song" --artist "Tester"
```

### 다중 파일 배치 처리 예시

```python
from api.services.analyze_and_save import analyze_and_save
import glob

for f in glob.glob("C:/audio/*.wav"):
    title = f.split("\\")[-1].replace(".wav", "")
    analyze_and_save(title=title, artist="Batch", audio_path=f)
```

---

## 📦 주요 의존성

| 패키지                         | 용도             |
| --------------------------- | -------------- |
| **SQLAlchemy**              | ORM (MySQL 연동) |
| **PyMySQL**                 | MySQL 드라이버     |
| **TorchCrepe**, **Librosa** | 오디오 피치/RMS 분석  |
| **NumPy**, **SciPy**        | 수치 계산          |
| **python-dotenv**           | 환경 변수 관리 (선택)  |

---

## 🔮 확장 아이디어

* 대량 분석 자동화 (배치 파이프라인)
* 기존 데이터 업데이트 (`upsert_song_by_title_artist`)
* 분석 로그 테이블 추가 (timestamp, source 등)
* REST API 확장 (FastAPI 도입 시 손쉽게 연결 가능)

---

## 📜 라이선스

MIT License © 2025 **정현준**

---

### 🧩 개발자 메모

> Vocal Wizard Backend는 “**분석 → 저장 자동화**”에 초점을 둔 순수 백엔드 시스템입니다.
> 프론트엔드나 시각화는 별도 구성되며,
> 본 프로젝트의 핵심은 **정확한 분석 파이프라인과 안정적인 DB 연동**입니다.

```

---


```
