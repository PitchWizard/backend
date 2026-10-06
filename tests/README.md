# PitchWizard 시스템 시험 코드

D07 시스템 시험 결과서의 **자동 시험 34건**을 다시 돌려볼 수 있는 코드입니다.
수동 시험 16건(마이크·브라우저 조작)은 사람이 직접 해야 합니다.

| 위치 | 대상 | D07 시험 ID |
|---|---|---|
| `wizard/tests/test_backend.py` | FastAPI API, 키 추천, 피치 요약 | STC-T01~T07 (24건 + 실제 변환 1건 STC-T06-012) |
| `frontend/tests/logic_test.cjs` | 피치 검출, 테시투라, 반주 채점, 키 범위 제한 | STC-T02-003~006, T05-005, T06-008~011 (9건) |

## 읽기 전에: 이 시험이 보장하는 것과 못 하는 것

- 기대값(`assert`)은 **AI가 코드를 읽고 쓴 것**입니다. 그래서 "코드가 지금 이렇게 동작한다"는 것만 확인됩니다.
  코드 자체가 요구사항과 다르면 시험도 같이 틀린 채로 통과합니다.
  → 팀원이 **D03 유즈케이스와 D04 백로그의 인수 조건을 보고**, 각 assert 값이 맞는지 직접 검토해 주세요.
- 문서의 첫 결과는 개발 PC가 아닌 다른 환경(MariaDB, 분석 모듈 대역)에서 나왔습니다.
  **팀 PC(venv311 + MySQL)에서 다시 돌린 결과로 D07을 갱신**하는 게 맞습니다.
- 곡 분석(유튜브 다운로드 → 음원 분리 → RMVPE)은 시간이 오래 걸리고 외부 의존이 커서 자동 시험에 넣지 않았습니다.

## 1. 백엔드 시험 (Windows, venv311)

### 1-1. 시험 전용 DB 만들기 (최초 1회)

시험은 `users`, `songs` 테이블을 **전부 지웁니다.** 반드시 따로 만든 DB에서 돌리세요.
DB 이름에 `test`가 없으면 시험이 시작되지 않게 막아 두었습니다.

```sql
CREATE DATABASE wizard_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

테이블은 시험 시작 시 자동으로 만들어집니다(`Base.metadata.create_all`).

### 1-2. 시험 도구 설치 (최초 1회)

```powershell
cd C:\Users\RC\wizard
.\venv311\Scripts\activate
pip install -r tests\requirements-test.txt
```

### 1-3. 실행

```powershell
cd C:\Users\RC\wizard
.\venv311\Scripts\activate
$env:TEST_DATABASE_URL = "mysql+pymysql://root:<비밀번호>@localhost/wizard_test?charset=utf8mb4"
python -m pytest tests -v
```

- `24 passed, 1 skipped` 가 나오면 정상입니다.
- 결과를 파일로 남기려면: `python -m pytest tests -v > tests\result_backend.txt`
- 실제 Rubber Band 변환까지 보려면(오래 걸림): `$env:PW_HEAVY = "1"` 후 다시 실행 → skipped 1건이 실행됩니다.
  (rubberband.exe 가 PATH 또는 프로젝트 폴더에 있어야 함. 이 시험은 아직 한 번도 실행해 보지 못했습니다.)
- torch 등을 불러오는 게 느리면 `$env:PW_STUB_ANALYZER = "1"` 로 분석 모듈을 대역으로 바꿀 수 있습니다.

### 1-4. 안전장치

- `outputs/` 폴더(실제 반주·피치 파일)는 시험 동안 임시 폴더로 바꿔 끼우므로 건드리지 않습니다.
- `api/database.py`의 실제 DB 주소는 사용하지 않습니다. `TEST_DATABASE_URL` 만 씁니다.
- 비밀번호가 들어간 명령은 커밋하지 마세요.

## 2. 프론트엔드 시험 (Node 22)

추가 설치는 필요 없습니다(vite가 이미 설치한 esbuild 사용).

```powershell
cd C:\Users\RC\wizard\frontend
node tests\logic_test.cjs
```

`9 passed, 0 failed` 가 나오면 정상입니다.
화면 소스에서 계산 함수(`autocorrelate`, `estimateTessitura`, `calcAnalysis`, `clampToAccompanimentRange`)만
뽑아서 실행하므로, 함수 이름을 바꾸면 `not found` 로 실패합니다. 그때는 시험 파일의 이름도 같이 고쳐 주세요.

## 3. D07에 반영하는 순서

1. 팀원이 assert 값을 D03/D04와 대조 → 이상한 값이 있으면 시험 또는 코드 수정
2. 팀 PC에서 위 명령으로 실행 → 결과 파일 저장
3. D07의 자동 시험 결과·시험 환경(운영체제, MySQL 버전)을 실제 값으로 갱신
4. D07 개정 이력에 **검수자 이름과 날짜** 기록

## 파일

```
wizard/tests/
  README.md              이 문서
  conftest.py            시험 DB 확인·교체, outputs 격리, 분석 모듈 대역
  test_backend.py        백엔드 시험 (함수 이름 = D07 시험 ID)
  requirements-test.txt  pytest, httpx
frontend/tests/
  logic_test.cjs         프론트엔드 계산 로직 시험
```
