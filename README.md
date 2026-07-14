# 주식 수익 최대화 AI

주가 상승 확률을 학습하고, 위험 제한을 적용해 백테스트와 모의매매로 검증하는 최소 구조의 AI 프로젝트입니다.

> 이 프로젝트의 첫 단계는 **수익 보장**이 아니라 데이터 누수 방지, 비용 반영, 손실 제한, 재현 가능한 검증입니다. 실거래 주문은 의도적으로 포함하지 않았습니다.

## 파일 구조

```text
Upbit_Ai/
├── main.py                 # 다운로드·학습·예측·백테스트 명령
├── stock_ai.py             # 특징 생성·AI 모델·위험관리·백테스트 통합
├── connectors.py           # 데이터/증권사 연결 규격 + 모의 증권사
├── tests/test_stock_ai.py  # 핵심 자동 테스트
├── pyproject.toml          # 의존성·검사 설정
├── .env.example            # 실행 설정 예시
├── .gitignore              # 키·모델·데이터 보호
└── .github/workflows/ci.yml
```

## 설계 원칙

- **AI 핵심과 외부 API 분리:** 증권사를 바꿔도 `stock_ai.py`는 수정하지 않습니다.
- **다음 봉 체결:** 현재 종가에서 계산한 예측을 다음 봉 시가에 적용해 미래 데이터 누수를 줄입니다.
- **거래비용 포함:** 수수료와 슬리피지를 백테스트에 반영합니다.
- **위험 우선:** 손절, 익절, 최대 투자 비중을 기본 설정으로 둡니다.
- **실거래 금지 상태:** 먼저 백테스트 → 모의매매 → 소액 검증 순서로 진행합니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
cp .env.example .env             # Windows: copy .env.example .env
```

## 사용 순서

### 1. 공개 데이터 다운로드

```bash
python main.py download --symbol AAPL --start 2018-01-01 --end 2026-01-01 --output aapl.csv
```

### 2. 모델 학습

```bash
python main.py train --data aapl.csv --model model.joblib
```

학습 평가는 시계열 순서를 유지한 마지막 구간에서 계산합니다. 정확도 하나만 보지 말고 ROC-AUC, 정밀도와 이후 백테스트 결과를 함께 판단해야 합니다.

### 3. 최신 예측

```bash
python main.py predict --data aapl.csv --model model.joblib
```

### 4. 백테스트

```bash
python main.py backtest --data aapl.csv --model model.joblib
```

### 5. 자동 테스트

```bash
ruff check .
pytest -q
```

## 실제 증권사 연결 방법

`connectors.py`의 `Broker` 규격을 구현하면 됩니다.

```python
class MyBroker:
    def get_cash(self) -> float: ...
    def get_position(self, symbol: str) -> float: ...
    def buy(self, symbol: str, quantity: float) -> str: ...
    def sell(self, symbol: str, quantity: float) -> str: ...
```

실제 주문 연결 전 반드시 다음 조건을 통과해야 합니다.

1. 여러 종목과 상승·하락·횡보 구간의 백테스트
2. 학습 구간과 완전히 분리된 워크포워드 검증
3. 최소 수 주의 모의매매
4. API 오류, 중복 주문, 장 마감, 네트워크 장애 대응
5. 일일 최대 손실 및 전체 거래 중지 기능

## 다음 개발 순서

1. 데이터 저장소와 종목 유니버스 추가
2. 워크포워드 학습과 모델 버전 관리
3. 여러 모델 앙상블 및 시장 국면 분류
4. 모의매매 실행 루프와 거래 기록 DB
5. 검증 완료 후에만 증권사 실계좌 어댑터 추가

투자 손실 가능성이 있으며, 모델 출력은 투자 수익을 보장하지 않습니다.
