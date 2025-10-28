"""
Risk & Position Sizing
- 계좌자본 대비 리스크 비중, 최소주문금액, 트레일링/손절 규칙을 일관되게 적용
- 실전 이전: cut-off 값(손절 %, 트레일 %)은 백테스트로 보정 권장

설정 연계 (.env)
- RISK_PER_TRADE: 계좌자본 대비 1건 위험 비중 (기본 0.02 = 2%)
- MAX_CONCURRENT_POS: 동시에 보유 가능한 최대 포지션 수 (기본 5)
- TRAIL_PCT: 트레일링 스탑 비율 (기본 0.05 = 5%)
- MIN_ORDER_KRW: 거래소 최소 주문 금액 보정 (기본 5,500 KRW)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

from ..config import SETTINGS
from ..exchange.upbit_client import UpbitClient


@dataclass
class PositionPlan:
    """
    매수/청산 계획 요약
    - krw_to_spend: 이번 진입에 투입할 원화 금액
    - trail_pct: 트레일링 스탑 비율(상승 시 손절선 상향)
    - stop_pct: 고정 손절 비율(진입가 대비)
    """
    ticker: str
    krw_to_spend: int
    trail_pct: float
    stop_pct: float
    meta: dict


class RiskManager:
    """
    포지션 사이징 및 방어 규칙 도출
    - 1/N 분할(기본): RISK_PER_TRADE × equity를 기준으로 하되,
      거래소 최소 주문 금액과 동일/초과하도록 보정
    - 트레일링/손절 기본값은 SETTINGS에 따름
    """
    def __init__(
        self,
        client: Optional[UpbitClient] = None,
        equity_krw: Optional[float] = None,
        default_stop_pct: float = 0.03,  # 3% 기본 손절
    ):
        self.client = client or UpbitClient()
        # 실매매 모드면 KRW 잔고를 참조, 드라이런에선 입력값/0 사용
        if equity_krw is None:
            self.equity = (self.client.krw_balance() if not self.client.dry_run else 0.0)
        else:
            self.equity = float(equity_krw)

        self.risk = float(SETTINGS.risk_per_trade)  # 0.02
        self.trail_pct = float(SETTINGS.trail_pct)  # 0.05
        self.default_stop_pct = float(default_stop_pct)
        self.min_order = float(SETTINGS.min_order_krw)
        self.max_pos = int(SETTINGS.max_pos)

    # -------------------------
    # 예산 산출
    # -------------------------
    def _budget_for_new_position(self) -> int:
        """
        예산 = max( equity * RISK_PER_TRADE, MIN_ORDER_KRW )
        (단순/보수적 접근: N-way equal weighting과 유사)
        """
        raw = float(self.equity) * self.risk
        if math.isnan(raw) or raw <= 0:
            raw = 0.0
        budget = max(self.min_order, raw)
        return int(budget)

    # -------------------------
    # 신규 진입 계획
    # -------------------------
    def plan_entry(self, ticker: str, stop_pct: Optional[float] = None) -> PositionPlan:
        """
        신규 포지션 진입 계획 도출
        - stop_pct: 미입력 시 default_stop_pct 사용
        """
        stop = float(stop_pct) if stop_pct is not None else self.default_stop_pct
        krw = self._budget_for_new_position()

        # Upbit 최소주문금액 보정(추가 안전장치)
        krw = max(int(self.min_order), int(krw))

        return PositionPlan(
            ticker=ticker,
            krw_to_spend=krw,
            trail_pct=self.trail_pct,
            stop_pct=stop,
            meta={
                "equity_snapshot": self.equity,
                "risk_per_trade": self.risk,
                "max_concurrent_pos": self.max_pos,
            },
        )

    # -------------------------
    # 트레일링 스탑 업데이트
    # -------------------------
    @staticmethod
    def update_trailing_stop(entry_price: float, highest_price: float, trail_pct: float) -> float:
        """
        진입 후 최고가(highest_price)를 기준으로 트레일링 스탑가 재계산
        stop_price = highest_price * (1 - trail_pct)
        단, 초기 stop은 entry_price * (1 - 기본 stop_pct) 보다 위로만 이동(하방 경직)
        """
        if highest_price <= 0 or trail_pct <= 0:
            return 0.0
        return float(highest_price) * (1.0 - float(trail_pct))

    # -------------------------
    # 손절가(고정 %) 계산
    # -------------------------
    @staticmethod
    def fixed_stop_price(entry_price: float, stop_pct: float) -> float:
        """
        고정 손절가: entry_price * (1 - stop_pct)
        """
        if entry_price <= 0 or stop_pct <= 0:
            return 0.0
        return float(entry_price) * (1.0 - float(stop_pct))

    # -------------------------
    # 체결 후 요약(로깅/리포트용)
    # -------------------------
    @staticmethod
    def summarize_fill(ticker: str, fill_price: float, krw_spent: float, trail_pct: float, stop_pct: float) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "fill_price": float(fill_price),
            "krw_spent": float(krw_spent),
            "trail_pct": float(trail_pct),
            "stop_pct": float(stop_pct),
            "trail_rule": "stop = highest * (1 - trail_pct)",
            "stop_rule": "stop = entry * (1 - stop_pct)",
        }
