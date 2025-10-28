"""
PnL Reporter (MVP)
- 체결/평가 내역을 받아 기간별 손익을 요약합니다.
- MVP에서는 간단한 ledger 프로토콜을 가정하고, 이후 SQLite로 확장 예정.

Ledger 인터페이스(예시)
- add_fill(side, ticker, price, volume, ts)  # 체결 기록
- add_mark(ticker, price, ts)                # 평가(마크) 기록
- iter_fills(start_ts, end_ts) -> iterable[dict]
- iter_marks(start_ts, end_ts) -> iterable[dict]

여기서는 최소 기능만 사용하는 것을 전제로 동작합니다.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Iterable, List

from ..utils.timeutil import now_kst


@dataclass
class PnLWindow:
    start_iso: str
    end_iso: str
    realized: float
    unrealized: float
    fills_count: int
    marks_count: int


class PnLReporter:
    """
    기간별 손익 요약을 생성하는 리포터.
    - realized: 매도 체결로 확정된 손익의 합 (단순 합산, 수수료/슬리피지 미반영)
    - unrealized: 보유분에 대한 평가손익 합 (마크 가격 기준, 단순 근사)
    """

    def __init__(self, ledger):
        """
        ledger: 위 인터페이스를 만족하는 원장 객체
        """
        self.ledger = ledger

    # -------------------------
    # 내부 헬퍼
    # -------------------------
    @staticmethod
    def _ts(dt: datetime) -> float:
        return dt.timestamp()

    def _collect_fills(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        if not hasattr(self.ledger, "iter_fills"):
            return []
        return list(self.ledger.iter_fills(self._ts(start), self._ts(end)))

    def _collect_marks(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        if not hasattr(self.ledger, "iter_marks"):
            return []
        return list(self.ledger.iter_marks(self._ts(start), self._ts(end)))

    # -------------------------
    # 평가손익 계산(근사)
    # -------------------------
    @staticmethod
    def _calc_unrealized_from_marks(marks: Iterable[Dict[str, Any]]) -> float:
        """
        매우 단순한 근사:
        - 동일 티커의 마지막 마크 가격만 사용
        - 보유 수량/평단이 없는 MVP 단계에선 0으로 둠
          → 추후 보유 수량/평단을 ledger가 제공하면 이 함수에서 실제 평가손익 계산
        """
        # TODO: 이후 ledger.get_positions() 등을 활용하여 실제 평가손익 계산
        return 0.0

    # -------------------------
    # 확정손익 계산(근사)
    # -------------------------
    @staticmethod
    def _calc_realized_from_fills(fills: Iterable[Dict[str, Any]]) -> float:
        """
        매우 단순한 근사:
        - 매수/매도 페어링을 하지 않고, 'side'가 'SELL'인 체결의 금액 합계를 이익으로 가정하지 않습니다.
        - MVP에서는 확정손익을 0으로 두고, 추후 포지션 페어링 로직을 추가합니다.
        """
        # TODO: FIFO/LIFO 페어링으로 realized 계산
        return 0.0

    # -------------------------
    # 공개 API
    # -------------------------
    def window_pnl(self, hours: int = 12) -> PnLWindow:
        """
        최근 N시간 구간의 손익 요약
        """
        end = now_kst()
        start = end - timedelta(hours=int(hours))
        fills = self._collect_fills(start, end)
        marks = self._collect_marks(start, end)

        realized = self._calc_realized_from_fills(fills)
        unrealized = self._calc_unrealized_from_marks(marks)

        return PnLWindow(
            start_iso=start.isoformat(),
            end_iso=end.isoformat(),
            realized=float(realized),
            unrealized=float(unrealized),
            fills_count=len(fills),
            marks_count=len(marks),
        )

    def pnl_12h(self) -> PnLWindow:
        return self.window_pnl(hours=12)

    def pnl_7d(self) -> PnLWindow:
        """
        최근 7일(168시간) 요약
        """
        return self.window_pnl(hours=24 * 7)

    def pnl_30d(self) -> PnLWindow:
        """
        최근 30일(720시간) 요약
        """
        return self.window_pnl(hours=24 * 30)


# ---------------------------------------------------------
# 간단한 인메모리 Ledger (옵션): MVP 테스트용
# ---------------------------------------------------------
class InMemoryLedger:
    """
    매우 단순한 인메모리 원장. 실전에서는 DB/파일로 교체하세요.
    """
    def __init__(self):
        self._fills: List[Dict[str, Any]] = []
        self._marks: List[Dict[str, Any]] = []

    # API
    def add_fill(self, side: str, ticker: str, price: float, volume: float, ts: Optional[float] = None):
        self._fills.append({
            "side": side.upper(),
            "ticker": ticker,
            "price": float(price),
            "volume": float(volume),
            "ts": ts if ts is not None else now_kst().timestamp(),
        })

    def add_mark(self, ticker: str, price: float, ts: Optional[float] = None):
        self._marks.append({
            "ticker": ticker,
            "price": float(price),
            "ts": ts if ts is not None else now_kst().timestamp(),
        })

    # 조회
    def iter_fills(self, start_ts: float, end_ts: float):
        for f in self._fills:
            if start_ts <= f["ts"] <= end_ts:
                yield f

    def iter_marks(self, start_ts: float, end_ts: float):
        for m in self._marks:
            if start_ts <= m["ts"] <= end_ts:
                yield m
