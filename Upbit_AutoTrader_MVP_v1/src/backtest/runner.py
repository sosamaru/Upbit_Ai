"""
일봉 백테스트 (OHLCV 종가 기반, 실전 로직 근사)
- 유니버스: (거래대금Top10 ∪ 상승률Top10)  ※ 근사(스냅샷 기반)
- 신규 진입: 레짐 필터 통과 + Signal score>=5 상위에서 최대 5개
- 사이징: RiskManager.plan_entry() (RPT × Equity, 최소주문 보정)
- 청산: 초기 손절(-2.5%), +4% 1차 익절(50%) & BE 승급, ATR(14) 트레일(2.5×ATR) [백업 6%]
- 수수료/슬리피지: 매수/매도 체결금액에 (1 - fee - slip) 반영
주의:
  * 실전과 동일한 '과거 시점 유니버스/신호'는 pyupbit 한계로 완전 복원 어렵습니다(근사).
  * OHLCV API 제약상 day count는 최대 200 전후이므로, 요청 구간과 겹치는 최근 ~200영업일만 시뮬합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd

from ..config import SETTINGS
from ..exchange.upbit_client import UpbitClient
from ..indicators.tech import atr, ema, macd, rsi
from ..portfolio.risk import RiskManager
from ..reporting.metrics import equity_to_metrics
from ..strategy.signal import SignalModel
from ..strategy.universe import UniverseBuilder


@dataclass
class Pos:
    avg: float
    qty: float
    high: float
    scaled: bool  # 1차 익절(50%) 실행 여부


def _regime_ok(client: UpbitClient, day: pd.Timestamp) -> bool:
    """KRW-BTC 일봉으로 레짐 평가: EMA20>50>100, MACD>Signal, RSI 45~75 중 2개 이상."""
    try:
        df = client.ohlcv(f"{client.base}-BTC", "day", 200)
        if df is None or df.empty:
            return True
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index <= day]
        if df.empty:
            return True
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        if len(close) < 100:
            return True  # 워밍업 구간은 차단하지 않음

        e20, e50, e100 = ema(close, 20), ema(close, 50), ema(close, 100)
        m, s, _ = macd(close, 12, 26, 9)
        r = rsi(close, 14)

        conds = 0
        conds += 1 if (e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1]) else 0
        conds += 1 if (m.iloc[-1] > s.iloc[-1]) else 0
        conds += 1 if (45 <= r.iloc[-1] <= 75) else 0
        return conds >= 2
    except Exception:
        return True


def _build_price_series(client: UpbitClient, ticker: str, count: int = 200) -> pd.Series:
    """해당 티커의 일봉 종가 시리즈(DatetimeIndex)."""
    df = client.ohlcv(ticker, "day", count)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df["close"], errors="coerce").astype(float)
    s.index = pd.to_datetime(df.index)
    return s


def _px_on(series: pd.Series, day: pd.Timestamp) -> float | None:
    """해당 날짜(day)까지 이용 가능한 가장 최근 종가를 반환."""
    if series is None or series.empty:
        return None
    series = series.sort_index()
    cut = series.loc[series.index <= day]
    if cut.empty:
        return None
    return float(cut.iloc[-1])


# ---- 워크포워드 성능 향상: 세그먼트 간 네트워크 캐시 재사용 (env로 OFF 가능) ----
_BACKTEST_SHARE = os.getenv("BACKTEST_SHARE_CLIENT", "1") == "1"
_SHARED_CLIENT = UpbitClient() if _BACKTEST_SHARE else None
_SHARED_UNI = UniverseBuilder(_SHARED_CLIENT) if _SHARED_CLIENT else None
_SHARED_MODEL = SignalModel() if _BACKTEST_SHARE else None


def backtest(
    start_date: str,
    end_date: str,
    initial_krw: float = 1_000_000.0,
    fee: float = 0.0005,
    slip: float = 0.0005,
) -> dict:
    # 공유 객체 사용(있으면), 아니면 새로 생성
    use_shared = os.getenv("BACKTEST_SHARE_CLIENT", "1") == "1"
    c = _SHARED_CLIENT if (use_shared and _SHARED_CLIENT) else UpbitClient()
    uni = _SHARED_UNI if (use_shared and _SHARED_UNI) else UniverseBuilder(c)
    model = _SHARED_MODEL if (use_shared and _SHARED_MODEL) else SignalModel()

    # ---- 시뮬레이션 캘린더 구성 (최근 ~200영업일로 자동 클리핑) ----
    req_days = pd.date_range(start=start_date, end=end_date, freq="D")
    pivot = c.ohlcv(f"{c.base}-BTC", "day", 200)
    if pivot is None or pivot.empty:
        # 데이터가 없으면 요청 캘린더를 그대로 사용(근사)
        days = req_days
    else:
        pivot_idx = pd.to_datetime(pivot.index)
        # 요청 구간과 겹치는 날짜만 사용
        days = pd.DatetimeIndex([d for d in req_days if d in pivot_idx])
        if len(days) == 0:
            days = pivot_idx  # 마지막 200일 근사

    cash = float(initial_krw)
    positions: Dict[str, Pos] = {}
    curve: List[float] = []

    # 가격 시리즈 캐시
    price_cache: Dict[str, pd.Series] = {}

    for day in days:
        day = pd.to_datetime(day)

        # --- 레짐 필터 (day 기준 과거 데이터로 평가) ---
        regime_ok = _regime_ok(c, day)

        # --- 유니버스 & 랭킹 (근사: 최신 기준 랭킹 사용) ---
        _, top_vol, top_chg = uni.pick()
        candidates = list(set(top_vol) | set(top_chg)) if regime_ok else []
        top5: List[str] = []
        if candidates:
            fetch_df = lambda t: c.ohlcv(t, "day", 200)
            ranked = model.rank(candidates, fetch_df)
            filtered = [s for s in ranked if s.score >= 5.0]
            top5 = [s.ticker for s in filtered[:5]]

        # ---- 총자산 산출 (평가) ----
        eq_positions = 0.0
        for t in positions:
            if t not in price_cache:
                price_cache[t] = _build_price_series(c, t, 200)
            px_t = _px_on(price_cache[t], day) or positions[t].avg
            eq_positions += px_t * positions[t].qty
        equity = cash + eq_positions

        # ---- RiskManager ----
        rm = RiskManager(client=c, equity_krw=float(equity))

        # ---- 신규 진입 (day 종가를 체결가로 사용) ----
        if regime_ok and top5:
            max_pos = int(SETTINGS.max_pos)
            cur_cnt = sum(1 for _t, p in positions.items() if p.qty > 0)
            slots_left = max(0, max_pos - cur_cnt)

            for t in top5:
                if slots_left <= 0:
                    break
                if t in positions and positions[t].qty > 0:
                    continue

                if t not in price_cache:
                    price_cache[t] = _build_price_series(c, t, 200)
                px = _px_on(price_cache[t], day)
                if px is None or px <= 0:
                    continue

                plan = rm.plan_entry(t)
                buy_amt = min(float(plan.krw_to_spend), cash)
                if buy_amt < SETTINGS.min_order_krw:
                    continue

                # 체결(수수료/슬립 반영)
                qty = (buy_amt * (1 - fee - slip)) / px
                if qty <= 0:
                    continue
                cash -= buy_amt
                positions[t] = Pos(avg=px, qty=qty, high=px, scaled=False)
                slots_left -= 1

        # ---- 보유 포지션 관리 (손절/익절/트레일: day 종가로 평가) ----
        to_close: List[str] = []
        for t, p in list(positions.items()):
            if t not in price_cache:
                price_cache[t] = _build_price_series(c, t, 200)
            px = _px_on(price_cache[t], day) or p.avg
            if px <= 0:
                continue
            p.high = max(p.high, px)
            gain = (px / p.avg - 1.0)

            # (1) 초기 손절 -2.5%
            if gain <= -0.025:
                cash += px * p.qty * (1 - fee - slip)
                to_close.append(t)
                continue

            # (2) 1차 익절 +4%: 50% 청산, BE 승급
            if (gain >= 0.04) and (not p.scaled) and p.qty > 0:
                half = p.qty * 0.5
                cash += px * half * (1 - fee - slip)
                p.qty -= half
                p.avg = max(p.avg, px * 0.999)  # 손익분기점 보호
                p.scaled = True

            # (3) ATR 트레일 (백업 6%) — ATR도 day 이전 데이터로 계산
            trail_stop = None
            df = c.ohlcv(t, "day", 60)
            if df is not None and not df.empty:
                try:
                    df = df.copy()
                    df.index = pd.to_datetime(df.index)
                    df = df.loc[df.index <= day]
                    if not df.empty:
                        a14 = atr(df, 14).iloc[-1]
                        if a14 and a14 > 0:
                            trail_stop = p.high - 2.5 * float(a14)
                except Exception:
                    trail_stop = None
            if trail_stop is None:
                trail_stop = p.high * 0.94  # ~6%
            trail_stop = max(trail_stop, p.avg)  # BE 보호

            if px <= trail_stop:
                cash += px * p.qty * (1 - fee - slip)
                to_close.append(t)

        for t in to_close:
            positions.pop(t, None)

        # ---- 일말 평가 ----
        eq_now = cash + sum((_px_on(price_cache[t], day) or p.avg) * p.qty for t, p in positions.items())
        curve.append(eq_now)

    curve = pd.Series(curve, index=days)
    return {"curve": curve, "metrics": equity_to_metrics(curve)}
