"""
Signal Scoring (전문가 체크리스트 방식)
- 입력: 후보 티커 목록 + fetch_df(ticker)->OHLCV(DataFrame)
- 출력: 각 티커의 점수와 사유(plain text), 점수 내림차순 정렬

체크리스트(최대 7점)
1) 추세: EMA20 > EMA50 > EMA100 (+1)
2) 모멘텀: MACD > Signal (+1)
3) 모멘텀: RSI in [50, 75] (+1)
4) 거래량: 최근 1일 거래량 / 20일 평균 > 1.5 (+1)
5) 변동성: ATR(14)/종가 < 0.06 (+1)
6) 돌파: 종가 > EMA20 * 1.01 (+1)
7) 보너스: 최근 3일 중 최고 종가가 오늘 종가와 1% 이내(강한 종가 마감) (+1)

실무 팁:
- 점수≥5를 우선 후보로 간주. 단, 급등 과열/테마성 이슈는 별도 알림/필터를 권장.
- 실매매 전, 백테스트에서 cut-off(5~6점)와 지표 파라미터를 교정하세요.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable
import math

import pandas as pd

from ..indicators.tech import ema, rsi, macd, atr


@dataclass
class Signal:
    ticker: str
    score: float
    reason: str


class SignalModel:
    def __init__(
        self,
        rsi_low: float = 50.0,
        rsi_high: float = 75.0,
        vol_ratio_thresh: float = 1.5,
        atr_close_thresh: float = 0.06,
        ema20_break_pct: float = 0.01,
        recent_close_within: float = 0.01,
        min_len: int = 120,
    ):
        self.rsi_low = float(rsi_low)
        self.rsi_high = float(rsi_high)
        self.vol_ratio_thresh = float(vol_ratio_thresh)
        self.atr_close_thresh = float(atr_close_thresh)
        self.ema20_break_pct = float(ema20_break_pct)
        self.recent_close_within = float(recent_close_within)
        self.min_len = int(min_len)

    # -------------------------
    # Core scoring for one ticker
    # -------------------------
    def score_one(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < self.min_len:
            return Signal(ticker, 0.0, "insufficient data")

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        volume = pd.to_numeric(df["volume"], errors="coerce").astype(float)

        e20 = ema(close, 20)
        e50 = ema(close, 50)
        e100 = ema(close, 100)

        macd_line, sig, _ = macd(close, 12, 26, 9)
        r = rsi(close, 14)
        a = atr(df, 14)

        score = 0.0
        reasons: List[str] = []

        # 1) 추세
        try:
            if e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1]:
                score += 1; reasons.append("uptrend (EMA20>50>100)")
        except Exception:
            pass

        # 2) MACD 모멘텀
        try:
            if macd_line.iloc[-1] > sig.iloc[-1]:
                score += 1; reasons.append("MACD above signal")
        except Exception:
            pass

        # 3) RSI 구간
        try:
            if self.rsi_low <= r.iloc[-1] <= self.rsi_high:
                score += 1; reasons.append(f"RSI in [{self.rsi_low:.0f},{self.rsi_high:.0f}]")
        except Exception:
            pass

        # 4) 거래량 확장
        try:
            vol_ma20 = volume.rolling(20).mean().iloc[-1]
            if vol_ma20 and not math.isnan(vol_ma20) and vol_ma20 > 0:
                vol_ratio = volume.iloc[-1] / vol_ma20
                if vol_ratio > self.vol_ratio_thresh:
                    score += 1; reasons.append(f"vol x{vol_ratio:.1f}")
        except Exception:
            pass

        # 5) 변동성(너무 큰 변동 회피)
        try:
            atr_pct = a.iloc[-1] / (close.iloc[-1] + 1e-12)
            if atr_pct < self.atr_close_thresh:
                score += 1; reasons.append(f"ATR%<{self.atr_close_thresh*100:.1f}%")
        except Exception:
            pass

        # 6) 단기 돌파
        try:
            if close.iloc[-1] > e20.iloc[-1] * (1.0 + self.ema20_break_pct):
                score += 1; reasons.append(f"close>EMA20+{self.ema20_break_pct*100:.0f}%")
        except Exception:
            pass

        # 7) 강한 종가 마감(최근 3일 최고가와 1% 이내)
        try:
            last3_max_close = close.iloc[-3:].max()
            if last3_max_close > 0 and (last3_max_close - close.iloc[-1]) / last3_max_close <= self.recent_close_within:
                score += 1; reasons.append(f"near 3D high (≤{self.recent_close_within*100:.0f}%)")
        except Exception:
            pass

        return Signal(ticker, float(score), ", ".join(reasons) if reasons else "no edge")

    # -------------------------
    # Rank a candidate list
    # -------------------------
    def rank(self, candidates: List[str], fetch_df: callable) -> List[Signal]:
        """
        candidates: 티커 문자열 리스트 (예: ['KRW-BTC', 'KRW-ETH', ...])
        fetch_df: 함수(ticker)->OHLCV df (일봉 200개 이상 권장)
        """
        out: List[Signal] = []
        for t in candidates:
            try:
                df = fetch_df(t)
                sig = self.score_one(t, df)
                out.append(sig)
            except Exception:
                out.append(Signal(t, 0.0, "fetch/score failed"))
        out.sort(key=lambda x: x.score, reverse=True)
        return out


# -------------------------
# 편의 함수: 유니버스에서 상위 5 추리기
# -------------------------
def select_top5_from_universe(
    top_vol: List[str],
    top_chg: List[str],
    fetch_df: Callable[[str], pd.DataFrame],
    model: SignalModel | None = None,
) -> List[Signal]:
    """
    거래량 Top10 ∪ 상승률 Top10 → 후보 set → 모델로 스코어링 → 상위 5개 반환
    """
    model = model or SignalModel()
    candidates = list(set(top_vol) | set(top_chg))
    ranked = model.rank(candidates, fetch_df)
    return ranked[:5]
