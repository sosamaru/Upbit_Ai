"""
기술적 지표 모듈 (EMA, RSI, MACD, ATR)
- NaN/분모 0 보호
- 벡터화 연산 위주 (pandas)
- 전략 레이어에서 그대로 import 하여 사용
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd


def _safe_series(s: pd.Series) -> pd.Series:
    """시리즈를 float로 캐스팅하고 복사본 반환"""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce").astype(float).copy()


# -------------------------
# 이동평균 (EMA)
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    """
    지수이동평균
    """
    s = _safe_series(series)
    if span <= 0 or s.empty:
        return pd.Series([np.nan] * len(s), index=s.index)
    return s.ewm(span=span, adjust=False).mean()


# -------------------------
# RSI (Wilder)
# -------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI (Wilder 방식 근사). 0~100 범위.
    """
    s = _safe_series(series)
    if period <= 0 or len(s) < period + 1:
        return pd.Series([np.nan] * len(s), index=s.index)

    delta = s.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    # Wilder's smoothing (EMA의 com=period-1에 해당)
    roll_up = up.ewm(com=period - 1, adjust=False).mean()
    roll_down = down.ewm(com=period - 1, adjust=False).mean()

    denom = roll_down.replace(0.0, np.nan)
    rs = roll_up / denom
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(0.0)


# -------------------------
# MACD (12, 26, 9 기본)
# -------------------------
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 라인, 시그널 라인, 히스토그램 반환
    """
    s = _safe_series(series)
    if min(fast, slow, signal) <= 0 or s.empty:
        idx = s.index if not s.empty else None
        nan = pd.Series([np.nan] * len(s), index=idx)
        return nan, nan, nan

    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -------------------------
# ATR (Average True Range)
# -------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR(평균 진폭). 입력 df에는 'high','low','close' 열이 필요.
    반환 값은 ATR 시계열.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    try:
        high = _safe_series(df["high"])
        low = _safe_series(df["low"])
        close = _safe_series(df["close"])
    except KeyError:
        # 필요한 열이 없을 경우 NaN 반환
        return pd.Series([np.nan] * len(df), index=df.index)

    prev_close = close.shift(1)

    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    if period <= 0 or len(tr) < period:
        return pd.Series([np.nan] * len(tr), index=tr.index)

    # 단순 이동평균(일반적) / Wilder EMA도 사용 가능
    atr_val = tr.rolling(period, min_periods=period).mean()
    return atr_val


# -------------------------
# 파생 유틸 (전략 편의)
# -------------------------
def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    전략에서 자주 쓰는 기본 지표 컬럼을 df에 추가하여 반환.
    - ema20/ema50/ema100
    - rsi14
    - macd/macd_signal/macd_hist (12,26,9)
    - atr14
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    close = _safe_series(out.get("close", pd.Series(index=out.index)))

    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema100"] = ema(close, 100)

    out["rsi14"] = rsi(close, 14)

    m, s, h = macd(close, 12, 26, 9)
    out["macd"] = m
    out["macd_signal"] = s
    out["macd_hist"] = h

    out["atr14"] = atr(out, 14)

    return out
