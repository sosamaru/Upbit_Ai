"""
빠른 점검 스크립트:
- KRW-BTC 일봉 200개 로드
- EMA20/50/100, RSI14, MACD(12,26,9) 계산
- 마지막 값 요약 출력
"""

from __future__ import annotations
import pandas as pd
import pyupbit

# 내부 지표 재사용
from ..indicators.tech import ema, rsi, macd

TICKER = "KRW-BTC"
INTERVAL = "day"
COUNT = 200

def main():
    df = pyupbit.get_ohlcv(TICKER, interval=INTERVAL, count=COUNT)
    if df is None or df.empty:
        print("[ERROR] OHLCV empty")
        return

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e100 = ema(close, 100)
    r = rsi(close, 14)
    m, s, h = macd(close, 12, 26, 9)

    print(f"[{TICKER}] 최근 종가: {close.iloc[-1]:,.0f} KRW")
    print(f"EMA20/50/100: {e20.iloc[-1]:,.0f} / {e50.iloc[-1]:,.0f} / {e100.iloc[-1]:,.0f}")
    print(f"RSI14: {r.iloc[-1]:.2f}")
    print(f"MACD: {m.iloc[-1]:.4f}, Signal: {s.iloc[-1]:.4f}, Hist: {h.iloc[-1]:.4f}")

    uptrend = e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1]
    macd_up = m.iloc[-1] > s.iloc[-1]
    rsi_mid = 50 <= r.iloc[-1] <= 75

    print("\n체크리스트:")
    print(f"- 추세(EMA20>50>100): {'OK' if uptrend else 'NO'}")
    print(f"- MACD>Signal: {'OK' if macd_up else 'NO'}")
    print(f"- RSI 50~75: {'OK' if rsi_mid else 'NO'}")

if __name__ == "__main__":
    main()
