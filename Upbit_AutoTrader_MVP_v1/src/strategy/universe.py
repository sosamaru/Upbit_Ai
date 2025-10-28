from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import pandas as pd

from ..exchange.upbit_client import UpbitClient


@dataclass
class SignalScore:
    ticker: str
    score: float
    reason: str


class UniverseBuilder:
    """
    Upbit /v1/ticker 24h 요약(배치) 기반의 경량 유니버스:
      - 거래대금 상위 10 (acc_trade_price_24h)
      - 24h 등락률 상위 10 (signed_change_rate)
      - 기본 필터: 가격 >= 20원, 거래대금 >= 1e8 (체결 안정성)
    """
    def __init__(self, client: UpbitClient):
        self.client = client

    def pick(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        df = self.client.market_summaries_24h()
        if df is None or df.empty:
            return pd.DataFrame(), [], []

        base = df[(df["trade_price"] >= 20) & (df["acc_trade_price_24h"] >= 1e8)].copy()
        if base.empty:
            return df, [], []

        top_vol = base.sort_values("acc_trade_price_24h", ascending=False).head(10).index.tolist()
        top_chg = base.sort_values("signed_change_rate", ascending=False).head(10).index.tolist()
        return base, top_vol, top_chg


class SignalModel:
    """
    간단 점수화: 최근 20일 수익률 + 거래대금(근사) 가중
    - fetch_df(ticker) 는 일봉 200개 정도를 반환한다고 가정
    """
    def rank(self, tickers: List[str], fetch_df: Callable[[str], pd.DataFrame]) -> List[SignalScore]:
        out: List[SignalScore] = []
        for t in tickers:
            try:
                df = fetch_df(t)
                if df is None or df.empty or len(df) < 50:
                    continue
                close = pd.to_numeric(df["close"], errors="coerce").astype(float)
                r20 = float((close.iloc[-1] / close.iloc[-20] - 1.0)) if len(close) >= 21 else 0.0
                vol = float(df["value"].iloc[-1]) if "value" in df.columns else 0.0
                score = 10.0 * r20 + (vol / 1e9)  # 가벼운 가중
                reason = f"r20={r20:.2%}"
                out.append(SignalScore(t, score, reason))
            except Exception:
                continue
        out.sort(key=lambda s: s.score, reverse=True)
        return out


def select_top5_from_universe(
    top_vol: List[str],
    top_chg: List[str],
    fetch_df: Callable[[str], pd.DataFrame],
    model: SignalModel,
) -> List[SignalScore]:
    tickers = list(set(top_vol) | set(top_chg))
    if not tickers:
        return []
    ranked = model.rank(tickers, fetch_df)
    return ranked[:5]
