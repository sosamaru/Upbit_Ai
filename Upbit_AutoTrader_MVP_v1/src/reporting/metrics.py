"""
백테스트/실거래 공용 성과 지표
- Equity 곡선(Series)에서 누적수익률, MDD, 샤프, 변동성 계산
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def equity_to_metrics(equity: pd.Series, rf: float = 0.0) -> dict:
    equity = equity.astype(float)
    rets = equity.pct_change().fillna(0.0)

    # 누적수익률
    cum_ret = (1.0 + rets).prod() - 1.0

    # 최대낙폭(MDD)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = dd.min() if len(dd) else 0.0

    # 샤프(일봉 가정 252 거래일)
    std = rets.std()
    if std and std > 0:
        sharpe = (rets.mean() - rf / 252.0) / std * np.sqrt(252.0)
    else:
        sharpe = 0.0

    return {
        "CumulativeReturn": float(cum_ret),
        "MDD": float(mdd),
        "Sharpe": float(sharpe),
        "Vol": float(std),
    }
