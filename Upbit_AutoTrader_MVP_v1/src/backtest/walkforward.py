from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List

from .runner import backtest  # 기존 backtest 러너 사용


def _parse(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str)
    except Exception as e:
        raise ValueError(f"Invalid date: {date_str}") from e


def walkforward(
    start: str,
    end: str,
    window_days: int = 120,
    step_days: int = 30,
    *,
    initial_krw: int = 2_000_000,
    fee: float = 0.0005,
    slip: float = 0.0005,
) -> Dict[str, Any]:
    """
    [start, end] 구간을 window_days 단위로 굴리며 step_days 간격으로 재시작.
    각 세그먼트는 이전 세그먼트의 최종 자본을 초기자본으로 인계합니다.
    """
    if window_days <= 0 or step_days <= 0:
        raise ValueError("window_days and step_days must be positive")

    s = _parse(start)
    e = _parse(end)
    if s >= e:
        raise ValueError("start must be earlier than end")

    cur = s
    equity = int(initial_krw)
    segments: List[Dict[str, Any]] = []

    while cur < e:
        w_end_dt = min(e, cur + timedelta(days=window_days))
        seg_start = cur.date().isoformat()
        seg_end = w_end_dt.date().isoformat()

        res = backtest(seg_start, seg_end, initial_krw=equity, fee=fee, slip=slip)
        metrics = dict(res.get("metrics", {}))
        cum_ret = float(metrics.get("CumulativeReturn", 0.0))

        eq_start = equity
        equity = int(round(equity * (1.0 + cum_ret)))

        segments.append({
            "range": f"{seg_start}~{seg_end}",
            "equity_start": eq_start,
            "equity_end": equity,
            "metrics": metrics,
        })

        cur = cur + timedelta(days=step_days)

    combined = (equity / float(initial_krw)) - 1.0
    return {
        "segments": segments,
        "initial_equity": int(initial_krw),
        "final_equity": int(equity),
        "combined_return": float(combined),
    }


__all__ = ["walkforward"]
