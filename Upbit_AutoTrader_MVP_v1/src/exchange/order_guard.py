"""
주문 보호/유틸 모듈
- 백오프 재시도 (429/일시 오류 대비)
- 업비트 호가단위 보정 (단순판)
- 안전한 가격/금액 정리
"""

from __future__ import annotations
import time
import math
from typing import Callable, Any


def backoff_retry(fn: Callable[..., Any], *args, retries: int = 3, sleep: float = 0.4, factor: float = 1.6, **kwargs) -> Any:
    """
    지수적 백오프 재시도 래퍼.
    fn이 예외를 던지면 최대 retries-1번까지 재시도.
    """
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if i == retries - 1:
                break
            time.sleep(sleep * (factor ** i))
    if last:
        raise last


def round_tick(price: float) -> float:
    """
    업비트 호가단위 보정(요약 단순판).
    실제 호가 규칙은 마켓/가격대에 따라 달라집니다.
    필요한 경우 정밀표로 교체하세요.
    """
    if price <= 0 or math.isnan(price):
        return 0.0
    if price < 10: step = 0.01
    elif price < 100: step = 0.1
    elif price < 1000: step = 1
    elif price < 10_000: step = 5
    elif price < 100_000: step = 10
    elif price < 500_000: step = 50
    else: step = 100
    return math.floor(price / step) * step


def clamp_min_order(krw_amount: float, min_order_krw: float) -> float:
    """
    최소 주문 금액 보정
    """
    if krw_amount is None or math.isnan(krw_amount) or krw_amount <= 0:
        return 0.0
    return max(float(min_order_krw), float(krw_amount))


def safe_div(a: float, b: float) -> float:
    """
    0 나누기 보호
    """
    if b is None or b == 0 or math.isnan(b):
        return 0.0
    return float(a) / float(b)
