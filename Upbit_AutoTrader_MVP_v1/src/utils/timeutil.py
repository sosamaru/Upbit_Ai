"""
시간 유틸리티
- now_kst(): 한국표준시(KST) 현재 시각 반환
- to_kst(), to_utc() 변환기 포함
"""

from __future__ import annotations
from datetime import datetime
import pytz

KST = pytz.timezone("Asia/Seoul")


def now_kst() -> datetime:
    """현재 한국표준시(datetime)"""
    return datetime.now(tz=KST)


def to_kst(dt: datetime) -> datetime:
    """datetime을 KST로 변환"""
    if dt.tzinfo is None:
        return KST.localize(dt)
    return dt.astimezone(KST)


def to_utc(dt: datetime) -> datetime:
    """datetime을 UTC로 변환"""
    return dt.astimezone(pytz.UTC)
