"""
환경설정 로더
- .env 파일을 읽어 Settings 객체에 담아 전역 SETTINGS로 제공
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


@dataclass
class Settings:
    upbit_access: str = os.getenv("UPBIT_ACCESS_KEY", "")
    upbit_secret: str = os.getenv("UPBIT_SECRET_KEY", "")

    tg_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    dry_run: bool = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")

    base_currency: str = os.getenv("BASE_CURRENCY", "KRW")
    max_pos: int = int(os.getenv("MAX_CONCURRENT_POS", "5"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    trail_pct: float = float(os.getenv("TRAIL_PCT", "0.05"))
    min_order_krw: float = float(os.getenv("MIN_ORDER_KRW", "5500"))


# 전역 SETTINGS 인스턴스
SETTINGS = Settings()
