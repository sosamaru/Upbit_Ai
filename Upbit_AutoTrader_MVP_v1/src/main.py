from __future__ import annotations

import sys
import asyncio
from dotenv import load_dotenv

from .utils.logging_setup import setup_logging
from .storage.sqlite_repo import SqliteRepo
from .portfolio.portfolio_loop import PortfolioLoop
from .scheduler_jobs import build_scheduler
from .telegram_bot import build_app


# Windows 콘솔 호환(telegram asyncio)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main():
    # 1) .env 로드 (TG/UPBIT 키 등)
    load_dotenv()

    # 2) 로깅 초기화
    setup_logging()

    # 3) 핵심 객체 생성
    repo = SqliteRepo()
    loop = PortfolioLoop(repo)

    # 4) 스케줄러/전역 주입 (중요: 텔레그램 봇보다 먼저)
    build_scheduler(loop, repo)

    # 5) 텔레그램 앱 빌드 & 실행(블로킹)
    app = build_app()
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
