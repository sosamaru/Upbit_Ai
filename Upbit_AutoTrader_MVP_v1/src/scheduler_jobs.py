from __future__ import annotations

import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone as tz

from .portfolio.portfolio_loop import PortfolioLoop
from .storage.sqlite_repo import SqliteRepo
from .config import SETTINGS
from .utils.timeutil import now_kst

# ---------------------------------------------------------------------
# 전역
# ---------------------------------------------------------------------
logger = logging.getLogger("scheduler_jobs")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

SCHED_TZ = tz("Asia/Seoul")  # 스케줄 실행 타임존(KST)
scheduler: Optional[BackgroundScheduler] = None
loop: Optional[PortfolioLoop] = None
repo: Optional[SqliteRepo] = None


# ---------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------
def send_telegram(text: str) -> None:
    """텔레그램 DM 전송. 토큰/채팅ID 없으면 로그만."""
    try:
        token = getattr(SETTINGS, "tg_token", None)
        chat_id = getattr(SETTINGS, "tg_chat_id", None)
        if not token or not chat_id:
            logger.info(f"[tg(log only)] {text}")
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
    except Exception as e:
        logger.warning(f"send_telegram failed: {e}")


def _kpi_since(since_utc: datetime) -> str:
    """간단 KPI 집계: 실현손익, 매수/매도 회전액"""
    try:
        realized = repo.realized_pnl(since=since_utc, until=datetime.utcnow()) if repo else 0.0
        buys = repo.turnover_krw(since=since_utc, until=None, side="BUY") if repo else 0.0
        sells = repo.turnover_krw(since=since_utc, until=None, side="SELL") if repo else 0.0
    except Exception as e:
        logger.warning(f"_kpi_since failed: {e}")
        realized = buys = sells = 0.0
    return f"PnL={int(realized):,} / BUY={int(buys):,} / SELL={int(sells):,}"


# ---------------------------------------------------------------------
# 잡 함수
# ---------------------------------------------------------------------
def portfolio_step_job():
    """포트폴리오 루프 한 스텝."""
    if loop is None:
        return
    try:
        status = loop.step()
        logger.info(f"[loop] {status}")
    except Exception as e:
        logger.exception(f"loop.step error: {e}")
        send_telegram(f"[loop] 오류: {e}")


def watchdog_job():
    """루프가 멈췄는지 감시."""
    if loop is None:
        return
    last = getattr(loop, "last_step_ts", None)
    if last is None:
        return
    try:
        minutes = float(getattr(SETTINGS, "watchdog_minutes", 10))
    except Exception:
        minutes = 10.0
    if (datetime.utcnow() - last) > timedelta(minutes=minutes):
        send_telegram(f"[watchdog] 루프가 {int(minutes)}분 이상 갱신되지 않았습니다.")


def report_12h_job():
    """최근 12시간 KPI."""
    since = datetime.utcnow() - timedelta(hours=12)
    send_telegram("[12h]\n" + _kpi_since(since))


def report_daily_job():
    """일일 리포트: 당일 00:00 KST부터."""
    kst_now = now_kst()
    kst0 = kst_now.replace(hour=0, minute=0, second=0, microsecond=0)
    utc0 = kst0 - timedelta(hours=9)
    send_telegram("[daily]\n" + _kpi_since(utc0))


def report_weekly_job():
    """주간 리포트: 최근 7일."""
    since = datetime.utcnow() - timedelta(days=7)
    send_telegram("[weekly]\n" + _kpi_since(since))


# ---------------------------------------------------------------------
# 스케줄러 초기화/시작
# ---------------------------------------------------------------------
def build_scheduler(loop_ref: PortfolioLoop, repo_ref: SqliteRepo) -> BackgroundScheduler:
    """루프/레포 주입형. 여러 번 호출해도 단일 스케줄러 유지."""
    global scheduler, loop, repo
    loop, repo = loop_ref, repo_ref
    if scheduler is None:
        scheduler = BackgroundScheduler(timezone=SCHED_TZ)
        # 5분 루프
        scheduler.add_job(portfolio_step_job, IntervalTrigger(minutes=5), id="loop_step", replace_existing=True)
        # 워치독 (3분마다)
        scheduler.add_job(watchdog_job, IntervalTrigger(minutes=3), id="watchdog", replace_existing=True)
        # 기존 08:00 / 20:00 12h 리포트 유지
        scheduler.add_job(report_12h_job, CronTrigger(hour=8, minute=0), id="r12h_morning", replace_existing=True)
        scheduler.add_job(report_12h_job, CronTrigger(hour=20, minute=0), id="r12h_evening", replace_existing=True)
        # 일일·주간 리포트 추가 (KST 기준)
        scheduler.add_job(report_daily_job, CronTrigger(hour=0, minute=10), id="daily", replace_existing=True)
        scheduler.add_job(report_weekly_job, CronTrigger(day_of_week="mon", hour=0, minute=20),
                          id="weekly", replace_existing=True)
        scheduler.start()
        logger.info("APScheduler started (Asia/Seoul)")
    return scheduler


# 하위호환: 기존 코드가 start_scheduler()만 호출하더라도 동작하도록 래퍼 제공
def start_scheduler(loop_ref: PortfolioLoop = None, repo_ref: SqliteRepo = None):
    if loop_ref is not None and repo_ref is not None:
        build_scheduler(loop_ref, repo_ref)
    elif loop is not None and repo is not None:
        build_scheduler(loop, repo)
    else:
        logger.warning("start_scheduler: loop/repo 미주입 — 스케줄러만 기동됨(잡은 등록되지 않음)")


def shutdown_scheduler(wait: bool = False):
    try:
        if scheduler:
            scheduler.shutdown(wait=wait)
            logger.info("APScheduler shutdown")
    except Exception:
        pass
