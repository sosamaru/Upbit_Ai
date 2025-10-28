from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

from .config import SETTINGS
from .exchange.upbit_client import UpbitClient
from .strategy.universe import UniverseBuilder
from .strategy.signal import SignalModel
from .indicators.tech import ema, rsi, macd
from .utils.timeutil import now_kst

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("tg_bot")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Global singletons (가격/유니버스/스코어링 공유)
# -----------------------------------------------------------------------------
client = UpbitClient()
uni = UniverseBuilder(client)
signal_model = SignalModel()

# -----------------------------------------------------------------------------
# LOOP/repo 프록시 (재할당 자동 추적)
# -----------------------------------------------------------------------------
from . import scheduler_jobs as _sched  # noqa: E402


class _LoopProxy:
    def __getattr__(self, name):
        obj = getattr(_sched, "loop", None)
        if obj is None:
            raise AttributeError("loop not initialized")
        return getattr(obj, name)


class _RepoProxy:
    def __getattr__(self, name):
        obj = getattr(_sched, "repo", None)
        if obj is None:
            raise AttributeError("repo not initialized")
        return getattr(obj, name)


LOOP = _LoopProxy()
repo = _RepoProxy()

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
def _auth_guard(update: Update) -> bool:
    allowed = str(getattr(SETTINGS, "tg_chat_id", "")).strip()
    return True if not allowed else (str(update.effective_chat.id) == allowed)


async def _reject_if_unauthorized(update: Update) -> bool:
    if not _auth_guard(update):
        try:
            await update.message.reply_text("권한이 없습니다. (.env TG_CHAT_ID 확인)")
        except Exception:
            pass
        return True
    return False


# -----------------------------------------------------------------------------
# 유지보수/운영
# -----------------------------------------------------------------------------
async def sync_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    bals = client.balances()
    n = 0
    for b in bals:
        if not isinstance(b, dict):
            continue
        cur = (b.get("currency") or "").upper()
        if cur == "KRW":
            continue
        vol = float(b.get("balance", 0))
        t = f"{client.base}-{cur}"
        if vol <= 0:
            if repo.get_position(t):
                repo.remove_position(t)
            continue
        px = client.last_price(t) or 1.0
        repo.upsert_position(t, avg_price=px, qty=vol, highest=px, partial_taken=False)
        n += 1
    await update.message.reply_text(f"동기화 완료: {n} 종목 반영")


# -----------------------------------------------------------------------------
# 기본
# -----------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    await update.message.reply_text("Upbit AutoTrader 준비 완료. /help 를 입력하세요.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    await update.message.reply_text(
        "/go — 포트폴리오 루프 ON (5분 주기)\n"
        "/stop — 포트폴리오 루프 OFF\n"
        "/loop_status — 루프 상태 확인\n"
        "/report — 유니버스 상위 후보(점수/사유)\n"
        "/holdings — 실계좌 잔고 요약\n"
        "/sell <KRW-XXX> <all|수량> — 지정 매도\n"
        "/value — 총 평가금액(실계좌+포지션)\n"
        "/pnl — 최근 12시간 실현손익\n"
        "/pnl7d — 최근 7일 실현손익\n"
        "/pnl30d — 최근 30일 실현손익\n"
        "/reserve <KRW> — 현금 예약(해당 금액은 투자 제외)\n"
        "/raise_cash <KRW> — 보유분 일부 매도로 현금 확보\n"
        "/regime — BTC 레짐 체크(EMA/MACD/RSI)\n"
        "/settings — 현재 설정/루프 상태\n"
        "/cooldown [h] — 재진입 쿨다운 조회/설정(패치 미적용 시 안내)\n"
        "/guard [pct <p>] [krw <v>] | off — 일일 손실 한도 가드\n"
        "/flat_all — 전량 청산 후 루프 OFF\n"
        "/halt [분] — 신규 진입 일시중지\n"
        "/limits [slip v] [depth v] [hourly v] [daily v] [split on|off]\n"
        "/ban KRW-XXX | /unban KRW-XXX | /banlist\n"
        "/health — 루프 상태/지연 확인\n"
        "/help — 이 도움말"
    )


# -----------------------------------------------------------------------------
# 루프 제어
# -----------------------------------------------------------------------------
async def go(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    LOOP.set_enabled(True)
    await update.message.reply_text("자동 포트폴리오 루프: ON (5분 주기)")


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    LOOP.set_enabled(False)
    await update.message.reply_text("자동 포트폴리오 루프: OFF")


async def loop_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    await update.message.reply_text(
        f"루프 상태: {'ON' if getattr(LOOP, 'enabled', False) else 'OFF'} "
        f"/ reserve={int(getattr(LOOP, 'reserve_krw', 0.0)):,} KRW"
    )


# -----------------------------------------------------------------------------
# 리포트/상태
# -----------------------------------------------------------------------------
async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    try:
        _, top_vol, top_chg = uni.pick()
        candidates = list(set(top_vol) | set(top_chg))
        if not candidates:
            await update.message.reply_text("후보 없음")
            return
        fetch_df = lambda t: client.ohlcv(t, "day", 200)
        ranked = signal_model.rank(candidates, fetch_df)
        top = [s for s in ranked if s.score >= 5.0][:5]
        if not top:
            await update.message.reply_text("후보 없음")
            return
        lines = ["[리포트] 유니버스 상위 후보 5개"]
        for s in top:
            lines.append(f"{s.ticker} | score={s.score:.1f} | {s.reason}")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"리포트 실패: {e}")


async def holdings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    bals = client.balances()
    if not bals:
        await update.message.reply_text("보유 잔고 없음 (API키/권한 또는 잔고 확인)")
        return
    lines: List[str] = []
    for b in bals:
        if not isinstance(b, dict):
            continue
        cur = b.get("currency")
        bal = float(b.get("balance", 0))
        locked = float(b.get("locked", 0))
        if bal > 0 or locked > 0:
            lines.append(f"{cur}: {bal} (locked {locked})")
    await update.message.reply_text("\n".join(lines) if lines else "보유 잔고 없음")


# -----------------------------------------------------------------------------
# 주문/자산
# -----------------------------------------------------------------------------
async def sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("사용법: /sell KRW-XXX all|수량")
        return
    ticker, qty_str = args[0], args[1]
    try:
        if qty_str.lower() == "all":
            vol = client.coin_balance(ticker)
            res = client.guarded_sell_market(ticker, vol)
        else:
            res = client.guarded_sell_market(ticker, float(qty_str))
        await update.message.reply_text(f"SELL 응답: ok={res.ok} msg={res.message or '-'}")
    except Exception as e:
        await update.message.reply_text(f"SELL 실패: {e}")


async def value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    krw = client.krw_balance() if not client.dry_run else 0.0
    tot = float(krw)
    lines = [f"KRW: {int(krw):,}"]
    for p in repo.list_positions():
        px = client.last_price(p.ticker) or p.avg_price
        v = px * p.qty
        tot += v
        lines.append(f"{p.ticker}: qty={p.qty:.6f} px≈{int(px):,} val≈{int(v):,}")
    lines.append(f"총 평가금액≈ {int(tot):,} KRW (reserve {int(getattr(LOOP, 'reserve_krw', 0.0)):,})")
    await update.message.reply_text("\n".join(lines))


# -----------------------------------------------------------------------------
# PnL
# -----------------------------------------------------------------------------
def _pnl_window(hours: int) -> tuple[datetime, datetime]:
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    return start, end


async def pnl12h(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    s, e = _pnl_window(12)
    realized = repo.realized_pnl(s, e)
    await update.message.reply_text(f"[PnL 12h] realized≈ {int(realized):,} KRW")


async def pnl7d(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    s, e = _pnl_window(24 * 7)
    realized = repo.realized_pnl(s, e)
    await update.message.reply_text(f"[PnL 7d] realized≈ {int(realized):,} KRW")


async def pnl30d(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    s, e = _pnl_window(24 * 30)
    realized = repo.realized_pnl(s, e)
    await update.message.reply_text(f"[PnL 30d] realized≈ {int(realized):,} KRW")


# -----------------------------------------------------------------------------
# Reserve & 현금화
# -----------------------------------------------------------------------------
async def reserve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    if not context.args:
        await update.message.reply_text(f"현재 reserve={int(getattr(LOOP, 'reserve_krw', 0.0)):,} KRW")
        return
    amt = max(0.0, float(context.args[0]))
    LOOP.set_reserve(amt)
    await update.message.reply_text(f"reserve 설정: {int(amt):,} KRW")


async def raise_cash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    if not context.args:
        await update.message.reply_text("사용법: /raise_cash <KRW>")
        return
    need = max(0.0, float(context.args[0]))

    # 1) LOOP에 구현되어 있으면 우선 사용
    if hasattr(LOOP, "raise_cash"):
        try:
            raised = LOOP.raise_cash(need)  # type: ignore[attr-defined]
            await update.message.reply_text(f"현금 확보: {int(raised):,} KRW")
            return
        except Exception as e:
            logger.warning(f"LOOP.raise_cash 실패, fallback 사용: {e}")

    # 2) Fallback: 보유 포지션에서 큰 것부터 매도
    raised = 0.0
    pos = sorted(
        repo.list_positions(),
        key=lambda p: (client.last_price(p.ticker) or p.avg_price) * p.qty,
        reverse=True,
    )
    for p in pos:
        if raised >= need:
            break
        px = client.last_price(p.ticker) or p.avg_price
        if px <= 0:
            continue
        remain = need - raised
        qty_to_sell = min(p.qty, (remain / px) * 1.02)  # 수수료 여유
        if qty_to_sell <= 0:
            continue
        res = client.guarded_sell_market(p.ticker, qty_to_sell)
        if not res.ok:
            continue
        raised += px * qty_to_sell
        new_qty = p.qty - qty_to_sell
        if new_qty <= 1e-12:
            repo.add_fill("SELL", p.ticker, px, qty_to_sell, datetime.utcnow())
            repo.remove_position(p.ticker)
        else:
            repo.add_fill("SELL", p.ticker, px, qty_to_sell, datetime.utcnow())
            repo.upsert_position(p.ticker, avg_price=p.avg_price, qty=new_qty, highest=max(p.highest, px))
    await update.message.reply_text(f"현금 확보 시도 완료: ≈{int(raised):,} KRW")


# -----------------------------------------------------------------------------
# 신규: 레짐 / 설정 / 쿨다운 / 가드
# -----------------------------------------------------------------------------
async def regime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    try:
        btc = f"{getattr(SETTINGS, 'base_currency', 'KRW')}-BTC"
        df = client.ohlcv(btc, "day", 200)
        if df is None or df.empty:
            await update.message.reply_text("[regime] 데이터가 없습니다.")
            return
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        e20, e50, e100 = ema(close, 20), ema(close, 50), ema(close, 100)
        m, s, _ = macd(close, 12, 26, 9)
        r = rsi(close, 14)
        f_ema = bool(e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1])
        f_macd = bool(m.iloc[-1] > s.iloc[-1])
        f_rsi = bool(45 <= r.iloc[-1] <= 75)
        ok = sum([f_ema, f_macd, f_rsi]) >= 2
        msg = [
            f"[regime] {btc}",
            f"EMA20>50>100 : {'✅' if f_ema else '❌'}",
            f"MACD>Signal  : {'✅' if f_macd else '❌'}",
            f"RSI 45~75    : {'✅' if f_rsi else '❌'}",
            f"→ 신규 진입 허용: {'YES ✅' if ok else 'NO ❌'}",
        ]
        await update.message.reply_text("\n".join(msg))
    except Exception as e:
        await update.message.reply_text(f"[regime] 오류: {e}")


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    reserve = getattr(LOOP, "reserve_krw", 0.0)
    enabled = getattr(LOOP, "enabled", False)
    dd_pct = getattr(LOOP, "dd_guard_pct", 0.0)
    dd_krw = getattr(LOOP, "dd_guard_krw", 0.0)
    guard = getattr(LOOP, "guard_active", False)
    msg = [
        "[settings]",
        f"DRY_RUN           : {getattr(SETTINGS, 'dry_run', False)}",
        f"BASE_CURRENCY     : {getattr(SETTINGS, 'base_currency', 'KRW')}",
        f"MAX_CONCURRENT_POS: {getattr(SETTINGS, 'max_pos', 5)}",
        f"RISK_PER_TRADE    : {getattr(SETTINGS, 'risk_per_trade', 0.01)}",
        f"TRAIL_PCT         : {getattr(SETTINGS, 'trail_pct', 0.06)}",
        f"MIN_ORDER_KRW     : {int(getattr(SETTINGS, 'min_order_krw', 5000))}",
        f"RESERVE_KRW(loop) : {int(reserve)}",
        f"LOOP_ENABLED      : {enabled}",
        f"DAILY_GUARD       : {'ON' if guard else 'OFF'} (pct={dd_pct:.4f}, krw={int(dd_krw):,})",
    ]
    await update.message.reply_text("\n".join(msg))


async def guard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    args = context.args or []
    # 금일 실현손익(KST 00:00~)
    kst = now_kst()
    kst0 = kst.replace(hour=0, minute=0, second=0, microsecond=0)
    utc_start = kst0 - timedelta(hours=9)
    realized = repo.realized_pnl(utc_start, datetime.utcnow())

    if not args:
        await update.message.reply_text(
            "[guard] "
            f"status={'ON' if getattr(LOOP, 'guard_active', False) else 'OFF'} / "
            f"pct={getattr(LOOP, 'dd_guard_pct', 0.0):.4f} / "
            f"krw={int(getattr(LOOP, 'dd_guard_krw', 0.0)):,} / "
            f"today_realized={int(realized):,} KRW"
        )
        return

    try:
        if args[0].lower() == "off":
            LOOP.dd_guard_pct = 0.0
            LOOP.dd_guard_krw = 0.0
            LOOP.guard_active = False
            await update.message.reply_text("[guard] 비활성화했습니다.")
            return
        if args[0].lower() == "pct" and len(args) >= 2:
            LOOP.dd_guard_pct = max(0.0, float(args[1]))
            await update.message.reply_text(f"[guard] pct={LOOP.dd_guard_pct:.4f}")
            return
        if args[0].lower() == "krw" and len(args) >= 2:
            LOOP.dd_guard_krw = max(0.0, float(args[1]))
            await update.message.reply_text(f"[guard] krw={int(LOOP.dd_guard_krw):,}")
            return
        await update.message.reply_text("사용법: /guard | /guard pct 0.02 | /guard krw 50000 | /guard off")
    except Exception as e:
        await update.message.reply_text(f"[guard] 오류: {e}")


async def cooldown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    try:
        if not hasattr(LOOP, "_cooldown"):
            await update.message.reply_text("이 기능은 패치 6 적용 이후에 사용 가능합니다.")
            return
        args = context.args or []
        if not args:
            h = int(LOOP._cooldown.total_seconds() // 3600)  # type: ignore[attr-defined]
            await update.message.reply_text(f"[cooldown] 현재 재진입 쿨다운 = {h}h")
            return
        new_h = int(args[0])
        if new_h < 0 or new_h > 168:
            await update.message.reply_text("허용 범위: 0 ~ 168 시간")
            return
        LOOP._cooldown = timedelta(hours=new_h)  # type: ignore[attr-defined]
        await update.message.reply_text(f"[cooldown] 재진입 쿨다운을 {new_h}h 로 설정했습니다.")
    except Exception as e:
        await update.message.reply_text(f"[cooldown] 오류: {e}")


# -----------------------------------------------------------------------------
# 운영 안전장치 / 제한
# -----------------------------------------------------------------------------
async def flat_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    n = LOOP.flat_all()
    LOOP.set_enabled(False)
    await update.message.reply_text(f"[flat_all] {n}개 포지션 청산, 루프 OFF")


async def halt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    m = 60
    if context.args and context.args[0].isdigit():
        m = max(1, int(context.args[0]))
    LOOP.set_halt(m)
    until = (datetime.utcnow() + timedelta(minutes=m)).strftime("%H:%M UTC")
    await update.message.reply_text(f"[halt] 신규 진입 {m}분 중지 (until {until})")


async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    args = [a.lower() for a in (context.args or [])]
    if not args:
        split_stat = "N/A"
        if hasattr(LOOP.client, "enable_smart_split"):
            split_stat = (
                f"ON(chunks={getattr(LOOP.client, 'split_chunks', '?')}, "
                f"delay_ms={getattr(LOOP.client, 'split_sleep_ms', '?')})"
                if getattr(LOOP.client, "enable_smart_split")
                else "OFF"
            )
        await update.message.reply_text(
            "[limits]\n"
            f"slip={LOOP.client.max_slip_pct:.3%}, depth={int(LOOP.client.min_depth_krw):,} KRW\n"
            f"hourly={LOOP.rebal_hour_pct:.2f}, daily={LOOP.rebal_day_pct:.2f}\n"
            f"smart_split={split_stat}"
        )
        return
    i = 0
    while i < len(args):
        k = args[i]
        v = args[i + 1] if i + 1 < len(args) else None
        if k in ("slip", "slippage") and v:
            LOOP.client.max_slip_pct = float(v)
        elif k in ("depth", "min_depth") and v:
            LOOP.client.min_depth_krw = float(v)
        elif k in ("hourly", "hour") and v:
            LOOP.rebal_hour_pct = float(v)
        elif k in ("daily", "day") and v:
            LOOP.rebal_day_pct = float(v)
        elif k == "split" and v in ("on", "off") and hasattr(LOOP.client, "enable_smart_split"):
            LOOP.client.enable_smart_split = (v == "on")
        i += 2 if v is not None else 1
    await update.message.reply_text("[limits] 업데이트 완료.")


async def ban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    if not context.args:
        await update.message.reply_text("사용법: /ban KRW-XXX")
        return
    t = context.args[0].upper()
    repo.ban_ticker(t, reason="manual")
    await update.message.reply_text(f"[ban] {t} 추가")


async def unban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    if not context.args:
        await update.message.reply_text("사용법: /unban KRW-XXX")
        return
    t = context.args[0].upper()
    repo.unban_ticker(t)
    await update.message.reply_text(f"[unban] {t} 제거")


async def banlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    bl = repo.list_blacklist()
    await update.message.reply_text("[banlist]\n" + ("\n".join(bl) if bl else "(empty)"))


async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_unauthorized(update):
        return
    now = datetime.utcnow()
    last = getattr(LOOP, "last_step_ts", None)
    age = (now - last).total_seconds() / 60 if last else None
    await update.message.reply_text(
        "[health]\n"
        f"loop={'ON' if getattr(LOOP, 'enabled', False) else 'OFF'}\n"
        f"halt_until={getattr(LOOP, 'halt_until', None)}\n"
        + (f"last_step_age_min={age:.1f}" if age is not None else "last_step: N/A")
    )


# -----------------------------------------------------------------------------
# 에러 & 앱
# -----------------------------------------------------------------------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    logger.error(f"[TG ERROR] {err}")
    try:
        if isinstance(update, Update) and update.effective_chat:
            await update.effective_chat.send_message("네트워크 지연/오류가 발생했습니다. 다시 시도해주세요.")
    except Exception:
        pass


def build_app():
    # Telegram HTTP 타임아웃 여유
    req = HTTPXRequest(connect_timeout=20, read_timeout=20, write_timeout=20, pool_timeout=20)
    app = ApplicationBuilder().token(SETTINGS.tg_token).request(req).build()

    # 기존
    app.add_handler(CommandHandler("sync", sync_cmd))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("go", go))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("loop_status", loop_status))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("holdings", holdings))
    app.add_handler(CommandHandler("sell", sell))
    app.add_handler(CommandHandler("value", value))
    app.add_handler(CommandHandler("pnl", pnl12h))
    app.add_handler(CommandHandler("pnl7d", pnl7d))
    app.add_handler(CommandHandler("pnl30d", pnl30d))
    app.add_handler(CommandHandler("reserve", reserve))
    app.add_handler(CommandHandler("raise_cash", raise_cash))

    # 신규
    app.add_handler(CommandHandler("regime", regime))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("cooldown", cooldown))
    app.add_handler(CommandHandler("flat_all", flat_all_cmd))
    app.add_handler(CommandHandler("halt", halt_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    app.add_handler(CommandHandler("ban", ban_cmd))
    app.add_handler(CommandHandler("unban", unban_cmd))
    app.add_handler(CommandHandler("banlist", banlist_cmd))
    app.add_handler(CommandHandler("health", health_cmd))

    app.add_error_handler(on_error)
    return app
