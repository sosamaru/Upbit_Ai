from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from ..exchange.upbit_client import UpbitClient
from ..strategy.universe import UniverseBuilder
from ..strategy.signal import SignalModel
from ..storage.sqlite_repo import SqliteRepo
from ..config import SETTINGS
from ..indicators.tech import ema, rsi, macd, atr
from ..utils.timeutil import now_kst

logger = logging.getLogger("portfolio_loop")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

KST_UTC_OFFSET_H = 9  # KST = UTC+9


class PortfolioLoop:
    """
    레짐/시그널/사이징/주문가드/리밸런싱/데일리가드/쿨다운/블랙리스트/트레일
    """
    def __init__(self, repo: Optional[SqliteRepo] = None):
        self.client = UpbitClient()
        self.uni = UniverseBuilder(self.client)
        self.model = SignalModel()
        self.repo = repo or SqliteRepo()

        # 실행 상태
        self.enabled: bool = False
        self.last_step_ts: Optional[datetime] = None

        # 운영 제어
        self.reserve_krw: float = 0.0
        self.halt_until: Optional[datetime] = None  # 신규 진입 일시 중지

        # Daily Loss Guard
        self.dd_guard_pct = float(os.getenv("DAILY_LOSS_PCT", "0.02"))
        self.dd_guard_krw = float(os.getenv("DAILY_LOSS_KRW", "0"))
        self.guard_active = False
        self._last_guard_day = None  # date(KST)

        # 리밸런싱 한도
        self.rebal_hour_pct = float(os.getenv("REBAL_LIMIT_HOURLY_PCT", "0.25"))
        self.rebal_day_pct = float(os.getenv("REBAL_LIMIT_DAILY_PCT", "1.00"))
        self.rebal_min_trade = float(os.getenv("REBAL_MIN_TRADE_KRW", str(SETTINGS.min_order_krw)))

        # 상관관계 캡
        self.corr_cap = float(os.getenv("CORR_CAP", "0.85"))

        # ATR 타겟팅 / RPT 스위치
        self.atr_k = float(os.getenv("ATR_K", "1.0"))                    # stop 거리 계수(ATR*atr_k)
        self.vol_target_pct = float(os.getenv("VOL_TARGET_PCT", "0"))    # 0=off, 예: 0.01
        self.rpt_floor = float(os.getenv("RPT_FLOOR", "0.003"))          # 최소 RPT(0.3%)

    # ----- 외부 제어 API -----
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on) 

    def set_reserve(self, krw: float) -> None:
        self.reserve_krw = max(0.0, float(krw))

    def set_api_keys(self, access_key: str, secret_key: str) -> None:
        try:
            self.client.set_api_keys(access_key, secret_key)
        except Exception as e:
            logger.error(f"PortfolioLoop.set_api_keys failed: {e}")

    def set_halt(self, minutes: int) -> None:
        self.halt_until = datetime.utcnow() + timedelta(minutes=max(1, int(minutes)))

    # ----- 내부 유틸 -----
    def _sync_from_exchange_once(self) -> None:
        """실계좌 보유 내역을 DB에 1회 반영(초기 구동 시)."""
        try:
            if self.client.dry_run:
                return
            synced = any(self.repo.list_positions())
            if synced:
                return
            for b in self.client.balances():
                if not isinstance(b, dict):
                    continue
                cur = (b.get("currency") or "").upper()
                if cur == "KRW":
                    continue
                qty = float(b.get("balance", 0))
                if qty <= 0:
                    continue
                t = f"{self.client.base}-{cur}"
                px = self.client.last_price(t) or 1.0
                self.repo.upsert_position(t, avg_price=px, qty=qty, highest=px, partial_taken=False)
        except Exception as e:
            logger.warning(f"sync_from_exchange_once skipped: {e}")

    def _regime_ok(self) -> bool:
        """멀티 타임프레임 레짐 (일봉 + 4H)"""
        try:
            btc = f"{self.client.base}-BTC"
            # 1) 일봉
            d = self.client.ohlcv(btc, "day", 200)
            ok_day = True
            if d is not None and not d.empty:
                close = d["close"]
                e20, e50, e100 = ema(close, 20), ema(close, 50), ema(close, 100)
                m, s, _ = macd(close, 12, 26, 9)
                r = rsi(close, 14)
                conds = 0
                conds += 1 if (e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1]) else 0
                conds += 1 if (m.iloc[-1] > s.iloc[-1]) else 0
                conds += 1 if (45 <= r.iloc[-1] <= 75) else 0
                ok_day = (conds >= 2)
            # 2) 4시간봉
            h4 = self.client.ohlcv(btc, "minute240", 300)
            ok_h4 = True
            if h4 is not None and not h4.empty:
                close = h4["close"]
                e20, e50 = ema(close, 20), ema(close, 50)
                m, s, _ = macd(close, 12, 26, 9)
                ok_h4 = (e20.iloc[-1] > e50.iloc[-1]) and (m.iloc[-1] > s.iloc[-1])
            return ok_day and ok_h4
        except Exception:
            # 데이터 이슈 시 과도한 차단을 피하기 위해 허용
            return True

    def _refresh_daily_guard(self, equity_krw: float) -> float:
        """금일(KST) 실현손익 기준 데일리 손실 가드 갱신."""
        kst_now = now_kst()
        today = kst_now.date()
        if self._last_guard_day != today:
            self.guard_active = False
            self._last_guard_day = today
        kst_start = kst_now.replace(hour=0, minute=0, second=0, microsecond=0)
        utc_start = kst_start - timedelta(hours=KST_UTC_OFFSET_H)
        try:
            realized = self.repo.realized_pnl(utc_start, datetime.utcnow())
        except Exception:
            realized = 0.0
        triggered = False
        if self.dd_guard_pct > 0:
            if realized <= -float(equity_krw) * float(self.dd_guard_pct):
                triggered = True
        if self.dd_guard_krw > 0:
            if realized <= -float(self.dd_guard_krw):
                triggered = True
        if triggered:
            self.guard_active = True
        return realized

    def _buy_capacity_krw(self, equity_krw: float) -> float:
        """시간/일 기준 리밸런싱 한도로 구매 가능 총액 산출."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        buy_1h = self.repo.turnover_krw(since=one_hour_ago, side="BUY")
        cap_1h = max(0.0, float(equity_krw) * self.rebal_hour_pct - buy_1h)

        kst0 = now_kst().replace(hour=0, minute=0, second=0, microsecond=0)
        utc0 = kst0 - timedelta(hours=KST_UTC_OFFSET_H)
        buy_day = self.repo.turnover_krw(since=utc0, side="BUY")
        cap_day = max(0.0, float(equity_krw) * self.rebal_day_pct - buy_day)

        return max(0.0, min(cap_1h, cap_day))

    def _rpt_factor(self, equity_krw: float) -> float:
        """최근 7일 성과로 RPT 스위치."""
        since = datetime.utcnow() - timedelta(days=7)
        pnl7 = self.repo.realized_pnl(since=since, until=datetime.utcnow())
        ratio = pnl7 / max(1.0, float(equity_krw))
        if ratio <= -0.05:
            return 0.25
        if ratio <= -0.02:
            return 0.50
        return 1.0

    def _passes_corr_cap(self, candidate: str, held: List[str]) -> bool:
        if not held:
            return True
        try:
            c_df = self.client.ohlcv(candidate, "day", 120)
            if c_df is None or c_df.empty:
                return True
            c_ret = pd.to_numeric(c_df["close"]).pct_change().dropna()
            for h in held:
                h_df = self.client.ohlcv(h, "day", 120)
                if h_df is None or h_df.empty:
                    continue
                h_ret = pd.to_numeric(h_df["close"]).pct_change().dropna()
                m = min(len(c_ret), len(h_ret))
                if m < 30:
                    continue
                corr = c_ret.tail(m).corr(h_ret.tail(m))
                if corr is not None and corr > self.corr_cap:
                    return False
        except Exception:
            return True
        return True

    # ----- 운영 명령 -----
    def flat_all(self) -> int:
        """전량 청산 후 DB 반영."""
        cnt = 0
        for p in self.repo.list_positions():
            if p.qty <= 0:
                continue
            res = self.client.guarded_sell_market(p.ticker, p.qty)
            if res.ok:
                px = self.client.last_price(p.ticker) or p.avg_price
                self.repo.add_fill("SELL", p.ticker, px, p.qty, datetime.utcnow())
                self.repo.remove_position(p.ticker)
                try:
                    self.repo.set_last_exit(p.ticker, datetime.utcnow())
                except Exception:
                    pass
                cnt += 1
        return cnt

    # ----- 메인 루프 -----
    def step(self) -> str:
        if not self.enabled:
            return "OFF"
        self.last_step_ts = datetime.utcnow()

        # 최초 1회: 실계좌 → DB 동기화(드라이런 제외)
        self._sync_from_exchange_once()

        # 현재 포지션/자산
        pos_list = self.repo.list_positions()
        positions = {p.ticker: p for p in pos_list}
        krw_total = self.client.krw_balance() if not self.client.dry_run else 1_000_000.0
        total_equity = float(krw_total)
        for t_symbol, p in positions.items():
            px_ = self.client.last_price(t_symbol) or p.avg_price
            total_equity += px_ * p.qty

        # 데일리 가드
        realized_today = self._refresh_daily_guard(total_equity)

        # 레짐 / 신규진입 일시정지 여부
        regime_ok = self._regime_ok()
        now = datetime.utcnow()
        halted = (self.halt_until is not None) and (now < self.halt_until)

        # 유니버스 / 후보 선정
        _, top_vol, top_chg = self.uni.pick()
        candidates = list(set(top_vol) | set(top_chg)) if regime_ok else []

        # 블랙리스트 / 쿨다운 필터
        bl = set(self.repo.list_blacklist())

        def valid_cand(t: str) -> bool:
            if t in bl:
                return False
            cd = self.repo.get_cooldown(t)
            return (cd is None) or (now >= cd)

        candidates = [t for t in candidates if valid_cand(t)]

        # 점수 상위 5
        top5: List[str] = []
        if candidates:
            fetch_df = lambda t: self.client.ohlcv(t, "day", 200)
            ranked = self.model.rank(candidates, fetch_df)
            filtered = [s for s in ranked if s.score >= 5.0]
            top = filtered[:5]
            top5 = [s.ticker for s in top]

        # 신규 진입 로직
        if (not halted) and regime_ok and top5 and (not self.guard_active):
            max_pos = int(SETTINGS.max_pos)
            cur_pos_cnt = sum(1 for p in pos_list if p.qty > 0)
            slots_left = max(0, max_pos - cur_pos_cnt)

            free_krw = self.client.krw_available()
            spendable_by_cash = max(0.0, free_krw - self.reserve_krw)
            spendable_by_cap = self._buy_capacity_krw(total_equity)
            spendable_global = max(0.0, min(spendable_by_cash, spendable_by_cap))

            for t in top5:
                if slots_left <= 0:
                    break
                if spendable_global < self.rebal_min_trade:
                    break
                if t in positions and positions[t].qty > 0:
                    continue
                if not self._passes_corr_cap(t, [k for k, p in positions.items() if p.qty > 0]):
                    continue

                # 변동성 과도 코인 차단(ATR%)
                df = self.client.ohlcv(t, "day", 60)
                if df is None or df.empty:
                    continue
                try:
                    close = float(df["close"].iloc[-1])
                    a14 = float(atr(df, 14).iloc[-1])
                    atr_pct = a14 / close if close > 0 else 1.0
                    if atr_pct > 0.06:
                        continue
                except Exception:
                    continue

                px = self.client.last_price(t)
                if not px or px <= 0:
                    continue

                # RPT + 성과 스위치
                base_rpt = float(getattr(SETTINGS, "risk_per_trade", 0.01))
                rpt = max(self.rpt_floor, base_rpt * self._rpt_factor(total_equity))
                buy_amt_rpt = float(total_equity) * rpt

                # ATR 타겟팅(옵션)
                if self.vol_target_pct > 0:
                    stop_pct = max(0.02, self.atr_k * atr_pct)  # 최소 2% stop 가정
                    target_krw = float(total_equity) * float(self.vol_target_pct)
                    buy_amt_vt = target_krw / stop_pct
                    buy_amt = min(buy_amt_rpt, buy_amt_vt, spendable_global)
                else:
                    buy_amt = min(buy_amt_rpt, spendable_global)

                if buy_amt < SETTINGS.min_order_krw:
                    continue

                res = self.client.smart_guarded_buy_market(t, buy_amt)
                if not res.ok:
                    logger.info(f"guarded_buy blocked: {t} {buy_amt} => {res.message}")
                    continue

                buy_qty = buy_amt / px
                self.repo.upsert_position(t, avg_price=px, qty=buy_qty, highest=px, partial_taken=False)
                self.repo.add_fill("BUY", t, px, buy_qty, datetime.utcnow())
                positions[t] = self.repo.get_position(t) or positions.get(t)
                slots_left -= 1
                spendable_global = max(0.0, spendable_global - buy_amt)

        # Top5 외 강제 청산(레짐이 좋을 때만)
        if regime_ok and top5:
            for t, p in list(positions.items()):
                if t not in top5 and p.qty > 0:
                    res = self.client.guarded_sell_market(t, p.qty)
                    if res.ok:
                        px_now = self.client.last_price(t) or p.avg_price
                        self.repo.add_fill("SELL", t, px_now, p.qty, datetime.utcnow())
                        self.repo.remove_position(t)
                        try:
                            self.repo.set_last_exit(t, datetime.utcnow())
                        except Exception:
                            pass

        # 보유 관리(손절/부분익절/트레일)
        for p in self.repo.list_positions():
            if p.qty <= 0:
                continue
            px = self.client.last_price(p.ticker) or 0.0
            if px <= 0:
                continue

            new_high = max(p.highest, px)
            self.repo.upsert_position(p.ticker, avg_price=p.avg_price, qty=p.qty, highest=new_high)

            gain = (px / p.avg_price - 1.0)
            # 즉시 손절
            if gain <= -0.025:
                res = self.client.guarded_sell_market(p.ticker, p.qty)
                if res.ok:
                    self.repo.add_fill("SELL", p.ticker, px, p.qty, datetime.utcnow())
                    self.repo.remove_position(p.ticker)
                    try:
                        self.repo.set_last_exit(p.ticker, datetime.utcnow())
                    except Exception:
                        pass
                continue

            # 부분 익절 (1차 50%)
            partial_done = False
            try:
                partial_done = self.repo.get_partial_taken(p.ticker)
            except Exception:
                pass

            if (gain >= 0.04) and (not partial_done) and (p.qty > 0):
                half = p.qty * 0.5
                res = self.client.guarded_sell_market(p.ticker, half)
                if res.ok:
                    self.repo.add_fill("SELL", p.ticker, px, half, datetime.utcnow())
                    remain = p.qty - half
                    be_avg = max(p.avg_price, px * 0.999)  # 손익분기점 상향
                    self.repo.upsert_position(
                        p.ticker, avg_price=be_avg, qty=remain, highest=max(new_high, px), partial_taken=True
                    )

            # 트레일링 스탑 (ATR 기반, 폴백 6%)
            trail_stop = None
            df = self.client.ohlcv(p.ticker, "day", 60)
            if df is not None and not df.empty:
                try:
                    a = atr(df, 14).iloc[-1]
                    if a and a > 0:
                        trail_stop = new_high - 2.5 * float(a)
                except Exception:
                    trail_stop = None
            if trail_stop is None:
                trail_stop = new_high * 0.94  # 폴백
            trail_stop = max(trail_stop, p.avg_price)

            if px <= trail_stop:
                res = self.client.guarded_sell_market(p.ticker, p.qty)
                if res.ok:
                    self.repo.add_fill("SELL", p.ticker, px, p.qty, datetime.utcnow())
                    self.repo.remove_position(p.ticker)
                    try:
                        self.repo.set_last_exit(p.ticker, datetime.utcnow())
                    except Exception:
                        pass

        # 마크 기록
        for p in self.repo.list_positions():
            px = self.client.last_price(p.ticker) or p.avg_price
            self.repo.add_mark(p.ticker, px, datetime.utcnow())

        if not regime_ok:
            return "REGIME_OFF"
        if self.guard_active:
            return "GUARD_ON"
        if halted:
            return "HALT"
        return "OK"
