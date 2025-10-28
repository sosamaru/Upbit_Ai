from __future__ import annotations

import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Tuple

import pandas as pd
import pyupbit
import requests

from ..config import SETTINGS


logger = logging.getLogger("upbit_client")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


@dataclass
class OrderResult:
    ok: bool
    raw: Dict[str, Any]
    message: str = ""


class UpbitClient:
    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        dry_run: Optional[bool] = None,
        base_currency: Optional[str] = None,
        retry: int = 3,
        retry_sleep: float = 0.5,
    ):
        # 기본 환경
        self.base = (base_currency or SETTINGS.base_currency).upper()
        self.dry_run = SETTINGS.dry_run if dry_run is None else bool(dry_run)
        self.retry, self.retry_sleep = int(retry), float(retry_sleep)

        # API 핸들
        access = access_key if access_key is not None else SETTINGS.upbit_access
        secret = secret_key if secret_key is not None else SETTINGS.upbit_secret
        self.api = None if self.dry_run else pyupbit.Upbit(access, secret)

        # 캐시들
        self._krw_tickers_cache: Optional[List[str]] = None
        self._krw_tickers_cache_time: float = 0.0
        self._cache_ttl: float = 60.0

        self._price_cache: Dict[str, tuple[float, float]] = {}   # ticker -> (ts, price)
        self._ohlcv_cache: Dict[tuple, tuple[float, pd.DataFrame]] = {}  # (ticker, interval, count) -> (ts, df)

        # 캐시 TTL (환경변수로 튜닝 가능)
        self._price_ttl = float(os.getenv("PRICE_CACHE_TTL_SEC", "600" if self.dry_run else "10"))
        self._ohlcv_ttl = float(os.getenv("OHLCV_CACHE_TTL_SEC", "900"))

        # 24h 요약 캐시
        self._summary_cache: Optional[tuple[float, pd.DataFrame]] = None
        self._summary_ttl: float = float(os.getenv("SUMMARY_TTL_SEC", "120"))

        # 주문 가드 파라미터 (환경변수로 제어)
        self.max_slip_pct = float(os.getenv("MAX_MARKET_SLIPPAGE_PCT", "0.015"))  # 1.5%
        self.min_depth_krw = float(os.getenv("MIN_ORDERBOOK_DEPTH_KRW", "100000"))  # 10만원
        self.enable_order_guard = os.getenv("ENABLE_ORDER_GUARD", "1") != "0"
                # 스마트 분할 매수 설정 (없으면 기본 OFF)
        self.enable_smart_split = os.getenv("ENABLE_SMART_SPLIT", "0") == "1"
        self.split_chunks = int(os.getenv("SMART_SPLIT_CHUNKS", "3"))
        self.split_sleep_ms = int(os.getenv("SMART_SPLIT_DELAY_MS", "200"))

        # 리트라이 환경변수 반영
        self.retry = int(os.getenv("UPBIT_RETRY", self.retry))
        self.retry_sleep = float(os.getenv("UPBIT_RETRY_SLEEP", self.retry_sleep))


    def set_api_keys(self, access_key: str, secret_key: str) -> None:
        """텔레그램에서 키 교체 시 런타임 반영 (DRY_RUN은 세션 생성 안 함)"""
        if self.dry_run:
            self.api = None
            return
        try:
            self.api = pyupbit.Upbit(access_key, secret_key)
            logger.info("Upbit API keys updated at runtime.")
        except Exception as e:
            logger.error(f"set_api_keys failed: {e}")

    # ---------------------------
    # 공통 리트라이
    # ---------------------------
    def _retry(self, fn: Callable, *args, **kwargs):
        last = None
        for i in range(self.retry):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                logger.warning(f"retry {i+1}/{self.retry} for {fn.__name__}: {e}")
                time.sleep(self.retry_sleep * (1 + i))
        raise last if last else RuntimeError("retry failed")

    # ---------------------------
    # 시세/캔들/호가
    # ---------------------------
    def krw_tickers(self, force: bool = False) -> List[str]:
        now = time.time()
        if (not force) and self._krw_tickers_cache and (now - self._krw_tickers_cache_time < self._cache_ttl):
            return self._krw_tickers_cache
        tickers = self._retry(pyupbit.get_tickers, fiat=self.base) or []
        tickers = [t for t in tickers if t.startswith(f"{self.base}-")]
        self._krw_tickers_cache, self._krw_tickers_cache_time = tickers, now
        return tickers

    def ohlcv(self, ticker: str, interval: str = "day", count: int = 200) -> pd.DataFrame:
        try:
            now = time.time()
            key = (ticker, interval, int(count))
            hit = self._ohlcv_cache.get(key)
            if hit and (now - hit[0] < self._ohlcv_ttl):
                return hit[1].copy()
            df = self._retry(pyupbit.get_ohlcv, ticker, interval=interval, count=count)
            df = pd.DataFrame() if df is None or df.empty else df
            if not df.empty:
                self._ohlcv_cache[key] = (now, df)
            return df
        except Exception as e:
            logger.error(f"ohlcv failed: {ticker} {e}")
            return pd.DataFrame()

    def last_price(self, ticker: str) -> Optional[float]:
        try:
            now = time.time()
            hit = self._price_cache.get(ticker)
            if hit and (now - hit[0] < self._price_ttl):
                return hit[1]

            px = self._retry(pyupbit.get_current_price, ticker)
            if px is None or float(px) <= 0:
                raise RuntimeError(f"bad price: {px}")
            v = float(px)
            self._price_cache[ticker] = (now, v)
            return v
        except Exception as e:
            # 실패 시: 최근 일봉 종가로 안전 폴백
            try:
                df = self.ohlcv(ticker, "day", 1)
                if df is not None and not df.empty:
                    v = float(df["close"].iloc[-1])
                    self._price_cache[ticker] = (time.time(), v)
                    logger.warning(f"last_price fallback to OHLCV close for {ticker}: {e}")
                    return v
            except Exception:
                pass
            logger.error(f"last_price failed: {ticker} {e}")
            return None

    def market_summaries_24h(self) -> pd.DataFrame:
        """
        /v1/ticker 배치로 24h 요약 수집
        """
        try:
            now = time.time()
            hit = self._summary_cache
            if hit and (now - hit[0] < self._summary_ttl):
                return hit[1].copy()

            markets = pyupbit.get_tickers(fiat=self.base) or []
            if not markets:
                return pd.DataFrame()

            rows: List[Dict[str, float | str]] = []
            for i in range(0, len(markets), 100):
                chunk = markets[i : i + 100]
                url = "https://api.upbit.com/v1/ticker"
                resp = requests.get(url, params={"markets": ",".join(chunk)}, timeout=10)
                resp.raise_for_status()
                data = resp.json() or []
                for d in data:
                    rows.append(
                        {
                            "market": d.get("market"),
                            "trade_price": float(d.get("trade_price") or 0.0),
                            "acc_trade_price_24h": float(d.get("acc_trade_price_24h") or 0.0),
                            "signed_change_rate": float(d.get("signed_change_rate") or 0.0),
                        }
                    )

            df = pd.DataFrame(rows).dropna()
            if not df.empty:
                df = df.set_index("market").sort_index()
                self._summary_cache = (now, df)
            return df
        except Exception as e:
            logger.error(f"market_summaries_24h failed: {e}")
            return pd.DataFrame()

    def orderbook(self, ticker: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        반환: (bids, asks) 각 [(price, size), ...] 상위~15호가
        """
        try:
            ob = self._retry(pyupbit.get_orderbook, ticker)
            d = ob[0] if isinstance(ob, (list, tuple)) and ob else ob
            units = d.get("orderbook_units", []) if isinstance(d, dict) else []
            bids = [(float(u["bid_price"]), float(u["bid_size"])) for u in units]
            asks = [(float(u["ask_price"]), float(u["ask_size"])) for u in units]
            return bids, asks
        except Exception as e:
            logger.error(f"orderbook failed: {ticker} {e}")
            return [], []

    # ---- 호가 기반 VWAP/슬리피지 추정 ----
    def _simulate_buy_vwap(self, ticker: str, krw_amount: float) -> Tuple[Optional[float], Optional[float], float]:
        """
        반환: (vwap, slip_pct, depth_krw)
        slip_pct = (vwap/last - 1) ; last_price 기준
        """
        last = self.last_price(ticker) or 0.0
        if last <= 0 or krw_amount <= 0:
            return None, None, 0.0
        _, asks = self.orderbook(ticker)
        remain = krw_amount
        cost = 0.0
        qty = 0.0
        depth_krw = 0.0
        for px, sz in sorted(asks, key=lambda x: x[0]):  # 저가 → 고가
            lot_krw = px * sz
            take_krw = min(remain, lot_krw)
            take_qty = take_krw / px
            cost += take_krw
            qty += take_qty
            depth_krw += lot_krw
            remain -= take_krw
            if remain <= 1e-6:
                break
        if qty <= 0:
            return None, None, depth_krw
        vwap = cost / qty
        slip = (vwap / last) - 1.0 if last > 0 else None
        return vwap, slip, depth_krw

    def _simulate_sell_vwap(self, ticker: str, qty: float) -> Tuple[Optional[float], Optional[float], float]:
        """
        반환: (vwap, slip_pct, depth_krw) ; slip_pct = (1 - vwap/last)
        """
        last = self.last_price(ticker) or 0.0
        if last <= 0 or qty <= 0:
            return None, None, 0.0
        bids, _ = self.orderbook(ticker)
        remain = qty
        proceeds = 0.0
        filled = 0.0
        depth_krw = 0.0
        for px, sz in sorted(bids, key=lambda x: x[0], reverse=True):  # 고가 → 저가
            take = min(remain, sz)
            proceeds += take * px
            filled += take
            depth_krw += px * sz
            remain -= take
            if remain <= 1e-9:
                break
        if filled <= 0:
            return None, None, depth_krw
        vwap = proceeds / filled
        slip = (1.0 - (vwap / last)) if last > 0 else None
        return vwap, slip, depth_krw

    # ---------------------------
    # 잔고/주문
    # ---------------------------
    def balances(self) -> List[Dict[str, Any]]:
        if self.dry_run:
            return []
        try:
            bals = self._retry(self.api.get_balances)
            if not isinstance(bals, (list, tuple)):
                logger.error(f"unexpected balances type: {type(bals)} {bals}")
                return []
            return list(bals) or []
        except Exception as e:
            logger.error(f"balances failed: {e}")
            return []

    def krw_balance(self) -> float:
        if self.dry_run:
            return 0.0
        total = 0.0
        for b in self.balances():
            if isinstance(b, dict) and b.get("currency") == "KRW":
                try:
                    total += float(b.get("balance", 0)) + float(b.get("locked", 0))
                except Exception:
                    pass
        return total

    def coin_balance(self, ticker: str) -> float:
        if self.dry_run:
            return 0.0
        cc = ticker.split("-")[-1].upper()
        for b in self.balances():
            if isinstance(b, dict) and (b.get("currency", "").upper() == cc):
                try:
                    return float(b.get("balance", 0))
                except Exception:
                    return 0.0
        return 0.0

    def krw_available(self) -> float:
        """주문가능 KRW(locked 제외)"""
        if self.dry_run:
            return 1_000_000.0
        avail = 0.0
        for b in self.balances():
            if isinstance(b, dict) and b.get("currency") == "KRW":
                try:
                    avail = float(b.get("balance", 0))
                except Exception:
                    pass
        return avail

    def coin_free(self, ticker: str) -> float:
        """특정 코인의 주문가능 수량(locked 제외)"""
        if self.dry_run:
            return 0.0
        cc = ticker.split("-")[-1].upper()
        for b in self.balances():
            if isinstance(b, dict) and b.get("currency", "").upper() == cc:
                try:
                    return float(b.get("balance", 0))
                except Exception:
                    return 0.0
        return 0.0

    # ---- 안전 주문 래퍼 ----
    def clamp_min_order(self, krw_amount: float) -> float:
        min_amt = float(SETTINGS.min_order_krw)
        if math.isnan(krw_amount) or krw_amount <= 0:
            return 0.0
        return max(min_amt, float(krw_amount))

    def safe_buy_market(self, ticker: str, krw_amount: float) -> OrderResult:
        """수수료/슬리피지 여유 0.1% 반영 + 최소주문 보정"""
        fee_buf = 0.001
        amt = self.clamp_min_order(krw_amount * (1.0 - fee_buf))
        if amt <= 0:
            return OrderResult(False, {}, "amount <= 0 after clamp")
        if self.dry_run:
            logger.info(f"[DRY] BUY {ticker} KRW {amt}")
            return OrderResult(True, {"dry": True, "action": "buy_market", "ticker": ticker, "krw": amt})
        try:
            res = self._retry(self.api.buy_market_order, ticker, amt)
            ok = isinstance(res, dict) and ("uuid" in res or "error" not in res)
            return OrderResult(bool(ok), res, "" if ok else "order error")
        except Exception as e:
            logger.error(f"buy_market failed: {ticker} {amt} {e}")
            return OrderResult(False, {}, str(e))

    def _safe_sellable(self, ticker: str, requested_qty: float) -> float:
        api_free = self.coin_free(ticker)  # locked 제외 가능한 수량
        base = api_free if api_free > 0 else requested_qty
        qty = base * 0.999  # 수수료/잔량 버퍼 0.1%
        qty = int(qty * 1_000_000) / 1_000_000  # 6자리 내림
        return max(0.0, qty)

    def sell_market(self, ticker: str, volume: float) -> OrderResult:
        volume = self._safe_sellable(ticker, volume)
        if volume <= 0:
            return OrderResult(False, {}, "no sellable volume")
        if self.dry_run:
            logger.info(f"[DRY] SELL {ticker} VOL {volume}")
            return OrderResult(True, {"dry": True, "action": "sell_market", "ticker": ticker, "vol": volume})
        try:
            res = self._retry(self.api.sell_market_order, ticker, volume)
            ok = isinstance(res, dict) and ("uuid" in res or "error" not in res)
            return OrderResult(bool(ok), res, "" if ok else "order error")
        except Exception as e:
            logger.error(f"sell_market failed: {ticker} {volume} {e}")
            return OrderResult(False, {}, str(e))

    # ---- 주문 가드 포함 버전 (호가 기반 슬리피지/깊이 체크) ----
    def guarded_buy_market(self, ticker: str, krw_amount: float,
                           max_slip_pct: Optional[float] = None,
                           min_depth_krw: Optional[float] = None) -> OrderResult:
        if not self.enable_order_guard:
            return self.safe_buy_market(ticker, krw_amount)

        mx = self.max_slip_pct if max_slip_pct is None else float(max_slip_pct)
        md = self.min_depth_krw if min_depth_krw is None else float(min_depth_krw)

        vwap, slip, depth = self._simulate_buy_vwap(ticker, krw_amount)
        if vwap is None or slip is None:
            return OrderResult(False, {}, "orderbook unavailable")

        if depth < md:
            return OrderResult(False, {}, f"depth < {int(md):,} KRW")

        if slip > mx:
            return OrderResult(False, {}, f"slippage {slip:.3%} > limit {mx:.3%}")

        return self.safe_buy_market(ticker, krw_amount)
    
    def smart_guarded_buy_market(self, ticker: str, krw_amount: float) -> OrderResult:
        """
        시장가 매수를 분할 실행하면서 기존 가드(슬리피지/깊이)를 그대로 적용.
        ENABLE_SMART_SPLIT=1 && SMART_SPLIT_CHUNKS>1 일 때만 동작.
        """
        if (not self.enable_smart_split) or self.split_chunks <= 1:
            return self.guarded_buy_market(ticker, krw_amount)

        # 분할 금액
        chunk = max(self.clamp_min_order(krw_amount / self.split_chunks), float(SETTINGS.min_order_krw))
        spent, ok_cnt, last = 0.0, 0, None
        for i in range(self.split_chunks):
            remain = max(0.0, krw_amount - spent)
            part = chunk if remain > chunk else remain
            if part < float(SETTINGS.min_order_krw):
                break
            res = self.guarded_buy_market(ticker, part)
            last = res
            if not res.ok:
                logger.info(f"[smart] split {i+1}/{self.split_chunks} blocked: {res.message}")
                break
            ok_cnt += 1
            spent += part
            time.sleep(self.split_sleep_ms / 1000.0)
        if ok_cnt <= 0 and last:
            return last
        return OrderResult(True, {"splits": ok_cnt, "spent": spent}, "")

    def guarded_sell_market(self, ticker: str, volume: float,
                            max_slip_pct: Optional[float] = None,
                            min_depth_krw: Optional[float] = None) -> OrderResult:
        if not self.enable_order_guard:
            return self.sell_market(ticker, volume)

        mx = self.max_slip_pct if max_slip_pct is None else float(max_slip_pct)
        md = self.min_depth_krw if min_depth_krw is None else float(min_depth_krw)

        vwap, slip, depth = self._simulate_sell_vwap(ticker, volume)
        if vwap is None or slip is None:
            return OrderResult(False, {}, "orderbook unavailable")

        if depth < md:
            return OrderResult(False, {}, f"depth < {int(md):,} KRW")

        if slip > mx:
            return OrderResult(False, {}, f"slippage {slip:.3%} > limit {mx:.3%}")

        return self.sell_market(ticker, volume)
