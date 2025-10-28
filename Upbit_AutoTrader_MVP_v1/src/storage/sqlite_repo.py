from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
import os

from sqlalchemy import (
    create_engine, Integer, Float, String, DateTime, LargeBinary, select, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


# ---------- ORM 스키마 ----------
class Base(DeclarativeBase):
    pass


class Fill(Base):
    __tablename__ = "fills"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY/SELL
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    price: Mapped[float] = mapped_column(Float)
    qty: Mapped[float] = mapped_column(Float)
    __table_args__ = (Index("ix_fills_ticker_ts", "ticker", "ts"),)


class Mark(Base):
    __tablename__ = "marks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    price: Mapped[float] = mapped_column(Float)
    __table_args__ = (Index("ix_marks_ticker_ts", "ticker", "ts"),)


class Position(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    avg_price: Mapped[float] = mapped_column(Float, default=0.0)
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    highest: Mapped[float] = mapped_column(Float, default=0.0)
    partial_taken: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    last_exit_ts: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Blacklist(Base):
    __tablename__ = "blacklist"
    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class Cooldown(Base):
    __tablename__ = "cooldowns"
    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    until: Mapped[datetime] = mapped_column(DateTime, index=True)
    reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


# ---------- Repo ----------
@dataclass
class RepoConfig:
    url: str = "sqlite:///data/trader.sqlite3"


@dataclass
class PositionView:
    ticker: str
    avg_price: float
    qty: float
    highest: float
    partial_taken: bool = False
    last_exit_ts: Optional[datetime] = None


class SqliteRepo:
    def __init__(self, cfg: RepoConfig | None = None):
        self.cfg = cfg or RepoConfig(url=os.getenv("SQLITE_URL", "sqlite:///data/trader.sqlite3"))
        if self.cfg.url.startswith("sqlite:///"):
            path = self.cfg.url.replace("sqlite:///", "", 1)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.engine = create_engine(self.cfg.url, echo=False, future=True)
        self._ensure_schema()

    def _ensure_schema(self):
        Base.metadata.create_all(self.engine)
        
        # 간단 마이그레이션: 누락 컬럼 보강
        with self.engine.connect() as conn:
            def _cols(table: str) -> set[str]:
                rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
                return {row[1] for row in rows}  # row[1] = column name
        # positions: 신규 컬럼 보장
            pcols = _cols("positions")
            if "partial_taken" not in pcols:
                try:
                    conn.exec_driver_sql("ALTER TABLE positions ADD COLUMN partial_taken INTEGER DEFAULT 0")
                except Exception:
                    pass
            if "last_exit_ts" not in pcols:
                try:
                    conn.exec_driver_sql("ALTER TABLE positions ADD COLUMN last_exit_ts DATETIME NULL")
                except Exception:
                    pass

            # fills: qty 컬럼 보장 (핵심 패치)
            fcols = _cols("fills")
            if "qty" not in fcols:
                try:
                    conn.exec_driver_sql("ALTER TABLE fills ADD COLUMN qty REAL DEFAULT 0.0")
                except Exception:
                    pass

            conn.commit()


    # ----- Position ops -----
    def upsert_position(
        self, ticker: str, *, avg_price: float, qty: float, highest: float,
        partial_taken: Optional[bool] = None, last_exit_ts: Optional[datetime] = None
    ) -> None:
        with Session(self.engine) as s:
            p = s.scalar(select(Position).where(Position.ticker == ticker))
            if p is None:
                p = Position(
                    ticker=ticker, avg_price=float(avg_price), qty=float(qty),
                    highest=float(highest),
                    partial_taken=1 if (partial_taken or False) else 0,
                    last_exit_ts=last_exit_ts
                )
                s.add(p)
            else:
                p.avg_price = float(avg_price)
                p.qty = float(qty)
                p.highest = float(highest)
                if partial_taken is not None:
                    p.partial_taken = 1 if partial_taken else 0
                if last_exit_ts is not None:
                    p.last_exit_ts = last_exit_ts
            s.commit()

    def remove_position(self, ticker: str) -> None:
        with Session(self.engine) as s:
            p = s.scalar(select(Position).where(Position.ticker == ticker))
            if p is not None:
                s.delete(p)
                s.commit()

    def get_position(self, ticker: str) -> Optional[PositionView]:
        with Session(self.engine) as s:
            p = s.scalar(select(Position).where(Position.ticker == ticker))
            if p is None:
                return None
            return PositionView(
                ticker=p.ticker,
                avg_price=float(p.avg_price),
                qty=float(p.qty),
                highest=float(p.highest),
                partial_taken=bool(p.partial_taken),
                last_exit_ts=p.last_exit_ts
            )

    def list_positions(self) -> List[PositionView]:
        with Session(self.engine) as s:
            rows = s.scalars(select(Position).order_by(Position.ticker)).all()
            return [
                PositionView(
                    ticker=r.ticker, avg_price=float(r.avg_price), qty=float(r.qty),
                    highest=float(r.highest), partial_taken=bool(r.partial_taken),
                    last_exit_ts=r.last_exit_ts
                )
                for r in rows
            ]

    # ----- Flags -----
    def get_partial_taken(self, ticker: str) -> bool:
        with Session(self.engine) as s:
            v = s.scalar(select(Position.partial_taken).where(Position.ticker == ticker))
            return bool(v or 0)

    def set_partial_taken(self, ticker: str, value: bool) -> None:
        with Session(self.engine) as s:
            p = s.scalar(select(Position).where(Position.ticker == ticker))
            if p is None:
                return
            p.partial_taken = 1 if value else 0
            s.commit()

    def get_last_exit(self, ticker: str) -> Optional[datetime]:
        with Session(self.engine) as s:
            return s.scalar(select(Position.last_exit_ts).where(Position.ticker == ticker))

    def set_last_exit(self, ticker: str, ts: datetime) -> None:
        with Session(self.engine) as s:
            p = s.scalar(select(Position).where(Position.ticker == ticker))
            if p is None:
                return
            p.last_exit_ts = ts
            s.commit()

    # ----- Blacklist / Cooldown -----
    def ban_ticker(self, ticker: str, reason: Optional[str] = None) -> None:
        with Session(self.engine) as s:
            if s.scalar(select(Blacklist).where(Blacklist.ticker == ticker)) is None:
                s.add(Blacklist(ticker=ticker, reason=reason, created_at=datetime.utcnow()))
            s.commit()

    def unban_ticker(self, ticker: str) -> None:
        with Session(self.engine) as s:
            row = s.scalar(select(Blacklist).where(Blacklist.ticker == ticker))
            if row:
                s.delete(row)
                s.commit()

    def list_blacklist(self) -> List[str]:
        with Session(self.engine) as s:
            return [r.ticker for r in s.scalars(select(Blacklist)).all()]

    def set_cooldown(self, ticker: str, until: datetime, reason: Optional[str] = None) -> None:
        with Session(self.engine) as s:
            row = s.scalar(select(Cooldown).where(Cooldown.ticker == ticker))
            if row is None:
                s.add(Cooldown(ticker=ticker, until=until, reason=reason))
            else:
                row.until = until
                row.reason = reason
            s.commit()

    def get_cooldown(self, ticker: str) -> Optional[datetime]:
        with Session(self.engine) as s:
            row = s.scalar(select(Cooldown).where(Cooldown.ticker == ticker))
            return row.until if row else None

    # ----- Fills / Marks -----
    def add_fill(self, side: str, ticker: str, price: float, qty: float, ts: datetime) -> None:
        with Session(self.engine) as s:
            s.add(Fill(ts=ts, side=side, ticker=ticker, price=float(price), qty=float(qty)))
            s.commit()

    def add_mark(self, ticker: str, price: float, ts: datetime) -> None:
        with Session(self.engine) as s:
            s.add(Mark(ts=ts, ticker=ticker, price=float(price)))
            s.commit()

    def list_fills(
        self, ticker: Optional[str] = None,
        since: Optional[datetime] = None, until: Optional[datetime] = None
    ) -> List[Fill]:
        with Session(self.engine) as s:
            stmt = select(Fill)
            if ticker:
                stmt = stmt.where(Fill.ticker == ticker)
            if since:
                stmt = stmt.where(Fill.ts >= since)
            if until:
                stmt = stmt.where(Fill.ts < until)
            stmt = stmt.order_by(Fill.ts.asc())
            return s.scalars(stmt).all()

    def list_marks(
        self, ticker: Optional[str] = None,
        since: Optional[datetime] = None, until: Optional[datetime] = None
    ) -> List[Mark]:
        with Session(self.engine) as s:
            stmt = select(Mark)
            if ticker:
                stmt = stmt.where(Mark.ticker == ticker)
            if since:
                stmt = stmt.where(Mark.ts >= since)
            if until:
                stmt = stmt.where(Mark.ts < until)
            stmt = stmt.order_by(Mark.ts.asc())
            return s.scalars(stmt).all()

    # ----- Realized PnL (간단 FIFO 근사) -----
    def realized_pnl(self, since: Optional[datetime] = None, until: Optional[datetime] = None) -> float:
        realized = 0.0
        stack: List[Tuple[float, float]] = []  # (price, qty)
        with Session(self.engine) as s:
            stmt = select(Fill).order_by(Fill.ts.asc())
            if since:
                stmt = stmt.where(Fill.ts >= since)
            if until:
                stmt = stmt.where(Fill.ts < until)
            for f in s.scalars(stmt).all():
                px = float(f.price); q = float(f.qty)
                if f.side.upper() == "BUY":
                    stack.append((px, q))
                elif f.side.upper() == "SELL":
                    remain = q; i = 0
                    while remain > 0 and i < len(stack):
                        bpx, bq = stack[i]
                        take = min(remain, bq)
                        realized += (px - bpx) * take
                        bq -= take; remain -= take
                        if bq <= 1e-12:
                            stack.pop(i)
                        else:
                            stack[i] = (bpx, bq)
                            i += 1
        return float(realized)

    # ----- Turnover(회전액) 집계 -----
    def turnover_krw(
        self, since: Optional[datetime] = None, until: Optional[datetime] = None, side: Optional[str] = None
    ) -> float:
        total = 0.0
        with Session(self.engine) as s:
            stmt = select(Fill)
            if since:
                stmt = stmt.where(Fill.ts >= since)
            if until:
                stmt = stmt.where(Fill.ts < until)
            if side:
                stmt = stmt.where(Fill.side == side.upper())
            for f in s.scalars(stmt).all():
                total += float(f.price) * float(f.qty)
        return float(total)
