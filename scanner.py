from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TZ = ZoneInfo("Asia/Kolkata")
LOG = logging.getLogger("bis20")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "runtime")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "bis20.sqlite3"
UNIVERSE_CACHE = Path(os.getenv("UNIVERSE_CACHE", str(ROOT / "data" / "liquid_universe.json")))
UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
KRONOS_REPO_DIR = Path(os.getenv("KRONOS_REPO_DIR", str(ROOT / "vendor" / "Kronos")))

NSE_EQUITY_CSV = os.getenv(
    "NSE_EQUITY_CSV",
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
)

# Calibrated for automatic continuous 5-minute scanning and reliable 1-2 daily trades
# from NSE's Top Intraday Movers.
CFG: Dict[str, Any] = {
    "PRICE_FLOOR": float(os.getenv("PRICE_FLOOR", "50")),
    "MIN_AVG_TURNOVER_INR": float(os.getenv("MIN_AVG_TURNOVER_INR", "50000000")),  # ₹5 Cr floor
    "LIQUIDITY_LOOKBACK_DAYS": 20,
    "DAILY_HISTORY_PERIOD": os.getenv("DAILY_HISTORY_PERIOD", "3mo"),
    "MAX_EQUITY_UNIVERSE": int(os.getenv("MAX_EQUITY_UNIVERSE", "2500")),
    "INTRADAY_UNIVERSE_SIZE": int(os.getenv("INTRADAY_UNIVERSE_SIZE", "200")),
    "INTRADAY_BATCH_SIZE": int(os.getenv("INTRADAY_BATCH_SIZE", "50")),
    "INTRADAY_PERIOD": "10d",
    "BAR_INTERVAL": "5m",
    "MIN_HISTORY_BARS": 45,
    "MIN_TIME_SLOT_SESSIONS": 3,
    "MIN_RELATIVE_VOLUME": 1.30,        # Same-slot 5m RVOL floor when paired with cumulative volume
    "MIN_CUMULATIVE_RVOL": 1.35,        # Cumulative session RVOL floor
    "BREAKOUT_LOOKBACK": 10,            # 10-bar (50-min) consolidation lookback
    "MAX_EXTENSION_ATR": 2.4,           # Allow strong momentum candles without chasing extreme spikes
    "MIN_DAY_MOVE_PCT": float(os.getenv("MIN_DAY_MOVE_PCT", "2.8")),   # Focus on stocks moving 2.8% to 9.8%
    "MAX_DAY_MOVE_PCT": float(os.getenv("MAX_DAY_MOVE_PCT", "9.8")),   # Before 10% circuit cap
    "MAX_BAR_AGE_MIN": 14,
    "MIN_RSI_BUY": 55.0,
    "MAX_RSI_BUY": 92.0,
    "MIN_RSI_SELL": 8.0,
    "MAX_RSI_SELL": 45.0,
    "MIN_ADX_TREND": 22.0,              # Directional trend strength
    "MIN_BODY_RATIO": 0.30,             # Clean candle body (was 0.45, which rejected valid continuation bars)
    "MIN_CLOSE_LOCATION_BUY": 0.60,     # Close in top 40% of 5m candle range
    "MAX_CLOSE_LOCATION_SELL": 0.40,    # Close in bottom 40% of 5m candle range
    "STOP_ATR_MULTIPLE": 1.50,
    "MIN_STOP_PCT": 0.0075,             # Minimum 0.75% stop distance
    "MAX_STOP_PCT": 0.0250,             # Maximum 2.50% stop distance
    "TARGET_R_MULTIPLE": 2.2,           # Primary target = 2.2R
    "RUNNER_R_MULTIPLE": 3.5,           # Runner target = 3.5R for 10%+ intraday movers
    "MIN_SCORE_TIER1": int(os.getenv("MIN_SCORE_TIER1", "76")),
    "MIN_SCORE_TIER2": int(os.getenv("MIN_SCORE_TIER2", "68")),
    "TOP_MOVERS_POOL_SIZE": int(os.getenv("TOP_MOVERS_POOL_SIZE", "10")),
    "MORNING_GUARD_HOUR": 9,            # Before 09:45 IST, allow at most 1 trade so slot #2 stays open for 09:45+
    "MORNING_GUARD_MINUTE": 45,
    "TIER2_START_HOUR": 10,             # Activate Tier-2 Top-Mover Continuation from 10:00 IST if <2 trades found
    "TIER2_START_MINUTE": 0,
    "LAST_ENTRY_HOUR": 14,              # No new intraday entries after 14:30 IST
    "LAST_ENTRY_MINUTE": 30,
    "MAX_SIGNALS_PER_CYCLE": int(os.getenv("MAX_SIGNALS_PER_CYCLE", "1")),
    "MAX_SIGNALS_PER_DAY": int(os.getenv("MAX_SIGNALS_PER_DAY", "2")),
    "CONTINUOUS_LOOP_MINUTES": int(os.getenv("CONTINUOUS_LOOP_MINUTES", "50")),  # Auto-loop every 5m for 50 mins per run
    "KRONOS_ENABLED": os.getenv("KRONOS_ENABLED", "true").lower() == "true",
    "KRONOS_REQUIRED": os.getenv("KRONOS_REQUIRED", "false").lower() == "true",
    "KRONOS_MODEL": os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-mini"),
    "KRONOS_TOKENIZER": os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-2k"),
    "KRONOS_LOOKBACK": int(os.getenv("KRONOS_LOOKBACK", "96")),
    "KRONOS_PRED_LEN": int(os.getenv("KRONOS_PRED_LEN", "3")),
    "KRONOS_SAMPLE_COUNT": int(os.getenv("KRONOS_SAMPLE_COUNT", "3")),
    "KRONOS_MIN_MOVE_PCT": float(os.getenv("KRONOS_MIN_MOVE_PCT", "0.05")),
    "UNIVERSE_MAX_AGE_HOURS": int(os.getenv("UNIVERSE_MAX_AGE_HOURS", "168")),
}

kronos_predictor = None
kronos_status = "not_loaded"


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            payload TEXT NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE,
            delivered INTEGER NOT NULL DEFAULT 0
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        conn.commit()


def persist_state_to_git() -> None:
    """Commit and push SQLite state immediately after delivering an alert inside a continuous loop."""
    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
        subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", "runtime/bis20.sqlite3", "data/liquid_universe.json"], check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", "BIS 2.0: persist scanner state [skip ci]"], check=False)
            subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
            subprocess.run(["git", "push", "origin", "HEAD:main"], check=False)
            LOG.info("Persisted updated SQLite state to GitHub main branch.")
    except Exception as exc:
        LOG.warning("Git state persistence warning: %s", exc)


def get_today_signal_count(day_str: Optional[str] = None) -> int:
    day = day_str or datetime.now(TZ).strftime("%Y-%m-%d")
    with db_connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM signals WHERE created_at LIKE ?", (day + "%",)).fetchone()
        return int(row["n"]) if row else 0


def get_today_alerted_symbols(day_str: Optional[str] = None) -> set:
    day = day_str or datetime.now(TZ).strftime("%Y-%m-%d")
    with db_connect() as conn:
        rows = conn.execute("SELECT symbol FROM signals WHERE created_at LIKE ?", (day + "%",)).fetchall()
        return {r["symbol"] for r in rows}


def load_kronos() -> None:
    """Load Kronos-mini model on CPU if available; degrade gracefully if KRONOS_REQUIRED=false."""
    global kronos_predictor, kronos_status
    if kronos_predictor is not None:
        return
    if not CFG["KRONOS_ENABLED"]:
        kronos_status = "disabled"
        if CFG["KRONOS_REQUIRED"]:
            raise RuntimeError("KRONOS_REQUIRED=true but KRONOS_ENABLED=false")
        return
    if not KRONOS_REPO_DIR.exists():
        kronos_status = f"missing_repo:{KRONOS_REPO_DIR}"
        LOG.warning("Kronos repository not found at %s; continuing in technical-only mode", KRONOS_REPO_DIR)
        if CFG["KRONOS_REQUIRED"]:
            raise RuntimeError(f"Kronos source repository not found: {KRONOS_REPO_DIR}")
        return
    sys.path.insert(0, str(KRONOS_REPO_DIR))
    try:
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        tokenizer = KronosTokenizer.from_pretrained(CFG["KRONOS_TOKENIZER"])
        model = Kronos.from_pretrained(CFG["KRONOS_MODEL"])
        kronos_predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=2048)
        kronos_status = "ready:cpu"
        LOG.info("Kronos-mini loaded on CPU")
    except Exception as exc:
        kronos_predictor = None
        kronos_status = "error:" + str(exc)[:240]
        LOG.warning("Kronos initialization failed (advisory mode=%s): %s", not CFG["KRONOS_REQUIRED"], exc)
        if CFG["KRONOS_REQUIRED"]:
            raise RuntimeError("Kronos is required but failed to load: " + str(exc)) from exc


def is_market_session(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 20) <= t <= dtime(15, 30)


def is_pre_market_wait_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(8, 30) <= t < dtime(9, 20)


def is_entry_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if not is_market_session(now):
        return False
    return now.time() <= dtime(CFG["LAST_ENTRY_HOUR"], CFG["LAST_ENTRY_MINUTE"])


def is_tier2_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if not is_entry_window(now):
        return False
    return now.time() >= dtime(CFG["TIER2_START_HOUR"], CFG["TIER2_START_MINUTE"])


def fetch_nse_equity_symbols() -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    resp = requests.get(NSE_EQUITY_CSV, headers=headers, timeout=25)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    cols = {str(c).strip().upper(): c for c in df.columns}
    sym_col = cols.get("SYMBOL")
    series_col = cols.get("SERIES")
    if not sym_col:
        raise ValueError("NSE equity CSV does not contain a SYMBOL column")
    if series_col:
        df = df[df[series_col].astype(str).str.upper().str.strip().eq("EQ")]
    symbols = (
        df[sym_col].astype(str).str.upper().str.strip()
        .loc[lambda s: s.str.fullmatch(r"[A-Z0-9&-]{1,30}")]
        .drop_duplicates().tolist()
    )
    symbols = [s for s in symbols if not s.endswith("-BE") and not s.endswith("-SM")]
    return symbols[:CFG["MAX_EQUITY_UNIVERSE"]]


def extract_ticker_frame(downloaded: pd.DataFrame, ticker: str, single: bool = False) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        for level in (0, 1):
            try:
                if ticker in downloaded.columns.get_level_values(level):
                    frame = downloaded.xs(ticker, axis=1, level=level, drop_level=True)
                    return frame.dropna(how="all")
            except Exception:
                pass
        return pd.DataFrame()
    if single:
        return downloaded.dropna(how="all")
    return pd.DataFrame()


def download_batch(symbols: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    tickers = [s + ".NS" for s in symbols]
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
            timeout=25,
        )
    except Exception as exc:
        LOG.warning("Yahoo batch failed (%s): %s", len(symbols), exc)
        return {s: pd.DataFrame() for s in symbols}
    out: Dict[str, pd.DataFrame] = {}
    for s, ticker in zip(symbols, tickers):
        frame = extract_ticker_frame(raw, ticker, single=(len(tickers) == 1))
        if not frame.empty:
            frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
            expected = ["open", "high", "low", "close", "volume"]
            if all(c in frame.columns for c in expected):
                frame = frame[expected].copy().dropna(subset=["open", "high", "low", "close"])
            else:
                frame = pd.DataFrame()
        out[s] = frame
    return out


def refresh_liquid_universe(force: bool = False) -> Dict[str, Any]:
    """Build candidate pool from NSE EQ symbols using 20-session turnover and volatility."""
    if not force and UNIVERSE_CACHE.exists():
        try:
            cached = json.loads(UNIVERSE_CACHE.read_text())
            refreshed = datetime.fromisoformat(cached["refreshed_at"])
            if datetime.now(TZ) - refreshed < timedelta(hours=CFG["UNIVERSE_MAX_AGE_HOURS"]):
                return cached
        except Exception:
            pass

    symbols = fetch_nse_equity_symbols()
    LOG.info("NSE main-board EQ symbols loaded: %d", len(symbols))
    metrics: List[Dict[str, Any]] = []
    batch_size = 60
    for start in range(0, len(symbols), batch_size):
        group = symbols[start:start + batch_size]
        data = download_batch(group, CFG["DAILY_HISTORY_PERIOD"], "1d")
        for symbol in group:
            df = data.get(symbol, pd.DataFrame())
            if df.empty or "close" not in df or "volume" not in df:
                continue
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert(TZ).tz_localize(None)
            keep = [x.date() < datetime.now(TZ).date() for x in idx]
            hist = df.loc[keep].copy()
            close = pd.to_numeric(hist["close"], errors="coerce").dropna()
            high = pd.to_numeric(hist["high"], errors="coerce").reindex(close.index).fillna(close)
            low = pd.to_numeric(hist["low"], errors="coerce").reindex(close.index).fillna(close)
            volume = pd.to_numeric(hist["volume"], errors="coerce").reindex(close.index).fillna(0)
            if len(close) < 50:
                continue
            turnover = (close * volume).replace([np.inf, -np.inf], np.nan).dropna()
            if len(turnover) < CFG["LIQUIDITY_LOOKBACK_DAYS"]:
                continue
            avg_turnover = float(turnover.tail(CFG["LIQUIDITY_LOOKBACK_DAYS"]).mean())
            last_price = float(close.iloc[-1])
            e20 = float(ema(close.values, 20))
            e50 = float(ema(close.values, 50))
            prev_close = float(close.iloc[-1])
            daily_range_pct = float(((high - low) / close.clip(lower=1e-6) * 100.0).tail(20).mean())
            if last_price >= CFG["PRICE_FLOOR"] and avg_turnover >= CFG["MIN_AVG_TURNOVER_INR"]:
                metrics.append({
                    "symbol": symbol,
                    "price": last_price,
                    "avg_turnover": avg_turnover,
                    "daily_ema20": e20,
                    "daily_ema50": e50,
                    "previous_close": prev_close,
                    "avg_daily_range_pct": round(daily_range_pct, 2),
                })
        LOG.info("Liquidity refresh progress %d/%d | qualified=%d", min(start + batch_size, len(symbols)), len(symbols), len(metrics))
        time.sleep(0.15)

    metrics.sort(key=lambda x: x["avg_turnover"], reverse=True)
    if not metrics:
        if UNIVERSE_CACHE.exists():
            old = json.loads(UNIVERSE_CACHE.read_text())
            old["refresh_warning"] = "Refresh returned no eligible stocks; retained previous cache"
            return old
        raise RuntimeError("No liquid universe was created. Check NSE/Yahoo data access.")

    top_liquid = metrics[:int(CFG["INTRADAY_UNIVERSE_SIZE"] * 0.6)]
    top_liquid_set = {m["symbol"] for m in top_liquid}
    remaining = [m for m in metrics[int(CFG["INTRADAY_UNIVERSE_SIZE"] * 0.6):450] if m["symbol"] not in top_liquid_set]
    remaining.sort(key=lambda x: (x.get("avg_daily_range_pct", 0.0), x["avg_turnover"]), reverse=True)
    combined = (top_liquid + remaining)[:CFG["INTRADAY_UNIVERSE_SIZE"]]

    result = {
        "refreshed_at": datetime.now(TZ).isoformat(),
        "source_symbol_count": len(symbols),
        "eligible_count": len(metrics),
        "scan_symbols": combined,
    }
    UNIVERSE_CACHE.write_text(json.dumps(result, indent=2))
    LOG.info("Liquid universe refreshed: %d eligible; scanning %d liquid & high-beta stocks", len(metrics), len(result["scan_symbols"]))
    return result


def ema(values: Any, period: int) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < period:
        return float("nan")
    alpha = 2.0 / (period + 1.0)
    result = float(np.mean(arr[:period]))
    for x in arr[period:]:
        result = alpha * float(x) + (1.0 - alpha) * result
    return result


def rsi(values: Any, period: int = 14) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= period:
        return float("nan")
    delta = np.diff(arr)
    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    value = tr.tail(period).mean()
    return float(value) if pd.notna(value) else float("nan")


def adx_di(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float]:
    """Calculate Wilder's ADX, +DI, and -DI on recent bars."""
    tail_df = df.tail(max(period * 4, 60))
    if len(tail_df) < period * 2:
        return 20.0, 20.0, 20.0
    high = tail_df["high"].astype(float).values
    low = tail_df["low"].astype(float).values
    close = tail_df["close"].astype(float).values

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])

    atr_smooth = float(np.mean(tr[:period]))
    plus_smooth = float(np.mean(plus_dm[:period]))
    minus_smooth = float(np.mean(minus_dm[:period]))

    dx_list: List[float] = []
    plus_di = 0.0
    minus_di = 0.0
    for i in range(period, len(tr)):
        atr_smooth = (atr_smooth * (period - 1) + float(tr[i])) / period
        plus_smooth = (plus_smooth * (period - 1) + float(plus_dm[i])) / period
        minus_smooth = (minus_smooth * (period - 1) + float(minus_dm[i])) / period
        if atr_smooth <= 1e-9:
            continue
        plus_di = 100.0 * (plus_smooth / atr_smooth)
        minus_di = 100.0 * (minus_smooth / atr_smooth)
        denom = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / denom if denom > 1e-9 else 0.0
        dx_list.append(dx)

    if not dx_list:
        return 20.0, 20.0, 20.0
    adx_val = float(np.mean(dx_list[-period:]))
    return adx_val, plus_di, minus_di


def normalize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    else:
        idx = idx.tz_convert(TZ)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def completed_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    cutoff = pd.Timestamp(now - timedelta(minutes=5)).tz_convert(TZ)
    return df.loc[df.index <= cutoff].copy()


def same_slot_relative_volume(df: pd.DataFrame, last_idx: pd.Timestamp, min_sessions: int) -> Tuple[float, int]:
    slot = last_idx.strftime("%H:%M")
    today = last_idx.date()
    same_slot = df[df.index.strftime("%H:%M") == slot]
    previous = same_slot[[x.date() < today for x in same_slot.index]].tail(5)
    vals = pd.to_numeric(previous["volume"], errors="coerce").dropna()
    vals = vals[vals > 0]
    current = float(df.loc[last_idx, "volume"])
    if len(vals) < min_sessions or float(vals.mean()) <= 0:
        return 0.0, int(len(vals))
    return current / float(vals.mean()), int(len(vals))


def session_cumulative_relative_volume(df: pd.DataFrame, last_idx: pd.Timestamp, min_sessions: int) -> Tuple[float, int]:
    """Compare today's cumulative volume up to this 5m slot with prior sessions."""
    today = last_idx.date()
    slot = last_idx.strftime("%H:%M")
    current = df[[x.date() == today and x.strftime("%H:%M") <= slot for x in df.index]]
    current_total = float(current["volume"].astype(float).sum())
    prior_totals: List[float] = []
    prior_dates = sorted({x.date() for x in df.index if x.date() < today})[-5:]
    for session_date in prior_dates:
        session = df[[x.date() == session_date and x.strftime("%H:%M") <= slot for x in df.index]]
        if not session.empty:
            vol_sum = float(session["volume"].astype(float).sum())
            if vol_sum > 0:
                prior_totals.append(vol_sum)
    if len(prior_totals) < min_sessions or float(np.mean(prior_totals)) <= 0:
        return 0.0, len(prior_totals)
    return current_total / float(np.mean(prior_totals)), len(prior_totals)


def session_vwap(df: pd.DataFrame) -> float:
    today = df.index[-1].date()
    session = df[[x.date() == today for x in df.index]]
    vol = session["volume"].astype(float).clip(lower=0)
    typical = (session["high"].astype(float) + session["low"].astype(float) + session["close"].astype(float)) / 3.0
    total = float(vol.sum())
    return float((typical * vol).sum() / total) if total > 0 else float("nan")


def compute_effective_day_move(df: pd.DataFrame, last_idx: pd.Timestamp, daily: Dict[str, Any], price: float) -> Tuple[float, float, float, float]:
    """Return (effective_move_pct, open_move_pct, session_high, session_low)."""
    today = last_idx.date()
    day_candles = df[[x.date() == today for x in df.index]]
    day_open = float(day_candles.iloc[0]["open"]) if not day_candles.empty else price
    session_high = float(day_candles["high"].max()) if not day_candles.empty else price
    session_low = float(day_candles["low"].min()) if not day_candles.empty else price

    prior_candles = df[[x.date() < today for x in df.index]]
    prev_close = float(prior_candles.iloc[-1]["close"]) if not prior_candles.empty else float(daily.get("previous_close") or day_open)

    open_move_pct = ((price / day_open) - 1.0) * 100.0 if day_open > 0 else 0.0
    prev_close_move_pct = ((price / prev_close) - 1.0) * 100.0 if prev_close > 0 else open_move_pct
    if abs(prev_close_move_pct) >= abs(open_move_pct):
        effective_move_pct = prev_close_move_pct
    else:
        effective_move_pct = open_move_pct
    return effective_move_pct, open_move_pct, session_high, session_low


def build_signal_dict(
    symbol: str,
    side: str,
    price: float,
    atr_value: float,
    score: int,
    rsi_value: float,
    adx_val: float,
    relvol: float,
    cumulative_relvol: float,
    sessions_compared: int,
    daily: Dict[str, Any],
    vwap_value: float,
    breakout_level: float,
    day_move: float,
    last_idx: pd.Timestamp,
    body_ratio: float,
    setup_label: str,
    tier: str,
) -> Dict[str, Any]:
    raw_risk = CFG["STOP_ATR_MULTIPLE"] * atr_value
    min_risk = price * CFG["MIN_STOP_PCT"]
    max_risk = price * CFG["MAX_STOP_PCT"]
    risk = float(np.clip(raw_risk, min_risk, max_risk))

    stop = price - risk if side == "BUY" else price + risk
    target = price + risk * CFG["TARGET_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["TARGET_R_MULTIPLE"]
    runner_target = price + risk * CFG["RUNNER_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["RUNNER_R_MULTIPLE"]
    reward_risk = abs(target - price) / max(abs(price - stop), 1e-9)

    return {
        "symbol": symbol,
        "side": side,
        "tier": tier,
        "price": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "runner_target": round(runner_target, 2),
        "reward_risk": round(reward_risk, 2),
        "score": int(min(100, max(0, score))),
        "rsi": round(rsi_value, 1),
        "adx": round(adx_val, 1),
        "relative_volume": round(relvol, 2),
        "cumulative_relative_volume": round(cumulative_relvol, 2),
        "volume_sessions_compared": int(sessions_compared),
        "avg_daily_turnover_crore": round(float(daily.get("avg_turnover", 0.0)) / 1e7, 2),
        "atr": round(atr_value, 2),
        "vwap": round(vwap_value, 2),
        "breakout_level": round(breakout_level, 2),
        "day_move_pct": round(day_move, 2),
        "candle_time": last_idx.isoformat(),
        "body_ratio": round(body_ratio, 2),
        "setup": setup_label,
        "forecast": None,
    }


def evaluate_symbol(
    symbol: str,
    source: pd.DataFrame,
    daily: Dict[str, Any],
    now: datetime,
    diagnostic: Optional[Dict[str, Any]] = None,
    allow_tier2: bool = False,
    pre_normalized: bool = False,
) -> Optional[Dict[str, Any]]:
    if source.empty:
        if diagnostic is not None:
            diagnostic["rejected"] = "no_candles"
        return None
    df = source if pre_normalized else normalize_intraday(source)
    df = completed_bars(df, now)
    if len(df) < CFG["MIN_HISTORY_BARS"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "insufficient_history"
        return None
    last_idx = df.index[-1]
    age = (pd.Timestamp(now).tz_convert(TZ) - last_idx).total_seconds() / 60.0
    if age < 5 or age > CFG["MAX_BAR_AGE_MIN"] + 5:
        if diagnostic is not None:
            diagnostic["rejected"] = "stale_candle"
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    if price < CFG["PRICE_FLOOR"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "price_floor"
        return None

    day_move, open_move, session_high, session_low = compute_effective_day_move(df, last_idx, daily, price)
    min_move_req = 2.5 if allow_tier2 else CFG["MIN_DAY_MOVE_PCT"]
    if abs(day_move) < min_move_req or abs(day_move) > CFG["MAX_DAY_MOVE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "day_move"
        return None

    relvol, slot_sessions = same_slot_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    cumulative_relvol, cumulative_sessions = session_cumulative_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    sessions_compared = max(slot_sessions, cumulative_sessions)
    if diagnostic is not None:
        diagnostic["volume"] = {
            "same_slot": round(relvol, 3),
            "cumulative": round(cumulative_relvol, 3),
            "sessions": sessions_compared,
        }

    # Volume confirmation: pass if both same-slot and cumulative RVOL are strong, OR cumulative RVOL >= 1.75x
    vol_pass = (
        (relvol >= CFG["MIN_RELATIVE_VOLUME"] and cumulative_relvol >= CFG["MIN_CUMULATIVE_RVOL"])
        or (cumulative_relvol >= 1.75 and relvol >= 0.85)
        or (relvol >= 2.20 and cumulative_relvol >= 1.15)
    )
    if not vol_pass:
        if diagnostic is not None:
            diagnostic["rejected"] = "relative_volume"
        return None

    closes = df["close"].astype(float).values
    fast = ema(closes, 9)
    slow = ema(closes, 21)
    rsi_value = rsi(closes, 14)
    atr_value = atr(df, 14)
    vwap_value = session_vwap(df)
    adx_val, plus_di, minus_di = adx_di(df, 14)
    if not np.isfinite([fast, slow, rsi_value, atr_value, vwap_value]).all() or atr_value <= 0:
        if diagnostic is not None:
            diagnostic["rejected"] = "indicator_invalid"
        return None

    prev = df.iloc[-(CFG["BREAKOUT_LOOKBACK"] + 1):-1]
    if len(prev) < CFG["BREAKOUT_LOOKBACK"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "breakout_history"
        return None
    prior_high = float(prev["high"].max())
    prior_low = float(prev["low"].min())

    bar_range = max(float(last["high"] - last["low"]), 1e-9)
    body_ratio = abs(float(last["close"] - last["open"])) / bar_range
    close_location = (float(last["close"]) - float(last["low"])) / bar_range

    bullish_candle = (
        float(last["close"]) > float(last["open"])
        and body_ratio >= CFG["MIN_BODY_RATIO"]
        and close_location >= CFG["MIN_CLOSE_LOCATION_BUY"]
    )
    bearish_candle = (
        float(last["close"]) < float(last["open"])
        and body_ratio >= CFG["MIN_BODY_RATIO"]
        and close_location <= CFG["MAX_CLOSE_LOCATION_SELL"]
    )

    # Tier-1 Breakout OR High-Tight Flag Continuation (within 0.65% of session high with strong RVOL & ADX)
    breakout_buy = (price >= prior_high * 0.999) or (
        price >= session_high * 0.9935 and cumulative_relvol >= 1.60 and adx_val >= 25.0
    )
    breakdown_sell = (price <= prior_low * 1.001) or (
        price <= session_low * 1.0065 and cumulative_relvol >= 1.60 and adx_val >= 25.0
    )

    daily_ema20 = float(daily.get("daily_ema20", price))
    daily_ema50 = float(daily.get("daily_ema50", price))
    daily_up = (price > daily_ema20) or (day_move >= 3.5 and cumulative_relvol >= 1.60)
    daily_down = (price < daily_ema20) or (day_move <= -3.5 and cumulative_relvol >= 1.60)

    above_vwap = price > vwap_value
    below_vwap = price < vwap_value
    adx_buy = adx_val >= CFG["MIN_ADX_TREND"] and plus_di >= minus_di * 0.95
    adx_sell = adx_val >= CFG["MIN_ADX_TREND"] and minus_di >= plus_di * 0.95

    side = None
    breakout_level = None
    tier = "TIER-1"
    setup_label = "Tier-1: Multi-factor VWAP + EMA9/21 + ADX RVOL Breakout"

    if (
        daily_up
        and fast > slow
        and above_vwap
        and breakout_buy
        and bullish_candle
        and adx_buy
        and CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"]
        and day_move >= CFG["MIN_DAY_MOVE_PCT"]
        and open_move >= 0.5
    ):
        extension = max(0.0, (price - prior_high) / atr_value)
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side, breakout_level = "BUY", min(prior_high, session_high)
    elif (
        daily_down
        and fast < slow
        and below_vwap
        and breakdown_sell
        and bearish_candle
        and adx_sell
        and CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"]
        and day_move <= -CFG["MIN_DAY_MOVE_PCT"]
        and open_move <= -0.5
    ):
        extension = max(0.0, (prior_low - price) / atr_value)
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side, breakout_level = "SELL", max(prior_low, session_low)

    # Tier-2 Top-Mover Continuation fallback (from 10:00 IST onward if <2 trades have fired today)
    if not side and allow_tier2:
        near_day_high = price >= session_high * 0.988
        near_day_low = price <= session_low * 1.012
        if (
            day_move >= 3.0
            and open_move >= 0.5
            and above_vwap
            and price > slow
            and fast >= slow * 0.998
            and near_day_high
            and float(last["close"]) > float(last["open"])
            and body_ratio >= 0.25
            and close_location >= 0.55
            and 54.0 <= rsi_value <= 92.0
        ):
            side, breakout_level = "BUY", session_high
            tier = "TIER-2"
            setup_label = "Tier-2: Top-5 Mover Intraday Trend Continuation"
        elif (
            day_move <= -3.0
            and open_move <= -0.5
            and below_vwap
            and price < slow
            and fast <= slow * 1.002
            and near_day_low
            and float(last["close"]) < float(last["open"])
            and body_ratio >= 0.25
            and close_location <= 0.45
            and 8.0 <= rsi_value <= 46.0
        ):
            side, breakout_level = "SELL", session_low
            tier = "TIER-2"
            setup_label = "Tier-2: Top-5 Mover Intraday Breakdown Continuation"

    if diagnostic is not None:
        diagnostic["conditions"] = {
            "daily_trend_buy": bool(daily_up), "daily_trend_sell": bool(daily_down),
            "ema_buy": bool(fast > slow), "ema_sell": bool(fast < slow),
            "vwap_buy": bool(above_vwap), "vwap_sell": bool(below_vwap),
            "breakout_buy": bool(breakout_buy), "breakout_sell": bool(breakdown_sell),
            "candle_buy": bool(bullish_candle), "candle_sell": bool(bearish_candle),
            "adx_buy": bool(adx_buy), "adx_sell": bool(adx_sell),
            "rsi_buy": bool(CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"]),
            "rsi_sell": bool(CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"]),
            "open_move_ok": bool(abs(open_move) >= 0.5),
        }
        diagnostic["day_move_pct"] = round(day_move, 2)
        diagnostic["open_move_pct"] = round(open_move, 2)

    if not side:
        if diagnostic is not None:
            diagnostic["rejected"] = "technical_alignment"
        return None

    score = 0
    score += 22 if relvol >= 2.2 else (16 if relvol >= 1.4 else 10)
    score += 18 if cumulative_relvol >= 2.0 else (14 if cumulative_relvol >= 1.4 else 8)
    score += 18 if 3.8 <= abs(day_move) <= 8.8 else (14 if abs(day_move) >= 2.8 else 8)
    score += 10 if (daily_ema20 > daily_ema50 if side == "BUY" else daily_ema20 < daily_ema50) else 5
    score += 15 if adx_val >= 32.0 else (12 if adx_val >= 22.0 else 6)
    score += 10 if body_ratio >= 0.55 else (7 if body_ratio >= 0.30 else 4)
    score += 7 if tier == "TIER-1" else 2

    min_req_score = CFG["MIN_SCORE_TIER1"] if tier == "TIER-1" else CFG["MIN_SCORE_TIER2"]
    if score < min_req_score:
        if diagnostic is not None:
            diagnostic["rejected"] = "low_score"
        return None

    return build_signal_dict(
        symbol=symbol,
        side=side,
        price=price,
        atr_value=atr_value,
        score=score,
        rsi_value=rsi_value,
        adx_val=adx_val,
        relvol=relvol,
        cumulative_relvol=cumulative_relvol,
        sessions_compared=sessions_compared,
        daily=daily,
        vwap_value=vwap_value,
        breakout_level=breakout_level,
        day_move=day_move,
        last_idx=last_idx,
        body_ratio=body_ratio,
        setup_label=setup_label,
        tier=tier,
    )


def kronos_forecast(
    symbol: str,
    raw_df: pd.DataFrame,
    side: str,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    if kronos_predictor is None:
        raise RuntimeError("Kronos model is not ready: " + kronos_status)
    df = normalize_intraday(raw_df)
    df = completed_bars(df, as_of or datetime.now(TZ)).tail(CFG["KRONOS_LOOKBACK"])
    if len(df) < 40:
        raise RuntimeError("Not enough completed candles for Kronos")

    x_df = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    x_df["amount"] = x_df["volume"] * ((x_df["high"] + x_df["low"] + x_df["close"]) / 3.0)
    x_timestamp = pd.Series(df.index.tz_localize(None))
    last_time = df.index[-1]
    y_index = pd.date_range(
        start=(last_time + pd.Timedelta(minutes=5)).tz_localize(None),
        periods=CFG["KRONOS_PRED_LEN"],
        freq="5min",
    )
    y_timestamp = pd.Series(y_index)
    prediction = kronos_predictor.predict(
        df=x_df.reset_index(drop=True),
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=CFG["KRONOS_PRED_LEN"],
        T=0.6,
        top_p=0.85,
        sample_count=CFG["KRONOS_SAMPLE_COUNT"],
    )
    predicted_close = float(prediction["close"].iloc[-1])
    current_close = float(df["close"].iloc[-1])
    change_pct = ((predicted_close / current_close) - 1.0) * 100.0
    aligned = change_pct >= CFG["KRONOS_MIN_MOVE_PCT"] if side == "BUY" else change_pct <= -CFG["KRONOS_MIN_MOVE_PCT"]
    return {
        "model": "Kronos-mini",
        "horizon_bars": CFG["KRONOS_PRED_LEN"],
        "forecast_close": round(predicted_close, 2),
        "forecast_change_pct": round(change_pct, 3),
        "direction_aligned": bool(aligned),
        "interpretation": "directional forecast (advisory)",
    }


def add_signal(payload: Dict[str, Any]) -> bool:
    day = datetime.now(TZ).strftime("%Y-%m-%d")
    dedupe = f"{day}|{payload['symbol']}"
    with db_connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM signals WHERE created_at LIKE ?", (day + "%",)).fetchone()["n"]
        if count >= CFG["MAX_SIGNALS_PER_DAY"]:
            return False
        try:
            conn.execute(
                "INSERT INTO signals(created_at,symbol,side,payload,dedupe_key,delivered) VALUES(?,?,?,?,?,0)",
                (datetime.now(TZ).isoformat(), payload["symbol"], payload["side"], json.dumps(payload), dedupe),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def format_alert_message(payload: Dict[str, Any]) -> str:
    forecast = payload.get("forecast") or {}
    if forecast.get("forecast_change_pct") is not None:
        aligned_tag = "✅ aligned" if forecast.get("direction_aligned") else "⚠️ neutral/divergent"
        forecast_line = f"Kronos-mini: {forecast.get('forecast_change_pct'):+.2f}% ({aligned_tag})"
    else:
        forecast_line = "Kronos-mini: advisory skipped"

    runner_str = f" | Runner (3.5R): ₹{payload['runner_target']}" if payload.get("runner_target") else ""
    tier_str = payload.get("tier", "TIER-1")
    adx_str = f" | ADX: {payload.get('adx', 'N/A')}" if payload.get("adx") is not None else ""

    return (
        f"BHARAT INTRADAY SCOUT 2.0 [{tier_str}]\n"
        f"{'🟢 BUY' if payload['side'] == 'BUY' else '🔴 SELL'} — {payload['symbol']} (Day Move: {payload['day_move_pct']:+.2f}%)\n"
        f"Entry reference: ₹{payload['price']}\n"
        f"Stop reference: ₹{payload['stop']} | Target (2.2R): ₹{payload['target']}{runner_str}\n"
        f"Planned R:R: {payload['reward_risk']}R | Conviction score: {payload['score']}/100\n"
        f"Same-slot 5m RVOL: {payload['relative_volume']}x | Cumulative RVOL: {payload['cumulative_relative_volume']}x\n"
        f"Avg daily turnover: ₹{payload['avg_daily_turnover_crore']} Cr\n"
        f"RSI: {payload['rsi']}{adx_str} | VWAP: ₹{payload['vwap']} | 5m ATR: ₹{payload['atr']}\n"
        f"{forecast_line}\n"
        f"Setup: {payload['setup']}\n"
        f"Candle time: {payload['candle_time']}\n"
        "Research alert only. Check live spread, price and execution risk. No order was placed."
    )


def send_telegram_message(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID GitHub Secret")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError("Telegram API rejected message: " + str(body.get("description", "unknown error")))


def deliver_pending_signals(limit: int = 5) -> int:
    """Deliver queued alerts, then mark them delivered in the persisted SQLite DB."""
    delivered = 0
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT id,payload FROM signals WHERE delivered=0 ORDER BY id ASC LIMIT ?",
            (max(1, min(int(limit), 20)),),
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload"])
            send_telegram_message(format_alert_message(payload))
            with db_connect() as conn:
                conn.execute("UPDATE signals SET delivered=1 WHERE id=?", (row["id"],))
                conn.commit()
            delivered += 1
            LOG.info("Telegram delivered | %s | %s", payload.get("symbol"), payload.get("side"))
        except Exception as exc:
            LOG.exception("Telegram delivery failed for queued signal id=%s: %s", row["id"], exc)
            break
    return delivered


def rank_top_intraday_movers(
    intraday_map: Dict[str, pd.DataFrame],
    daily_map: Dict[str, Dict[str, Any]],
    now: datetime,
) -> List[Tuple[float, str]]:
    """Rank stocks by absolute intraday move % among liquid stocks moving between 2.5% and 9.8%."""
    movers: List[Tuple[float, str]] = []
    for symbol, raw_df in intraday_map.items():
        if raw_df is None or raw_df.empty:
            continue
        df = completed_bars(normalize_intraday(raw_df), now)
        if len(df) < CFG["MIN_HISTORY_BARS"]:
            continue
        last_idx = df.index[-1]
        price = float(df.iloc[-1]["close"])
        day_move, _, _, _ = compute_effective_day_move(df, last_idx, daily_map.get(symbol, {}), price)
        if 2.5 <= abs(day_move) <= CFG["MAX_DAY_MOVE_PCT"]:
            movers.append((abs(day_move), symbol))
    movers.sort(key=lambda x: x[0], reverse=True)
    return movers


def run_scan_cycle() -> Dict[str, Any]:
    started = time.time()
    now = datetime.now(TZ)
    if not is_market_session(now):
        return {"status": "outside_market_hours", "at": now.isoformat()}

    delivered_prior = deliver_pending_signals(limit=5)
    existing_today = get_today_signal_count()
    if existing_today >= CFG["MAX_SIGNALS_PER_DAY"]:
        return {
            "status": "daily_target_reached",
            "at": now.isoformat(),
            "signals_today": existing_today,
            "telegram_delivered": delivered_prior,
        }

    # Morning slot pacing: allow at most 1 trade before 09:45 IST so slot #2 stays open for 09:45+
    if now.time() < dtime(CFG["MORNING_GUARD_HOUR"], CFG["MORNING_GUARD_MINUTE"]) and existing_today >= 1:
        return {
            "status": "morning_slot_used_waiting_for_0945",
            "at": now.isoformat(),
            "signals_today": existing_today,
            "telegram_delivered": delivered_prior,
        }

    if not is_entry_window(now):
        return {
            "status": "past_last_entry_window",
            "at": now.isoformat(),
            "signals_today": existing_today,
            "telegram_delivered": delivered_prior,
        }

    if not UNIVERSE_CACHE.exists():
        LOG.info("Universe cache missing; building initial universe cache...")
        refresh_liquid_universe(force=True)

    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    if not entries:
        raise RuntimeError("Liquid universe cache is empty")

    symbols = [x["symbol"] for x in entries]
    daily_map = {x["symbol"]: x for x in entries}
    intraday_all: Dict[str, pd.DataFrame] = {}

    for start in range(0, len(symbols), CFG["INTRADAY_BATCH_SIZE"]):
        group = symbols[start:start + CFG["INTRADAY_BATCH_SIZE"]]
        intraday_all.update(download_batch(group, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"]))
        time.sleep(0.08)

    alerted_today = get_today_alerted_symbols()
    top_movers = rank_top_intraday_movers(intraday_all, daily_map, now)
    top_mover_pool = {sym for _, sym in top_movers[:CFG["TOP_MOVERS_POOL_SIZE"] * 2]}

    candidates: List[Tuple[int, float, Dict[str, Any], pd.DataFrame]] = []
    top5_diagnostics: Dict[str, Any] = {}

    # Pass 1: Tier-1 Multi-Factor Breakout Scan focused on today's Top Movers
    for symbol in symbols:
        if symbol in alerted_today or (top_mover_pool and symbol not in top_mover_pool):
            continue
        frame = intraday_all.get(symbol, pd.DataFrame())
        diag: Dict[str, Any] = {}
        signal = evaluate_symbol(symbol, frame, daily_map[symbol], now, diagnostic=diag, allow_tier2=False)
        if symbol in {s for _, s in top_movers[:5]}:
            top5_diagnostics[symbol] = "PASS_TIER1" if signal else diag
        if signal:
            candidates.append((signal["score"], abs(signal["day_move_pct"]), signal, frame))

    # Pass 2: Tier-2 Top-Mover Continuation fallback after 10:00 IST if <2 signals have fired today
    tier2_triggered = False
    if not candidates and is_tier2_window(now) and existing_today < CFG["MAX_SIGNALS_PER_DAY"]:
        tier2_triggered = True
        top_mover_symbols = [sym for _, sym in top_movers[:CFG["TOP_MOVERS_POOL_SIZE"]]]
        for symbol in top_mover_symbols:
            if symbol in alerted_today:
                continue
            frame = intraday_all.get(symbol, pd.DataFrame())
            diag_t2: Dict[str, Any] = {}
            signal = evaluate_symbol(symbol, frame, daily_map[symbol], now, diagnostic=diag_t2, allow_tier2=True)
            if symbol in {s for _, s in top_movers[:5]} and signal:
                top5_diagnostics[symbol] = "PASS_TIER2"
            if signal:
                candidates.append((signal["score"], abs(signal["day_move_pct"]), signal, frame))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    technical_candidates = len(candidates)
    kronos_rejected = 0
    inserted = 0
    remaining_slots = max(0, CFG["MAX_SIGNALS_PER_DAY"] - existing_today)
    cycle_limit = min(CFG["MAX_SIGNALS_PER_CYCLE"], remaining_slots)

    for _, _, signal, candles in candidates:
        if inserted >= cycle_limit:
            break
        if CFG["KRONOS_ENABLED"] and kronos_predictor is not None:
            try:
                forecast = kronos_forecast(signal["symbol"], candles, signal["side"])
                signal["forecast"] = forecast
                if forecast["direction_aligned"]:
                    signal["score"] = min(100, signal["score"] + 5)
                elif CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue
            except Exception as exc:
                signal["forecast"] = {"status": "unavailable", "detail": str(exc)[:180]}
                if CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue
        elif CFG["KRONOS_REQUIRED"]:
            kronos_rejected += 1
            continue

        if add_signal(signal):
            inserted += 1

    delivered = delivered_prior + deliver_pending_signals(limit=5)
    if inserted > 0 or delivered > 0:
        persist_state_to_git()

    summary = {
        "status": "complete",
        "at": now.isoformat(),
        "symbols_considered": len(symbols),
        "top_5_intraday_movers": [{"symbol": s, "abs_move_pct": round(m, 2)} for m, s in top_movers[:5]],
        "top_5_diagnostics": top5_diagnostics,
        "technical_candidates": technical_candidates,
        "tier2_fallback_evaluated": tier2_triggered,
        "kronos_rejected_or_unavailable": kronos_rejected,
        "new_signals_queued": inserted,
        "signals_today_total": existing_today + inserted,
        "telegram_delivered": delivered,
        "duration_seconds": round(time.time() - started, 1),
        "universe_refreshed_at": universe.get("refreshed_at"),
        "kronos": kronos_status,
    }
    LOG.info("SCAN SUMMARY %s", json.dumps(summary))
    return summary


def seconds_until_next_5m_slot(now: Optional[datetime] = None) -> float:
    """Calculate sleep seconds until 12 seconds after the next 5-minute boundary (e.g. 09:25:12)."""
    now = now or datetime.now(TZ)
    minute_mod = now.minute % 5
    mins_to_add = 5 - minute_mod
    next_slot = (now + timedelta(minutes=mins_to_add)).replace(second=12, microsecond=0)
    wait_sec = (next_slot - now).total_seconds()
    if wait_sec < 15:
        wait_sec += 300.0
    return wait_sec


def run_continuous_market_session() -> Dict[str, Any]:
    """Run 5-minute scan cycles automatically in a loop so the user never has to trigger runs manually."""
    session_start = time.monotonic()
    max_seconds = CFG["CONTINUOUS_LOOP_MINUTES"] * 60.0
    now = datetime.now(TZ)

    # If triggered between 08:30 and 09:20 IST, wait until 09:20:10 IST automatically
    if is_pre_market_wait_window(now):
        target_open = now.replace(hour=9, minute=20, second=10, microsecond=0)
        wait_open = max(0.0, (target_open - now).total_seconds())
        LOG.info("Pre-market start detected at %s; waiting %.0fs until 09:20:10 IST market scan...", now.strftime("%H:%M:%S"), wait_open)
        time.sleep(wait_open)
        now = datetime.now(TZ)

    if not is_market_session(now):
        LOG.info("Outside NSE market session (%s); scan skipped cleanly", now.strftime("%H:%M:%S"))
        return {"status": "outside_market_hours", "at": now.isoformat()}

    load_kronos()
    if CFG["KRONOS_REQUIRED"] and kronos_predictor is None:
        raise RuntimeError("Kronos is required but not ready: " + kronos_status)

    cycles_run = 0
    last_result: Dict[str, Any] = {"status": "idle"}

    while True:
        now = datetime.now(TZ)
        if not is_entry_window(now):
            LOG.info("Reached end of intraday entry window (%s); stopping session loop.", now.strftime("%H:%M:%S"))
            break

        cycles_run += 1
        LOG.info("=== AUTO-SCAN CYCLE #%d at %s IST ===", cycles_run, now.strftime("%H:%M:%S"))
        last_result = run_scan_cycle()

        if last_result.get("status") == "daily_target_reached" or last_result.get("signals_today_total", 0) >= CFG["MAX_SIGNALS_PER_DAY"]:
            LOG.info("Daily target of %d confirmed trades reached! Stopping session loop.", CFG["MAX_SIGNALS_PER_DAY"])
            break

        elapsed = time.monotonic() - session_start
        wait_sec = seconds_until_next_5m_slot(datetime.now(TZ))
        if elapsed + wait_sec >= max_seconds:
            LOG.info("Session loop window (%.0fm) completed after %d cycles; handing off to next scheduled/chained job.", CFG["CONTINUOUS_LOOP_MINUTES"], cycles_run)
            break

        LOG.info("Sleeping %.0fs until next 5-minute candle close...", wait_sec)
        time.sleep(wait_sec)

    last_result["cycles_run"] = cycles_run
    return last_result


def run_dry_run_v2() -> Dict[str, Any]:
    """Historical full-session diagnostic replay with chronological live-pacing simulation."""
    from collections import Counter
    started = time.monotonic()
    if not UNIVERSE_CACHE.exists():
        raise RuntimeError("Missing data/liquid_universe.json. Run refresh-universe first.")
    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    if not entries:
        raise RuntimeError("Universe cache has no scan symbols")

    frames = {}
    for offset in range(0, len(entries), CFG["INTRADAY_BATCH_SIZE"]):
        group = [e["symbol"] for e in entries[offset:offset + CFG["INTRADAY_BATCH_SIZE"]]]
        frames.update(download_batch(group, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"]))
        LOG.info("DRY RUN V2 downloaded %s/%s symbols", min(offset + len(group), len(entries)), len(entries))

    normalized = {}
    latest_sessions = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        clean = normalize_intraday(frame)
        clean = clean.between_time("09:15", "15:25")
        if clean.empty:
            continue
        normalized[symbol] = clean
        latest_sessions.append(clean.index[-1].date())
    if not latest_sessions:
        raise RuntimeError("Yahoo returned no usable NSE 5-minute candles")
    replay_day = max(latest_sessions)
    actual_now = datetime.now(TZ)
    end = datetime.combine(replay_day, dtime(CFG["LAST_ENTRY_HOUR"], CFG["LAST_ENTRY_MINUTE"]), tzinfo=TZ)
    if replay_day == actual_now.date():
        end = min(end, actual_now)

    rejection = Counter()
    conditions_failed = Counter()
    candidates = []
    total_checks = 0
    symbols_with_session = 0
    replayed_bars = 0

    for entry in entries:
        symbol = entry["symbol"]
        frame = normalized.get(symbol)
        if frame is None:
            rejection["missing_intraday_data"] += 1
            continue
        today = frame[[x.date() == replay_day for x in frame.index]]
        if today.empty:
            rejection["missing_replay_session"] += 1
            continue
        symbols_with_session += 1
        for candle_start in today.index:
            anchor = candle_start.to_pydatetime() + timedelta(minutes=5)
            if anchor > end or anchor.time() < dtime(9, 20):
                continue
            total_checks += 1
            replayed_bars += 1
            prefix = frame.loc[frame.index <= candle_start]
            diag: Dict[str, Any] = {}
            allow_t2 = anchor.time() >= dtime(CFG["TIER2_START_HOUR"], CFG["TIER2_START_MINUTE"])
            signal = evaluate_symbol(
                symbol, prefix, entry, anchor, diagnostic=diag, allow_tier2=allow_t2, pre_normalized=True
            )
            if signal:
                candidates.append((signal["score"], abs(signal["day_move_pct"]), signal, prefix, anchor))
                rejection["technical_pass"] += 1
            else:
                rejection[diag.get("rejected", "unknown")] += 1
            if diag.get("rejected") == "technical_alignment":
                flags = diag.get("conditions", {})
                for key, passed in flags.items():
                    if not passed:
                        conditions_failed[key] += 1

    candidates_by_rank = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]["candle_time"], item[2]["symbol"]))
    seen_symbols = set()
    unique_daily_picks = []
    for item in candidates_by_rank:
        sym = item[2]["symbol"]
        if sym not in seen_symbols:
            seen_symbols.add(sym)
            unique_daily_picks.append(item)

    candidates_by_time = sorted(candidates, key=lambda item: (item[2]["candle_time"], -item[0], -item[1]))
    live_simulated_picks = []
    live_seen = set()
    used_times = set()
    for item in candidates_by_time:
        if len(live_simulated_picks) >= CFG["MAX_SIGNALS_PER_DAY"]:
            break
        sig = item[2]
        anchor_dt = item[4]
        if sig["symbol"] in live_seen or sig["candle_time"] in used_times:
            continue
        if anchor_dt.time() < dtime(CFG["MORNING_GUARD_HOUR"], CFG["MORNING_GUARD_MINUTE"]) and len(live_simulated_picks) >= 1:
            continue
        live_seen.add(sig["symbol"])
        used_times.add(sig["candle_time"])
        live_simulated_picks.append(item)

    independent = {"status": "not_run"}
    eligible = []
    for entry in entries:
        f = normalized.get(entry["symbol"])
        if f is not None and len(completed_bars(f.loc[f.index.date <= replay_day], end)) >= 40:
            eligible.append((entry["symbol"], f))
    if eligible and kronos_predictor is not None:
        sym, frame = eligible[0]
        try:
            pred = kronos_forecast(sym, frame, "BUY", as_of=end)
            independent = {"status": "success", "symbol": sym, "forecast": pred}
        except Exception as exc:
            independent = {"status": "failed", "symbol": sym, "error": str(exc)[:350]}
            LOG.warning("DRY RUN V2 independent Kronos inference failed: %s", exc)
    elif not eligible:
        independent = {"status": "no_sufficient_history"}
    else:
        independent = {"status": "skipped_kronos_not_loaded"}

    forecast_samples = []
    for _, _, signal, prefix, anchor in unique_daily_picks[:CFG["MAX_SIGNALS_PER_DAY"]]:
        item = dict(signal)
        if kronos_predictor is not None:
            try:
                item["forecast"] = kronos_forecast(signal["symbol"], prefix, signal["side"], as_of=anchor)
            except Exception as exc:
                item["forecast"] = {"status": "error", "detail": str(exc)[:250]}
        forecast_samples.append(item)

    report = {
        "mode": "DRY_RUN_V2_NO_ALERTS",
        "session_date_ist": str(replay_day),
        "symbols_requested": len(entries),
        "symbols_with_data": len(normalized),
        "symbols_with_replay_session": symbols_with_session,
        "candles_evaluated": replayed_bars,
        "checks_total": total_checks,
        "primary_rejection_counts": dict(rejection.most_common()),
        "alignment_condition_failures": dict(conditions_failed.most_common()),
        "technical_candidates_bars": len(candidates),
        "unique_symbol_candidates": len(unique_daily_picks),
        "simulated_live_daily_trades": [x[2] for x in live_simulated_picks],
        "selected_top_2_daily_trades": [x[2] for x in unique_daily_picks[:CFG["MAX_SIGNALS_PER_DAY"]]],
        "top_10_unique_candidates": [x[2] for x in unique_daily_picks[:10]],
        "kronos_independent_test": independent,
        "kronos_candidate_forecasts": forecast_samples,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    path = ROOT / "dry_run_v2_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOG.info("DRY RUN V2 SUMMARY %s", json.dumps({
        k: v for k, v in report.items()
        if k in (
            "session_date_ist", "symbols_requested", "symbols_with_replay_session",
            "candles_evaluated", "primary_rejection_counts", "unique_symbol_candidates",
            "simulated_live_daily_trades", "selected_top_2_daily_trades", "elapsed_seconds",
        )
    }, default=str))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bharat Intraday Scout 2.0 automatic session runner")
    parser.add_argument(
        "--task",
        choices=["scan", "refresh-universe", "test-telegram", "test-kronos", "dry-run-v2"],
        default="scan",
    )
    args = parser.parse_args()
    if args.task == "dry-run-v2":
        load_kronos()
        run_dry_run_v2()
        return 0

    init_db()

    if args.task == "refresh-universe":
        result = refresh_liquid_universe(force=True)
        LOG.info(
            "UNIVERSE READY | eligible=%s | selected=%s | refreshed=%s",
            result.get("eligible_count"),
            len(result.get("scan_symbols", [])),
            result.get("refreshed_at"),
        )
        return 0

    if args.task == "test-telegram":
        send_telegram_message("✅ Bharat Intraday Scout 2.0 Telegram test successful. Auto-session loop active.")
        LOG.info("Telegram test succeeded")
        return 0

    if args.task == "test-kronos":
        load_kronos()
        if kronos_predictor is None:
            raise RuntimeError("Kronos failed to load: " + kronos_status)
        LOG.info("Kronos self-test succeeded: %s", kronos_status)
        return 0

    result = run_continuous_market_session()
    LOG.info("RUN RESULT %s", json.dumps(result))
    if result.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
