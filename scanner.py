from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
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

# Pre-Explosion Ignition & fmzquant + Kronos Confluence Configuration:
# Strictly rejects stocks that have ALREADY moved >5.0% ("already_moved_too_far").
# Catches stocks EARLY (0.4% to 4.5% day move) right as they break out of a volatility squeeze
# with institutional volume/MFI + MACD + ADX + ATR Channel + Kronos forward trajectory alignment.
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
    "MIN_RELATIVE_VOLUME": 1.35,        # 5m volume surge at ignition
    "MIN_CUMULATIVE_RVOL": 1.20,        # Session volume accumulation
    "BREAKOUT_LOOKBACK": 12,            # 12-bar (1-hour) coil breakout lookback
    "MAX_EXTENSION_ATR": 2.0,           # Do not chase overextended candles
    "MIN_DAY_MOVE_PCT": float(os.getenv("MIN_DAY_MOVE_PCT", "0.4")),   # Catch early ignition starting from +0.4%
    "IDEAL_MAX_MOVE_PCT": float(os.getenv("IDEAL_MAX_MOVE_PCT", "3.6")),  # Sweet spot: +0.4% to +3.6% (leaves 8-12% room!)
    "MAX_DAY_MOVE_PCT": float(os.getenv("MAX_DAY_MOVE_PCT", "5.0")),   # HARD CAP: Reject stocks that already moved >5.0%!
    "MAX_VWAP_DISTANCE_PCT": 2.8,       # Must be within 2.8% of Session VWAP (near institutional cost basis)
    "MAX_BAR_AGE_MIN": 14,
    "MIN_RSI_BUY": 54.0,
    "MAX_RSI_BUY": 82.0,                # Avoid already-exhausted >82 RSI tops
    "MIN_RSI_SELL": 18.0,
    "MAX_RSI_SELL": 46.0,
    "MIN_MFI_BUY": 52.0,                # fmzquant EMA-VWAP-MFI strategy: positive money flow accumulation
    "MAX_MFI_SELL": 48.0,
    "MIN_ADX_TREND": 18.0,              # fmzquant ADX Trend Breakout Momentum strategy
    "MIN_CONFLUENCE_PILLARS": int(os.getenv("MIN_CONFLUENCE_PILLARS", "4")),  # Must pass at least 4 of 5 fmzquant pillars
    "MIN_BODY_RATIO": 0.32,
    "MIN_CLOSE_LOCATION_BUY": 0.62,
    "MAX_CLOSE_LOCATION_SELL": 0.38,
    "STOP_ATR_MULTIPLE": 1.50,
    "MIN_STOP_PCT": 0.0080,             # 0.80% min stop
    "MAX_STOP_PCT": 0.0220,             # 2.20% max stop
    "TARGET1_R_MULTIPLE": 2.5,          # T1 = 2.5R
    "TARGET2_R_MULTIPLE": 5.0,          # T2 = 5.0R (mid-expansion)
    "EXPANSION_TARGET_PCT": 10.0,       # T3 = +10% full intraday breakout projection
    "MIN_SCORE_TIER1": int(os.getenv("MIN_SCORE_TIER1", "78")),
    "MIN_SCORE_TIER2": int(os.getenv("MIN_SCORE_TIER2", "70")),
    "TIER2_START_HOUR": 9,
    "TIER2_START_MINUTE": 45,
    "LAST_ENTRY_HOUR": 15,
    "LAST_ENTRY_MINUTE": 20,
    "LIMIT_BEFORE_10AM": int(os.getenv("LIMIT_BEFORE_10AM", "10")),
    "LIMIT_10AM_TO_1PM": int(os.getenv("LIMIT_10AM_TO_1PM", "10")),
    "LIMIT_1PM_TO_2PM": int(os.getenv("LIMIT_1PM_TO_2PM", "5")),
    "LIMIT_AFTER_2PM": int(os.getenv("LIMIT_AFTER_2PM", "2")),
    "MAX_SIGNALS_PER_CYCLE": int(os.getenv("MAX_SIGNALS_PER_CYCLE", "3")),
    "CONTINUOUS_LOOP_MINUTES": int(os.getenv("CONTINUOUS_LOOP_MINUTES", "350")),
    "KRONOS_ENABLED": os.getenv("KRONOS_ENABLED", "true").lower() == "true",
    "KRONOS_REQUIRED": os.getenv("KRONOS_REQUIRED", "true").lower() == "true",  # Enforce Kronos confirmation when model is loaded
    "KRONOS_MODEL": os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-mini"),
    "KRONOS_TOKENIZER": os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-2k"),
    "KRONOS_LOOKBACK": int(os.getenv("KRONOS_LOOKBACK", "96")),
    "KRONOS_PRED_LEN": int(os.getenv("KRONOS_PRED_LEN", "6")),          # 6 candles (30-min forward trajectory)
    "KRONOS_SAMPLE_COUNT": int(os.getenv("KRONOS_SAMPLE_COUNT", "3")),  # 3-sample ensemble average
    "KRONOS_MIN_MOVE_PCT": float(os.getenv("KRONOS_MIN_MOVE_PCT", "0.08")),
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
    """Conflict-free persistence of SQLite DB and universe cache to GitHub main branch."""
    try:
        tmp_db = Path("/tmp/bis20.sqlite3.bak")
        tmp_uni = Path("/tmp/liquid_universe.json.bak")
        if DB_PATH.exists():
            shutil.copy2(DB_PATH, tmp_db)
        if UNIVERSE_CACHE.exists():
            shutil.copy2(UNIVERSE_CACHE, tmp_uni)

        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
        subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=False)
        subprocess.run(["git", "fetch", "origin", "main"], check=False)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=False)

        if tmp_db.exists():
            shutil.copy2(tmp_db, DB_PATH)
        if tmp_uni.exists():
            shutil.copy2(tmp_uni, UNIVERSE_CACHE)

        subprocess.run(["git", "add", "runtime/bis20.sqlite3", "data/liquid_universe.json"], check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", "BIS 2.0: persist scanner state [skip ci]"], check=False)
            subprocess.run(["git", "push", "origin", "HEAD:main"], check=False)
            LOG.info("Persisted updated SQLite state to GitHub main branch.")
    except Exception as exc:
        LOG.warning("Git state persistence warning (non-fatal): %s", exc)


def get_time_window_info(now: Optional[datetime] = None) -> Tuple[str, dtime, dtime, int]:
    """Return (window_name, start_time, end_time, max_trades_in_window) for current IST time."""
    now = now or datetime.now(TZ)
    t = now.time()
    if t < dtime(10, 0):
        return "BEFORE_10AM", dtime(0, 0), dtime(9, 59, 59), CFG["LIMIT_BEFORE_10AM"]
    if t < dtime(13, 0):
        return "10AM_TO_1PM", dtime(10, 0), dtime(12, 59, 59), CFG["LIMIT_10AM_TO_1PM"]
    if t < dtime(14, 0):
        return "1PM_TO_2PM", dtime(13, 0), dtime(13, 59, 59), CFG["LIMIT_1PM_TO_2PM"]
    return "AFTER_2PM", dtime(14, 0), dtime(23, 59, 59), CFG["LIMIT_AFTER_2PM"]


def get_window_signal_counts(now: Optional[datetime] = None) -> Tuple[str, int, int, int]:
    """Return (window_name, used_in_window, limit_in_window, total_today)."""
    now = now or datetime.now(TZ)
    day_str = now.strftime("%Y-%m-%d")
    win_name, win_start, win_end, win_limit = get_time_window_info(now)

    with db_connect() as conn:
        rows = conn.execute("SELECT created_at FROM signals WHERE created_at LIKE ?", (day_str + "%",)).fetchall()

    total_today = len(rows)
    used_in_window = 0
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["created_at"]).astimezone(TZ)
            if win_start <= dt.time() <= win_end:
                used_in_window += 1
        except Exception:
            pass

    return win_name, used_in_window, win_limit, total_today


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
    """Load Kronos-mini model on CPU if available; degrade gracefully if KRONOS_REPO_DIR is missing."""
    global kronos_predictor, kronos_status
    if kronos_predictor is not None:
        return
    if not CFG["KRONOS_ENABLED"]:
        kronos_status = "disabled"
        return
    if not KRONOS_REPO_DIR.exists():
        kronos_status = f"missing_repo:{KRONOS_REPO_DIR}"
        LOG.warning("Kronos repository not found at %s; continuing with 5-pillar fmzquant confluence", KRONOS_REPO_DIR)
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
        LOG.warning("Kronos initialization failed; falling back to 5-pillar technical confluence: %s", exc)


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
    """Build candidate pool from NSE EQ symbols using 20-session turnover and high ADR% potential."""
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

    top_liquid = metrics[:int(CFG["INTRADAY_UNIVERSE_SIZE"] * 0.5)]
    top_liquid_set = {m["symbol"] for m in top_liquid}
    remaining = [m for m in metrics[int(CFG["INTRADAY_UNIVERSE_SIZE"] * 0.5):500] if m["symbol"] not in top_liquid_set]
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


def ema_series(values: np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema(values: Any, period: int) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < period:
        return float("nan")
    s = ema_series(arr, period)
    return float(s[-1])


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


def mfi(df: pd.DataFrame, period: int = 14) -> float:
    """Money Flow Index (from fmzquant 200-EMA-VWAP-MFI-Trend-Following-Strategy)."""
    if len(df) <= period:
        return 50.0
    tail = df.tail(period + 15)
    tp = (tail["high"].astype(float) + tail["low"].astype(float) + tail["close"].astype(float)) / 3.0
    vol = tail["volume"].astype(float).clip(lower=0.0)
    rmf = tp * vol
    diff = tp.diff()
    pos_flow = rmf.where(diff > 0, 0.0).tail(period).sum()
    neg_flow = rmf.where(diff < 0, 0.0).tail(period).sum()
    if neg_flow <= 1e-9:
        return 100.0 if pos_flow > 0 else 50.0
    mr = float(pos_flow / neg_flow)
    return 100.0 - (100.0 / (1.0 + mr))


def bollinger_squeeze_ignition(closes: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[bool, bool, bool, float]:
    """Adaptive Bollinger Band Squeeze & Breakout (from fmzquant Adaptive-Bollinger-Bands-Trend-Following-Strategy).

    Returns (squeeze_coil, bb_breakout_buy, bb_breakdown_sell, current_bandwidth_pct).
    """
    if len(closes) < period + 6:
        return False, False, False, 0.0
    recent = closes[-(period + 6):]
    bw_history: List[float] = []
    upper_now = 0.0
    lower_now = 0.0
    for offset in range(7):
        window = recent[offset:offset + period]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        up = mean + mult * std
        lo = mean - mult * std
        bw = ((up - lo) / max(mean, 1e-9)) * 100.0
        bw_history.append(bw)
        if offset == 6:
            upper_now = up
            lower_now = lo

    curr_bw = bw_history[-1]
    prior_min_bw = min(bw_history[:-1])
    # Squeeze coil: prior 5 bars had tight bandwidth (< 2.2% on 5m) or current bandwidth is expanding from coil
    squeeze_coil = prior_min_bw <= 2.4 or (curr_bw >= prior_min_bw * 1.12)
    price = float(closes[-1])
    # Breakout through or into top 15% of upper band during bandwidth expansion
    band_span = max(upper_now - lower_now, 1e-9)
    pos_in_band = (price - lower_now) / band_span
    bb_buy = squeeze_coil and pos_in_band >= 0.82 and curr_bw >= bw_history[-2]
    bb_sell = squeeze_coil and pos_in_band <= 0.18 and curr_bw >= bw_history[-2]
    return squeeze_coil, bb_buy, bb_sell, round(curr_bw, 2)


def macd_impulse(closes: np.ndarray) -> Tuple[bool, bool, float]:
    """MACD Histogram Impulse (from fmzquant 1-2-3-Pattern-with-EMAs-MACD strategy)."""
    if len(closes) < 35:
        return False, False, 0.0
    e12 = ema_series(closes, 12)
    e26 = ema_series(closes, 26)
    macd_line = e12 - e26
    valid = macd_line[~np.isnan(macd_line)]
    if len(valid) < 10:
        return False, False, 0.0
    sig_line = ema_series(valid, 9)
    hist = valid - sig_line
    if np.isnan(hist[-1]) or np.isnan(hist[-2]):
        return False, False, 0.0
    h_now = float(hist[-1])
    h_prev = float(hist[-2])
    macd_buy = h_now > 0 and h_now >= h_prev
    macd_sell = h_now < 0 and h_now <= h_prev
    return macd_buy, macd_sell, round(h_now, 4)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    value = tr.tail(period).mean()
    return float(value) if pd.notna(value) else float("nan")


def adx_di_slope(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float, bool]:
    """Calculate Wilder's ADX, +DI, -DI, and whether ADX is rising (from fmzquant ADX Trend Breakout Momentum)."""
    tail_df = df.tail(max(period * 4, 60))
    if len(tail_df) < period * 2:
        return 20.0, 20.0, 20.0, True
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

    if len(dx_list) < 2:
        return 20.0, 20.0, 20.0, True
    adx_val = float(np.mean(dx_list[-period:]))
    adx_prev = float(np.mean(dx_list[-(period + 1):-1])) if len(dx_list) > period else adx_val
    adx_rising = adx_val >= adx_prev * 0.98 or dx_list[-1] >= adx_val
    return adx_val, plus_di, minus_di, adx_rising


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
    mfi_value: float,
    adx_val: float,
    bb_bw: float,
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
    pillars_matched: List[str],
) -> Dict[str, Any]:
    raw_risk = CFG["STOP_ATR_MULTIPLE"] * atr_value
    min_risk = price * CFG["MIN_STOP_PCT"]
    max_risk = price * CFG["MAX_STOP_PCT"]
    risk = float(np.clip(raw_risk, min_risk, max_risk))

    stop = price - risk if side == "BUY" else price + risk
    target = price + risk * CFG["TARGET1_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["TARGET1_R_MULTIPLE"]
    runner_target = price + risk * CFG["TARGET2_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["TARGET2_R_MULTIPLE"]
    expansion_10pct_target = price * (1.0 + CFG["EXPANSION_TARGET_PCT"] / 100.0) if side == "BUY" else price * (1.0 - CFG["EXPANSION_TARGET_PCT"] / 100.0)
    remaining_room_to_12pct = max(4.0, round(12.5 - abs(day_move), 2))
    reward_risk = abs(target - price) / max(abs(price - stop), 1e-9)

    return {
        "symbol": symbol,
        "side": side,
        "tier": tier,
        "price": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "runner_target": round(runner_target, 2),
        "expansion_target": round(expansion_10pct_target, 2),
        "remaining_room_pct": remaining_room_to_12pct,
        "reward_risk": round(reward_risk, 2),
        "score": int(min(100, max(0, score))),
        "rsi": round(rsi_value, 1),
        "mfi": round(mfi_value, 1),
        "adx": round(adx_val, 1),
        "bb_bandwidth_pct": round(bb_bw, 2),
        "relative_volume": round(relvol, 2),
        "cumulative_relative_volume": round(cumulative_relvol, 2),
        "volume_sessions_compared": int(sessions_compared),
        "avg_daily_turnover_crore": round(float(daily.get("avg_turnover", 0.0)) / 1e7, 2),
        "avg_daily_range_pct": round(float(daily.get("avg_daily_range_pct", 2.8)), 2),
        "atr": round(atr_value, 2),
        "vwap": round(vwap_value, 2),
        "breakout_level": round(breakout_level, 2),
        "day_move_pct": round(day_move, 2),
        "candle_time": last_idx.isoformat(),
        "body_ratio": round(body_ratio, 2),
        "pillars_matched": pillars_matched,
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

    # CRITICAL PRE-EXPLOSION FILTER:
    # Reject stocks that have ALREADY moved > 5.0% ("already_moved_too_far")!
    # We only want stocks in the early ignition phase (0.4% to 5.0% move) that still have 10-15% room ahead.
    day_move, open_move, session_high, session_low = compute_effective_day_move(df, last_idx, daily, price)
    if abs(day_move) > CFG["MAX_DAY_MOVE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "already_moved_too_far"
            diagnostic["day_move_pct"] = round(day_move, 2)
        return None
    if abs(day_move) < CFG["MIN_DAY_MOVE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "no_initial_ignition"
            diagnostic["day_move_pct"] = round(day_move, 2)
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

    # Volume accumulation / ignition check
    vol_pass = (
        (relvol >= CFG["MIN_RELATIVE_VOLUME"] and cumulative_relvol >= CFG["MIN_CUMULATIVE_RVOL"])
        or (relvol >= 1.85 and cumulative_relvol >= 1.00)
        or (cumulative_relvol >= 1.65 and relvol >= 0.90)
    )
    if not vol_pass:
        if diagnostic is not None:
            diagnostic["rejected"] = "relative_volume"
        return None

    closes = df["close"].astype(float).values
    fast = ema(closes, 9)
    slow = ema(closes, 21)
    ema50 = ema(closes, 50) if len(closes) >= 50 else slow
    rsi_value = rsi(closes, 14)
    mfi_value = mfi(df, 14)
    atr_value = atr(df, 14)
    vwap_value = session_vwap(df)
    adx_val, plus_di, minus_di, adx_rising = adx_di_slope(df, 14)
    squeeze_coil, bb_buy, bb_sell, bb_bw = bollinger_squeeze_ignition(closes, 20, 2.0)
    macd_buy, macd_sell, _ = macd_impulse(closes)

    if not np.isfinite([fast, slow, rsi_value, atr_value, vwap_value]).all() or atr_value <= 0:
        if diagnostic is not None:
            diagnostic["rejected"] = "indicator_invalid"
        return None

    # Must not be overextended from Session VWAP (enter near institutional VWAP base before the 10-15% run)
    vwap_dist_pct = abs((price - vwap_value) / max(vwap_value, 1e-9)) * 100.0
    if vwap_dist_pct > CFG["MAX_VWAP_DISTANCE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "vwap_overextended"
        return None

    prev = df.iloc[-(CFG["BREAKOUT_LOOKBACK"] + 1):-1]
    if len(prev) < CFG["BREAKOUT_LOOKBACK"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "breakout_history"
        return None
    prior_high = float(prev["high"].max())
    prior_low = float(prev["low"].min())

    # Multi-session 3-day high/low channel (from fmzquant 5-day-High-Low-Breakout-Price-Channel-Strategy)
    prior_days_df = df[[x.date() < last_idx.date() for x in df.index]].tail(225)
    multi_day_high = float(prior_days_df["high"].max()) if not prior_days_df.empty else prior_high
    multi_day_low = float(prior_days_df["low"].min()) if not prior_days_df.empty else prior_low

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

    # Evaluate the 5 Quantitative Strategy Pillars from fmzquant/strategies:
    # Pillar 1: Adaptive Bollinger Squeeze & ATR Channel Ignition
    atr_chan_buy = price >= (slow + 1.0 * atr_value)
    atr_chan_sell = price <= (slow - 1.0 * atr_value)
    p1_buy = bb_buy or (squeeze_coil and atr_chan_buy)
    p1_sell = bb_sell or (squeeze_coil and atr_chan_sell)

    # Pillar 2: Multi-EMA + Session VWAP + Money Flow Index (MFI) Institutional Accumulation
    p2_buy = price > vwap_value and fast > slow and slow >= ema50 * 0.997 and mfi_value >= CFG["MIN_MFI_BUY"]
    p2_sell = price < vwap_value and fast < slow and slow <= ema50 * 1.003 and mfi_value <= CFG["MAX_MFI_SELL"]

    # Pillar 3: ADX + DI Directional Trend Acceleration
    p3_buy = adx_val >= CFG["MIN_ADX_TREND"] and plus_di > minus_di and adx_rising
    p3_sell = adx_val >= CFG["MIN_ADX_TREND"] and minus_di > plus_di and adx_rising

    # Pillar 4: MACD Histogram Impulse + RSI Sweet-Spot Ignition (not overbought/oversold yet)
    p4_buy = macd_buy and (CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"])
    p4_sell = macd_sell and (CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"])

    # Pillar 5: Consolidation Coil / Multi-Day Channel Breakout with Volume Surge
    p5_buy = (price >= prior_high * 0.999 or price >= multi_day_high * 0.998 or price >= session_high * 0.995) and relvol >= 1.35
    p5_sell = (price <= prior_low * 1.001 or price <= multi_day_low * 1.002 or price <= session_low * 1.005) and relvol >= 1.35

    buy_pillars: List[str] = []
    if p1_buy:
        buy_pillars.append("BB-Squeeze/ATR")
    if p2_buy:
        buy_pillars.append("EMA-VWAP-MFI")
    if p3_buy:
        buy_pillars.append("ADX-DI-Surge")
    if p4_buy:
        buy_pillars.append("MACD-RSI-Impulse")
    if p5_buy:
        buy_pillars.append("Channel-Breakout")

    sell_pillars: List[str] = []
    if p1_sell:
        sell_pillars.append("BB-Squeeze/ATR")
    if p2_sell:
        sell_pillars.append("EMA-VWAP-MFI")
    if p3_sell:
        sell_pillars.append("ADX-DI-Surge")
    if p4_sell:
        sell_pillars.append("MACD-RSI-Impulse")
    if p5_sell:
        sell_pillars.append("Channel-Breakdown")

    daily_ema20 = float(daily.get("daily_ema20", price))
    daily_ema50 = float(daily.get("daily_ema50", price))
    daily_up = price >= daily_ema20 * 0.99
    daily_down = price <= daily_ema20 * 1.01

    req_pillars = CFG["MIN_CONFLUENCE_PILLARS"] if not allow_tier2 else max(3, CFG["MIN_CONFLUENCE_PILLARS"] - 1)

    side = None
    breakout_level = None
    matched_pillars: List[str] = []
    tier = "TIER-1"
    setup_label = "Pre-Explosion Ignition: fmzquant 5-Pillar Confluence + Kronos"

    if (
        daily_up
        and price > vwap_value
        and fast > slow
        and bullish_candle
        and CFG["MIN_DAY_MOVE_PCT"] <= day_move <= CFG["MAX_DAY_MOVE_PCT"]
        and open_move >= 0.25
        and len(buy_pillars) >= req_pillars
    ):
        extension = max(0.0, (price - prior_high) / atr_value)
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side = "BUY"
            breakout_level = prior_high
            matched_pillars = buy_pillars
            tier = "TIER-1" if len(buy_pillars) >= CFG["MIN_CONFLUENCE_PILLARS"] else "TIER-2"
    elif (
        daily_down
        and price < vwap_value
        and fast < slow
        and bearish_candle
        and -CFG["MAX_DAY_MOVE_PCT"] <= day_move <= -CFG["MIN_DAY_MOVE_PCT"]
        and open_move <= -0.25
        and len(sell_pillars) >= req_pillars
    ):
        extension = max(0.0, (prior_low - price) / atr_value)
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side = "SELL"
            breakout_level = prior_low
            matched_pillars = sell_pillars
            tier = "TIER-1" if len(sell_pillars) >= CFG["MIN_CONFLUENCE_PILLARS"] else "TIER-2"

    if diagnostic is not None:
        diagnostic["day_move_pct"] = round(day_move, 2)
        diagnostic["buy_pillars"] = buy_pillars
        diagnostic["sell_pillars"] = sell_pillars
        diagnostic["conditions"] = {
            "p1_bb_squeeze": bool(p1_buy or p1_sell),
            "p2_vwap_mfi": bool(p2_buy or p2_sell),
            "p3_adx_di": bool(p3_buy or p3_sell),
            "p4_macd_rsi": bool(p4_buy or p4_sell),
            "p5_channel_break": bool(p5_buy or p5_sell),
            "candle_confirm": bool(bullish_candle or bearish_candle),
        }

    if not side:
        if diagnostic is not None:
            diagnostic["rejected"] = "technical_alignment"
        return None

    # Pre-Explosion Conviction Score (rewards EARLY entries 0.6%-3.6% + high ADR% + multi-pillar confluence)
    adr_pct = float(daily.get("avg_daily_range_pct", 2.8))
    score = 0
    # 1. Early Ignition Bonus (up to 25 pts for catching BEFORE the big move: 0.6% to 3.6%)
    score += 25 if 0.6 <= abs(day_move) <= CFG["IDEAL_MAX_MOVE_PCT"] else 15
    # 2. fmzquant Strategy Pillar Confluence (up to 25 pts)
    score += 25 if len(matched_pillars) == 5 else (20 if len(matched_pillars) == 4 else 14)
    # 3. Relative Volume Ignition (up to 20 pts)
    score += 20 if relvol >= 2.2 else (15 if relvol >= 1.5 else 10)
    # 4. Money Flow & ADX Acceleration (up to 15 pts)
    score += 15 if (mfi_value >= 62 if side == "BUY" else mfi_value <= 38) and adx_val >= 24 else 10
    # 5. High-Beta / Multi-Day Breakout Potential (up to 15 pts)
    score += 15 if (adr_pct >= 3.0 or (price >= multi_day_high if side == "BUY" else price <= multi_day_low)) else 9

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
        mfi_value=mfi_value,
        adx_val=adx_val,
        bb_bw=bb_bw,
        relvol=relvol,
        cumulative_relvol=cumulative_relvol,
        sessions_compared=sessions_compared,
        daily=daily,
        vwap_value=vwap_value,
        breakout_level=breakout_level,
        day_move=day_move,
        last_idx=last_idx,
        body_ratio=body_ratio,
        setup_label=f"{setup_label} ({len(matched_pillars)}/5 Pillars)",
        tier=tier,
        pillars_matched=matched_pillars,
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
        T=0.5,
        top_p=0.85,
        sample_count=CFG["KRONOS_SAMPLE_COUNT"],
    )
    current_close = float(df["close"].iloc[-1])
    predicted_close = float(prediction["close"].iloc[-1])
    pred_high = float(prediction["high"].max()) if "high" in prediction else float(prediction["close"].max())
    pred_low = float(prediction["low"].min()) if "low" in prediction else float(prediction["close"].min())

    change_pct = ((predicted_close / current_close) - 1.0) * 100.0
    peak_move_pct = (
        ((pred_high / current_close) - 1.0) * 100.0
        if side == "BUY"
        else ((pred_low / current_close) - 1.0) * 100.0
    )
    aligned = (
        (change_pct >= CFG["KRONOS_MIN_MOVE_PCT"] or peak_move_pct >= CFG["KRONOS_MIN_MOVE_PCT"] * 2.5)
        if side == "BUY"
        else (change_pct <= -CFG["KRONOS_MIN_MOVE_PCT"] or peak_move_pct <= -CFG["KRONOS_MIN_MOVE_PCT"] * 2.5)
    )
    return {
        "model": "Kronos-mini",
        "horizon_bars": CFG["KRONOS_PRED_LEN"],
        "forecast_close": round(predicted_close, 2),
        "forecast_change_pct": round(change_pct, 3),
        "forecast_peak_move_pct": round(peak_move_pct, 3),
        "direction_aligned": bool(aligned),
    }


def add_signal(payload: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    day = now.strftime("%Y-%m-%d")
    dedupe = f"{day}|{payload['symbol']}"
    _, used_in_win, win_limit, _ = get_window_signal_counts(now)
    if used_in_win >= win_limit:
        return False
    with db_connect() as conn:
        try:
            conn.execute(
                "INSERT INTO signals(created_at,symbol,side,payload,dedupe_key,delivered) VALUES(?,?,?,?,?,0)",
                (now.isoformat(), payload["symbol"], payload["side"], json.dumps(payload), dedupe),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def format_alert_message(payload: Dict[str, Any]) -> str:
    forecast = payload.get("forecast") or {}
    if forecast.get("forecast_change_pct") is not None:
        aligned_tag = "✅ CONFIRMED" if forecast.get("direction_aligned") else "⚠️ neutral"
        peak_str = f" | Peak 30m: {forecast.get('forecast_peak_move_pct', 0.0):+.2f}%"
        forecast_line = (
            f"Kronos-mini ({forecast.get('horizon_bars')} bars): "
            f"{forecast.get('forecast_change_pct'):+.2f}%{peak_str} ({aligned_tag})"
        )
    else:
        forecast_line = "Kronos-mini: technical-confluence mode"

    pillars = ", ".join(payload.get("pillars_matched") or [])
    exp_str = f" | 10% Expansion: ₹{payload['expansion_target']}" if payload.get("expansion_target") else ""

    return (
        f"🚀 BHARAT PRE-EXPLOSION SCOUT 2.0 [{payload.get('tier', 'TIER-1')}]\n"
        f"{'🟢 EARLY BUY IGNITION' if payload['side'] == 'BUY' else '🔴 EARLY SELL IGNITION'} — {payload['symbol']}\n"
        f"Current Day Move: {payload['day_move_pct']:+.2f}% (Remaining 10-15% Runway: ~{payload.get('remaining_room_pct', 10.0)}%)\n"
        f"Entry reference: ₹{payload['price']} (VWAP: ₹{payload['vwap']})\n"
        f"Stop: ₹{payload['stop']} | T1 (2.5R): ₹{payload['target']} | T2 (5R): ₹{payload['runner_target']}{exp_str}\n"
        f"Conviction Score: {payload['score']}/100 | Planned R:R: {payload['reward_risk']}R\n"
        f"5m RVOL: {payload['relative_volume']}x | Session RVOL: {payload['cumulative_relative_volume']}x\n"
        f"RSI: {payload['rsi']} | MFI: {payload.get('mfi', 'N/A')} | ADX: {payload.get('adx', 'N/A')} | BB Width: {payload.get('bb_bandwidth_pct', 'N/A')}%\n"
        f"Avg Turnover: ₹{payload['avg_daily_turnover_crore']} Cr | 20d Avg Range: {payload.get('avg_daily_range_pct', 'N/A')}%\n"
        f"{forecast_line}\n"
        f"fmzquant Pillars: {pillars}\n"
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


def deliver_pending_signals(limit: int = 10) -> int:
    """Deliver queued alerts immediately, then mark them delivered in SQLite."""
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


def rank_pre_explosion_candidates(
    intraday_map: Dict[str, pd.DataFrame],
    daily_map: Dict[str, Dict[str, Any]],
    now: datetime,
) -> List[Dict[str, Any]]:
    """Rank stocks that are in the EARLY ignition zone (0.4% to 5.0% move) with surging volume."""
    ignition_list: List[Dict[str, Any]] = []
    for symbol, raw_df in intraday_map.items():
        if raw_df is None or raw_df.empty:
            continue
        df = completed_bars(normalize_intraday(raw_df), now)
        if len(df) < CFG["MIN_HISTORY_BARS"]:
            continue
        last_idx = df.index[-1]
        price = float(df.iloc[-1]["close"])
        day_move, _, _, _ = compute_effective_day_move(df, last_idx, daily_map.get(symbol, {}), price)
        if CFG["MIN_DAY_MOVE_PCT"] <= abs(day_move) <= CFG["MAX_DAY_MOVE_PCT"]:
            relvol, _ = same_slot_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
            if relvol >= 1.25:
                ignition_list.append({
                    "symbol": symbol,
                    "day_move_pct": round(day_move, 2),
                    "rvol": round(relvol, 2),
                })
    ignition_list.sort(key=lambda x: x["rvol"], reverse=True)
    return ignition_list


def run_scan_cycle() -> Dict[str, Any]:
    started = time.time()
    now = datetime.now(TZ)
    if not is_market_session(now):
        return {"status": "outside_market_hours", "at": now.isoformat()}

    delivered_prior = deliver_pending_signals(limit=10)
    win_name, used_in_win, win_limit, existing_today = get_window_signal_counts(now)

    if used_in_win >= win_limit:
        LOG.info("Window %s quota full (%d/%d); continuing loop for next window.", win_name, used_in_win, win_limit)
        return {
            "status": "window_quota_full_waiting_next_window",
            "at": now.isoformat(),
            "time_window": win_name,
            "window_signals": f"{used_in_win}/{win_limit}",
            "signals_today_total": existing_today,
            "telegram_delivered": delivered_prior,
        }

    if not is_entry_window(now):
        return {
            "status": "past_last_entry_window",
            "at": now.isoformat(),
            "signals_today_total": existing_today,
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
    early_ignitions = rank_pre_explosion_candidates(intraday_all, daily_map, now)

    candidates: List[Tuple[int, float, Dict[str, Any], pd.DataFrame]] = []

    # Pass 1: Scan all 200 stocks for Early Pre-Explosion Ignition (4+ of 5 fmzquant pillars, day move <= 5.0%)
    for symbol in symbols:
        if symbol in alerted_today:
            continue
        frame = intraday_all.get(symbol, pd.DataFrame())
        signal = evaluate_symbol(symbol, frame, daily_map[symbol], now, allow_tier2=False)
        if signal:
            candidates.append((signal["score"], signal["relative_volume"], signal, frame))

    # Pass 2: Tier-2 (3 of 5 fmzquant pillars) on top early volume-ignition stocks if Pass 1 had 0 matches
    tier2_triggered = False
    if not candidates and is_tier2_window(now):
        tier2_triggered = True
        for item in early_ignitions[:15]:
            symbol = item["symbol"]
            if symbol in alerted_today:
                continue
            frame = intraday_all.get(symbol, pd.DataFrame())
            signal = evaluate_symbol(symbol, frame, daily_map[symbol], now, allow_tier2=True)
            if signal:
                candidates.append((signal["score"], signal["relative_volume"], signal, frame))

    # Rank by highest Confluence Score, then highest Relative Volume Surge (NOT highest prior move!)
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    technical_candidates = len(candidates)
    kronos_rejected = 0
    inserted = 0
    remaining_window_slots = max(0, win_limit - used_in_win)
    cycle_limit = min(CFG["MAX_SIGNALS_PER_CYCLE"], remaining_window_slots)

    # Evaluate top candidates through Kronos-mini 6-bar forward trajectory predictor
    for _, _, signal, candles in candidates[:max(8, cycle_limit * 3)]:
        if inserted >= cycle_limit:
            break
        if CFG["KRONOS_ENABLED"] and kronos_predictor is not None:
            try:
                forecast = kronos_forecast(signal["symbol"], candles, signal["side"])
                signal["forecast"] = forecast
                if forecast["direction_aligned"]:
                    signal["score"] = min(100, signal["score"] + 6)
                elif CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue
            except Exception as exc:
                signal["forecast"] = {"status": "unavailable", "detail": str(exc)[:180]}
                if CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue

        if add_signal(signal, now=now):
            inserted += 1

    delivered = delivered_prior + deliver_pending_signals(limit=10)
    if inserted > 0 or delivered > 0:
        persist_state_to_git()

    summary = {
        "status": "complete",
        "at": now.isoformat(),
        "time_window": win_name,
        "window_signals": f"{used_in_win + inserted}/{win_limit}",
        "symbols_considered": len(symbols),
        "top_early_ignition_watchlist": early_ignitions[:6],
        "technical_candidates": technical_candidates,
        "tier2_fallback_evaluated": tier2_triggered,
        "kronos_rejected": kronos_rejected,
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
    """Calculate sleep seconds until 12 seconds after the next 5-minute boundary."""
    now = now or datetime.now(TZ)
    minute_mod = now.minute % 5
    mins_to_add = 5 - minute_mod
    next_slot = (now + timedelta(minutes=mins_to_add)).replace(second=12, microsecond=0)
    wait_sec = (next_slot - now).total_seconds()
    if wait_sec < 15:
        wait_sec += 300.0
    return wait_sec


def run_continuous_market_session() -> Dict[str, Any]:
    """Run 5-minute scan cycles continuously all day until 15:20 IST without stopping early."""
    session_start = time.monotonic()
    max_seconds = CFG["CONTINUOUS_LOOP_MINUTES"] * 60.0
    now = datetime.now(TZ)

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

    cycles_run = 0
    last_result: Dict[str, Any] = {"status": "idle"}

    while True:
        now = datetime.now(TZ)
        if not is_entry_window(now):
            LOG.info("Reached end of intraday entry window (%s IST); stopping session loop.", now.strftime("%H:%M:%S"))
            break

        cycles_run += 1
        LOG.info("=== AUTO-SCAN CYCLE #%d at %s IST ===", cycles_run, now.strftime("%H:%M:%S"))
        try:
            last_result = run_scan_cycle()
        except Exception as exc:
            LOG.exception("Scan cycle #%d encountered transient error (continuing loop): %s", cycles_run, exc)
            last_result = {"status": "transient_error", "detail": str(exc)[:200]}

        elapsed = time.monotonic() - session_start
        wait_sec = seconds_until_next_5m_slot(datetime.now(TZ))
        if elapsed + wait_sec >= max_seconds:
            LOG.info("Session loop window (%.0fm) completed after %d cycles.", CFG["CONTINUOUS_LOOP_MINUTES"], cycles_run)
            break

        LOG.info("Sleeping %.0fs until next 5-minute candle close...", wait_sec)
        time.sleep(wait_sec)

    last_result["cycles_run"] = cycles_run
    return last_result


def run_dry_run_v2() -> Dict[str, Any]:
    """Historical full-session diagnostic replay for Pre-Explosion Ignition + Kronos."""
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
        session_end_high = float(today["high"].max())
        session_end_low = float(today["low"].min())
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
                # Calculate actual post-signal excursion achieved later in the same session!
                future_bars = today.loc[today.index > candle_start]
                if not future_bars.empty:
                    if signal["side"] == "BUY":
                        max_after = float(future_bars["high"].max())
                        signal["actual_move_after_signal_pct"] = round(((max_after / signal["price"]) - 1.0) * 100.0, 2)
                    else:
                        min_after = float(future_bars["low"].min())
                        signal["actual_move_after_signal_pct"] = round(((signal["price"] - min_after) / signal["price"]) * 100.0, 2)
                else:
                    signal["actual_move_after_signal_pct"] = 0.0
                candidates.append((signal["score"], signal["relative_volume"], signal, prefix, anchor))
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

    # Filter through Kronos-mini on top unique picks
    kronos_confirmed_picks = []
    for _, _, signal, prefix, anchor in unique_daily_picks[:15]:
        item = dict(signal)
        if kronos_predictor is not None:
            try:
                fc = kronos_forecast(signal["symbol"], prefix, signal["side"], as_of=anchor)
                item["forecast"] = fc
                if fc["direction_aligned"] or not CFG["KRONOS_REQUIRED"]:
                    kronos_confirmed_picks.append(item)
            except Exception as exc:
                item["forecast"] = {"status": "error", "detail": str(exc)[:250]}
        else:
            kronos_confirmed_picks.append(item)

    report = {
        "mode": "DRY_RUN_V2_PRE_EXPLOSION_IGNITION",
        "session_date_ist": str(replay_day),
        "symbols_requested": len(entries),
        "symbols_with_replay_session": symbols_with_session,
        "candles_evaluated": replayed_bars,
        "primary_rejection_counts": dict(rejection.most_common()),
        "alignment_condition_failures": dict(conditions_failed.most_common()),
        "unique_symbol_candidates": len(unique_daily_picks),
        "kronos_confirmed_pre_explosion_trades": kronos_confirmed_picks[:8],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    path = ROOT / "dry_run_v2_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOG.info("DRY RUN V2 SUMMARY %s", json.dumps({
        k: v for k, v in report.items()
        if k in (
            "session_date_ist", "symbols_requested", "candles_evaluated",
            "primary_rejection_counts", "unique_symbol_candidates",
            "kronos_confirmed_pre_explosion_trades", "elapsed_seconds",
        )
    }, default=str))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bharat Intraday Scout 2.0 Pre-Explosion Ignition Runner")
    parser.add_argument(
        "--task",
        choices=["scan", "refresh-universe", "test-telegram", "test-kronos", "dry-run-v2", "reset-today-quota"],
        default="scan",
    )
    args = parser.parse_args()
    if args.task == "dry-run-v2":
        load_kronos()
        run_dry_run_v2()
        return 0

    init_db()

    if args.task == "reset-today-quota":
        day = datetime.now(TZ).strftime("%Y-%m-%d")
        with db_connect() as conn:
            conn.execute("DELETE FROM signals WHERE created_at LIKE ?", (day + "%",))
            conn.commit()
        persist_state_to_git()
        LOG.info("Cleared today's signal counter (%s); scanner will now resume alerting today.", day)
        return 0

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
        send_telegram_message("✅ Bharat Intraday Scout 2.0 Pre-Explosion Ignition + Kronos Engine active.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
