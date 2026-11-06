from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sqlite3
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


def _env_time(name: str, default: str) -> dtime:
    h, m = os.getenv(name, default).split(":")
    return dtime(int(h), int(m))


# Strategy: "Top-Mover Momentum". Every scan looks at the whole liquid NSE universe,
# finds stocks that are already moving >= MOVER_MIN_PCT from the previous close, and
# alerts only when one of them also shows a confirmed intraday continuation setup.
CFG: Dict[str, Any] = {
    # ---- universe / data ----
    "PRICE_FLOOR": float(os.getenv("PRICE_FLOOR", "50")),
    "MIN_AVG_TURNOVER_INR": float(os.getenv("MIN_AVG_TURNOVER_INR", "50000000")),   # Rs 5 crore, 20-session avg
    "LIQUIDITY_LOOKBACK_DAYS": 20,
    "DAILY_HISTORY_PERIOD": os.getenv("DAILY_HISTORY_PERIOD", "3mo"),
    "MAX_EQUITY_UNIVERSE": int(os.getenv("MAX_EQUITY_UNIVERSE", "2500")),
    "INTRADAY_UNIVERSE_SIZE": int(os.getenv("INTRADAY_UNIVERSE_SIZE", "600")),
    "INTRADAY_BATCH_SIZE": int(os.getenv("INTRADAY_BATCH_SIZE", "100")),
    "INTRADAY_PERIOD": "10d",
    "BAR_INTERVAL": "5m",
    "MIN_HISTORY_BARS": 60,
    "MIN_TIME_SLOT_SESSIONS": 4,
    "MAX_BAR_AGE_MIN": 14,
    # ---- stage 1: mover screen ----
    "MOVER_MIN_PCT": float(os.getenv("MOVER_MIN_PCT", "10")),
    "MOVER_RELAXED_PCT": float(os.getenv("MOVER_RELAXED_PCT", "7")),
    "RELAX_AFTER": _env_time("RELAX_AFTER", "11:30"),
    "SHORTLIST_MARGIN_PCT": 2.0,
    "MAX_SHORTLIST": 40,
    # ---- stage 2: confirmation ----
    "MIN_CUM_RELVOL": float(os.getenv("MIN_CUM_RELVOL", "2.0")),
    "MIN_SLOT_RELVOL": float(os.getenv("MIN_SLOT_RELVOL", "0.7")),
    "MIN_TODAY_TURNOVER_INR": float(os.getenv("MIN_TODAY_TURNOVER_INR", "100000000")),
    "BREAKOUT_LOOKBACK": 12,
    "MAX_EXTENSION_ATR": 1.5,
    "MAX_VWAP_STRETCH_PCT": 6.0,
    "MAX_STOP_PCT": 3.0,
    "MIN_RSI_BUY": 55.0, "MAX_RSI_BUY": 90.0,
    "MIN_RSI_SELL": 10.0, "MAX_RSI_SELL": 45.0,
    "MIN_BODY_RATIO": 0.25,
    "MIN_CLOSE_LOCATION_BUY": 0.65,
    "MAX_CLOSE_LOCATION_SELL": 0.35,
    "STOP_ATR_MULTIPLE": 1.5,
    "TARGET_R_MULTIPLE": 2.0,
    "MIN_SCORE": int(os.getenv("MIN_SCORE", "60")),
    # ---- timing / limits ----
    "ENTRY_START": _env_time("ENTRY_START", "09:45"),
    "ENTRY_END": _env_time("ENTRY_END", "14:45"),
    "DIGEST_AFTER": _env_time("DIGEST_AFTER", "14:30"),
    "MAX_SIGNALS_PER_CYCLE": 1,
    "MAX_SIGNALS_PER_DAY": int(os.getenv("MAX_SIGNALS_PER_DAY", "2")),
    # ---- Kronos (optional, advisory unless KRONOS_REQUIRED=true) ----
    "KRONOS_ENABLED": os.getenv("KRONOS_ENABLED", "false").lower() == "true",
    "KRONOS_REQUIRED": os.getenv("KRONOS_REQUIRED", "false").lower() == "true",
    "KRONOS_MODEL": os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-mini"),
    "KRONOS_TOKENIZER": os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-2k"),
    "KRONOS_LOOKBACK": int(os.getenv("KRONOS_LOOKBACK", "96")),
    "KRONOS_PRED_LEN": int(os.getenv("KRONOS_PRED_LEN", "3")),
    "KRONOS_SAMPLE_COUNT": int(os.getenv("KRONOS_SAMPLE_COUNT", "1")),
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


def state_get(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_connect() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def state_set(key: str, value: str) -> None:
    with db_connect() as conn:
        conn.execute("INSERT INTO app_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        conn.commit()


def load_kronos() -> None:
    """Load the real Kronos-mini model on CPU from the shallow-cloned repo."""
    global kronos_predictor, kronos_status
    if not CFG["KRONOS_ENABLED"]:
        kronos_status = "disabled"
        if CFG["KRONOS_REQUIRED"]:
            raise RuntimeError("KRONOS_REQUIRED=true but KRONOS_ENABLED=false")
        return
    if not KRONOS_REPO_DIR.exists():
        raise RuntimeError(f"Kronos source repository not found: {KRONOS_REPO_DIR}")
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
        LOG.exception("Kronos initialization failed")
        if CFG["KRONOS_REQUIRED"]:
            raise RuntimeError("Kronos is required but failed to load: " + str(exc)) from exc


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
    """Build candidate pool from all NSE EQ symbols using 20-session turnover.

    Run weekly; the result is cached on disk and committed back to the repo.
    """
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
            if last_price >= CFG["PRICE_FLOOR"] and avg_turnover >= CFG["MIN_AVG_TURNOVER_INR"]:
                metrics.append({
                    "symbol": symbol,
                    "price": last_price,
                    "avg_turnover": avg_turnover,
                    "daily_ema20": e20,
                    "daily_ema50": e50,
                    "previous_close": prev_close,
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

    result = {
        "refreshed_at": datetime.now(TZ).isoformat(),
        "source_symbol_count": len(symbols),
        "eligible_count": len(metrics),
        "scan_symbols": metrics[:CFG["INTRADAY_UNIVERSE_SIZE"]],
    }
    UNIVERSE_CACHE.write_text(json.dumps(result, indent=2))
    LOG.info("Liquid universe refreshed: %d eligible; scanning top %d by average turnover", len(metrics), len(result["scan_symbols"]))
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
    # Yahoo timestamps label the start of a 5-minute candle; discard the candle
    # until its full five minutes have elapsed.
    cutoff = pd.Timestamp(now - timedelta(minutes=5)).tz_convert(TZ)
    return df.loc[df.index <= cutoff].copy()


def same_slot_relative_volume(df: pd.DataFrame, last_idx: pd.Timestamp, min_sessions: int) -> Tuple[float, int]:
    slot = last_idx.strftime("%H:%M")
    today = last_idx.date()
    same_slot = df[df.index.strftime("%H:%M") == slot]
    previous = same_slot[[x.date() < today for x in same_slot.index]].tail(5)
    vals = pd.to_numeric(previous["volume"], errors="coerce").dropna()
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
            prior_totals.append(float(session["volume"].astype(float).sum()))
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


# ---------------------------------------------------------------------------
# Stage 1: find today's movers from the whole liquid universe (cheap daily bars)
# ---------------------------------------------------------------------------
def scan_window_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return CFG["ENTRY_START"] <= now.time() <= CFG["ENTRY_END"]


def current_threshold(now: datetime, sent_today: int) -> Tuple[float, bool]:
    """10% by default; drops to the relaxed level after 11:30 IST if nothing was sent yet."""
    if sent_today == 0 and now.time() >= CFG["RELAX_AFTER"]:
        return CFG["MOVER_RELAXED_PCT"], True
    return CFG["MOVER_MIN_PCT"], False


def find_movers(entries: List[Dict[str, Any]], now: datetime, threshold: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return today's biggest movers vs previous close, plus data-coverage stats."""
    symbols = [e["symbol"] for e in entries]
    gate = max(threshold - CFG["SHORTLIST_MARGIN_PCT"], 3.0)
    movers: List[Dict[str, Any]] = []
    with_data = 0
    for start in range(0, len(symbols), CFG["INTRADAY_BATCH_SIZE"]):
        group = symbols[start:start + CFG["INTRADAY_BATCH_SIZE"]]
        data = download_batch(group, "5d", "1d")
        for sym in group:
            df = data.get(sym, pd.DataFrame())
            if df is None or df.empty or len(df) < 2:
                continue
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert(TZ).tz_localize(None)
            if idx[-1].date() != now.date():
                continue          # no live daily bar for today (holiday / stale feed)
            with_data += 1
            close = pd.to_numeric(df["close"], errors="coerce")
            prev_close, last = float(close.iloc[-2]), float(close.iloc[-1])
            if not np.isfinite([prev_close, last]).all() or prev_close <= 0:
                continue
            chg = (last / prev_close - 1.0) * 100.0
            if abs(chg) >= gate:
                movers.append({"symbol": sym, "change_pct": round(chg, 2), "prev_close": prev_close, "last": last})
        time.sleep(0.1)
    movers.sort(key=lambda m: abs(m["change_pct"]), reverse=True)
    stats = {"symbols": len(symbols), "with_live_daily_bar": with_data,
             "coverage_pct": round(100.0 * with_data / max(len(symbols), 1), 1)}
    return movers[:CFG["MAX_SHORTLIST"]], stats


# ---------------------------------------------------------------------------
# Stage 2: confirmation on 5-minute candles
# ---------------------------------------------------------------------------
def evaluate_mover(symbol: str, source: pd.DataFrame, entry: Dict[str, Any], now: datetime,
                   threshold: float, prev_close: Optional[float] = None,
                   diagnostic: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    def reject(reason: str) -> None:
        if diagnostic is not None:
            diagnostic["rejected"] = reason
        return None

    if source is None or source.empty:
        return reject("no_candles")
    df = completed_bars(normalize_intraday(source), now)
    if len(df) < CFG["MIN_HISTORY_BARS"]:
        return reject("insufficient_history")
    last_idx = df.index[-1]
    age = (pd.Timestamp(now).tz_convert(TZ) - last_idx).total_seconds() / 60.0
    if age < 5 or age > CFG["MAX_BAR_AGE_MIN"]:
        return reject("stale_candle")
    last = df.iloc[-1]
    price = float(last["close"])
    if price < CFG["PRICE_FLOOR"]:
        return reject("price_floor")

    today_mask = np.array([x.date() == last_idx.date() for x in df.index])
    day = df[today_mask]
    if len(day) < 4:
        return reject("too_early")
    if prev_close is None:
        prior = df[~today_mask]
        if prior.empty:
            return reject("no_prev_close")
        prev_close = float(prior["close"].iloc[-1])
    day_change = (price / prev_close - 1.0) * 100.0
    if abs(day_change) < threshold:
        return reject("move_below_threshold")
    side = "BUY" if day_change > 0 else "SELL"

    # Circuit-locked stocks cannot actually be traded: last 3 candles with no range.
    tail = day.tail(3)
    if len(tail) == 3 and bool(((tail["high"] - tail["low"]).abs() < 1e-9).all()):
        return reject("circuit_locked")

    day_turnover = float((day["close"].astype(float) * day["volume"].astype(float)).sum())
    if day_turnover < CFG["MIN_TODAY_TURNOVER_INR"]:
        return reject("low_turnover_today")

    relvol, slot_n = same_slot_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    cum_relvol, cum_n = session_cumulative_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    if cum_relvol < CFG["MIN_CUM_RELVOL"] or relvol < CFG["MIN_SLOT_RELVOL"]:
        return reject("relative_volume")

    closes = df["close"].astype(float).values
    fast, slow = ema(closes, 9), ema(closes, 21)
    rsi_v, atr_v, vwap_v = rsi(closes, 14), atr(df, 14), session_vwap(df)
    if not np.isfinite([fast, slow, rsi_v, atr_v, vwap_v]).all() or atr_v <= 0:
        return reject("indicator_invalid")

    prev = df.iloc[-(CFG["BREAKOUT_LOOKBACK"] + 1):-1]
    prior_high, prior_low = float(prev["high"].max()), float(prev["low"].min())
    rng = max(float(last["high"] - last["low"]), 1e-9)
    body = abs(float(last["close"] - last["open"])) / rng
    loc = (float(last["close"]) - float(last["low"])) / rng

    if side == "BUY":
        checks = {
            "trend_ema": fast > slow, "vwap": price > vwap_v, "breakout": price > prior_high,
            "candle": float(last["close"]) > float(last["open"]) and body >= CFG["MIN_BODY_RATIO"] and loc >= CFG["MIN_CLOSE_LOCATION_BUY"],
            "rsi": CFG["MIN_RSI_BUY"] <= rsi_v <= CFG["MAX_RSI_BUY"],
        }
        level = prior_high
        extension = (price - prior_high) / atr_v
        stretch = (price / vwap_v - 1.0) * 100.0
    else:
        checks = {
            "trend_ema": fast < slow, "vwap": price < vwap_v, "breakout": price < prior_low,
            "candle": float(last["close"]) < float(last["open"]) and body >= CFG["MIN_BODY_RATIO"] and loc <= CFG["MAX_CLOSE_LOCATION_SELL"],
            "rsi": CFG["MIN_RSI_SELL"] <= rsi_v <= CFG["MAX_RSI_SELL"],
        }
        level = prior_low
        extension = (prior_low - price) / atr_v
        stretch = (vwap_v / price - 1.0) * 100.0
    if diagnostic is not None:
        diagnostic["conditions"] = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        return reject("setup_not_confirmed")
    if extension > CFG["MAX_EXTENSION_ATR"]:
        return reject("extended_chasing")
    if stretch > CFG["MAX_VWAP_STRETCH_PCT"]:
        return reject("too_far_from_vwap")

    risk = CFG["STOP_ATR_MULTIPLE"] * atr_v
    if risk / price * 100.0 > CFG["MAX_STOP_PCT"]:
        return reject("stop_too_wide")
    stop = price - risk if side == "BUY" else price + risk
    target = price + risk * CFG["TARGET_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["TARGET_R_MULTIPLE"]

    score = 0
    score += int(min(abs(day_change), 20.0) / 20.0 * 20)
    score += 20 if cum_relvol >= 5 else 15 if cum_relvol >= 3 else 10
    score += 10 if relvol >= 2 else 5
    score += 10                                   # VWAP aligned (hard gate above)
    score += 10 if body >= 0.6 else 5
    score += 10 if ((rsi_v >= 65) if side == "BUY" else (rsi_v <= 35)) else 5
    score += 10 if extension <= 0.5 else 5
    score += 10 if day_turnover >= 5e8 else 5
    score += 5 if slot_n >= 5 else 2
    if score < CFG["MIN_SCORE"]:
        return reject("score_below_minimum")

    return {
        "symbol": symbol, "side": side, "price": round(price, 2),
        "stop": round(stop, 2), "target": round(target, 2),
        "reward_risk": CFG["TARGET_R_MULTIPLE"], "risk_pct": round(risk / price * 100.0, 2),
        "score": int(score), "rsi": round(rsi_v, 1),
        "day_change_pct": round(day_change, 2), "prev_close": round(prev_close, 2),
        "relative_volume": round(relvol, 2), "cumulative_relative_volume": round(cum_relvol, 2),
        "today_turnover_crore": round(day_turnover / 1e7, 1),
        "avg_daily_turnover_crore": round(float(entry.get("avg_turnover", 0)) / 1e7, 1),
        "atr": round(atr_v, 2), "vwap": round(vwap_v, 2), "breakout_level": round(level, 2),
        "candle_time": last_idx.isoformat(), "body_ratio": round(body, 2),
        "setup": f"Top-mover momentum {'long' if side == 'BUY' else 'short'}: {day_change:+.1f}% on day, {cum_relvol:.1f}x volume, 12-bar breakout above/below VWAP",
        "forecast": None,
    }


def kronos_forecast(symbol: str, raw_df: pd.DataFrame, side: str,
                    as_of: Optional[datetime] = None) -> Dict[str, Any]:
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
    y_index = pd.date_range(start=(last_time + pd.Timedelta(minutes=5)).tz_localize(None), periods=CFG["KRONOS_PRED_LEN"], freq="5min")
    y_timestamp = pd.Series(y_index)
    prediction = kronos_predictor.predict(
        df=x_df.reset_index(drop=True),
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=CFG["KRONOS_PRED_LEN"],
        T=1.0,
        top_p=0.9,
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
        "interpretation": "directional forecast, not probability",
    }


# ---------------------------------------------------------------------------
# Signals, Telegram, scan cycle
# ---------------------------------------------------------------------------
def today_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def signals_sent_today() -> int:
    with db_connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM signals WHERE created_at LIKE ?", (today_str() + "%",)).fetchone()["n"])


def add_signal(payload: Dict[str, Any]) -> bool:
    day = today_str()
    # One idea per symbol per day (no flipping long/short on the same stock).
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


def format_alert_message(p: Dict[str, Any]) -> str:
    f = p.get("forecast") or {}
    fline = f"\nKronos (advisory): {f['forecast_change_pct']}% over {f['horizon_bars']} candles" if f.get("forecast_change_pct") is not None else ""
    side = "🟢 BUY (long)" if p["side"] == "BUY" else "🔴 SELL (short, intraday)"
    return (
        "BHARAT INTRADAY SCOUT — TOP MOVER\n"
        f"{side} — {p['symbol']}  ({p['day_change_pct']:+.1f}% today)\n"
        f"Entry ref: ₹{p['price']}  | Prev close: ₹{p['prev_close']}\n"
        f"Stop: ₹{p['stop']} ({p['risk_pct']}% risk)  | Target: ₹{p['target']} ({p['reward_risk']}R)\n"
        f"Score: {p['score']}/100 (ranking only, not win probability)\n"
        f"Volume: {p['cumulative_relative_volume']}x vs usual by this time; last candle {p['relative_volume']}x\n"
        f"Traded today: ₹{p['today_turnover_crore']} cr | RSI {p['rsi']} | VWAP ₹{p['vwap']}\n"
        f"Breakout level: ₹{p['breakout_level']}{fline}\n"
        "Square off before 3:15 PM. Check live price/spread first (stock may be near circuit). Research alert only; no order placed."
    )


def send_telegram_message(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID GitHub Secret")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError("Telegram API rejected message: " + str(body.get("description", "unknown error")))


def deliver_pending_signals(limit: int = 5) -> int:
    delivered = 0
    with db_connect() as conn:
        rows = conn.execute("SELECT id,payload FROM signals WHERE delivered=0 ORDER BY id ASC LIMIT ?",
                            (max(1, min(int(limit), 20)),)).fetchall()
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
            LOG.exception("Telegram delivery failed for signal id=%s: %s", row["id"], exc)
            break
    return delivered


def once_per_day(key: str) -> bool:
    """True the first time it is called on a given IST day (persisted in SQLite)."""
    day = today_str()
    if state_get(key) == day:
        return False
    state_set(key, day)
    return True


def run_scan_cycle() -> Dict[str, Any]:
    started = time.time()
    now = datetime.now(TZ)
    if not scan_window_open(now):
        return {"status": "outside_entry_window", "at": now.isoformat()}
    if not UNIVERSE_CACHE.exists():
        raise RuntimeError("No liquid universe cache. Run workflow_dispatch with task=refresh-universe first.")
    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    if not entries:
        raise RuntimeError("Liquid universe cache is empty")
    by_symbol = {e["symbol"]: e for e in entries}

    sent_before = signals_sent_today()
    threshold, relaxed = current_threshold(now, sent_before)
    movers, cov = find_movers(entries, now, threshold)
    LOG.info("Movers shortlist: %d | coverage %s | threshold %.1f%% (relaxed=%s)", len(movers), cov, threshold, relaxed)

    # Data-feed health: tell the owner once per day if Yahoo returned too little data.
    if cov["coverage_pct"] < 50 and once_per_day("warned_low_coverage"):
        try:
            send_telegram_message(f"⚠️ Scout: only {cov['coverage_pct']}% of symbols returned live data from Yahoo. Alerts may be missed today.")
        except Exception:
            LOG.exception("coverage warning failed")

    if once_per_day("heartbeat_day"):
        try:
            send_telegram_message(f"✅ Scout is live ({now.strftime('%H:%M')} IST). Movers ≥{threshold:.0f}% so far: "
                                  f"{sum(1 for m in movers if abs(m['change_pct']) >= threshold)}. Data coverage {cov['coverage_pct']}%.")
        except Exception:
            LOG.exception("heartbeat failed")

    rejections: Dict[str, int] = {}
    candidates: List[Tuple[int, Dict[str, Any], pd.DataFrame]] = []
    if movers and sent_before < CFG["MAX_SIGNALS_PER_DAY"]:
        mover_syms = [m["symbol"] for m in movers]
        intraday = download_batch(mover_syms, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"])
        for m in movers:
            diag: Dict[str, Any] = {}
            sig = evaluate_mover(m["symbol"], intraday.get(m["symbol"], pd.DataFrame()), by_symbol[m["symbol"]],
                                 now, threshold, prev_close=m["prev_close"], diagnostic=diag)
            if sig:
                candidates.append((sig["score"], sig, intraday[m["symbol"]]))
            else:
                r = diag.get("rejected", "unknown")
                rejections[r] = rejections.get(r, 0) + 1
    candidates.sort(key=lambda x: x[0], reverse=True)

    inserted = kronos_rejected = 0
    for _, sig, candles in candidates:
        if inserted >= CFG["MAX_SIGNALS_PER_CYCLE"]:
            break
        if CFG["KRONOS_ENABLED"]:
            try:
                fc = kronos_forecast(sig["symbol"], candles, sig["side"])
                sig["forecast"] = fc
                if CFG["KRONOS_REQUIRED"] and not fc["direction_aligned"]:
                    kronos_rejected += 1
                    continue
            except Exception as exc:
                sig["forecast"] = {"status": "unavailable", "detail": str(exc)[:180]}
                if CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue
        if add_signal(sig):
            inserted += 1
    delivered = deliver_pending_signals(limit=5)

    # End-of-window digest so silence is never ambiguous.
    digest_sent = False
    if now.time() >= CFG["DIGEST_AFTER"] and once_per_day("digest_day"):
        sent_now = signals_sent_today()
        top = ", ".join(f"{m['symbol']} {m['change_pct']:+.1f}%" for m in movers[:5]) or "none"
        msg = (f"📋 Scout daily digest: {sent_now} confirmed alert(s) today.\nTop movers now: {top}\n"
               + ("" if sent_now else "No mover passed all confirmation filters (volume/VWAP/breakout/not-extended). Not a malfunction.\n")
               + (f"Rejections: {json.dumps(rejections)}" if rejections else ""))
        try:
            send_telegram_message(msg)
            digest_sent = True
        except Exception:
            LOG.exception("digest failed")

    summary = {
        "status": "complete", "at": now.isoformat(), "threshold_pct": threshold, "relaxed": relaxed,
        "coverage": cov, "movers": [(m["symbol"], m["change_pct"]) for m in movers[:8]],
        "technical_candidates": len(candidates), "rejections": rejections,
        "kronos_rejected": kronos_rejected, "new_signals_queued": inserted,
        "telegram_delivered": delivered, "digest_sent": digest_sent,
        "duration_seconds": round(time.time() - started, 1),
    }
    LOG.info("SCAN SUMMARY %s", json.dumps(summary))
    return summary


def run_dry_run_v2() -> Dict[str, Any]:
    """Replay the latest session on the mover shortlist. No Telegram, no DB writes."""
    from collections import Counter
    started = time.monotonic()
    if not UNIVERSE_CACHE.exists():
        raise RuntimeError("Missing data/liquid_universe.json. Run refresh-universe first.")
    entries = json.loads(UNIVERSE_CACHE.read_text()).get("scan_symbols", [])
    frames: Dict[str, pd.DataFrame] = {}
    for off in range(0, len(entries), CFG["INTRADAY_BATCH_SIZE"]):
        group = [e["symbol"] for e in entries[off:off + CFG["INTRADAY_BATCH_SIZE"]]]
        frames.update(download_batch(group, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"]))
    by_symbol = {e["symbol"]: e for e in entries}
    norm = {}
    for s, f in frames.items():
        if f is not None and not f.empty:
            c = normalize_intraday(f).between_time("09:15", "15:25")
            if not c.empty:
                norm[s] = c
    if not norm:
        raise RuntimeError("Yahoo returned no usable NSE 5-minute candles")
    replay_day = max(c.index[-1].date() for c in norm.values())
    rejection: Counter = Counter()
    hits = []
    movers_final = []
    for s, f in norm.items():
        today = f[[x.date() == replay_day for x in f.index]]
        prior = f[[x.date() < replay_day for x in f.index]]
        if today.empty or prior.empty:
            continue
        prev_close = float(prior["close"].iloc[-1])
        end_chg = (float(today["close"].iloc[-1]) / prev_close - 1) * 100
        if abs(end_chg) < CFG["MOVER_RELAXED_PCT"] - 2:
            continue
        movers_final.append((s, round(end_chg, 2)))
        for ts in today.index:
            anchor = ts.to_pydatetime() + timedelta(minutes=5)
            if not (CFG["ENTRY_START"] <= anchor.time() <= CFG["ENTRY_END"]):
                continue
            prefix = f.loc[f.index <= ts]
            diag: Dict[str, Any] = {}
            sig = evaluate_mover(s, prefix, by_symbol[s], anchor, CFG["MOVER_RELAXED_PCT"], prev_close, diag)
            if sig:
                hits.append(sig); rejection["PASS"] += 1
            else:
                rejection[diag.get("rejected", "unknown")] += 1
    hits.sort(key=lambda x: x["candle_time"])
    first_per_symbol = {}
    for h in hits:
        first_per_symbol.setdefault(h["symbol"], h)
    report = {
        "mode": "DRY_RUN_NO_ALERTS", "session_date_ist": str(replay_day),
        "symbols_with_data": len(norm), "movers_at_close": sorted(movers_final, key=lambda x: -abs(x[1]))[:15],
        "rejection_counts": dict(rejection.most_common()),
        "first_signal_per_symbol": list(first_per_symbol.values())[:10],
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "caveats": ["Single-session replay, not a backtest; no P&L.", "Yahoo data is unofficial.",
                    "Mover set is chosen with end-of-day hindsight, so signal counts are optimistic."],
    }
    (ROOT / "dry_run_v2_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOG.info("DRY RUN SUMMARY %s", json.dumps({k: report[k] for k in ("session_date_ist", "movers_at_close", "rejection_counts")}, default=str))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bharat Intraday Scout 2.1 one-shot runner")
    parser.add_argument("--task", choices=["scan", "refresh-universe", "test-telegram", "test-kronos", "dry-run-v2"], default="scan")
    args = parser.parse_args()
    if args.task == "dry-run-v2":
        run_dry_run_v2()
        return 0
    init_db()
    if args.task == "refresh-universe":
        r = refresh_liquid_universe(force=True)
        LOG.info("UNIVERSE READY | eligible=%s | selected=%s", r.get("eligible_count"), len(r.get("scan_symbols", [])))
        return 0
    if args.task == "test-telegram":
        send_telegram_message("✅ Bharat Intraday Scout Telegram test successful. No trade was placed.")
        return 0
    if args.task == "test-kronos":
        CFG["KRONOS_ENABLED"] = True
        load_kronos()
        if kronos_predictor is None:
            raise RuntimeError("Kronos failed to load: " + kronos_status)
        return 0
    if not scan_window_open(datetime.now(TZ)):
        LOG.info("Outside entry window (%s-%s IST, weekdays); scan skipped", CFG["ENTRY_START"], CFG["ENTRY_END"])
        return 0
    if CFG["KRONOS_ENABLED"]:
        load_kronos()
    result = run_scan_cycle()
    LOG.info("RUN RESULT %s", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
