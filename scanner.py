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

# Selective defaults: broad NSE EQ universe is filtered by 20-session turnover,
# then the most liquid 100 names are checked every five minutes.
CFG: Dict[str, Any] = {
    "PRICE_FLOOR": float(os.getenv("PRICE_FLOOR", "60")),
    "MIN_AVG_TURNOVER_INR": float(os.getenv("MIN_AVG_TURNOVER_INR", "100000000")),
    "LIQUIDITY_LOOKBACK_DAYS": 20,
    "DAILY_HISTORY_PERIOD": os.getenv("DAILY_HISTORY_PERIOD", "3mo"),
    "MAX_EQUITY_UNIVERSE": int(os.getenv("MAX_EQUITY_UNIVERSE", "2500")),
    "INTRADAY_UNIVERSE_SIZE": int(os.getenv("INTRADAY_UNIVERSE_SIZE", "100")),
    "INTRADAY_BATCH_SIZE": int(os.getenv("INTRADAY_BATCH_SIZE", "40")),
    "INTRADAY_PERIOD": "10d",
    "BAR_INTERVAL": "5m",
    "MIN_HISTORY_BARS": 60,
    "MIN_TIME_SLOT_SESSIONS": 5,
    "MIN_RELATIVE_VOLUME": 0.50,
    "BREAKOUT_LOOKBACK": 12,
    "MAX_EXTENSION_ATR": 2,
    "MAX_DAY_MOVE_PCT": 8.0,
    "MAX_BAR_AGE_MIN": 9,
    "MIN_RSI_BUY": 45.0,
    "MAX_RSI_BUY": 80.0,
    "MIN_RSI_SELL": 45.0,
    "MAX_RSI_SELL": 55.0,
    "MIN_BODY_RATIO": 0.20,
    "MIN_CLOSE_LOCATION_BUY": 0.78,
    "MAX_CLOSE_LOCATION_SELL": 0.22,
    "STOP_ATR_MULTIPLE": 1.20,
    "TARGET_R_MULTIPLE": 2.0,
    "MAX_SIGNALS_PER_CYCLE": 2,
    "MAX_SIGNALS_PER_DAY": 3,
    "KRONOS_ENABLED": os.getenv("KRONOS_ENABLED", "true").lower() == "true",
    "KRONOS_REQUIRED": os.getenv("KRONOS_REQUIRED", "true").lower() == "true",
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


def is_market_session(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 20) <= t <= dtime(15, 30)


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
    # Avoid non-stock NSE instrument symbols if present in the file.
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

    Runs once daily; the result is cached on disk. It is intentionally not run
    from every intraday request.
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
            # Use completed prior sessions only. Do not let a partial current-day
            # daily candle distort the average turnover or daily trend.
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
        # Do not destroy a prior valid cache during a Yahoo outage.
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
        # yfinance usually returns an exchange-local timezone for NSE symbols;
        # if timezone metadata is absent, interpret timestamps as Asia/Kolkata.
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


def evaluate_symbol(symbol: str, source: pd.DataFrame, daily: Dict[str, Any], now: datetime,
                    diagnostic: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if source.empty:
        if diagnostic is not None:
            diagnostic["rejected"] = "no_candles"
        return None
    df = normalize_intraday(source)
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

    # Same clock-time volume comparison across the previous five sessions.
    relvol, slot_sessions = same_slot_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    cumulative_relvol, cumulative_sessions = session_cumulative_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    if diagnostic is not None:
        diagnostic["volume"] = {"same_slot": round(relvol, 3), "cumulative": round(cumulative_relvol, 3),
                                "sessions": min(slot_sessions, cumulative_sessions)}
    if relvol < CFG["MIN_RELATIVE_VOLUME"] or cumulative_relvol < 1.25:
        if diagnostic is not None:
            diagnostic["rejected"] = "relative_volume"
        return None

    closes = df["close"].astype(float).values
    fast = ema(closes, 9)
    slow = ema(closes, 21)
    rsi_value = rsi(closes, 14)
    atr_value = atr(df, 14)
    vwap_value = session_vwap(df)
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
    breakout = price > prior_high
    breakdown = price < prior_low

    day_candles = df[[x.date() == last_idx.date() for x in df.index]]
    day_open = float(day_candles.iloc[0]["open"]) if not day_candles.empty else price
    day_move = ((price / day_open) - 1.0) * 100.0 if day_open else 0.0
    if abs(day_move) > CFG["MAX_DAY_MOVE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "day_move"
        return None

    bar_range = max(float(last["high"] - last["low"]), 1e-9)
    body_ratio = abs(float(last["close"] - last["open"])) / bar_range
    close_location = (float(last["close"]) - float(last["low"])) / bar_range
    bullish_candle = float(last["close"]) > float(last["open"]) and body_ratio >= CFG["MIN_BODY_RATIO"] and close_location >= CFG["MIN_CLOSE_LOCATION_BUY"]
    bearish_candle = float(last["close"]) < float(last["open"]) and body_ratio >= CFG["MIN_BODY_RATIO"] and close_location <= CFG["MAX_CLOSE_LOCATION_SELL"]

    daily_up = daily["daily_ema20"] > daily["daily_ema50"] and price > daily["daily_ema20"]
    daily_down = daily["daily_ema20"] < daily["daily_ema50"] and price < daily["daily_ema20"]
    above_vwap = price > vwap_value
    below_vwap = price < vwap_value

    side = None
    breakout_level = None
    if (daily_up and fast > slow and above_vwap and breakout and bullish_candle
            and CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"]):
        extension = (price - prior_high) / atr_value
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side, breakout_level = "BUY", prior_high
    elif (daily_down and fast < slow and below_vwap and breakdown and bearish_candle
            and CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"]):
        extension = (prior_low - price) / atr_value
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side, breakout_level = "SELL", prior_low

    if diagnostic is not None:
        diagnostic["conditions"] = {
            "daily_trend_buy": bool(daily_up), "daily_trend_sell": bool(daily_down),
            "ema_buy": bool(fast > slow), "ema_sell": bool(fast < slow),
            "vwap_buy": bool(above_vwap), "vwap_sell": bool(below_vwap),
            "breakout_buy": bool(breakout), "breakout_sell": bool(breakdown),
            "candle_buy": bool(bullish_candle), "candle_sell": bool(bearish_candle),
            "rsi_buy": bool(CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"]),
            "rsi_sell": bool(CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"]),
        }
        diagnostic["volume"] = {"same_slot": round(relvol, 3), "cumulative": round(cumulative_relvol, 3)}
    if not side:
        if diagnostic is not None:
            diagnostic["rejected"] = "technical_alignment"
        return None

    risk = CFG["STOP_ATR_MULTIPLE"] * atr_value
    stop = price - risk if side == "BUY" else price + risk
    target = price + risk * CFG["TARGET_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["TARGET_R_MULTIPLE"]
    reward_risk = abs(target - price) / max(abs(price - stop), 1e-9)
    if reward_risk < 2.0:
        if diagnostic is not None:
            diagnostic["rejected"] = "risk_reward"
        return None

    # Transparent ranking score, not a predicted probability.
    score = 0
    score += 15 if relvol >= 2.0 else 10
    score += 10 if cumulative_relvol >= 1.50 else 5
    score += 10 if (daily_up if side == "BUY" else daily_down) else 0
    score += 15 if (above_vwap if side == "BUY" else below_vwap) else 0
    score += 15 if body_ratio >= 0.65 else 10
    score += 15 if (rsi_value >= 58 if side == "BUY" else rsi_value <= 42) else 10
    score += 10 if abs((price - breakout_level) / atr_value) <= 0.25 else 5
    score += 5 if slot_sessions >= 5 else 2

    return {
        "symbol": symbol,
        "side": side,
        "price": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": round(reward_risk, 2),
        "score": int(score),
        "rsi": round(rsi_value, 1),
        "relative_volume": round(relvol, 2),
        "cumulative_relative_volume": round(cumulative_relvol, 2),
        "volume_sessions_compared": min(slot_sessions, cumulative_sessions),
        "avg_daily_turnover_crore": round(float(daily["avg_turnover"]) / 1e7, 2),
        "atr": round(atr_value, 2),
        "vwap": round(vwap_value, 2),
        "breakout_level": round(breakout_level, 2),
        "day_move_pct": round(day_move, 2),
        "candle_time": last_idx.isoformat(),
        "body_ratio": round(body_ratio, 2),
        "setup": "5-session relative-volume breakout + multi-timeframe trend",
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


def add_signal(payload: Dict[str, Any]) -> bool:
    day = datetime.now(TZ).strftime("%Y-%m-%d")
    dedupe = f"{day}|{payload['symbol']}|{payload['side']}"
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
    forecast_line = (
        f"Kronos-mini: {forecast.get('forecast_change_pct')}% over {forecast.get('horizon_bars')} candles"
        if forecast.get("forecast_change_pct") is not None
        else "Kronos-mini: unavailable"
    )
    return (
        "BHARAT INTRADAY SCOUT 2.0\n"
        f"{'🟢 BUY' if payload['side'] == 'BUY' else '🔴 SELL'} — {payload['symbol']}\n"
        f"Entry reference: ₹{payload['price']}\n"
        f"Stop reference: ₹{payload['stop']} | Target reference: ₹{payload['target']}\n"
        f"Planned R:R: {payload['reward_risk']}R\n"
        f"Quality score: {payload['score']}/100 (not win probability)\n"
        f"Same-time 5-session 5m RVOL: {payload['relative_volume']}x\n"
        f"Cumulative 5-session RVOL: {payload['cumulative_relative_volume']}x\n"
        f"Average daily turnover: ₹{payload['avg_daily_turnover_crore']} crore\n"
        f"RSI: {payload['rsi']} | VWAP: ₹{payload['vwap']} | ATR: ₹{payload['atr']}\n"
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
            # Stop here; keep this and later signals queued for the next run.
            break
    return delivered


def run_scan_cycle() -> Dict[str, Any]:
    started = time.time()
    now = datetime.now(TZ)
    if not is_market_session(now):
        return {"status": "outside_market_hours", "at": now.isoformat()}
    if not UNIVERSE_CACHE.exists():
        raise RuntimeError("No liquid universe cache. Run workflow_dispatch with task=refresh-universe first.")

    # Read the weekly universe cache only. Never trigger a multi-thousand-stock
    # liquidity refresh inside a five-minute intraday scan.
    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    if not entries:
        raise RuntimeError("Liquid universe cache is empty")

    candidates: List[Tuple[int, Dict[str, Any], pd.DataFrame]] = []
    symbols = [x["symbol"] for x in entries]
    for start in range(0, len(symbols), CFG["INTRADAY_BATCH_SIZE"]):
        group = symbols[start:start + CFG["INTRADAY_BATCH_SIZE"]]
        intraday = download_batch(group, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"])
        daily_map = {x["symbol"]: x for x in entries[start:start + len(group)]}
        for symbol in group:
            signal = evaluate_symbol(symbol, intraday.get(symbol, pd.DataFrame()), daily_map[symbol], now)
            if signal:
                candidates.append((signal["score"], signal, intraday[symbol]))
        time.sleep(0.10)

    candidates.sort(key=lambda x: x[0], reverse=True)
    technical_candidates = len(candidates)
    kronos_rejected = 0
    inserted = 0
    for _, signal, candles in candidates[:CFG["MAX_SIGNALS_PER_CYCLE"]]:
        if CFG["KRONOS_ENABLED"]:
            try:
                forecast = kronos_forecast(signal["symbol"], candles, signal["side"])
                signal["forecast"] = forecast
                if CFG["KRONOS_REQUIRED"] and not forecast["direction_aligned"]:
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

    delivered = deliver_pending_signals(limit=5)
    summary = {
        "status": "complete",
        "at": now.isoformat(),
        "symbols_considered": len(symbols),
        "technical_candidates": technical_candidates,
        "kronos_rejected_or_unavailable": kronos_rejected,
        "new_signals_queued": inserted,
        "telegram_delivered": delivered,
        "duration_seconds": round(time.time() - started, 1),
        "universe_refreshed_at": universe.get("refreshed_at"),
        "data_source": "Yahoo Finance via yfinance (unofficial)",
        "kronos": kronos_status,
    }
    LOG.info("SCAN SUMMARY %s", json.dumps(summary))
    return summary



def run_dry_run_v2() -> Dict[str, Any]:
    """Historical full-session diagnostic. No Telegram, orders or signal DB writes.

    Daily trend and universe use current cache, so this is NOT a look-ahead-free
    backtest. Forecasts are exploratory and do not represent win probabilities.
    """
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
    # A market-open day may have incomplete candles; replay only completed bars.
    actual_now = datetime.now(TZ)
    end = datetime.combine(replay_day, dtime(15, 30), tzinfo=TZ)
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
        # Exclude any bars from future sessions; use only history available as of each replay.
        today = frame[[x.date() == replay_day for x in frame.index]]
        if today.empty:
            rejection["missing_replay_session"] += 1
            continue
        symbols_with_session += 1
        for candle_start in today.index:
            anchor = candle_start.to_pydatetime() + timedelta(minutes=5)
            if anchor > end:
                continue
            total_checks += 1
            replayed_bars += 1
            # Do not allow the evaluation function to inspect later bars.
            prefix = frame.loc[frame.index <= candle_start]
            diag = {}
            signal = evaluate_symbol(symbol, prefix, entry, anchor, diagnostic=diag)
            if signal:
                candidates.append((signal["score"], signal, prefix, anchor))
                rejection["technical_pass"] += 1
            else:
                rejection[diag.get("rejected", "unknown")] += 1
            if diag.get("rejected") == "technical_alignment":
                flags = diag.get("conditions", {})
                for key, passed in flags.items():
                    if not passed:
                        conditions_failed[key] += 1
    candidates.sort(key=lambda item: (-item[0], item[1]["candle_time"], item[1]["symbol"]))
    # Diagnostic only: always test one actual Kronos inference on a recent stock
    # even if there are zero technical candidates.
    independent = {"status": "not_run"}
    eligible = []
    for entry in entries:
        f = normalized.get(entry["symbol"])
        if f is not None and len(completed_bars(f.loc[f.index.date <= replay_day], end)) >= 40:
            eligible.append((entry["symbol"], f))
    if eligible:
        sym, frame = eligible[0]
        try:
            pred = kronos_forecast(sym, frame, "BUY", as_of=end)
            independent = {"status": "success", "symbol": sym, "forecast": pred}
        except Exception as exc:
            independent = {"status": "failed", "symbol": sym, "error": str(exc)[:350]}
            LOG.exception("DRY RUN V2 independent Kronos inference failed")
    else:
        independent = {"status": "no_sufficient_history"}

    # Only the top technical candidates are tested through Kronos, matching
    # live per-cycle selection size; no Telegram or DB changes.
    forecast_samples = []
    for _, signal, prefix, anchor in candidates[:CFG["MAX_SIGNALS_PER_CYCLE"]]:
        item = dict(signal)
        try:
            item["forecast"] = kronos_forecast(signal["symbol"], prefix, signal["side"], as_of=anchor)
        except Exception as exc:
            item["forecast"] = {"status": "error", "detail": str(exc)[:250]}
        forecast_samples.append(item)

    report = {
        "mode": "DRY_RUN_V2_NO_ALERTS", "session_date_ist": str(replay_day),
        "symbols_requested": len(entries), "symbols_with_data": len(normalized),
        "symbols_with_replay_session": symbols_with_session,
        "candles_evaluated": replayed_bars, "checks_total": total_checks,
        "primary_rejection_counts": dict(rejection.most_common()),
        "alignment_condition_failures": dict(conditions_failed.most_common()),
        "technical_candidates": len(candidates),
        "top_technical_candidates": [x[1] for x in candidates[:10]],
        "kronos_independent_test": independent,
        "kronos_candidate_forecasts": forecast_samples,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "caveats": [
            "Not a point-in-time backtest: cached universe and daily EMA values may look ahead.",
            "Replays only the most recent available trading session; no P&L/win-rate estimate.",
            "Yahoo Finance intraday data is unofficial and may have gaps or delays.",
            "No Telegram messages, signal database writes, trades or scheduled activation.",
        ],
    }
    path = ROOT / "dry_run_v2_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOG.info("DRY RUN V2 SUMMARY %s", json.dumps({k: v for k, v in report.items()
             if k in ("session_date_ist", "symbols_requested", "symbols_with_replay_session",
                      "candles_evaluated", "primary_rejection_counts", "alignment_condition_failures",
                      "technical_candidates", "kronos_independent_test", "elapsed_seconds")}, default=str))
    LOG.info("DRY RUN V2 report saved: %s", path)
    if independent["status"] != "success":
        raise RuntimeError("Independent Kronos inference did not pass; see report and log")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bharat Intraday Scout 2.0 one-shot runner")
    parser.add_argument("--task", choices=["scan", "refresh-universe", "test-telegram", "test-kronos", "dry-run-v2"], default="scan")
    args = parser.parse_args()
    if args.task == "dry-run-v2":
        load_kronos()
        run_dry_run_v2()
        return 0

    init_db()

    if args.task == "refresh-universe":
        result = refresh_liquid_universe(force=True)
        LOG.info("UNIVERSE READY | eligible=%s | selected=%s | refreshed=%s",
                 result.get("eligible_count"), len(result.get("scan_symbols", [])), result.get("refreshed_at"))
        return 0

    if args.task == "test-telegram":
        send_telegram_message("✅ Bharat Intraday Scout 2.0 Telegram test successful. No trade was placed.")
        LOG.info("Telegram test succeeded")
        return 0

    if args.task == "test-kronos":
        load_kronos()
        if kronos_predictor is None:
            raise RuntimeError("Kronos failed to load: " + kronos_status)
        LOG.info("Kronos self-test succeeded: %s", kronos_status)
        return 0

    if not is_market_session():
        LOG.info("Outside NSE market session; scan skipped")
        return 0

    load_kronos()
    if CFG["KRONOS_REQUIRED"] and kronos_predictor is None:
        raise RuntimeError("Kronos is required but not ready: " + kronos_status)
    result = run_scan_cycle()
    LOG.info("RUN RESULT %s", json.dumps(result))
    if result.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
