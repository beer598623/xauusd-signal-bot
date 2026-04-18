"""
XAUUSD MTF BB Pullback — Signal Generator
Strategy: BB Touch + MTF Filter (5m entry, 1H filter)
Backtest: WR 51.4% | PF 4.25 | MaxDD -2.03% | Sep 2025–Apr 2026
"""

import os
import time
import csv
import json
import asyncio
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("XAUUSD")

# ─────────────────────────────────────────────
#  SECRETS — NEVER HARDCODE, ALWAYS os.environ
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]   # BotFather token
TELEGRAM_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]     # Your chat / group ID
TWELVEDATA_API_KEY  = os.environ["TWELVEDATA_API_KEY"]   # TwelveData free tier
GITHUB_TOKEN        = os.environ.get("GITHUB_TOKEN", "") # For auto-commit
GITHUB_REPO         = os.environ.get("GITHUB_REPO", "")  # e.g. username/repo
RISK_PERCENT        = float(os.environ.get("RISK_PERCENT", "1.0"))
ACCOUNT_BALANCE     = float(os.environ.get("ACCOUNT_BALANCE", "1000.0"))

# ─────────────────────────────────────────────
#  CONSTANTS — FROM XLSX SPEC
# ─────────────────────────────────────────────
# Indicator params
BB_PERIOD       = 20
BB_STDDEV       = 2.0
RSI_PERIOD      = 14
ATR_PERIOD      = 14
EMA_FAST        = 21    # 1H
EMA_SLOW        = 50    # 1H
ADX_PERIOD      = 14    # 1H
ADX_MIN         = 22.0

# Risk params
SL_BUFFER_MULT  = 0.3   # × ATR
MAX_SL_MULT     = 3.0   # × ATR  — skip if wider
TP_RR           = 2.5   # risk:reward
BE_TRIGGER_MULT = 1.0   # × ATR  — move SL to entry
MIN_LOT         = 0.01

# Session filter (UTC)
SESSIONS_UTC = [
    (7, 12),   # London Morning  → Thai 14:00–19:00
    (13, 20),  # NY + Overlap    → Thai 20:00–03:00
]

# Loop config
LOOP_INTERVAL_SEC = 30    # check every 30s
RUNTIME_MINUTES   = 50    # self-terminate after 50 min

# TwelveData endpoint
TD_BASE = "https://api.twelvedata.com"
SYMBOL  = "XAU/USD"

# Performance log
LOG_FILE = Path("performance_log.csv")

# ─────────────────────────────────────────────
#  MARKET DATA
# ─────────────────────────────────────────────

def fetch_ohlcv(interval: str, outputsize: int = 200) -> pd.DataFrame:
    """Fetch OHLCV candles from TwelveData REST API."""
    url = f"{TD_BASE}/time_series"
    params = {
        "symbol":     SYMBOL,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     TWELVEDATA_API_KEY,
        "order":      "ASC",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise ValueError(f"TwelveData error: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df.set_index("datetime", inplace=True)
    return df


def fetch_price_realtime() -> float:
    """Fetch current bid price."""
    url = f"{TD_BASE}/price"
    r = requests.get(url, params={"symbol": SYMBOL, "apikey": TWELVEDATA_API_KEY}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


# ─────────────────────────────────────────────
#  INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_bb(close: pd.Series, period: int = 20, std: float = 2.0):
    """Returns (upper, mid, lower)."""
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std(ddof=0)
    return mid + std * sigma, mid, mid - std * sigma


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed ADX."""
    tr   = calc_atr(high, low, close, period)
    up   = high.diff()
    dn   = -low.diff()
    pdm  = np.where((up > dn) & (up > 0), up, 0.0)
    ndm  = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm_s = pd.Series(pdm, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    ndm_s = pd.Series(ndm, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    pdi   = 100 * pdm_s / tr.replace(0, np.nan)
    ndi   = 100 * ndm_s / tr.replace(0, np.nan)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def build_indicators_1h(df: pd.DataFrame) -> dict:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    ema21 = calc_ema(close, EMA_FAST)
    ema50 = calc_ema(close, EMA_SLOW)
    adx   = calc_adx(high, low, close, ADX_PERIOD)
    return {
        "ema21": float(ema21.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "adx":   float(adx.iloc[-1]),
    }


def build_indicators_5m(df: pd.DataFrame) -> dict:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    bb_u, bb_m, bb_l = calc_bb(close, BB_PERIOD, BB_STDDEV)
    rsi  = calc_rsi(close, RSI_PERIOD)
    atr  = calc_atr(high, low, close, ATR_PERIOD)
    return {
        # current bar (index -1)
        "close":    float(close.iloc[-1]),
        "high":     float(high.iloc[-1]),
        "low":      float(low.iloc[-1]),
        "bb_upper": float(bb_u.iloc[-1]),
        "bb_mid":   float(bb_m.iloc[-1]),
        "bb_lower": float(bb_l.iloc[-1]),
        "rsi":      float(rsi.iloc[-1]),
        "atr":      float(atr.iloc[-1]),
        # previous bar (index -2)
        "prev_close":    float(close.iloc[-2]),
        "prev_high":     float(high.iloc[-2]),
        "prev_low":      float(low.iloc[-2]),
        "prev_bb_upper": float(bb_u.iloc[-2]),
        "prev_bb_lower": float(bb_l.iloc[-2]),
        "prev_rsi":      float(rsi.iloc[-2]),
    }


# ─────────────────────────────────────────────
#  SESSION FILTER
# ─────────────────────────────────────────────

def is_trading_session(dt_utc: datetime) -> bool:
    hour = dt_utc.hour
    for start, end in SESSIONS_UTC:
        if start <= hour < end:
            return True
    return False


# ─────────────────────────────────────────────
#  SIGNAL LOGIC — ALL 7 CONDITIONS
# ─────────────────────────────────────────────

def check_signal(ind_1h: dict, ind_5m: dict, dt_utc: datetime) -> dict | None:
    """
    Returns signal dict or None.
    Direction: 'BUY' or 'SELL'
    All 7 conditions must pass.
    """
    # ── Condition 1: Session gate ──
    if not is_trading_session(dt_utc):
        log.debug("SKIP: off-session")
        return None

    # ── Determine trend direction ──
    ema_gap_pct = abs(ind_1h["ema21"] - ind_1h["ema50"]) / ind_1h["ema50"] * 100
    if ema_gap_pct < 0.1:
        log.debug("SKIP: EMA transition zone")
        return None

    bullish_trend = ind_1h["ema21"] > ind_1h["ema50"]

    # ── Condition 3: ADX strength ──
    if ind_1h["adx"] < ADX_MIN:
        log.debug(f"SKIP: ADX {ind_1h['adx']:.1f} < {ADX_MIN}")
        return None

    atr = ind_5m["atr"]

    # ────────── BUY SIGNAL ──────────
    if bullish_trend:
        # C2: 1H trend = bullish ✓
        # C4: Previous bar Low <= BB Lower
        if ind_5m["prev_low"] > ind_5m["prev_bb_lower"]:
            log.debug("BUY C4 fail: prev low did not touch BB lower")
            return None
        # C5: Current bar Close > BB Lower (rejection)
        if ind_5m["close"] <= ind_5m["bb_lower"]:
            log.debug("BUY C5 fail: current close still below BB lower")
            return None
        # C6: RSI < 45 turning up
        if ind_5m["prev_rsi"] >= 45:
            log.debug(f"BUY C6 fail: prev RSI {ind_5m['prev_rsi']:.1f} not < 45")
            return None
        if ind_5m["rsi"] <= ind_5m["prev_rsi"]:
            log.debug("BUY C6 fail: RSI not turning up")
            return None
        # C7: SL width check
        sl_price = ind_5m["prev_low"] - SL_BUFFER_MULT * atr
        sl_dist  = ind_5m["close"] - sl_price
        if sl_dist > MAX_SL_MULT * atr:
            log.debug(f"BUY C7 fail: SL too wide ({sl_dist:.2f} > {MAX_SL_MULT*atr:.2f})")
            return None
        tp_price = ind_5m["close"] + TP_RR * sl_dist
        lot      = calc_lot_size(sl_dist)
        return {
            "direction": "BUY",
            "entry":     round(ind_5m["close"], 2),
            "sl":        round(sl_price, 2),
            "tp":        round(tp_price, 2),
            "sl_dist":   round(sl_dist, 2),
            "be_level":  round(ind_5m["close"] + BE_TRIGGER_MULT * atr, 2),
            "lot":       lot,
            "atr":       round(atr, 2),
            **ind_1h,
            **{f"5m_{k}": v for k, v in ind_5m.items()},
            "timestamp": dt_utc.isoformat(),
        }

    # ────────── SELL SIGNAL ──────────
    else:
        # C2: 1H trend = bearish ✓
        # C4: Previous bar High >= BB Upper
        if ind_5m["prev_high"] < ind_5m["prev_bb_upper"]:
            log.debug("SELL C4 fail: prev high did not touch BB upper")
            return None
        # C5: Current bar Close < BB Upper (rejection)
        if ind_5m["close"] >= ind_5m["bb_upper"]:
            log.debug("SELL C5 fail: current close still above BB upper")
            return None
        # C6: RSI > 55 turning down
        if ind_5m["prev_rsi"] <= 55:
            log.debug(f"SELL C6 fail: prev RSI {ind_5m['prev_rsi']:.1f} not > 55")
            return None
        if ind_5m["rsi"] >= ind_5m["prev_rsi"]:
            log.debug("SELL C6 fail: RSI not turning down")
            return None
        # C7: SL width check
        sl_price = ind_5m["prev_high"] + SL_BUFFER_MULT * atr
        sl_dist  = sl_price - ind_5m["close"]
        if sl_dist > MAX_SL_MULT * atr:
            log.debug(f"SELL C7 fail: SL too wide ({sl_dist:.2f} > {MAX_SL_MULT*atr:.2f})")
            return None
        tp_price = ind_5m["close"] - TP_RR * sl_dist
        lot      = calc_lot_size(sl_dist)
        return {
            "direction": "SELL",
            "entry":     round(ind_5m["close"], 2),
            "sl":        round(sl_price, 2),
            "tp":        round(tp_price, 2),
            "sl_dist":   round(sl_dist, 2),
            "be_level":  round(ind_5m["close"] - BE_TRIGGER_MULT * atr, 2),
            "lot":       lot,
            "atr":       round(atr, 2),
            **ind_1h,
            **{f"5m_{k}": v for k, v in ind_5m.items()},
            "timestamp": dt_utc.isoformat(),
        }


# ─────────────────────────────────────────────
#  RISK MANAGEMENT — POSITION SIZING
# ─────────────────────────────────────────────

def calc_lot_size(sl_pts: float) -> float:
    """
    Lot = FLOOR( Risk_USD / (SL_pts × 100) / 0.01 ) × 0.01
    XAUUSD: 1 lot = 100 oz, 1 pt move = $100/lot
    """
    risk_usd = ACCOUNT_BALANCE * RISK_PERCENT / 100
    if sl_pts <= 0:
        return MIN_LOT
    raw = risk_usd / (sl_pts * 100)
    lot = max(MIN_LOT, round(int(raw / MIN_LOT) * MIN_LOT, 2))
    return lot


# ─────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

_pending_signal: dict | None = None  # store signal awaiting user response


def send_signal_alert(signal: dict) -> int | None:
    """Send formatted signal with inline confirm/skip buttons. Returns message_id."""
    direction = signal["direction"]
    emoji = "🟢" if direction == "BUY" else "🔴"
    thai_time = (datetime.fromisoformat(signal["timestamp"])
                 .replace(tzinfo=timezone.utc) + timedelta(hours=7)
                 ).strftime("%d/%m/%Y %H:%M")

    text = (
        f"{emoji} *XAUUSD {direction} SIGNAL*\n"
        f"⏰ {thai_time} (ไทย)\n\n"
        f"📍 *Entry:* `{signal['entry']}`\n"
        f"🛑 *Stop Loss:* `{signal['sl']}` ({signal['sl_dist']} pts)\n"
        f"🎯 *Take Profit:* `{signal['tp']}` (RR 1:{TP_RR})\n"
        f"⚡ *Break-Even:* `{signal['be_level']}`\n"
        f"📦 *Lot Size:* `{signal['lot']}` lot\n\n"
        f"📊 *1H Indicators*\n"
        f"  EMA21: `{signal['ema21']:.2f}` | EMA50: `{signal['ema50']:.2f}`\n"
        f"  ADX: `{signal['adx']:.1f}` {'✅' if signal['adx'] >= ADX_MIN else '⚠️'}\n\n"
        f"📊 *5m Indicators*\n"
        f"  RSI: `{signal['5m_rsi']:.1f}` → `{signal['5m_prev_rsi']:.1f}`\n"
        f"  ATR: `{signal['atr']:.2f}`\n"
        f"  BB: `{signal['5m_bb_lower']:.2f}` / `{signal['5m_bb_upper']:.2f}`\n\n"
        f"_กด ✅ เพื่อบันทึกการยืนยัน หรือ ❌ เพื่อข้ามสัญญาณ_"
    )

    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
        "reply_markup": json.dumps({
            "inline_keyboard": [[
                {"text": "✅ ยืนยันเข้าเทรด", "callback_data": "confirm"},
                {"text": "❌ ข้ามสัญญาณ",    "callback_data": "skip"},
            ]]
        }),
    }
    resp = requests.post(f"{TG_BASE}/sendMessage", json=payload, timeout=10)
    data = resp.json()
    if data.get("ok"):
        log.info(f"✅ Telegram alert sent — message_id: {data['result']['message_id']}")
        return data["result"]["message_id"]
    else:
        log.error(f"Telegram send failed: {data}")
        return None


def send_text(text: str) -> None:
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(f"{TG_BASE}/sendMessage", json=payload, timeout=10)


def get_updates(offset: int = 0) -> list:
    resp = requests.get(f"{TG_BASE}/getUpdates", params={"offset": offset, "timeout": 5}, timeout=15)
    if resp.ok:
        return resp.json().get("result", [])
    return []


def answer_callback(callback_id: str, text: str) -> None:
    requests.post(f"{TG_BASE}/answerCallbackQuery",
                  json={"callback_query_id": callback_id, "text": text}, timeout=10)


# ─────────────────────────────────────────────
#  PERFORMANCE LOG
# ─────────────────────────────────────────────
LOG_COLUMNS = [
    "timestamp", "action", "direction", "entry", "sl", "tp",
    "sl_dist", "lot", "atr", "ema21", "ema50", "adx",
    "rsi", "prev_rsi", "bb_lower", "bb_upper", "be_level",
]


def ensure_log() -> None:
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def append_log(signal: dict, action: str) -> None:
    ensure_log()
    row = {
        "timestamp": signal.get("timestamp"),
        "action":    action,  # "CONFIRM" | "SKIP"
        "direction": signal.get("direction"),
        "entry":     signal.get("entry"),
        "sl":        signal.get("sl"),
        "tp":        signal.get("tp"),
        "sl_dist":   signal.get("sl_dist"),
        "lot":       signal.get("lot"),
        "atr":       signal.get("atr"),
        "ema21":     signal.get("ema21"),
        "ema50":     signal.get("ema50"),
        "adx":       signal.get("adx"),
        "rsi":       signal.get("5m_rsi"),
        "prev_rsi":  signal.get("5m_prev_rsi"),
        "bb_lower":  signal.get("5m_bb_lower"),
        "bb_upper":  signal.get("5m_bb_upper"),
        "be_level":  signal.get("be_level"),
    }
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)
    log.info(f"📝 Log written: {action} | {signal.get('direction')} @ {signal.get('entry')}")


# ─────────────────────────────────────────────
#  GITHUB AUTO-COMMIT
# ─────────────────────────────────────────────

def git_commit_log() -> None:
    """Commit performance_log.csv back to GitHub repo via git CLI."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.warning("GITHUB_TOKEN or GITHUB_REPO not set — skipping auto-commit")
        return
    try:
        # Configure git identity for the Actions bot
        subprocess.run(["git", "config", "user.email", "actions@github.com"], check=True)
        subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"], check=True)
        subprocess.run(["git", "add", str(LOG_FILE)], check=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], capture_output=True
        )
        if result.returncode == 0:
            log.info("No changes to commit.")
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(
            ["git", "commit", "-m", f"perf: auto-log update {now}"], check=True
        )
        remote = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        subprocess.run(["git", "push", remote, "HEAD:main"], check=True)
        log.info("✅ performance_log.csv committed & pushed to GitHub")
    except subprocess.CalledProcessError as e:
        log.error(f"Git commit failed: {e}")


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def main() -> None:
    log.info("🚀 XAUUSD Signal Generator starting…")
    ensure_log()

    global _pending_signal
    _pending_signal = None

    start_time     = time.time()
    deadline       = start_time + RUNTIME_MINUTES * 60
    last_update_id = 0
    last_signal_ts = None  # avoid duplicate alerts in same 5m bar

    while time.time() < deadline:
        loop_start = time.time()

        # ── 1. Poll Telegram callbacks ──
        try:
            updates = get_updates(offset=last_update_id + 1)
            for upd in updates:
                last_update_id = upd["update_id"]
                if "callback_query" in upd and _pending_signal is not None:
                    cb   = upd["callback_query"]
                    data = cb["data"]
                    answer_callback(cb["id"], "✅ บันทึกแล้ว!" if data == "confirm" else "❌ ข้ามสัญญาณ")
                    action = "CONFIRM" if data == "confirm" else "SKIP"
                    append_log(_pending_signal, action)
                    git_commit_log()
                    reply = (
                        "✅ บันทึกการยืนยันเข้าเทรดแล้ว — โชคดีนะ! 🍀"
                        if data == "confirm"
                        else "❌ ข้ามสัญญาณนี้แล้ว — รอสัญญาณถัดไป"
                    )
                    send_text(reply)
                    _pending_signal = None
        except Exception as e:
            log.warning(f"Telegram poll error: {e}")

        # ── 2. Fetch data & compute signals ──
        try:
            dt_utc = datetime.now(timezone.utc)
            if not is_trading_session(dt_utc):
                log.info(f"⏳ Off-session ({dt_utc.strftime('%H:%M UTC')}) — sleeping")
            else:
                df_1h = fetch_ohlcv("1h", outputsize=100)
                df_5m = fetch_ohlcv("5min", outputsize=100)
                ind_1h = build_indicators_1h(df_1h)
                ind_5m = build_indicators_5m(df_5m)

                # Dedup: skip if we already alerted on this 5m bar
                current_bar_ts = df_5m.index[-1].isoformat()
                if current_bar_ts == last_signal_ts:
                    log.debug("Same 5m bar — no new signal check")
                else:
                    signal = check_signal(ind_1h, ind_5m, dt_utc)
                    if signal:
                        log.info(f"🔔 SIGNAL: {signal['direction']} @ {signal['entry']}")
                        _pending_signal = signal
                        last_signal_ts  = current_bar_ts
                        send_signal_alert(signal)
                    else:
                        log.info(
                            f"No signal | ADX={ind_1h['adx']:.1f} | "
                            f"EMA21={'>' if ind_1h['ema21'] > ind_1h['ema50'] else '<'}EMA50 | "
                            f"RSI={ind_5m['rsi']:.1f}"
                        )
        except Exception as e:
            log.error(f"Data/signal error: {e}", exc_info=True)

        # ── 3. Sleep until next interval ──
        elapsed = time.time() - loop_start
        sleep   = max(0, LOOP_INTERVAL_SEC - elapsed)
        time.sleep(sleep)

    remaining = int(deadline - time.time())
    log.info(f"⏹ Runtime limit reached ({RUNTIME_MINUTES} min) — exiting. Next run via GH Actions.")


if __name__ == "__main__":
    main()
