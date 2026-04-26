"""
XAUUSD MTF BB Pullback - Signal Generator v2.4
Changes from v2.3:
- FIX-1: RSI loss=0 → replace 1e-10 (ป้องกัน NaN signal drop เงียบๆ)
- FIX-2: API retry 3 ครั้ง + exponential backoff
- FIX-3: Cache 1H data ทุก 60 นาที (ลด API call ~50%)
- FIX-4: Signal expiry 10 นาที (auto-expire stale signal)
- FIX-5: เพิ่ม hold_time_min, session, rr_planned ใน log
- FIX-6: ตัด cron 00:00 และ 18:00 UTC → แก้ใน yml แยก
         เพิ่ม session guard (6, 18) UTC ใน code เป็น 2nd layer
- FIX-7: Price-based expiry เพิ่มบน time-based (0.5 ATR threshold)
- FIX-8: เพิ่ม regime tag "trend"/"range" ใน signal + log
"""

import os
import time
import csv
import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("XAUUSD")

TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
TWELVEDATA_API_KEY  = os.environ["TWELVEDATA_API_KEY"]
GITHUB_TOKEN        = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO         = os.environ.get("GITHUB_REPO", "")
RISK_PERCENT        = float(os.environ.get("RISK_PERCENT", "1.0"))
ACCOUNT_BALANCE     = float(os.environ.get("ACCOUNT_BALANCE", "100.0"))

BB_PERIOD           = 20
BB_STDDEV           = 2.0
RSI_PERIOD          = 14
ATR_PERIOD          = 14
EMA_FAST            = 21
EMA_SLOW            = 50
ADX_PERIOD          = 10
ADX_MIN             = 22.0
RSI_BUY_MAX         = 48
RSI_SELL_MIN        = 52
SL_BUFFER_MULT      = 0.3
MAX_SL_MULT         = 3.0
TP_RR               = 2.5
BE_TRIGGER_MULT     = 1.0
MIN_LOT             = 0.01
ATR_EXPANSION_MULT  = 1.2
MAX_LOSS_STREAK     = 3
ROLLOVER_START      = (21, 59)
ROLLOVER_END        = (22, 10)

# FIX-6: session guard เป็น 2nd layer ป้องกัน Asia session
# Cron จะตัด 00:00 และ 18:00 UTC ออกแล้ว
# Guard นี้เป็น safety net กรณี workflow_dispatch ผิดเวลา
ACTIVE_SESSION_UTC  = (6, 18)      # London open → NY close

SESSIONS_UTC        = [(0, 24)]    # keep เดิม (guard จัดการแทน)
LOOP_INTERVAL_SEC   = 300
RUNTIME_MINUTES     = 355

# FIX-3: 1H cache — refresh ทุก 60 นาทีเท่านั้น
H1_CACHE_MINUTES    = 60

# FIX-4/7: Signal expiry — time-based + price-based
SIGNAL_EXPIRY_MIN       = 10    # หมดอายุถ้าเกิน 10 นาที
SIGNAL_EXPIRY_ATR_MULT  = 0.5   # หรือถ้าราคาวิ่งเกิน 0.5 ATR จาก entry

# FIX-2: API retry
API_MAX_RETRIES     = 3
API_RETRY_DELAY     = 5            # seconds (จะ x2 ทุก retry)

TD_BASE             = "https://api.twelvedata.com"
SYMBOL              = "XAU/USD"
LOG_FILE            = Path("performance_log.csv")

# FIX-5: เพิ่ม columns สำหรับ quant analysis
LOG_COLUMNS = [
    "timestamp", "action", "direction", "entry", "sl", "tp",
    "sl_dist", "lot", "atr", "ema21", "ema50", "adx",
    "rsi", "prev_rsi", "bb_lower", "bb_upper", "be_level",
    "exit_price", "actual_entry",
    # --- v2.3 additions ---
    "rr_planned",       # TP_RR ที่ตั้งไว้ ณ วันนั้น
    "session",          # London / NY / Other
    "hold_time_min",    # กี่นาทีจาก entry ถึง exit (กรอกตอน exit)
    # --- v2.4 additions ---
    "regime",           # FIX-8: "trend" (ADX≥25) / "range" (ADX<25)
]


# ─────────────────────────────────────────────
# FIX-2: fetch with retry + exponential backoff
# ─────────────────────────────────────────────
def fetch_ohlcv(interval: str, outputsize: int = 100) -> pd.DataFrame:
    url = f"{TD_BASE}/time_series"
    params = {
        "symbol":     SYMBOL,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     TWELVEDATA_API_KEY,
        "order":      "ASC",
    }
    last_exc = None
    delay = API_RETRY_DELAY
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
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
        except Exception as e:
            last_exc = e
            if attempt < API_MAX_RETRIES:
                log.warning(f"fetch_ohlcv [{interval}] attempt {attempt} failed: {e} — retry in {delay}s")
                time.sleep(delay)
                delay *= 2   # exponential backoff
            else:
                log.error(f"fetch_ohlcv [{interval}] failed after {API_MAX_RETRIES} attempts: {e}")
    raise last_exc


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_bb(close: pd.Series, period: int = 20, std: float = 2.0):
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std(ddof=0)
    return mid + std * sigma, mid, mid - std * sigma


# FIX-1: RSI — loss=0 → 1e-10 ป้องกัน NaN เงียบๆ
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    loss  = loss.replace(0, 1e-10)   # FIX-1: ไม่ให้ NaN
    rs    = gain / loss
    return 100 - 100 / (1 + rs)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    tr    = calc_atr(high, low, close, period)
    up    = high.diff()
    dn    = -low.diff()
    pdm   = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index).ewm(alpha=1/period, adjust=False).mean()
    ndm   = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index).ewm(alpha=1/period, adjust=False).mean()
    pdi   = 100 * pdm / tr.replace(0, np.nan)
    ndi   = 100 * ndm / tr.replace(0, np.nan)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def build_indicators_1h(df: pd.DataFrame) -> dict:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    return {
        "ema21": float(calc_ema(close, EMA_FAST).iloc[-1]),
        "ema50": float(calc_ema(close, EMA_SLOW).iloc[-1]),
        "adx":   float(calc_adx(high, low, close, ADX_PERIOD).iloc[-1]),
    }


def build_indicators_5m(df: pd.DataFrame) -> dict:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    bb_u, bb_m, bb_l = calc_bb(close, BB_PERIOD, BB_STDDEV)
    rsi     = calc_rsi(close, RSI_PERIOD)
    atr     = calc_atr(high, low, close, ATR_PERIOD)
    atr_avg = float(atr.rolling(20).mean().iloc[-1])
    return {
        "close":         float(close.iloc[-1]),
        "high":          float(high.iloc[-1]),
        "low":           float(low.iloc[-1]),
        "bb_upper":      float(bb_u.iloc[-1]),
        "bb_mid":        float(bb_m.iloc[-1]),
        "bb_lower":      float(bb_l.iloc[-1]),
        "rsi":           float(rsi.iloc[-1]),
        "atr":           float(atr.iloc[-1]),
        "atr_avg":       atr_avg,
        "prev_high":     float(high.iloc[-2]),
        "prev_low":      float(low.iloc[-2]),
        "prev_bb_upper": float(bb_u.iloc[-2]),
        "prev_bb_lower": float(bb_l.iloc[-2]),
        "prev_rsi":      float(rsi.iloc[-2]),
    }


# FIX-6: session guard helper
def get_session_name(dt_utc: datetime) -> str:
    h = dt_utc.hour
    if 6 <= h < 12:
        return "London"
    elif 12 <= h < 18:
        return "NY"
    elif 18 <= h < 22:
        return "NY_Late"
    else:
        return "Asia"


def is_active_session(dt_utc: datetime) -> bool:
    """FIX-6: 2nd-layer guard — เฉพาะ London + NY core"""
    return ACTIVE_SESSION_UTC[0] <= dt_utc.hour < ACTIVE_SESSION_UTC[1]


def is_trading_session(dt_utc: datetime) -> bool:
    for start, end in SESSIONS_UTC:
        if start <= dt_utc.hour < end:
            return True
    return False


def is_market_open(dt_utc: datetime) -> bool:
    wd = dt_utc.weekday()
    if wd == 5:
        return False
    if wd == 6:
        return False
    if wd == 4 and dt_utc.hour >= 23:
        return False
    return True


def is_rollover(dt_utc: datetime) -> bool:
    h, m = dt_utc.hour, dt_utc.minute
    if h == ROLLOVER_START[0] and m >= ROLLOVER_START[1]:
        return True
    if h == ROLLOVER_END[0] and m <= ROLLOVER_END[1]:
        return True
    return False


def get_loss_streak() -> int:
    if not LOG_FILE.exists():
        return 0
    try:
        rows = []
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("action") == "CONFIRM" and row.get("exit_price"):
                    rows.append(row)
        if not rows:
            return 0
        streak = 0
        for row in reversed(rows):
            try:
                entry_str = row.get("actual_entry") or row.get("entry")
                entry = float(entry_str)
                lot   = float(row["lot"])
                exit_ = float(row["exit_price"])
                if row["direction"] == "BUY":
                    p = (exit_ - entry) * lot * 100
                else:
                    p = (entry - exit_) * lot * 100
                if p < 0:
                    streak += 1
                else:
                    break
            except:
                break
        return streak
    except:
        return 0


def is_market_active(ind_5m: dict) -> bool:
    atr     = ind_5m["atr"]
    atr_avg = ind_5m["atr_avg"]
    if atr_avg <= 0 or pd.isna(atr_avg):
        return False   # fail-safe (ไม่ใช่ True เหมือนเดิม)
    return atr >= atr_avg * ATR_EXPANSION_MULT


# FIX-4/7: ตรวจ signal expiry — time-based + price-based
def is_signal_expired(signal: dict, current_price: float | None = None) -> bool:
    """
    Expire signal ถ้า:
    (A) อายุเกิน SIGNAL_EXPIRY_MIN นาที  — time-based
    (B) ราคาตลาดวิ่งห่างจาก entry เกิน 0.5 ATR — price-based
        (normalize ตาม volatility วันนั้น ดีกว่าตัวเลขคงที่)
    """
    try:
        ts  = datetime.fromisoformat(signal["timestamp"]).replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age > SIGNAL_EXPIRY_MIN:
            return True   # (A) time expired
    except:
        return True       # อ่าน timestamp ไม่ได้ = expired

    # (B) price-based: ต้องการ current_price และ ATR
    if current_price is not None:
        try:
            atr       = float(signal.get("atr", 0))
            entry     = float(signal["entry"])
            threshold = atr * SIGNAL_EXPIRY_ATR_MULT
            if threshold > 0 and abs(current_price - entry) > threshold:
                return True
        except:
            pass   # ถ้าอ่านค่าไม่ได้ ไม่ expire จาก price check

    return False


def check_signal(ind_1h: dict, ind_5m: dict, dt_utc: datetime) -> dict | None:
    if not is_market_open(dt_utc):
        return None
    # FIX-6: session guard
    if not is_active_session(dt_utc):
        log.info(f"SKIP: outside active session ({get_session_name(dt_utc)}, {dt_utc.hour:02d}:xx UTC)")
        return None
    if is_rollover(dt_utc):
        log.info("SKIP: rollover 21:59-22:10 UTC")
        return None
    streak = get_loss_streak()
    if streak >= MAX_LOSS_STREAK:
        log.info(f"SKIP: loss streak cooldown ({streak} losses)")
        return None
    if abs(ind_1h["ema21"] - ind_1h["ema50"]) / ind_1h["ema50"] * 100 < 0.1:
        return None
    if ind_1h["adx"] < ADX_MIN:
        return None
    if not is_market_active(ind_5m):
        log.info(f"SKIP: low vol ATR={ind_5m['atr']:.2f} avg={ind_5m['atr_avg']:.2f}")
        return None

    atr     = ind_5m["atr"]
    bullish = ind_1h["ema21"] > ind_1h["ema50"]
    session = get_session_name(dt_utc)          # FIX-5
    regime  = "trend" if ind_1h["adx"] >= 25 else "range"  # FIX-8

    if bullish:
        if ind_5m["prev_low"] > ind_5m["prev_bb_lower"]: return None
        if ind_5m["close"] <= ind_5m["bb_lower"]: return None
        if ind_5m["prev_rsi"] >= RSI_BUY_MAX: return None
        if ind_5m["rsi"] <= ind_5m["prev_rsi"]: return None
        sl_price = ind_5m["prev_low"] - SL_BUFFER_MULT * atr
        sl_dist  = ind_5m["close"] - sl_price
        if sl_dist > MAX_SL_MULT * atr: return None
        return {
            "direction":  "BUY",
            "entry":      round(ind_5m["close"], 2),
            "sl":         round(sl_price, 2),
            "tp":         round(ind_5m["close"] + TP_RR * sl_dist, 2),
            "sl_dist":    round(sl_dist, 2),
            "be_level":   round(ind_5m["close"] + BE_TRIGGER_MULT * atr, 2),
            "lot":        calc_lot_size(sl_dist),
            "atr":        round(atr, 2),
            "rr_planned": TP_RR,        # FIX-5
            "session":    session,      # FIX-5
            "regime":     regime,       # FIX-8
            **ind_1h,
            **{f"5m_{k}": v for k, v in ind_5m.items()},
            "timestamp":  dt_utc.isoformat(),
        }
    else:
        if ind_5m["prev_high"] < ind_5m["prev_bb_upper"]: return None
        if ind_5m["close"] >= ind_5m["bb_upper"]: return None
        if ind_5m["prev_rsi"] <= RSI_SELL_MIN: return None
        if ind_5m["rsi"] >= ind_5m["prev_rsi"]: return None
        sl_price = ind_5m["prev_high"] + SL_BUFFER_MULT * atr
        sl_dist  = sl_price - ind_5m["close"]
        if sl_dist > MAX_SL_MULT * atr: return None
        return {
            "direction":  "SELL",
            "entry":      round(ind_5m["close"], 2),
            "sl":         round(sl_price, 2),
            "tp":         round(ind_5m["close"] - TP_RR * sl_dist, 2),
            "sl_dist":    round(sl_dist, 2),
            "be_level":   round(ind_5m["close"] - BE_TRIGGER_MULT * atr, 2),
            "lot":        calc_lot_size(sl_dist),
            "atr":        round(atr, 2),
            "rr_planned": TP_RR,        # FIX-5
            "session":    session,      # FIX-5
            "regime":     regime,       # FIX-8
            **ind_1h,
            **{f"5m_{k}": v for k, v in ind_5m.items()},
            "timestamp":  dt_utc.isoformat(),
        }


def get_current_balance() -> float:
    if not LOG_FILE.exists():
        return ACCOUNT_BALANCE
    try:
        total_pnl = 0.0
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exit_price = row.get("exit_price", "").strip()
                if not exit_price:
                    continue
                try:
                    entry_str = row.get("actual_entry", "").strip() or row.get("entry", "").strip()
                    entry = float(entry_str)
                    lot   = float(row["lot"])
                    exit_ = float(exit_price)
                    if row["direction"] == "BUY":
                        total_pnl += (exit_ - entry) * lot * 100
                    elif row["direction"] == "SELL":
                        total_pnl += (entry - exit_) * lot * 100
                except:
                    continue
        balance = ACCOUNT_BALANCE + total_pnl
        log.info(f"Balance: ${balance:.2f} (PnL: ${total_pnl:+.2f})")
        return balance
    except Exception as e:
        log.warning(f"Could not read balance: {e}")
        return ACCOUNT_BALANCE


def calc_lot_size(sl_pts: float) -> float:
    balance  = get_current_balance()
    risk_usd = balance * RISK_PERCENT / 100
    if sl_pts <= 0:
        return MIN_LOT
    return max(MIN_LOT, round(int(risk_usd / (sl_pts * 100) / MIN_LOT) * MIN_LOT, 2))


TG_BASE        = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
_pending_queue: list = []


def send_signal_alert(signal: dict) -> None:
    direction  = signal["direction"]
    thai_time  = (datetime.fromisoformat(signal["timestamp"])
                  .replace(tzinfo=timezone.utc) + timedelta(hours=7)
                  ).strftime("%d/%m/%Y %H:%M")
    streak     = get_loss_streak()
    session    = signal.get("session", "")
    queue_size = len(_pending_queue)

    text = (
        f"XAUUSD {direction} SIGNAL\n"
        f"Time: {thai_time} (Thai) | Session: {session}\n\n"
        f"Entry: {signal['entry']}\n"
        f"Stop Loss: {signal['sl']} ({signal['sl_dist']} pts)\n"
        f"Take Profit: {signal['tp']} (RR 1:{TP_RR})\n"
        f"Break-Even: {signal['be_level']}\n"
        f"Lot Size: {signal['lot']} lot\n\n"
        f"1H: EMA21={signal['ema21']:.2f} EMA50={signal['ema50']:.2f} ADX={signal['adx']:.1f}\n"
        f"5m: RSI={signal['5m_rsi']:.1f} ATR={signal['atr']:.2f}\n\n"
        f"Streak: {streak}L"
        + (f" | Queue: {queue_size} pending" if queue_size > 1 else "")
        + "\n\nกด Confirm หรือ Skip"
    )
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "reply_markup": json.dumps({
            "inline_keyboard": [[
                {"text": "✅ Confirm", "callback_data": "confirm"},
                {"text": "❌ Skip",    "callback_data": "skip"},
            ]]
        }),
    }
    resp = requests.post(f"{TG_BASE}/sendMessage", json=payload, timeout=10)
    if not resp.json().get("ok"):
        log.error(f"Telegram send failed: {resp.json()}")


def send_text(text: str) -> None:
    requests.post(f"{TG_BASE}/sendMessage",
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)


def get_updates(offset: int = 0) -> list:
    resp = requests.get(f"{TG_BASE}/getUpdates",
                        params={"offset": offset, "timeout": 5}, timeout=15)
    return resp.json().get("result", []) if resp.ok else []


def answer_callback(callback_id: str, text: str) -> None:
    requests.post(f"{TG_BASE}/answerCallbackQuery",
                  json={"callback_query_id": callback_id, "text": text}, timeout=10)


def ensure_log() -> None:
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()
    else:
        # migrate: เพิ่ม column ใหม่ถ้า CSV เก่าไม่มี
        with open(LOG_FILE, "r", newline="") as f:
            existing_cols = csv.DictReader(f).fieldnames or []
        new_cols = [c for c in LOG_COLUMNS if c not in existing_cols]
        if new_cols:
            _migrate_csv(new_cols)


def _migrate_csv(new_cols: list) -> None:
    """เพิ่ม columns ใหม่เข้า CSV เก่าโดยไม่ลบข้อมูล"""
    import shutil
    backup = LOG_FILE.with_suffix(".bak.csv")
    shutil.copy(LOG_FILE, backup)
    rows = []
    with open(LOG_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in new_cols:
                row.setdefault(col, "")
            rows.append(row)
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV migrated: added columns {new_cols} | backup → {backup}")


def append_log(signal: dict, action: str) -> None:
    ensure_log()
    row = {
        "timestamp":    signal.get("timestamp"),
        "action":       action,
        "direction":    signal.get("direction"),
        "entry":        signal.get("entry"),
        "sl":           signal.get("sl"),
        "tp":           signal.get("tp"),
        "sl_dist":      signal.get("sl_dist"),
        "lot":          signal.get("lot"),
        "atr":          signal.get("atr"),
        "ema21":        signal.get("ema21"),
        "ema50":        signal.get("ema50"),
        "adx":          signal.get("adx"),
        "rsi":          signal.get("5m_rsi"),
        "prev_rsi":     signal.get("5m_prev_rsi"),
        "bb_lower":     signal.get("5m_bb_lower"),
        "bb_upper":     signal.get("5m_bb_upper"),
        "be_level":     signal.get("be_level"),
        "exit_price":   "",
        "actual_entry": "",
        # v2.3
        "rr_planned":   signal.get("rr_planned", TP_RR),
        "session":      signal.get("session", ""),
        "hold_time_min": "",          # กรอกตอน exit
        # v2.4
        "regime":       signal.get("regime", ""),   # FIX-8
    }
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)
    log.info(f"Log written: {action} | {signal.get('direction')} @ {signal.get('entry')} | {row['session']}")


def git_commit_log() -> None:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        subprocess.run(["git", "config", "user.email", "actions@github.com"], check=True)
        subprocess.run(["git", "config", "user.name", "GitHub Actions Bot"], check=True)
        subprocess.run(["git", "add", str(LOG_FILE)], check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True).returncode == 0:
            log.info("No changes to commit.")
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(["git", "commit", "-m", f"perf: auto-log update {now}"], check=True)
        remote = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        subprocess.run(["git", "fetch", remote, "main"], check=True)
        subprocess.run(["git", "merge", "-X", "ours", "FETCH_HEAD",
                        "--no-edit", "-m", "merge: keep local log"], check=True)
        subprocess.run(["git", "push", remote, "HEAD:main"], check=True)
        log.info("performance_log.csv pushed to GitHub")
    except subprocess.CalledProcessError as e:
        log.error(f"Git commit failed: {e}")
        subprocess.run(["git", "rebase", "--abort"], capture_output=True)
        subprocess.run(["git", "merge", "--abort"], capture_output=True)


def main() -> None:
    log.info("XAUUSD Signal Generator v2.4 starting...")
    log.info(f"Balance: ${ACCOUNT_BALANCE} | Risk: {RISK_PERCENT}% | ADX min: {ADX_MIN}")
    log.info(f"Loop: {LOOP_INTERVAL_SEC}s | Active session: {ACTIVE_SESSION_UTC[0]}:00–{ACTIVE_SESSION_UTC[1]}:00 UTC")
    log.info(f"Signal expiry: {SIGNAL_EXPIRY_MIN} min | Loss streak limit: {MAX_LOSS_STREAK}")
    ensure_log()

    global _pending_queue
    _pending_queue = []

    start_time     = time.time()
    deadline       = start_time + RUNTIME_MINUTES * 60
    last_update_id = 0
    last_signal_ts = None

    # FIX-3: 1H cache
    df_1h_cache:    pd.DataFrame | None = None
    last_1h_fetch:  datetime | None     = None
    # FIX-7: เก็บ price ล่าสุดที่รู้ ใช้ใน price-based expiry check
    _last_price:    float | None        = None

    while time.time() < deadline:
        loop_start = time.time()

        # ── Poll Telegram callbacks ──
        try:
            for upd in get_updates(offset=last_update_id + 1):
                last_update_id = upd["update_id"]
                if "callback_query" in upd and _pending_queue:
                    cb     = upd["callback_query"]
                    data   = cb["data"]
                    signal = _pending_queue.pop(0)

                    # FIX-4/7: ตรวจ expiry — time + price based
                    # _last_price อัปเดตทุก loop จาก df_5m (ดูใน signal section)
                    if is_signal_expired(signal, _last_price):
                        answer_callback(cb["id"], "Signal expired!")
                        send_text(
                            f"⚠️ Signal {signal['direction']} @ {signal['entry']} หมดอายุแล้ว\n"
                            f"(เกิน {SIGNAL_EXPIRY_MIN} นาที — ไม่บันทึก)"
                        )
                        log.warning(f"Expired signal discarded: {signal['direction']} @ {signal['entry']}")
                        continue

                    answer_callback(cb["id"], "Saved!" if data == "confirm" else "Skipped")
                    action = "CONFIRM" if data == "confirm" else "SKIP"
                    append_log(signal, action)
                    git_commit_log()
                    msg = "✅ Confirmed - Good luck!" if data == "confirm" else "❌ Skipped"
                    if _pending_queue:
                        # FIX-4/7: ทำความสะอาด expired signals ออกจาก queue ก่อน
                        _pending_queue = [s for s in _pending_queue if not is_signal_expired(s, _last_price)]
                        if _pending_queue:
                            msg += f"\n\nNext signal waiting ({len(_pending_queue)} in queue)"
                            send_text(msg)
                            send_signal_alert(_pending_queue[0])
                        else:
                            send_text(msg + "\n\n(สัญญาณที่รออยู่หมดอายุแล้ว)")
                    else:
                        send_text(msg)
        except Exception as e:
            log.warning(f"Telegram poll error: {e}")

        # ── Check signals ──
        try:
            dt_utc = datetime.now(timezone.utc)
            if not is_market_open(dt_utc):
                log.info("Market closed (weekend) - sleeping")
            elif is_rollover(dt_utc):
                log.info("Rollover window - sleeping")
            elif not is_active_session(dt_utc):
                # FIX-6: Asia / late night — skip ประหยัด API
                log.info(f"Outside active session ({get_session_name(dt_utc)}) - sleeping")
            else:
                # FIX-3: ดึง 1H เฉพาะตอนเริ่ม หรือเมื่อผ่านไป 60 นาที
                need_1h_refresh = (
                    df_1h_cache is None or
                    last_1h_fetch is None or
                    (dt_utc - last_1h_fetch).total_seconds() >= H1_CACHE_MINUTES * 60
                )
                if need_1h_refresh:
                    df_1h_cache   = fetch_ohlcv("1h", outputsize=100)
                    last_1h_fetch = dt_utc
                    log.info("1H data refreshed")

                df_5m  = fetch_ohlcv("5min", outputsize=100)
                ind_1h = build_indicators_1h(df_1h_cache)
                ind_5m = build_indicators_5m(df_5m)
                _last_price = ind_5m["close"]   # FIX-7: อัปเดตทุก loop

                current_bar_ts = df_5m.index[-1].isoformat()
                if current_bar_ts != last_signal_ts:
                    signal = check_signal(ind_1h, ind_5m, dt_utc)
                    if signal:
                        log.info(f"SIGNAL: {signal['direction']} @ {signal['entry']} | {signal['session']}")
                        _pending_queue.append(signal)
                        last_signal_ts = current_bar_ts
                        if len(_pending_queue) == 1:
                            send_signal_alert(signal)
                        else:
                            log.info(f"Signal queued ({len(_pending_queue)} total)")
                    else:
                        streak = get_loss_streak()
                        log.info(
                            f"No signal | ADX={ind_1h['adx']:.1f} | "
                            f"EMA={'>' if ind_1h['ema21'] > ind_1h['ema50'] else '<'} | "
                            f"RSI={ind_5m['rsi']:.1f} | "
                            f"ATR={ind_5m['atr']:.2f}/{ind_5m['atr_avg']:.2f} | "
                            f"Streak={streak}L | Queue={len(_pending_queue)} | "
                            f"Session={get_session_name(dt_utc)}"
                        )
        except Exception as e:
            log.error(f"Data/signal error: {e}", exc_info=True)

        time.sleep(max(0, LOOP_INTERVAL_SEC - (time.time() - loop_start)))

    log.info(f"Runtime limit reached ({RUNTIME_MINUTES} min) - exiting.")


if __name__ == "__main__":
    main()
