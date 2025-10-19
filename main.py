import os
import time
import math
import logging
from datetime import datetime, timedelta, timezone, time as dtime
from typing import List, Dict, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, AssetClass
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# -----------------------
# Config
# -----------------------
API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")
PAPER = os.environ.get("APCA_PAPER", "true").lower() in ("1", "true", "yes")

RUN_EVERY_SECONDS = int(os.environ.get("RUN_EVERY_SECONDS", "3600"))
BARS_NEEDED = 300  # a bit more for MACD/EMA warmup

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
ETF_ALLOWLIST = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "ARKK"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alpaca-momo-bot")

ET_TZ = ZoneInfo("America/New_York")

# -----------------------
# Indicator helpers
# -----------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# -----------------------
# Universe helpers
# -----------------------

def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i:i+size] for i in range(0, len(seq), size)]


def unique_upper(symbols: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in symbols:
        u = s.strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_sp500_symbols() -> List[str]:
    try:
        df = pd.read_csv(SP500_URL)
        col = 'Symbol' if 'Symbol' in df.columns else ('Ticker' if 'Ticker' in df.columns else None)
        if not col:
            raise RuntimeError("S&P 500 CSV missing Symbol/Ticker column")
        syms = unique_upper(df[col].tolist())
        if not syms:
            raise RuntimeError("S&P 500 list parsed empty symbol set")
        return syms
    except Exception as e:
        raise RuntimeError(f"Failed to fetch S&P 500 symbols: {e}")


def build_universe(trading: TradingClient) -> List[str]:
    base = set(fetch_sp500_symbols()) | set(ETF_ALLOWLIST)
    req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = trading.get_assets(req)
    tradable = {a.symbol for a in assets if getattr(a, 'tradable', False)}
    universe = sorted(base & tradable)
    logger.info(f"Built universe of {len(universe)} tradable symbols")
    return universe

# -----------------------
# Momentum buy signal
# -----------------------

def compute_signals(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Momentum rider: look for healthy uptrends likely to continue.
    Conditions (15m bars):
      1) Trend: MA60 > MA240 (uptrend filter)
      2) Price above MA60 (participating in the trend)
      3) RSI14 in 55–70 and rising vs previous bar (momentum without being overbought)
      4) MACD histogram > 0 and increasing vs previous bar (acceleration)
    """
    out: Dict[str, Dict[str, float]] = {}
    for symbol, sdf in df.groupby(level=0):
        closes = sdf['close']
        if closes.size < 260:
            continue
        ma60 = closes.rolling(window=60, min_periods=60).mean()
        ma240 = closes.rolling(window=240, min_periods=240).mean()
        rsi14 = rsi(closes, 14)
        macd_line, macd_sig, macd_hist = macd(closes)

        last_idx = -1
        prev_idx = -2
        try:
            c_last = float(closes.iloc[last_idx])
            ma60_last = float(ma60.iloc[last_idx])
            ma240_last = float(ma240.iloc[last_idx])
            rsi_last = float(rsi14.iloc[last_idx])
            rsi_prev = float(rsi14.iloc[prev_idx])
            hist_last = float(macd_hist.iloc[last_idx])
            hist_prev = float(macd_hist.iloc[prev_idx])
        except Exception:
            continue

        trend_up = (not math.isnan(ma60_last)) and (not math.isnan(ma240_last)) and ma60_last > ma240_last
        price_above_ma = (not math.isnan(ma60_last)) and c_last > ma60_last
        rsi_ok = (not math.isnan(rsi_last)) and (55 <= rsi_last <= 70) and (not math.isnan(rsi_prev)) and rsi_last > rsi_prev
        macd_ok = (not math.isnan(hist_last)) and (not math.isnan(hist_prev)) and hist_last > 0 and hist_last > hist_prev

        buy = bool(trend_up and price_above_ma and rsi_ok and macd_ok)

        out[symbol] = {
            "ma60": ma60_last,
            "ma240": ma240_last,
            "rsi14": rsi_last,
            "rsi14_prev": rsi_prev,
            "macd_hist": hist_last,
            "macd_hist_prev": hist_prev,
            "price": c_last,
            "buy_signal": buy,
        }
    return out

# -----------------------
# Session gating
# -----------------------

def is_trading_session_now(trading: TradingClient) -> bool:
    """Return True only during Tue–Fri, 09:30–16:00 ET, and when Alpaca says market is open."""
    now_et = datetime.now(ET_TZ)
    if now_et.weekday() not in (1, 2, 3, 4):
        logger.info("Outside Tue–Fri; skipping this cycle.")
        return False
    try:
        clock = trading.get_clock()
        if not clock.is_open:
            logger.info(f"Market closed (next open: {clock.next_open}). Skipping.")
            return False
    except Exception as e:
        logger.warning(f"Clock check failed ({e}); falling back to time window only.")
    t = now_et.time()
    if not (dtime(9, 30) <= t < dtime(16, 0)):
        logger.info("Outside regular hours 09:30–16:00 ET; skipping this cycle.")
        return False
    return True

# -----------------------
# Core loop
# -----------------------

def run_once():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set.")

    trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    if not is_trading_session_now(trading):
        return

    account = trading.get_account()
    available_cash = float(getattr(account, 'cash', account.buying_power))
    logger.info(f"Available cash: ${available_cash:,.2f}")

    if available_cash < 1.0:
        logger.info("Available cash < $1. Nothing to do.")
        return

    symbols = build_universe(trading)
    if not symbols:
        logger.error("No symbols in universe after filtering. Exiting this cycle.")
        return

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=15 * (BARS_NEEDED + 5))

    all_frames = []
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
    for batch in chunked(symbols, BATCH_SIZE):
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed="sip",
                limit=BARS_NEEDED,
            )
            bars = data_client.get_stock_bars(req)
            records = []
            for symbol, sb in bars.data.items():
                for b in sb:
                    records.append({
                        "symbol": symbol,
                        "timestamp": b.timestamp,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": int(b.volume),
                    })
            if records:
                frame = pd.DataFrame.from_records(records)
                frame.set_index(["symbol", "timestamp"], inplace=True)
                all_frames.append(frame)
        except Exception as e:
            logger.error(f"Data fetch error for batch {batch}: {e}")

    if not all_frames:
        logger.warning("No bars fetched. Exiting this cycle.")
        return

    df = pd.concat(all_frames).sort_index()
    sigs = compute_signals(df)

    notional_per_trade = available_cash * 0.05
    if notional_per_trade < 1.0:
        logger.info(f"5% of cash (${notional_per_trade:.2f}) is < $1. Skipping orders.")
        return

    for sym, s in sigs.items():
        if s.get("buy_signal"):
            try:
                order = MarketOrderRequest(
                    symbol=sym,
                    notional=round(notional_per_trade, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                submitted = trading.submit_order(order)
                logger.info(
                    f"BUY {sym}: ${notional_per_trade:.2f} | price={s.get('price')}, ma60={s.get('ma60')}, ma240={s.get('ma240')}, rsi={s.get('rsi14')}↑{s.get('rsi14_prev')}, macd_hist={s.get('macd_hist')}↑{s.get('macd_hist_prev')} | order id {submitted.id}"
                )
            except Exception as e:
                logger.error(f"Failed to submit order for {sym}: {e}")
        else:
            logger.info(
                f"NO BUY {sym}: price={s.get('price')}, ma60={s.get('ma60')}, ma240={s.get('ma240')}, rsi={s.get('rsi14')} vs {s.get('rsi14_prev')}, macd_hist={s.get('macd_hist')} vs {s.get('macd_hist_prev')}"
            )


def main():
    logger.info("Starting hourly loop (momentum mode)...")
    while True:
        try:
            run_once()
        except Exception:
            logger.exception("Cycle error")
        finally:
            time.sleep(RUN_EVERY_SECONDS)


if __name__ == "__main__":
    main()
