"""
prep_data.py — Build clean, validated real-data series for the DCA day-of-week study.

Sources (all REAL, plain-committed public data reachable under the sandbox network policy):
  1. CoinMetrics community reference rate (daily, UTC)  -> BTC 2010+, ETH 2015+, to 2026-05-24
  2. Bitfinex BTCUSD 1-min candles (2013-04 .. 2017-12) -> real exchange intraday
  3. Coinbase BTCUSD 1-hour candles (2017-07 .. 2019-07) -> real exchange intraday

Exchange APIs / data.binance.vision / all recent intraday (LFS) are BLOCKED by egress policy;
see REPORT for the full data-access ledger. This script only touches confirmed-real files.
"""
import pandas as pd, numpy as np, glob, os, sys

SCRATCH = "/tmp/claude-0/-home-user-test1/788aa9d5-a0f7-5fbc-a121-e615483b7031/scratchpad"
OUT = "/home/user/test1/analysis/data"
os.makedirs(OUT, exist_ok=True)

def log(*a): print(*a, flush=True)

# ---------------------------------------------------------------- CoinMetrics daily
def prep_coinmetrics():
    for asset in ["btc", "eth"]:
        f = f"{SCRATCH}/test-cm/csv/{asset}.csv"
        df = pd.read_csv(f, usecols=["time", "PriceUSD", "ReferenceRateUSD"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        # Prefer ReferenceRateUSD (populated to the tail); fall back to PriceUSD.
        df["price"] = df["ReferenceRateUSD"].fillna(df["PriceUSD"])
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0].sort_values("time").reset_index(drop=True)
        df[["time", "price"]].to_csv(f"{OUT}/cm_{asset}_daily.csv", index=False)
        log(f"[coinmetrics] {asset}: n={len(df)}  {df.time.iloc[0].date()} .. {df.time.iloc[-1].date()}")

# ---------------------------------------------------------------- Bitfinex 1m -> 1h
def prep_bitfinex():
    # columns: ts_ms, open, close, high, low, volume  (Bitfinex candle order)
    files = sorted(glob.glob(f"{SCRATCH}/bitfinex/BTCUSD/Candles_1m/*/merged.csv"))
    if not files:
        log("[bitfinex] no files"); return None
    parts = []
    for fp in files:
        d = pd.read_csv(fp, header=None,
                        names=["ts", "open", "close", "high", "low", "vol"])
        parts.append(d)
    m = pd.concat(parts, ignore_index=True)
    m["time"] = pd.to_datetime(m["ts"], unit="ms", utc=True)
    m = m.dropna(subset=["close"]).sort_values("time").drop_duplicates("time")
    m = m[(m["close"] > 0)]
    m = m.set_index("time")
    # resample to hourly OHLC + volume + close*vol for VWAP approx
    g = m.resample("1h")
    h = pd.DataFrame({
        "open":  g["open"].first(),
        "high":  g["high"].max(),
        "low":   g["low"].min(),
        "close": g["close"].last(),
        "vol":   g["vol"].sum(),
        "pv":    (m["close"] * m["vol"]).resample("1h").sum(),
    }).dropna(subset=["close"])
    h["vwap"] = np.where(h["vol"] > 0, h["pv"] / h["vol"], h["close"])
    h = h.drop(columns="pv").reset_index()
    h.to_csv(f"{OUT}/btc_bitfinex_1h.csv", index=False)
    log(f"[bitfinex] BTC 1h: n={len(h)}  {h.time.iloc[0]} .. {h.time.iloc[-1]}")
    return h

# ---------------------------------------------------------------- Coinbase 1h
def prep_coinbase():
    f = f"{SCRATCH}/rltrader/data/input/coinbase-1h-btc-usd.csv"
    d = pd.read_csv(f)
    d["time"] = pd.to_datetime(d["Date"], utc=True)
    d = d.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                          "VolumeFrom": "vol"})
    d = d[["time", "open", "high", "low", "close", "vol"]].dropna(subset=["close"])
    d = d[d["close"] > 0].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    d["vwap"] = d["close"]  # no intra-hour info; VWAP proxy = close
    d.to_csv(f"{OUT}/btc_coinbase_1h.csv", index=False)
    log(f"[coinbase] BTC 1h: n={len(d)}  {d.time.iloc[0]} .. {d.time.iloc[-1]}")
    return d

# ---------------------------------------------------------------- Binance 1h (real, requested exchange)
def prep_binance():
    f = f"{SCRATCH}/hunt_binance/binance-BTCUSDT-1h.csv"
    if not os.path.exists(f):
        log("[binance] file missing"); return None
    d = pd.read_csv(f)
    d["time"] = pd.to_datetime(d["open_timestamp_utc"], unit="s", utc=True)
    d = d[["time", "open", "high", "low", "close", "volume"]].rename(columns={"volume": "vol"})
    d = d.dropna(subset=["close"]); d = d[d["close"] > 0].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    d["vwap"] = d["close"]  # no intra-hour detail
    d.to_csv(f"{OUT}/btc_binance_1h.csv", index=False)
    log(f"[binance] BTC 1h: n={len(d)}  {d.time.iloc[0]} .. {d.time.iloc[-1]}")
    return d

def _xval(a, b, na, nb):
    j = a[["time","close"]].rename(columns={"close":na}).merge(
        b[["time","close"]].rename(columns={"close":nb}), on="time", how="inner")
    if len(j) > 10:
        corr = j[na].corr(j[nb]); rd = ((j[na]-j[nb]).abs()/j[nb]).median()
        log(f"[xval] {na} vs {nb}: n={len(j)} corr={corr:.5f} median|reldiff|={rd*100:.3f}%")

# ---------------------------------------------------------------- cross-validate + merge
def crossval_and_merge(bfx, cb, bnc=None):
    # overlap window
    lo = max(bfx.time.min(), cb.time.min())
    hi = min(bfx.time.max(), cb.time.max())
    log(f"[xval] overlap {lo} .. {hi}")
    a = bfx[(bfx.time >= lo) & (bfx.time <= hi)][["time", "close"]].rename(columns={"close": "bfx"})
    b = cb[(cb.time >= lo) & (cb.time <= hi)][["time", "close"]].rename(columns={"close": "cb"})
    j = a.merge(b, on="time", how="inner")
    if len(j) > 10:
        corr = j["bfx"].corr(j["cb"])
        reldiff = ((j["bfx"] - j["cb"]).abs() / j["cb"]).median()
        log(f"[xval] n={len(j)} hourly-close corr(bitfinex,coinbase)={corr:.5f}  median|reldiff|={reldiff*100:.3f}%")
    # three-way cross-validation in overlaps
    if bnc is not None:
        _xval(bfx, bnc, "bitfinex", "binance")
        _xval(cb, bnc, "coinbase", "binance")
    # merge: Bitfinex up to Binance start (2017-08-17), then Binance (requested exchange) onward
    cols = ["time", "open", "high", "low", "close", "vwap", "vol"]
    if bnc is not None:
        cut = bnc.time.min()
        merged = pd.concat([bfx[bfx.time < cut][cols], bnc[bnc.time >= cut][cols]],
                           ignore_index=True)
        src = "Bitfinex(<2017-08-17)+Binance(onward)"
    else:
        cut = pd.Timestamp("2017-07-01", tz="UTC")
        merged = pd.concat([bfx[bfx.time < cut][cols], cb[cb.time >= cut][cols]], ignore_index=True)
        src = "Bitfinex+Coinbase"
    merged = merged.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    merged.to_csv(f"{OUT}/btc_intraday_1h.csv", index=False)
    log(f"[merge] BTC intraday 1h ({src}): n={len(merged)}  {merged.time.iloc[0]} .. {merged.time.iloc[-1]}")
    return merged

if __name__ == "__main__":
    prep_coinmetrics()
    bfx = prep_bitfinex()
    cb = prep_coinbase()
    bnc = prep_binance()
    if bfx is not None and cb is not None:
        crossval_and_merge(bfx, cb, bnc)
    log("DONE prep")
