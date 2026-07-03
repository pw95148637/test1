"""run1_core.py — Angles A, B, C, J, L, I(week-def/perm-count). Emits outputs/res_core.json."""
import pandas as pd, numpy as np, json, dca_core as C
from dca_core import significance_suite as SIG, add_week_ratio, daily_from_intraday, slice_window

OUT = "/home/user/test1/analysis/outputs"
def load(p):
    d = pd.read_csv(p); d["time"] = pd.to_datetime(d["time"], utc=True); return d

btc_d = load("data/cm_btc_daily.csv")
eth_d = load("data/cm_eth_daily.csv")
btc_ih = load("data/btc_intraday_1h.csv")   # real exchange 1h, 2013-2019
R = {}

def cell(daily, tz, week_def="iso", price_col="price", n_perm=2000, n_boot=1000, seed=42):
    d = add_week_ratio(daily, price_col, "time", tz=(tz if tz != "UTC" else None) if _is_intraday_daily(daily) else None,
                       week_def=week_def)
    return SIG(d, n_perm=n_perm, n_boot=n_boot, seed=seed)

def _is_intraday_daily(_): return False  # daily frames already have local day key; handled per-call

# ---- helper: build a one-price-per-day frame for a given source/tz/convention/granularity
def daily_series(source, tz="UTC", price="close", gran="1h"):
    """source: 'btc_daily','eth_daily', or intraday df. Returns df[time, price] one row/local-day."""
    if source == "btc_daily":
        return btc_d.rename(columns={"price": "price"})[["time", "price"]]
    if source == "eth_daily":
        return eth_d[["time", "price"]]
    # intraday BTC
    df = btc_ih
    if gran == "4h":
        g = df.set_index("time").resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "vwap": "mean", "vol": "sum"}).dropna(subset=["close"]).reset_index()
        df = g
    elif gran == "daily":
        g = df.set_index("time").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "vwap": "mean", "vol": "sum"}).dropna(subset=["close"]).reset_index()
        df = g
    return daily_from_intraday(df, tz=tz, price=price)

def suite_on(source, tz="UTC", price="close", gran="1h", week_def="iso", years=None,
             n_perm=2000, n_boot=1000, seed=42):
    ds = daily_series(source, tz=tz, price=price, gran=gran)
    ds = slice_window(ds, years)
    # ds time is local-day-normalized already; add_week_ratio with tz=None keeps that calendar
    d = add_week_ratio(ds, "price", "time", tz=None, week_def=week_def)
    return SIG(d, n_perm=n_perm, n_boot=n_boot, seed=seed)

def _rp(v):
    """Round preserving small p-values (3 significant figures) instead of crushing to 0.0."""
    if not isinstance(v, float):
        return v
    if v == 0.0 or not np.isfinite(v):
        return v
    import math
    return round(v, max(4, -int(math.floor(math.log10(abs(v)))) + 2))

def fmt(r):
    return {k: _rp(v) for k, v in r.items()
            if k not in ("day_means_pct", "ci_lo_pct", "ci_hi_pct")} | {
        "day_means_pct": [round(x, 3) for x in r["day_means_pct"]],
        "ci_lo_pct": [round(x, 3) for x in r["ci_lo_pct"]],
        "ci_hi_pct": [round(x, 3) for x in r["ci_hi_pct"]]}

# =============================================================== C: significance master table
print("=== C: significance master table ===", flush=True)
C_tab = {}
# (a) daily reference-rate, UTC only, BTC & ETH x {full,3yr,1yr}
for src, name in [("btc_daily", "BTC-daily"), ("eth_daily", "ETH-daily")]:
    for yrs, lab in [(None, "full"), (3, "3yr"), (1, "1yr")]:
        r = suite_on(src, tz="UTC", years=yrs)
        C_tab[f"{name}|UTC|{lab}"] = fmt(r)
        print(f"{name:9s} UTC {lab:4s} nwk={r['n_weeks']:4d} spread={r['obs_spread_pct']:.3f}% "
              f"A={r['anova_p']:.3g} K={r['kw_p']:.3g} P={r['perm_p']:.3g} cheap={r['cheapest']} pricey={r['priciest']}", flush=True)
# (b) real intraday BTC x {full, 2013-15, 2016-19} x {UTC, Asia/Taipei}
splits = {"full": (None, None),
          "2013-2015": (pd.Timestamp("2013-01-01", tz="UTC"), pd.Timestamp("2016-01-01", tz="UTC")),
          "2016-2019": (pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"))}
for tz in ["UTC", "Asia/Taipei"]:
    for lab, (lo, hi) in splits.items():
        ds = daily_series("intraday", tz=tz, price="close", gran="1h")
        if lo is not None:
            ds = ds[(ds["time"] >= lo) & (ds["time"] < hi)]
        d = add_week_ratio(ds, "price", "time", tz=None, week_def="iso")
        r = SIG(d, n_perm=2000, n_boot=1000, seed=42)
        C_tab[f"BTC-intraday|{tz}|{lab}"] = fmt(r)
        print(f"BTC-intra {tz:11s} {lab:9s} nwk={r['n_weeks']:4d} spread={r['obs_spread_pct']:.3f}% "
              f"A={r['anova_p']:.3g} K={r['kw_p']:.3g} P={r['perm_p']:.3g} cheap={r['cheapest']} pricey={r['priciest']}", flush=True)
R["C_significance"] = C_tab

# =============================================================== B: three-indicator consistency
print("\n=== B: three-indicator consistency ===", flush=True)
B_tab = {}
def three_ind(source, tz="UTC", years=None, gran="1h"):
    ds = slice_window(daily_series(source, tz=tz, price="close", gran=gran), years)
    d = add_week_ratio(ds, "price", "time", tz=None, week_def="iso")
    ratio_means = d.groupby("dow")["ratio"].mean().reindex(range(7)).values
    rank_ratio = C.ranking_of(ratio_means, ascending=True)          # low ratio = cheap
    lr = C.logret_by_dow(ds, "price", "time")                       # per-dow mean logret
    rank_lr = C.ranking_of(-lr.values, ascending=True)              # most-negative return day = cheapest to buy? -> low price
    hc = C.harmonic_cost_by_dow(ds, "price", "time", detrend="ma", ma_win=90)
    rank_hc = C.ranking_of(hc.values, ascending=True)               # low harmonic cost = cheap
    return {"ratio_means_pct": [round((x-1)*100,3) for x in ratio_means],
            "rank_ratio": rank_ratio, "rank_logret": rank_lr, "rank_harmonic": rank_hc,
            "cheapest_ratio": rank_ratio[0], "cheapest_harmonic": rank_hc[0],
            "agree_cheapest": rank_ratio[0]==rank_hc[0]}
for src, name in [("btc_daily","BTC-daily"),("eth_daily","ETH-daily")]:
    for yrs,lab in [(None,"full"),(3,"3yr")]:
        B_tab[f"{name}|{lab}"] = three_ind(src, years=yrs)
        t=B_tab[f"{name}|{lab}"]
        print(f"{name} {lab}: ratio_cheap={t['cheapest_ratio']} harm_cheap={t['cheapest_harmonic']} agree={t['agree_cheapest']}", flush=True)
        print(f"   rank_ratio   {t['rank_ratio']}\n   rank_harmonic{t['rank_harmonic']}", flush=True)
for tz in ["UTC","Asia/Taipei"]:
    B_tab[f"BTC-intraday|{tz}|full"] = three_ind("intraday", tz=tz)
    t=B_tab[f"BTC-intraday|{tz}|full"]
    print(f"BTC-intra {tz}: ratio_cheap={t['cheapest_ratio']} harm_cheap={t['cheapest_harmonic']} agree={t['agree_cheapest']}", flush=True)
R["B_three_indicator"] = B_tab

# =============================================================== J: price convention
print("\n=== J: price convention (close/open/vwap) on intraday BTC full ===", flush=True)
J_tab={}
for price in ["close","open","vwap"]:
    for tz in ["UTC","Asia/Taipei"]:
        r = suite_on("intraday", tz=tz, price=price, gran="1h")
        J_tab[f"{price}|{tz}"]=fmt(r)
        print(f"{price:5s} {tz:11s} spread={r['obs_spread_pct']:.3f}% A={r['anova_p']:.3g} K={r['kw_p']:.3g} P={r['perm_p']:.3g} cheap={r['cheapest']} pricey={r['priciest']}", flush=True)
R["J_convention"]=J_tab

# =============================================================== L: granularity
print("\n=== L: granularity (1h/4h/daily bars -> daily close) ===", flush=True)
L_tab={}
for gran in ["1h","4h","daily"]:
    for tz in ["UTC","Asia/Taipei"]:
        r = suite_on("intraday", tz=tz, price="close", gran=gran)
        L_tab[f"{gran}|{tz}"]=fmt(r)
        print(f"{gran:5s} {tz:11s} spread={r['obs_spread_pct']:.3f}% A={r['anova_p']:.3g} K={r['kw_p']:.3g} P={r['perm_p']:.3g} cheap={r['cheapest']} pricey={r['priciest']}", flush=True)
R["L_granularity"]=L_tab

# =============================================================== I: week-def + perm-count sensitivity
print("\n=== I: week-def (iso vs roll7) + perm-count (2000 vs 5000) ===", flush=True)
I_tab={}
for src,name in [("btc_daily","BTC-daily"),("eth_daily","ETH-daily")]:
    for yrs,lab in [(None,"full"),(3,"3yr")]:
        for wd in ["iso","roll7"]:
            r=suite_on(src, tz="UTC", years=yrs, week_def=wd)
            I_tab[f"{name}|{lab}|{wd}"]=fmt(r)
            print(f"{name} {lab} {wd:5s}: spread={r['obs_spread_pct']:.3f}% A={r['anova_p']:.3g} K={r['kw_p']:.3g} P={r['perm_p']:.3g} cheap={r['cheapest']}", flush=True)
# perm-count convergence on two key cells
for src,name,yrs,lab in [("btc_daily","BTC-daily",None,"full"),("btc_daily","BTC-daily",3,"3yr")]:
    r2=suite_on(src, tz="UTC", years=yrs, n_perm=2000, seed=1)
    r5=suite_on(src, tz="UTC", years=yrs, n_perm=5000, seed=1)
    I_tab[f"{name}|{lab}|perm2000"]={"perm_p":round(r2["perm_p"],4)}
    I_tab[f"{name}|{lab}|perm5000"]={"perm_p":round(r5["perm_p"],4)}
    print(f"{name} {lab}: perm_p 2000={r2['perm_p']:.4f} 5000={r5['perm_p']:.4f}", flush=True)
R["I_sensitivity"]=I_tab

json.dump(R, open(f"{OUT}/res_core.json","w"), indent=1)
print("\nSAVED res_core.json", flush=True)
