"""run3_extra.py — Angle K (cross-asset generalization) + M (named market-cycle segments).
Emits outputs/res_extra.json."""
import pandas as pd, numpy as np, json
import dca_core as C
from dca_core import significance_suite as SIG, add_week_ratio, slice_window

SCRATCH = "/tmp/claude-0/-home-user-test1/788aa9d5-a0f7-5fbc-a121-e615483b7031/scratchpad"
OUT = "/home/user/test1/analysis/outputs"
R = {}

def load_cm(asset):
    f = f"{SCRATCH}/test-cm/csv/{asset}.csv"
    head = pd.read_csv(f, nrows=0).columns
    cols = ["time"] + [c for c in ("ReferenceRateUSD", "PriceUSD") if c in head]
    d = pd.read_csv(f, usecols=cols)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    if "ReferenceRateUSD" in d and "PriceUSD" in d:
        d["price"] = d["ReferenceRateUSD"].fillna(d["PriceUSD"])
    else:
        d["price"] = d[cols[1]]
    d = d.dropna(subset=["price"]); d = d[d["price"] > 0].sort_values("time").reset_index(drop=True)
    return d[["time", "price"]]

def core_cell(daily, years=None, n_perm=2000):
    ds = slice_window(daily, years)
    if len(ds) < 120:
        return None
    d = add_week_ratio(ds, "price", "time", tz=None, week_def="iso")
    r = SIG(d, n_perm=n_perm, n_boot=600, seed=11)
    ratio_means = d.groupby("dow")["ratio"].mean().reindex(range(7)).values
    hc = C.harmonic_cost_by_dow(ds, "price", "time", detrend="ma", ma_win=90)
    return {"n_weeks": r["n_weeks"], "spread_pct": round(r["obs_spread_pct"], 3),
            "anova_p": round(r["anova_p"], 4), "kw_p": round(r["kw_p"], 4), "perm_p": round(r["perm_p"], 4),
            "cheapest_ratio": r["cheapest"], "priciest_ratio": r["priciest"],
            "cheapest_harmonic": C.ranking_of(hc.values, ascending=True)[0]}

# =============================================================== K: cross-asset
print("=== K: cross-asset generalization ===", flush=True)
K_tab = {}
for asset in ["sol", "bnb", "ada", "xrp", "ltc", "doge"]:
    d = load_cm(asset)
    for yrs, lab in [(None, "full"), (3, "3yr")]:
        r = core_cell(d, yrs)
        if r:
            K_tab[f"{asset.upper()}|{lab}"] = r
            print(f"{asset.upper():5s} {lab:4s}: nwk={r['n_weeks']:4d} spread={r['spread_pct']}% "
                  f"A={r['anova_p']} K={r['kw_p']} P={r['perm_p']} cheap(ratio)={r['cheapest_ratio']} cheap(harm)={r['cheapest_harmonic']}", flush=True)
R["K_cross_asset"] = K_tab

# =============================================================== M: named market cycles
print("\n=== M: named market-cycle segments ===", flush=True)
cycles = [
    ("2017-18_bull_crash", "2017-01-01", "2018-12-31"),
    ("2018-20_bear_base",  "2019-01-01", "2020-03-31"),
    ("2020-21_covid_inst_bull", "2020-04-01", "2021-11-30"),
    ("2022_terra_ftx_bear", "2021-12-01", "2022-12-31"),
    ("2023-24_recovery",   "2023-01-01", "2024-12-31"),
    ("2024-25_etf_era",    "2024-01-01", "2025-06-30"),
    ("2025-26_now",        "2025-01-01", "2026-05-24"),
]
btc = load_cm("btc"); eth = load_cm("eth")
M_tab = {}
for name, lo, hi in cycles:
    lo = pd.Timestamp(lo, tz="UTC"); hi = pd.Timestamp(hi, tz="UTC")
    for daily, asset in [(btc, "BTC"), (eth, "ETH")]:
        sub = daily[(daily["time"] >= lo) & (daily["time"] <= hi)]
        if len(sub) < 120:
            continue
        d = add_week_ratio(sub[["time", "price"]], "price", "time", tz=None, week_def="iso")
        r = SIG(d, n_perm=2000, n_boot=500, seed=13)
        M_tab[f"{asset}|{name}"] = {"n_weeks": r["n_weeks"], "spread_pct": round(r["obs_spread_pct"], 3),
            "anova_p": round(r["anova_p"], 4), "kw_p": round(r["kw_p"], 4), "perm_p": round(r["perm_p"], 4),
            "cheapest": r["cheapest"], "priciest": r["priciest"]}
        t = M_tab[f"{asset}|{name}"]
        print(f"{asset} {name:26s}: nwk={t['n_weeks']:3d} spread={t['spread_pct']:.3f}% A={t['anova_p']} K={t['kw_p']} P={t['perm_p']} cheap={t['cheapest']}", flush=True)
R["M_cycles"] = M_tab

json.dump(R, open(f"{OUT}/res_extra.json", "w"), indent=1)
print("\nSAVED res_extra.json", flush=True)
