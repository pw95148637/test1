"""run2_stress.py — Angles D(rolling), E(regime), F(placebo), G(walk-forward), H(cost).
Emits outputs/res_stress.json and outputs/fig_rolling.png."""
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
import dca_core as C
from dca_core import significance_suite as SIG, add_week_ratio, slice_window, simulate_gbm, DOW_NAMES

OUT = "/home/user/test1/analysis/outputs"
def load(p):
    d = pd.read_csv(p); d["time"] = pd.to_datetime(d["time"], utc=True); return d
btc_d = load("data/cm_btc_daily.csv"); eth_d = load("data/cm_eth_daily.csv")
R = {}

# =============================================================== D: rolling stability
print("=== D: rolling stability ===", flush=True)
def rolling_series(daily, win_weeks=52):
    d = add_week_ratio(daily, "price", "time", tz=None, week_def="iso")
    # per-week per-dow ratio table
    wk_order = d.drop_duplicates("wk")["wk"].tolist()
    piv = d.pivot_table(index="wk", values="ratio", columns="dow", aggfunc="mean").reindex(wk_order)
    spreads, cheap = [], []
    dates = []
    wk_dates = d.groupby("wk")["time"].min().reindex(wk_order)
    for i in range(win_weeks, len(wk_order)+1):
        block = piv.iloc[i-win_weeks:i]
        m = block.mean(axis=0)
        spreads.append((m.max()-m.min())*100)
        cheap.append(int(m.idxmin()))
        dates.append(wk_dates.iloc[i-1])
    return pd.Series(spreads, index=dates), pd.Series(cheap, index=dates)

D_res = {}
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
for ax, (daily, name) in zip(axes, [(btc_d, "BTC"), (eth_d, "ETH")]):
    for win in [26, 52, 104]:
        sp, ch = rolling_series(daily, win)
        ax.plot(sp.index, sp.values, lw=1.1, label=f"{win}wk window", alpha=0.85)
        if win == 52:
            # cheapest-day flip count
            flips = int((ch.values[1:] != ch.values[:-1]).sum())
            D_res[f"{name}|52wk"] = {"n_points": len(sp),
                "spread_mean_pct": round(float(sp.mean()),3),
                "spread_last_pct": round(float(sp.iloc[-1]),3),
                "cheapest_day_flips": flips,
                "cheapest_day_flip_rate": round(flips/max(1,len(ch)-1),3),
                "distinct_cheapest_days": int(pd.Series(ch.values).nunique())}
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_title(f"{name}: rolling max-min day-of-week spread (weekly-normalized ratio)")
    ax.set_ylabel("spread %"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_rolling.png", dpi=110); plt.close()
for k,v in D_res.items(): print(k, v, flush=True)
R["D_rolling"] = D_res

# =============================================================== E: regime split
print("\n=== E: regime split (200d MA bull/bear + vol terciles) ===", flush=True)
def regime_suite(daily, name):
    d = daily.copy().sort_values("time").reset_index(drop=True)
    d["ma200"] = d["price"].rolling(200, min_periods=100).mean()
    d["bull"] = d["price"] > d["ma200"]
    d["lr"] = np.log(d["price"]).diff()
    d["rv"] = d["lr"].rolling(30, min_periods=15).std()
    out = {}
    for lab, mask in [("bull", d["bull"] == True), ("bear", d["bull"] == False)]:
        sub = d[mask].dropna(subset=["price"])
        dd = add_week_ratio(sub[["time","price"]], "price","time", tz=None, week_def="iso")
        r = SIG(dd, n_perm=2000, n_boot=800, seed=7)
        out[lab] = {"n_weeks": r["n_weeks"], "spread_pct": round(r["obs_spread_pct"],3),
                    "anova_p": round(r["anova_p"],4), "kw_p": round(r["kw_p"],4),
                    "perm_p": round(r["perm_p"],4), "cheapest": r["cheapest"], "priciest": r["priciest"]}
    # vol terciles
    q1, q2 = d["rv"].quantile([1/3, 2/3])
    for lab, mask in [("lowvol", d["rv"]<=q1), ("midvol", (d["rv"]>q1)&(d["rv"]<=q2)), ("highvol", d["rv"]>q2)]:
        sub = d[mask].dropna(subset=["price"])
        dd = add_week_ratio(sub[["time","price"]], "price","time", tz=None, week_def="iso")
        r = SIG(dd, n_perm=2000, n_boot=600, seed=7)
        out[lab] = {"n_weeks": r["n_weeks"], "spread_pct": round(r["obs_spread_pct"],3),
                    "anova_p": round(r["anova_p"],4), "kw_p": round(r["kw_p"],4),
                    "perm_p": round(r["perm_p"],4), "cheapest": r["cheapest"], "priciest": r["priciest"]}
    for k,v in out.items(): print(f"{name} {k:8s}: nwk={v['n_weeks']:4d} spread={v['spread_pct']}% A={v['anova_p']} K={v['kw_p']} P={v['perm_p']} cheap={v['cheapest']}", flush=True)
    return out
R["E_regime"] = {"BTC": regime_suite(btc_d, "BTC"), "ETH": regime_suite(eth_d, "ETH")}

# =============================================================== F: placebo (GBM null)
print("\n=== F: placebo GBM null ===", flush=True)
def _placebo_one(args):
    n_days, mu, sig, seed = args
    df = simulate_gbm(n_days, mu, sig, 100.0, seed=seed)
    d = add_week_ratio(df, "price", "time", tz=None, week_def="iso")
    r = SIG(d, n_perm=400, n_boot=0 if False else 1, seed=seed)  # n_boot tiny (unused)
    return (r["obs_spread_pct"], r["anova_p"], r["kw_p"], r["perm_p"])

def placebo(daily, name, years, n_sims=200):
    sub = slice_window(daily, years)
    lr = np.log(sub["price"]).diff().dropna()
    sig = float(lr.std()*np.sqrt(365)); mu = 0.0
    n_days = len(sub)
    # observed
    dd = add_week_ratio(sub[["time","price"]], "price","time", tz=None, week_def="iso")
    ro = SIG(dd, n_perm=2000, n_boot=1, seed=42)
    args = [(n_days, mu, sig, 1000+i) for i in range(n_sims)]
    with Pool(min(8, n_sims)) as pool:
        res = pool.map(_placebo_one, args)
    arr = np.array(res)  # cols: spread, anova, kw, perm
    spreads = arr[:,0]
    pct = (spreads < ro["obs_spread_pct"]).mean()*100
    out = {"sigma_annual": round(sig,3), "n_days": int(n_days), "n_sims": n_sims,
           "obs_spread_pct": round(ro["obs_spread_pct"],3),
           "placebo_spread_median_pct": round(float(np.median(spreads)),3),
           "placebo_spread_p95_pct": round(float(np.percentile(spreads,95)),3),
           "obs_spread_percentile_in_null": round(float(pct),1),
           "placebo_anova_p_median": round(float(np.median(arr[:,1])),3),
           "frac_placebo_anova_sig05": round(float((arr[:,1]<0.05).mean()),3)}
    print(f"{name} {years}yr: sigma={out['sigma_annual']} obs_spread={out['obs_spread_pct']}% "
          f"null_median={out['placebo_spread_median_pct']}% null_p95={out['placebo_spread_p95_pct']}% "
          f"obs@pctile={out['obs_spread_percentile_in_null']}", flush=True)
    return out
F_res = {}
for daily, name in [(btc_d,"BTC"),(eth_d,"ETH")]:
    F_res[f"{name}|full"] = placebo(daily, name, None, 200)
    F_res[f"{name}|3yr"]  = placebo(daily, name, 3, 200)
R["F_placebo"] = F_res

# =============================================================== G: walk-forward OOS
print("\n=== G: walk-forward out-of-sample ===", flush=True)
def walk_forward(daily, name, step_months=6, min_train_weeks=100, test_weeks=13):
    d = add_week_ratio(daily, "price","time", tz=None, week_def="iso")
    wk_order = d.drop_duplicates("wk")["wk"].tolist()
    wk_date = d.groupby("wk")["time"].min().reindex(wk_order)
    piv = d.pivot_table(index="wk", values="ratio", columns="dow", aggfunc="mean").reindex(wk_order)
    hits, tests = 0, 0
    detail = []
    i = min_train_weeks
    while i + test_weeks <= len(wk_order):
        train = piv.iloc[:i]; test = piv.iloc[i:i+test_weeks]
        best_day = int(train.mean(axis=0).idxmin())   # cheapest day in-sample
        # OOS: is best_day's mean ratio < 1 (beats weekly mean) more often? use mean ratio over test
        test_bestday = test[best_day].mean()
        win = test_bestday < 1.0    # beat that-week average price
        hits += int(win); tests += 1
        detail.append((str(wk_date.iloc[i].date()), DOW_NAMES[best_day], round((test_bestday-1)*100,3), bool(win)))
        i += max(1, int(step_months*4.33))
    out = {"n_tests": tests, "hits": hits, "hit_rate": round(hits/max(1,tests),3),
           "detail": detail[:40]}
    print(f"{name}: walk-forward n_tests={tests} hit_rate={out['hit_rate']} "
          f"(fraction of OOS periods where the in-sample cheapest day beat that period's weekly avg)", flush=True)
    return out
R["G_walkforward"] = {"BTC": walk_forward(btc_d,"BTC"), "ETH": walk_forward(eth_d,"ETH")}

# =============================================================== H: cost conversion
print("\n=== H: cost conversion ===", flush=True)
# realistic execution costs for a retail spot DCA
cost_scen = {
    "binance_taker_0.10pct": 0.10, "binance_bnb_0.075pct": 0.075,
    "coinbase_adv_taker_0.05-0.6pct": 0.30, "typical_spread_slippage_majors": 0.02,
}
H = {"execution_cost_bps": {k: round(v*100) for k,v in cost_scen.items()},
     "note": "spreads below are the EV edge of best-vs-worst day; a DCA buyer only ever captures "
             "at most (weekly_mean - best_day) ~ half the max-min spread, and only if the best day is knowable ex-ante."}
for lab, spread in [("full_history_BTC", 1.198), ("recent_3yr_BTC", 0.444),
                    ("full_history_ETH", 1.078), ("recent_3yr_ETH", 0.501),
                    ("real_intraday_BTC_2013-19", 0.797)]:
    capturable = spread/2  # best day vs weekly mean, optimistic
    H[lab] = {"maxmin_spread_pct": spread, "optimistic_capturable_vs_weekmean_pct": round(capturable,3),
              "vs_binance_taker": round(capturable - 0.10, 3),
              "net_after_taker_and_spread_pct": round(capturable - 0.10 - 0.02, 3)}
    print(f"{lab}: spread={spread}% capturable~{capturable:.3f}% net_after_costs={H[lab]['net_after_taker_and_spread_pct']}%", flush=True)
R["H_cost"] = H

json.dump(R, open(f"{OUT}/res_stress.json","w"), indent=1)
print("\nSAVED res_stress.json + fig_rolling.png", flush=True)
