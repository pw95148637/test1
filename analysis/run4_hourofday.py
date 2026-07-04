"""run4_hourofday.py — Hour-of-day EV effect + liquidity(volume)-by-hour on REAL intraday BTC.
Same methodology as the day-of-week study, one level finer: normalize WITHIN each day
(ratio = hourly close / that day's mean), group by hour-of-day, 3-line significance
(permutation shuffles hour labels WITHIN each day, preserving day structure)."""
import pandas as pd, numpy as np, json
from scipy import stats

d = pd.read_csv("data/btc_intraday_1h.csv"); d["time"] = pd.to_datetime(d["time"], utc=True)
print(f"data: n={len(d)}  {d.time.min()} .. {d.time.max()}  (real exchange: Bitfinex+Binance)")

def perm_p_k(ratio, day_codes, hod, obs_spread, k, n_perm=2000, seed=42):
    rng = np.random.default_rng(seed); n = len(ratio)
    base = np.argsort(day_codes, kind="stable")
    cnt = 0
    for _ in range(n_perm):
        key = rng.random(n)
        perm = base[np.lexsort((key[base], day_codes[base]))]
        nh = np.empty(n, np.int64); nh[base] = hod[perm]
        s = np.bincount(nh, weights=ratio, minlength=k); c = np.bincount(nh, minlength=k).astype(float)
        m = np.divide(s, c, out=np.full(k, np.nan), where=c > 0)
        if (np.nanmax(m) - np.nanmin(m)) >= obs_spread: cnt += 1
    return cnt / n_perm

def analyse(tz, sub=None, lab="full"):
    x = d.copy()
    t = x["time"].dt.tz_convert(tz)
    if sub: t0, t1 = sub; keep = (x["time"] >= t0) & (x["time"] < t1); x = x[keep]; t = t[keep]
    x["day"] = t.dt.normalize(); x["hod"] = t.dt.hour
    dm = x.groupby("day")["close"].transform("mean")
    x["ratio"] = x["close"] / dm
    means = x.groupby("hod")["ratio"].mean().reindex(range(24)).values
    spread = np.nanmax(means) - np.nanmin(means)
    groups = [x[x.hod == h]["ratio"].values for h in range(24)]
    _, ap = stats.f_oneway(*groups); _, kp = stats.kruskal(*groups)
    pp = perm_p_k(x["ratio"].to_numpy(float), pd.factorize(x["day"])[0], x["hod"].to_numpy(np.int64),
                  spread, 24, n_perm=2000)
    vol = x.groupby("hod")["vol"].mean().reindex(range(24))
    volshare = (vol / vol.sum() * 100).values
    cheap_h = int(np.nanargmin(means)); dear_h = int(np.nanargmax(means))
    return {"tz": tz, "lab": lab, "n": len(x), "n_days": x["day"].nunique(),
            "spread_pct": spread*100, "anova_p": ap, "kw_p": kp, "perm_p": pp,
            "cheapest_hour": cheap_h, "priciest_hour": dear_h,
            "hour_dev_pct": (means-1)*100, "vol_share_pct": volshare,
            "min_vol_hour": int(np.nanargmin(vol.values)), "max_vol_hour": int(np.nanargmax(vol.values)),
            "vol_ratio_max_min": float(np.nanmax(vol.values)/np.nanmin(vol.values))}

R = {}
for tz in ["UTC", "Asia/Taipei"]:
    r = analyse(tz)
    R[tz] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    print(f"\n=== {tz} (full 2013-2019) ===")
    print(f"n={r['n']} days={r['n_days']} hour-of-day spread={r['spread_pct']:.3f}% "
          f"ANOVA={r['anova_p']:.3g} KW={r['kw_p']:.3g} perm={r['perm_p']:.3g}")
    print(f"cheapest hour={r['cheapest_hour']:02d}:00  priciest hour={r['priciest_hour']:02d}:00 ({tz})")
    print(f"volume: busiest={r['max_vol_hour']:02d}:00  thinnest={r['min_vol_hour']:02d}:00  "
          f"busiest/thinnest vol ratio={r['vol_ratio_max_min']:.2f}x")
    dev = r["hour_dev_pct"]; vs = r["vol_share_pct"]
    print("hour  price_dev%   vol_share%")
    for h in range(24):
        bar = "#" * int(round(vs[h]))
        print(f" {h:02d}:00  {dev[h]:+7.3f}   {vs[h]:5.2f} {bar}")

json.dump(R, open("outputs/res_hourofday.json", "w"), indent=1)
print("\nSAVED res_hourofday.json")
