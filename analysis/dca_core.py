"""
dca_core.py — Core engine for the BTC/ETH weekly-DCA day-of-week study.

Design decisions (locked per task spec):
  * Within-week NORMALIZE first: ratio = daily_price / weekly_mean_price. All "which day is
    cheap/expensive" conclusions are based on this ratio, never on raw cross-era prices.
  * Significance THREE ways: ANOVA (f_oneway), Kruskal-Wallis, permutation (shuffle dow labels
    WITHIN each week, preserving week structure), >=2000 perms.
  * Bootstrap CI by resampling WHOLE WEEKS with replacement (not per-row).

Everything is tz-aware and convention-parametrized so UTC vs Asia/Taipei, close/open/vwap,
ISO-week vs rolling-7-day can all be swept.
"""
import numpy as np, pandas as pd
from scipy import stats

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ----------------------------------------------------------------------------- builders
def daily_from_intraday(df1h, tz="UTC", price="close"):
    """Collapse tz-aware hourly OHLCV into ONE price per LOCAL calendar day under `tz`.
       price in {'close','open','vwap'}. Returns df[date(local), price]."""
    d = df1h.copy()
    t = pd.to_datetime(d["time"], utc=True).dt.tz_convert(tz)
    d["lday"] = t.dt.normalize()          # local midnight = day key
    d = d.sort_values("time")
    if price == "close":
        out = d.groupby("lday")["close"].last()
    elif price == "open":
        out = d.groupby("lday")["open"].first()
    elif price == "vwap":
        v = d["vol"].clip(lower=0).fillna(0.0)
        pv = (d["vwap"] * v).groupby(d["lday"]).sum()
        vv = v.groupby(d["lday"]).sum()
        out = np.where(vv > 0, pv / vv, d.groupby("lday")["close"].last())
        out = pd.Series(out, index=vv.index)
    else:
        raise ValueError(price)
    out = out.rename("price").reset_index().rename(columns={"lday": "time"})
    return out

def add_week_ratio(df, price_col="price", time_col="time", tz=None, week_def="iso"):
    """Given a ONE-price-per-day frame, attach dow, wk, ratio(price/weekly_mean).
       tz: if the daily frame is not already localized to the analysis tz, convert.
           (For intraday-derived frames pass tz=None; the day key is already local.)
       week_def: 'iso' (Mon-Sun ISO weeks) or 'roll7' (rolling non-aligned 7-day blocks)."""
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True) if not isinstance(
        d[time_col].dtype, pd.DatetimeTZDtype) else d[time_col]
    if d[time_col].dt.tz is None:
        d[time_col] = d[time_col].dt.tz_localize("UTC")
    if tz is not None:
        d[time_col] = d[time_col].dt.tz_convert(tz)
    d = d.sort_values(time_col).reset_index(drop=True)
    d["dow"] = d[time_col].dt.dayofweek
    if week_def == "iso":
        iso = d[time_col].dt.isocalendar()
        d["wk"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    elif week_def == "roll7":
        d0 = d[time_col].dt.normalize()
        block = ((d0 - d0.iloc[0]).dt.days // 7)
        d["wk"] = "R" + block.astype(str)
    else:
        raise ValueError(week_def)
    wm = d.groupby("wk")[price_col].transform("mean")
    d["ratio"] = d[price_col] / wm
    return d

def slice_window(df, years=None, time_col="time", asof=pd.Timestamp("2026-05-24", tz="UTC")):
    """Keep only rows within the last `years` before asof (None => all)."""
    if years is None:
        return df
    lo = asof - pd.Timedelta(days=int(round(years * 365.25)))
    t = df[time_col]
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    return df[t >= lo].copy()

# ----------------------------------------------------------------------------- significance
def _perm_spreads(ratio, week_codes, dow, n_perm, rng):
    """Vectorized within-week permutation of dow labels -> null distribution of max-min spread."""
    n = len(ratio)
    base_idx = np.argsort(week_codes, kind="stable")   # rows grouped by week (orig order)
    out = np.empty(n_perm)
    for i in range(n_perm):
        key = rng.random(n)
        perm_idx = base_idx[np.lexsort((key[base_idx], week_codes[base_idx]))]
        new_dow = np.empty(n, np.int64)
        new_dow[base_idx] = dow[perm_idx]
        sums = np.bincount(new_dow, weights=ratio, minlength=7)
        cnts = np.bincount(new_dow, minlength=7).astype(float)
        m = np.divide(sums, cnts, out=np.full(7, np.nan), where=cnts > 0)
        out[i] = np.nanmax(m) - np.nanmin(m)
    return out

def significance_suite(df, ratio_col="ratio", dow_col="dow", wk_col="wk",
                       n_perm=2000, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    obs_means = df.groupby(dow_col)[ratio_col].mean().reindex(range(7)).values
    obs_spread = np.nanmax(obs_means) - np.nanmin(obs_means)
    groups = [df[df[dow_col] == d][ratio_col].values for d in range(7)]
    groups = [g for g in groups if len(g) > 0]
    try:
        f_stat, anova_p = stats.f_oneway(*groups)
    except Exception:
        f_stat, anova_p = np.nan, np.nan
    try:
        kw_stat, kw_p = stats.kruskal(*groups)
    except Exception:
        kw_stat, kw_p = np.nan, np.nan

    ratio = df[ratio_col].to_numpy(float)
    dow = df[dow_col].to_numpy(np.int64)
    week_codes = pd.factorize(df[wk_col])[0]
    perm = _perm_spreads(ratio, week_codes, dow, n_perm, rng)
    perm_p = (perm >= obs_spread).mean()

    # bootstrap whole weeks
    weeks = df[wk_col].unique()
    wk_idx = {w: np.where(df[wk_col].values == w)[0] for w in weeks}
    boot = np.empty((n_boot, 7))
    dvals = df[dow_col].values
    for b in range(n_boot):
        samp = rng.choice(weeks, size=len(weeks), replace=True)
        idxs = np.concatenate([wk_idx[w] for w in samp])
        sub_r = ratio[idxs]; sub_d = dvals[idxs]
        s = np.bincount(sub_d, weights=sub_r, minlength=7)
        c = np.bincount(sub_d, minlength=7).astype(float)
        boot[b] = np.divide(s, c, out=np.full(7, np.nan), where=c > 0)
    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

    best = int(np.nanargmin(obs_means)); worst = int(np.nanargmax(obs_means))
    return {
        "n": len(df), "n_weeks": len(weeks),
        "obs_spread_pct": obs_spread * 100,
        "anova_p": anova_p, "kw_p": kw_p, "perm_p": perm_p,
        "cheapest": DOW_NAMES[best], "priciest": DOW_NAMES[worst],
        "day_means_pct": (obs_means - 1) * 100,
        "ci_lo_pct": (ci_lo - 1) * 100, "ci_hi_pct": (ci_hi - 1) * 100,
        "all_ci_cover_zero": bool(np.all((ci_lo <= 1) & (ci_hi >= 1))),
    }

# ----------------------------------------------------------------------------- other indicators
def logret_by_dow(daily, price_col="price", time_col="time", tz=None):
    """Method (ii): mean daily log-return grouped by day-of-week (of the day itself)."""
    d = daily.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True)
    if tz:
        d[time_col] = d[time_col].dt.tz_convert(tz)
    d = d.sort_values(time_col)
    d["lr"] = np.log(d[price_col]).diff()
    d["dow"] = d[time_col].dt.dayofweek
    m = d.groupby("dow")["lr"].mean().reindex(range(7))
    return m  # mean log-return per dow

def harmonic_cost_by_dow(daily, price_col="price", time_col="time", tz=None,
                         detrend="ma", ma_win=90, week_def="iso"):
    """Method (iii): $1 weekly DCA harmonic average cost per dow, on DE-TRENDED price.
       Lower harmonic mean => cheaper day. Returns Series indexed by dow (relative cost)."""
    d = add_week_ratio(daily, price_col, time_col, tz=tz, week_def=week_def)
    p = d[price_col].to_numpy(float)
    if detrend == "ma":
        trend = pd.Series(p).rolling(ma_win, center=True, min_periods=ma_win // 2).mean().to_numpy()
        pr = p / trend
    elif detrend == "linlog":
        x = np.arange(len(p)); lp = np.log(p)
        b, a = np.polyfit(x, lp, 1); pr = np.exp(lp - (a + b * x))
    else:
        pr = p
    d["_pr"] = pr
    d = d.dropna(subset=["_pr"])
    hm = d.groupby("dow")["_pr"].apply(lambda s: len(s) / np.sum(1.0 / s)).reindex(range(7))
    return hm / hm.mean()  # normalized: <1 cheaper

def ranking_of(series_like, ascending=True):
    """Return dow order best->worst given a per-dow score (lower=better if ascending)."""
    s = pd.Series(series_like).reindex(range(7))
    order = s.sort_values(ascending=ascending).index.tolist()
    return [DOW_NAMES[i] for i in order]

# ----------------------------------------------------------------------------- placebo
def simulate_gbm(n_days, mu_annual, sigma_annual, start_price, seed=None):
    rng = np.random.default_rng(seed)
    dt = 1 / 365.0
    rets = rng.normal((mu_annual - 0.5 * sigma_annual**2) * dt, sigma_annual * np.sqrt(dt), n_days)
    prices = start_price * np.exp(np.cumsum(rets))
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"time": dates, "price": prices})
