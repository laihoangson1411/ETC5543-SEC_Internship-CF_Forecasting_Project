"""
Scenario utilities.

LGC price and the LGC-based shrink factor for negative prices, the price forecast (half-hourly and monthly TWAP),
and the scenario feature rows (same calendar month of each training FY re-targeted to a TWAP).
All inputs are passed as arguments.
"""

import pandas as pd
import numpy as np

from utilsData import month_end_labels


def shrink_factor(m_hist, m_future, lgc_m, mode, depth):
    """Shrink factor for negative prices when reusing m_hist prices for m_future.

    mode 'lgc_ratio': LGC(future) / LGC(hist); assumes ALL negative depth is LGC-driven.
    mode 'empirical': ratio of the predicted depths (a + b*LGC); keeps the non-LGC floor `a`, the average over the
        year. (Adding the seasonal term c1*sin + c2*cos of the month to both depths was tested: it made the FY26
        forecast of every model worse, so it is not used.)

    Args:
        m_hist, m_future: Month-start timestamps (keys of `lgc_m`).
        lgc_m: Monthly LGC price Series on a month-start index.
        mode: 'empirical' or 'lgc_ratio' (see above).
        depth: (a, b) of the fitted depth model (fit_depth_model()['a'], ['b']).

    Returns:
        Float clipped to [0, 1]; 1.0 (no change) if an LGC price is missing or zero.
    """
    a, b = depth
    lgc_s = lgc_m.get(m_hist, np.nan)
    lgc_t = lgc_m.get(m_future, np.nan)
    if pd.isna(lgc_s) or pd.isna(lgc_t) or lgc_s == 0:
        return 1.0
    if mode == 'lgc_ratio':
        f = lgc_t / lgc_s
    else:
        denom = a + b * lgc_s
        f = (a + b * lgc_t) / denom if denom > 0 else 1.0
    return float(np.clip(f, 0.0, 1.0))


def load_lgc_monthly(path: str) -> pd.Series:
    """Monthly LGC price on a month-start index (the keys used by shrink_factor).

    utf-8-sig strips the BOM that Excel adds to CSV files.
    """
    lgc = pd.read_csv(path, index_col=0, parse_dates=True, dayfirst=True, encoding='utf-8-sig')
    return lgc['LGC'].resample('MS').mean().rename('lgc')


def load_lgc_forecast(path: str) -> pd.Series:
    """Forecast LGC price by calendar year (ForecastedLGCPrice.csv: TENOR 'Cal 26' -> 2026, MID POINT INDEX)."""
    lgc = pd.read_csv(path, encoding='utf-8-sig')
    lgc['year'] = 2000 + lgc['TENOR'].str.extract(r'(\d+)')[0].astype(int)
    return lgc.set_index('year')['MID POINT INDEX']


def fit_depth_model(price: pd.Series, lgc: pd.Series, how=0.05) -> dict:
    """Fit depth = a + b*LGC + c1*sin + c2*cos on the months where both depth and LGC exist.

    depth is monthly_depth(price, how); a is the non-LGC floor (average over the year) and b the LGC
    pass-through, the (a, b) pair used by shrink_factor; c1, c2 control the season in the fit.

    Returns:
        dict with a, b, c1, c2, months, r_raw (correlation of depth and LGC), r_within (mean correlation
        within each season: 0=DJF, 1=MAM, 2=JJA, 3=SON) and `data` (depth, lgc, sin, cos, season).
    """
    d = pd.concat([monthly_depth(price, how), lgc], axis=1).dropna()
    d = d[d.index <= price.index.max()]
    month_num = d.index.month
    d['sin'] = np.sin(2 * np.pi * month_num / 12)
    d['cos'] = np.cos(2 * np.pi * month_num / 12)
    d['season'] = month_num % 12 // 3
    X = np.column_stack([np.ones(len(d)), d['lgc'], d['sin'], d['cos']])
    coef, *_ = np.linalg.lstsq(X, d['depth'].values, rcond=None)
    r_within = np.nanmean([g['depth'].corr(g['lgc']) for _, g in d.groupby('season')])
    return {'a': coef[0], 'b': coef[1], 'c1': coef[2], 'c2': coef[3], 'months': len(d),
            'r_raw': d['depth'].corr(d['lgc']), 'r_within': r_within, 'data': d}


def load_forecast_price_hh(path: str) -> pd.Series:
    """Half-hourly price forecast on a time-starting index, like the generation data.

    The file has a 'Time-ending' timestamp (dd/mm/yyyy h:mm) and a 'Price' column;
    each half-hour is moved back 30 min so it is labelled by its start.
    """
    fc = pd.read_csv(path)
    fc['Time-ending'] = pd.to_datetime(fc['Time-ending'], format='%d/%m/%Y %H:%M')
    start = fc['Time-ending'] - pd.Timedelta(minutes=30)
    return pd.Series(fc['Price'].values, index=pd.DatetimeIndex(start, name='Time-starting'), name='Price').sort_index()


def load_forecast_twap_monthly(path: str) -> pd.Series:
    """Monthly TWAP of the 30-min price forecast.

    Each half-hour belongs to the month in which it STARTS (see load_forecast_price_hh).
    The index uses month_end_labels(), like the other monthly tables.
    """
    price = load_forecast_price_hh(path)
    labels = month_end_labels(price.index)

    # Completeness check: each month should have days_in_month * 48 half-hours
    counts = price.groupby(labels).size()
    bad = counts[counts != counts.index.day * 48]
    if len(bad) > 0:
        print(f'WARNING: {len(bad)} months with an unexpected number of half-hours:')
        print(bad.head().to_string())
    return price.groupby(labels).mean().rename('twap_forecast')


def adjust_month_prices(p_hist, s, twap_target):
    """LGC adjustment of the half-hourly prices of one historical month (the two steps of the slides).

    Step 1: every negative half-hour is multiplied by s (the LGC shrink factor).
    Step 2: every non-negative half-hour is multiplied by k, so that the month's TWAP equals twap_target:
        k = (twap_target - n_s * mean(P'_neg)) / ((1 - n_s) * mean(P_pos)),   n_s = share of negative half-hours.
    Scaling only the positive side stops the TWAP rescale from undoing the LGC adjustment.

    Returns:
        (adjusted half-hourly prices, k); k = 1 if the month has no non-negative half-hour.
    """
    p = p_hist.dropna()
    neg = p < 0
    p1 = p.where(~neg, s * p)                                            # step 1
    ns = neg.mean()
    neg_part = ns * p1[neg].mean() if neg.any() else 0.0
    pos_part = (1 - ns) * p[~neg].mean() if (~neg).any() else 0.0
    k = (twap_target - neg_part) / pos_part if pos_part != 0 else 1.0
    p2 = p1.where(neg, k * p1)                                           # step 2
    if pos_part != 0:                                                    # the adjusted month must hit the target TWAP
        assert abs(p2.mean() - twap_target) < 1e-6, f'TWAP mismatch: {p2.mean():.3f} vs {twap_target:.3f}'
    return p2, k


def build_scenario_features(d, p_hist, m_h, m_f, twap_target, adjust_neg, lgc_m, mode, depth):
    """Features of historical month m_h re-targeted to the TWAP of future month m_f.

    The half-hourly prices of m_h are adjusted by adjust_month_prices (negative ones shrunk by the LGC factor if
    adjust_neg, positive ones scaled to hit twap_target), then the monthly price features are recomputed from them.

    Args:
        d: Monthly feature table (df_wind or df_solar).
        p_hist: Half-hourly prices of month m_h.
        m_h: Historical month (index of d). m_f: Month being forecast.
        twap_target: Target monthly TWAP T.
        adjust_neg: Apply the LGC shrink to the negative prices.
        lgc_m, mode, depth: LGC settings passed on to shrink_factor (depth = (a, b)).

    Returns:
        dict of feature values (a copy of row m_h with twap_neg, twap_pos and twap replaced). A month with no
        negative (or no positive) half-hour keeps the value of d for that bucket (NaN for wind, 0 for solar).
    """
    row = d.loc[m_h].to_dict()
    s = shrink_factor(m_h.to_period('M').to_timestamp(), m_f.to_period('M').to_timestamp(),
                      lgc_m, mode, depth) if adjust_neg else 1.0            # lgc_m is on a month-start index
    p_adj, _ = adjust_month_prices(p_hist, s, twap_target)
    neg = p_hist.dropna() < 0                     # buckets by the original sign (s = 0 turns a negative price into 0)
    if neg.any():
        row['twap_neg'] = p_adj[neg].mean()
    if (~neg).any():
        row['twap_pos'] = p_adj[~neg].mean()
    row['twap'] = twap_target
    return row


def build_scenario_table(d, price_hh, t_target, scenario_fys, lgc_m, mode, depth,
                         versions=(('raw', False), ('lgc_adj', True))):
    """One feature row per (test month, scenario FY, version).

    Args:
        d: Monthly feature table (df_wind or df_solar).
        price_hh: Half-hourly price; each half-hour belongs to the month in which it starts.
        t_target: Target TWAP per test month (Series indexed by month).
        scenario_fys: Financial years whose months are reused as scenarios.
        lgc_m, mode, depth: LGC settings passed on to shrink_factor (depth = (a, b)).
        versions: (tag, adjust_neg) pairs to build.

    Meta columns start with '_' (_month, _fy, _version, _actual, _T). _actual is NaN for a month
    that is not in d (a future month).
    The scenario rows are copies of the historical month m_h, so every feature not touched by
    build_scenario_features (weather, month_sin/cos, ...) keeps its historical value.
    """
    hh_by_month = dict(list(price_hh.groupby(month_end_labels(price_hh.index))))
    rows = []
    for m_f in t_target.index:
        for fy in scenario_fys:
            # Same calendar month in the scenario FY
            cands = d.index[((d.index.month == m_f.month) & (d['FY'] == fy)).values]
            if len(cands) == 0:
                continue
            m_h = cands[0]
            for tag, adj in versions:
                row = build_scenario_features(d, hh_by_month[m_h], m_h, m_f, t_target[m_f], adj, lgc_m, mode, depth)
                row.update({'_month': m_f, '_fy': fy, '_version': tag,
                            '_actual': d['target'].get(m_f, np.nan), '_T': t_target[m_f]})
                rows.append(row)
    return pd.DataFrame(rows)


def monthly_depth(price, how):
    """Monthly depth of negative prices (positive = deeper): a quantile, or the mean of negatives."""
    if how == 'neg_mean':
        return (-price.where(price < 0).resample('MS').mean()).rename('depth')
    return (-price.resample('MS').quantile(how)).rename('depth')


def replay_example(price_hh, m_h, m_f, t_target, lgc_hist, lgc_future, mode, depth):
    """Price structure of historical month m_h before and after it is replayed into future month m_f.

    Uses adjust_month_prices on the half-hourly prices of m_h, as the scenario forecast does: the negative
    half-hours are shrunk by s (LGC) and the positive ones scaled by k so that the month's TWAP equals t_target.

    Returns:
        dict with neg_share, s, k and (historical, adjusted) pairs for lgc, twap_neg, twap_pos and twap.
    """
    lgc = pd.Series([lgc_hist, lgc_future], index=[m_h.to_period('M').to_timestamp(), m_f.to_period('M').to_timestamp()])
    p_hist = price_hh[month_end_labels(price_hh.index) == m_h].dropna()
    s = shrink_factor(*lgc.index, lgc, mode, depth)
    p_adj, k = adjust_month_prices(p_hist, s, t_target)
    neg = p_hist < 0
    return {'neg_share': neg.mean(), 's': s, 'k': k, 'lgc': (lgc_hist, lgc_future),
            'twap_neg': (p_hist[neg].mean(), p_adj[neg].mean()), 'twap_pos': (p_hist[~neg].mean(), p_adj[~neg].mean()),
            'twap': (p_hist.mean(), p_adj.mean())}
