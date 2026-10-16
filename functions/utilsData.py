"""
Data utilities.

Timestamp cleaning, 30-min regridding, calendar features, generation loading
(MLF and capacity scaling) and the monthly contract-factor target table.
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utilsContractFactors import (calcMonthlyContractFactors, aggregateContractFactorsByGenType,
                                  convertUnits, getFY, getSeason)
from utilsGeneration import filter_generation_data


def to_naive_timestamp(values: pd.Series) -> pd.Series:
    """Parse timestamps, convert tz-aware ones to Brisbane time, drop tz, floor to 30 min."""
    ts = pd.to_datetime(values, errors='coerce')
    if getattr(ts.dt, 'tz', None) is not None:
        # Brisbane has no daylight saving, so the offset stays fixed
        ts = ts.dt.tz_convert('Australia/Brisbane').dt.tz_localize(None)
    return ts.dt.floor('30min')


def reindex_and_interpolate(df: pd.DataFrame, start: str, end: str, limit: int) -> pd.DataFrame:
    """Reindex to a full 30-min grid and linearly fill gaps of up to `limit` intervals."""
    full_index = pd.date_range(start=start, end=end, freq='30min')
    out = df.sort_index().reindex(full_index)
    for col in out.columns:
        out[col] = out[col].interpolate(limit=limit, limit_direction='both')
    out.index.name = 'Time-starting'
    return out


def _station_wide(raw: pd.DataFrame, metrics: list, cfg: Dict) -> pd.DataFrame:
    """Long station data -> one column per metric and station ('<metric>_<station>') on the 30-min grid."""
    wide = raw.pivot_table(index='ts_local', columns='station_id', values=metrics, aggfunc='last')
    wide.columns = [f'{metric}_{station}' for metric, station in wide.columns]
    return reindex_and_interpolate(wide, cfg['analysis_start'], cfg['analysis_end'], cfg['weather_interp_limit'])


def month_end_labels(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each timestamp to the last day of its month."""
    idx = pd.DatetimeIndex(index)
    return idx.to_period('M').to_timestamp('M')


def month_start(index) -> pd.DatetimeIndex:
    """Move month-end timestamps (e.g. 2022-07-31) to the 1st of the same month (2022-07-01), for plotting."""
    return pd.DatetimeIndex(index).to_period('M').to_timestamp()


def monthly_station_mean(weather_hh: pd.DataFrame, metric: str, stations: list) -> pd.Series:
    """Monthly mean of one metric: average the given stations per half-hour, then average each month."""
    cols = [f'{metric}_{s}' for s in stations if f'{metric}_{s}' in weather_hh.columns]
    return weather_hh[cols].mean(axis=1).groupby(month_end_labels(weather_hh.index)).mean()


def build_twap_split_monthly(price: pd.Series, fill_empty: bool = False) -> pd.DataFrame:
    """Monthly mean of the negative and of the non-negative half-hourly prices.

    twap_neg: mean of half-hours with price < 0.  twap_pos: mean of half-hours with price >= 0.
    neg_share: share of half-hours with price < 0.
    fill_empty: False (wind) leaves NaN in a month with an empty bucket; True (solar) sets it to 0.
    """
    labels = month_end_labels(price.index)
    out = pd.DataFrame({
        'twap_neg': price.where(price < 0).groupby(labels).mean(),
        'twap_pos': price.where(price >= 0).groupby(labels).mean(),
        'neg_share': (price < 0).groupby(labels).mean(),
    })
    if fill_empty:
        out[['twap_neg', 'twap_pos']] = out[['twap_neg', 'twap_pos']].fillna(0)
    out.index.name = 'Time-ending'
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with month, cyclical month (sin/cos), FY and Season columns.

    FY is the Australian financial year (Jul-Jun) labelled by its end year.
    Season: 1=Summer, 2=Autumn, 3=Winter, 4=Spring. Index must be a DatetimeIndex.
    """
    out = df.copy()
    month_num = out.index.month
    out['month'] = month_num
    # sin/cos encoding makes December and January neighbours
    out['month_sin'] = np.sin(2 * np.pi * month_num / 12.0)
    out['month_cos'] = np.cos(2 * np.pi * month_num / 12.0)
    out['FY'] = getFY(out.index)
    out['Season'] = getSeason(out.index)
    return out


# DTN variables read from HistoricWeather_30min.csv (dp_c and rainfall_mm are only used by EDA.qmd)
DTN_METRICS = ['wind_speed_kph', 'wind_gust_kph', 'temp_c', 'rh', 'wind_dir_degrees', 'dp_c', 'rainfall_mm']


def load_weather_history(cfg: Dict, data_dir: str, apply_qm: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load half-hourly weather from local CSVs; return (weather_wide, wind_raw).

    Reads raw/HistoricWeather_30min.csv, raw/HistoricSolarIrradiance_30min.csv and
    processed/QM_HistoricWeather_30min.csv from `data_dir`.

    weather_wide has one column per metric and station (e.g. 'wind_speed_kph_89002'),
    plus wind speed in m/s, wind-direction sin/cos and solar columns, on a 30-min grid.
    If apply_qm is True, wind_speed_kph for stations 89002/86338 is overridden by the
    quantile-mapped (QM) file; set it to False to get the raw DTN data.
    """
    wind_csv = os.path.join(data_dir, 'raw', 'HistoricWeather_30min.csv')
    solar_csv = os.path.join(data_dir, 'raw', 'HistoricSolarIrradiance_30min.csv')
    qm_path = os.path.join(data_dir, 'processed', 'QM_HistoricWeather_30min.csv')

    start, end = pd.Timestamp(cfg['analysis_start']), pd.Timestamp(cfg['analysis_end'])
    stations = cfg['station_ids']

    # Read only the needed columns and rename them to a common schema
    wind_raw = pd.read_csv(
        wind_csv,
        usecols=['related_location', 'obs_time_local'] + DTN_METRICS,
    ).rename(columns={'related_location': 'station_id', 'obs_time_local': 'ts_local'})
    solar_raw = pd.read_csv(
        solar_csv,
        usecols=['station_number', 'timestamp_local', 'air_temp_c', 'dni', 'ghi'],
    ).rename(columns={'station_number': 'station_id', 'timestamp_local': 'ts_local'})

    # Clean timestamps and station ids, then keep the analysis window and chosen stations
    for raw in (wind_raw, solar_raw):
        raw['ts_local'] = to_naive_timestamp(raw['ts_local'])
        raw['station_id'] = pd.to_numeric(raw['station_id'], errors='coerce')
    wind_raw = wind_raw[wind_raw['station_id'].isin(stations) & wind_raw['ts_local'].between(start, end)].copy()
    solar_raw = solar_raw[solar_raw['station_id'].isin(stations) & solar_raw['ts_local'].between(start, end)].copy()

    # Wide format: one column per metric and station
    wind_wide = _station_wide(wind_raw, DTN_METRICS, cfg)

    # QM override: only wind_speed_kph for 89002/86338 is replaced;
    # temp_c, rh and wind_dir_degrees keep the original data
    if apply_qm:
        qm_df = pd.read_csv(qm_path, parse_dates=['datetime'], index_col='datetime')
        print(f'QM file loaded: {len(qm_df):,} rows, cols: {list(qm_df.columns)}')
        overlap = wind_wide.index.intersection(qm_df.index)
        for stn in [89002, 86338]:
            ws_col = f'wind_speed_kph_{stn}'
            if ws_col in qm_df.columns and ws_col in wind_wide.columns:
                wind_wide.loc[overlap, ws_col] = qm_df.loc[overlap, ws_col].values
        n_qm = int(qm_df['qm_corrected'].sum()) if 'qm_corrected' in qm_df.columns else 0
        print(f'QM overlay applied to wind_speed_kph only: {len(overlap):,} HH matched, {n_qm:,} HH were QM-corrected')

    # Derived wind features per station
    for station in stations:
        col = f'wind_speed_kph_{station}'
        if col in wind_wide.columns:
            wind_wide[f'wind_speed_ms_{station}'] = wind_wide[col] / 3.6  # km/h -> m/s
        dir_col = f'wind_dir_degrees_{station}'
        if dir_col in wind_wide.columns:
            # sin/cos avoids the 359 -> 0 degree jump in the direction
            wind_wide[f'sin_wind_dir_{station}'] = np.sin(np.deg2rad(wind_wide[dir_col]))
            wind_wide[f'cos_wind_dir_{station}'] = np.cos(np.deg2rad(wind_wide[dir_col]))

    solar_wide = _station_wide(solar_raw, ['air_temp_c', 'dni', 'ghi'], cfg)

    weather = wind_wide.join(solar_wide, how='outer')
    return weather, wind_raw


def load_generation_inputs(cfg: Dict, data_dir: str):
    """Load generation from local CSVs; exclusion periods are left as NaN.

    Args:
        cfg: Config dict (uses analysis window, wind_assets and solar_assets).
        data_dir: Data folder; input CSVs are read from its 'raw' subfolder.

    Returns:
        gen_scaled: Half-hourly generation (MLF- and capacity-scaled) for all assets plus VIC1Price.
        vret: VRET asset table indexed by asset name.
        capacity: Dict of asset capacity (MW), plus 'WIND' and 'SOLAR' totals.
    """
    all_assets = cfg['wind_assets'] + cfg['solar_assets']

    # The NEMSight export is already time-starting (index column 'Time-starting', first row 00:00): no shift
    gen_raw = pd.read_csv(os.path.join(data_dir, 'raw', 'GenerationMW_FY19-26.csv'),
                          index_col=0, parse_dates=True)
    vret = pd.read_csv(os.path.join(data_dir, 'raw', 'Asset.csv'), index_col=0)     # asset metadata by DUID
    exclusion_periods = pd.read_csv(os.path.join(data_dir, 'raw', 'ExclusionPeriods.csv'),
                                    parse_dates=['ExclusionStart', 'ExclusionEnd'])

    # MLF table: one row per FY, one column per asset; the CSV has no FY column, so label the rows (row 0 first)
    mlf = pd.read_csv(os.path.join(data_dir, 'raw', 'MLF_FY23-26.csv'))
    mlf.index = pd.Index([2022, 2023, 2024, 2025, 2026], name='FY')

    price_data = gen_raw['VIC1Price'].copy()
    gen_assets = gen_raw[all_assets].copy()

    # Capacity (MW) = peak observed output x contracted share
    capacity = {asset: gen_assets[asset].max() * vret.loc[asset, 'Contracted Capacity'] for asset in all_assets}

    # MW -> MWh per interval, then set exclusion/commissioning periods to NaN
    gen_assets = convertUnits(gen_assets, 'MW', 'MWh')
    gen_assets, _ = filter_generation_data(gen_assets, exclusion_periods, extendExclusions=True, assets=all_assets)
    gen_assets['FY'] = getFY(gen_assets.index)

    # Apply the yearly MLF and contracted share to each asset.
    # .map() keeps every row (FY without an MLF -> NaN), unlike merge which would drop rows.
    for asset in all_assets:
        gen_assets[asset] = (gen_assets[asset]
                             * gen_assets['FY'].map(mlf[asset])
                             * vret.loc[asset, 'Contracted Capacity'])
    gen_scaled = gen_assets.loc[cfg['analysis_start']:cfg['analysis_end'], all_assets].join(price_data)
    # Manually blank BULGANA1 for Dec 2025 - Jan 2026 (kept from the original notebook)
    gen_scaled.loc['2025-12-01':'2026-01-31', 'BULGANA1'] = np.nan

    capacity['WIND'] = sum(capacity[a] for a in cfg['wind_assets'])
    capacity['SOLAR'] = sum(capacity[a] for a in cfg['solar_assets'])
    return gen_scaled, vret, capacity


def build_targets(gen_scaled: pd.DataFrame, cfg: Dict, capacity: Dict[str, float]) -> pd.DataFrame:
    """Build the monthly target table (contract factors and TWAP) from half-hourly generation.

    Args:
        gen_scaled: Half-hourly generation for all assets plus VIC1Price.
        cfg: Config dict (uses 'zero_dollar_floor', 'wind_assets' and 'solar_assets').
        capacity: Asset capacity in MW, keyed by asset name.

    Returns:
        Month-end indexed DataFrame with per-asset and WIND fleet columns
        (e.g. 'WIND_ContractFactor', 'VIC1Price_TWAP') and calendar features.
    """
    all_assets = cfg['wind_assets'] + cfg['solar_assets']

    # Monthly generation, TWAP, DWAP and contract factor for every asset
    monthly = calcMonthlyContractFactors(gen_scaled, all_assets, capacity, extraCol=None,
                                         zeroDollarFloor=cfg['zero_dollar_floor'])
    monthly.index.name = 'Time-ending'

    # Combine the wind assets into one 'WIND' fleet (generation-weighted)
    monthly = aggregateContractFactorsByGenType(monthly, cfg['wind_assets'], 'WIND')
    monthly = add_calendar_features(monthly)
    return monthly