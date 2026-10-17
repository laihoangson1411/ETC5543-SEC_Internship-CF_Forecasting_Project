"""
Project configuration shared by every notebook.

Paths (ROOT, DATA_DIR) and one base `cfg` dict with the analysis window, assets, weather
stations, input files and model settings. A notebook takes a copy with `make_cfg()` and
overrides only what differs, e.g. `make_cfg(train_fys=[2023, 2024, 2025, 2026])`.
"""

import copy
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (parent of functions/)
DATA_DIR = os.path.join(ROOT, 'data')                                  # folder with raw/, processed/ and forecast/
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FORECAST_DIR = os.path.join(DATA_DIR, 'forecast')

BASE_CFG = {
    'random_seed': 42,
    'zero_dollar_floor': True,                    # floor prices at $0 when computing contract revenue

    # Train/test split by financial year (FY ends in June)
    'train_fys': [2023, 2024, 2025],
    'test_fys': [2026],
    'analysis_start': '2022-07-01 00:00:00',
    'analysis_end': '2026-06-30 23:30:00',

    # Assets and weather stations
    'wind_assets': ['DUNDWF3', 'BRYB1WF1', 'BULGANA1', 'MRTLSWF1'],
    'solar_assets': ['WINTSF1', 'GLENSF1', 'COHUNSF1'],
    'station_ids': [85280, 86338, 89002],         # stations read from the weather files
    'wind_stations': [86338, 89002],              # DTN stations averaged into the monthly wind features (IDA_Weather)
    'solar_stations': [85280, 86338, 89002],      # Solcast stations averaged into the monthly solar features (IDA_Weather)
    'weather_interp_limit': 2,                    # max consecutive 30-min gaps to interpolate
    'corruption_window': ('2025-10-07', '2026-04-14'),   # DTN wind speed is squashed here (Wind_QuantileMapping.qmd)

    # Input and output files
    'wind_csv': os.path.join(RAW_DIR, 'HistoricWeather_30min.csv'),            # raw DTN weather (not QM)
    'locations_csv': os.path.join(RAW_DIR, 'Locations.csv'),                   # coordinates of the stations and farms
    'cf_csv': os.path.join(PROCESSED_DIR, 'HistoricContractFactors.csv'),      # monthly CF (IDA_ContractFactors.qmd)
    'lgc_csv': os.path.join(RAW_DIR, 'HistoricLGCPrice.csv'),                  # monthly LGC price, history
    'lgc_fc_csv': os.path.join(RAW_DIR, 'ForecastedLGCPrice.csv'),             # LGC price, forecast
    'price_forecast_csv': os.path.join(RAW_DIR, 'PriceForecast_30min.csv'),     # 30-min price forecast
    'asx_csv': os.path.join(RAW_DIR, 'PriceForecast_Month.csv'),               # ASX adjusted monthly price
    'forecast_dir': FORECAST_DIR,                                              # output folder of the forecasts

    # Weather-driven models and their features (EDIT HERE; read by the Wind/Solar analysis and forecast files)
    'models': ['Linear', 'Ridge', 'RF', 'XGBoost'],        # the Ensemble (mean of RF and XGBoost) is added on top
    'wind_features': ['wind_speed_kph', 'twap_neg', 'twap_pos', 'rh', 'month_sin', 'month_cos'],
    'solar_features': ['ghi', 'twap_neg', 'twap_pos', 'month_sin', 'month_cos'],

    # Model settings
    'rf_params': {'n_estimators': 300, 'max_depth': 4, 'min_samples_leaf': 2, 'random_state': 42},
    'xgb_params': {'n_estimators': 250, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.9,
                   'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'random_state': 42},
    'alpha_grid': np.logspace(0, 2, 15),          # candidate alphas for RidgeCV
}


def make_cfg(**overrides) -> dict:
    """Return a copy of BASE_CFG with the given keys replaced, so a notebook never edits the shared dict."""
    cfg = copy.deepcopy(BASE_CFG)
    unknown = [k for k in overrides if k not in cfg]
    assert not unknown, f'unknown cfg key(s) {unknown}; add them to BASE_CFG first'
    cfg.update(overrides)
    return cfg
