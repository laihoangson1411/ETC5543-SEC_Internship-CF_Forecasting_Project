"""
Model utilities for the WIND Contract Factor notebook.

One fit/predict function per model (`run_*`), the Exponential baseline, the runner
`run_all_models`, forecast metrics and two prediction helpers for the scenario cells.
Model settings are read from the `cfg` dict, so nothing here depends on notebook globals.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utilsContractFactors import expfunctionCF

# Optional boosting libraries: flags let the run_* functions return NaN if missing
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMRegressor = None


def calculate_metrics(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    """Return MAE, RMSE, MAPE (%) and R2 on the timestamps where both series are non-null.

    Zero actuals are skipped in MAPE. All metrics are NaN if fewer than 2 points align.
    """
    aligned = pd.concat([actual.rename('actual'), pred.rename('pred')], axis=1).dropna()
    if len(aligned) < 2:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    mae = mean_absolute_error(aligned['actual'], aligned['pred'])
    rmse = np.sqrt(mean_squared_error(aligned['actual'], aligned['pred']))
    # Zero actuals become NaN so they drop out of the MAPE average
    denom = aligned['actual'].replace(0, np.nan).abs()
    mape = ((aligned['pred'] - aligned['actual']).abs() / denom).dropna().mean() * 100 if denom.notna().any() else np.nan
    r2 = r2_score(aligned['actual'], aligned['pred'])
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def run_linear(train, test, feats, target_col, cfg):
    """Ordinary least squares on `feats`."""
    m = LinearRegression().fit(train[feats], train[target_col])
    return pd.Series(m.predict(test[feats]), index=test.index), m


def run_ridge(train, test, feats, target_col, cfg):
    """RidgeCV on standardised features; the scaler is kept as `m.scaler_`."""
    scaler = StandardScaler().fit(train[feats].values)
    m = RidgeCV(alphas=cfg['alpha_grid'], cv=min(5, max(2, len(train) // 4)))
    m.fit(scaler.transform(train[feats].values), train[target_col].values)
    m.scaler_ = scaler
    pred = m.predict(scaler.transform(test[feats].values))
    return pd.Series(pred, index=test.index), m


def run_lgbm(train, test, feats, target_col, cfg):
    """LightGBM regressor (NaN output if the library is missing)."""
    if not HAS_LGBM:
        return pd.Series(np.nan, index=test.index), None
    m = LGBMRegressor(**cfg['lgbm_params']).fit(train[feats], train[target_col])
    return pd.Series(m.predict(test[feats]), index=test.index), m


def run_rf(train, test, feats, target_col, cfg):
    """Random forest regressor."""
    m = RandomForestRegressor(**cfg['rf_params']).fit(train[feats], train[target_col])
    return pd.Series(m.predict(test[feats]), index=test.index), m


def run_xgb(train, test, feats, target_col, cfg):
    """XGBoost regressor (NaN output if the library is missing)."""
    if not HAS_XGBOOST:
        return pd.Series(np.nan, index=test.index), None
    m = XGBRegressor(**cfg['xgb_params']).fit(train[feats], train[target_col])
    return pd.Series(m.predict(test[feats]), index=test.index), m


def run_sarimax(train, test, feats, target_col, cfg):
    """SARIMAX with the features as exogenous variables; forecasts the test rows.

    The series is fitted on a plain integer index so that gaps (rows dropped for
    missing features) do not break the time-series index checks.
    """
    try:
        fit = SARIMAX(train[target_col].reset_index(drop=True), exog=train[feats].reset_index(drop=True),
                      order=cfg['sarimax_order'], enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = fit.forecast(steps=len(test), exog=test[feats].values)
        return pd.Series(np.asarray(pred), index=test.index), fit
    except Exception as e:
        print(f'  SARIMAX failed: {e}')
        return pd.Series(np.nan, index=test.index), None


# Weather-driven models fitted by run_all_models (name -> function)
MODEL_FNS = {'Linear': run_linear, 'Ridge': run_ridge, 'LightGBM': run_lgbm,
             'SARIMAX': run_sarimax, 'RF': run_rf, 'XGBoost': run_xgb}


def fit_exponential(df, target_col, twap_col, season_col, train_fys):
    """Fit CF = f(TWAP) with `expfunctionCF` separately per season on the training years.

    Returns the prediction for every row of df and the fitted parameters per season.
    """
    params = {}
    for season in sorted(df[season_col].dropna().unique()):
        train = df[df['FY'].isin(train_fys) & (df[season_col] == season)]
        sub = train[[target_col, twap_col]].dropna()
        sub = sub[(sub[target_col] > 0) & (sub[target_col] < 3) & (sub[twap_col] > 0)]
        try:
            popt, _ = curve_fit(expfunctionCF, sub[twap_col], sub[target_col], p0=[1.0, 1.0], maxfev=20000)
            params[season] = {'a': popt[0], 'b': popt[1]}
        except Exception:
            params[season] = {'a': np.nan, 'b': np.nan}

    pred = pd.Series(np.nan, index=df.index)
    for season, p in params.items():
        if np.isnan(p['a']):
            continue
        mask = df[season_col] == season
        pred.loc[mask] = expfunctionCF(df.loc[mask, twap_col], p['a'], p['b'])
    return pred, params


def run_all_models(df, cfg, features, target_col='target'):
    """Fit all models on cfg['train_fys'] and evaluate them on cfg['test_fys'].

    Models: Seasonal Naive, Exponential, the six weather-driven models in `features`,
    and the Ensemble (mean of RF and XGBoost).

    Args:
        df: Monthly feature table (needs FY, Season, twap, CF_lag12 and the target).
        cfg: Config dict (train/test FYs and model settings).
        features: Dict model name -> feature list.
        target_col: Target column in df.

    Returns:
        rows: list of dicts (method, n, MAE, RMSE, MAPE, R2), one per method.
        fitted: dict method -> {'pred': test predictions, 'model': fitted object, 'feats': features}.
    """
    train_fys, test_fys = cfg['train_fys'], cfg['test_fys']
    train = df[df['FY'].isin(train_fys)]
    test = df[df['FY'].isin(test_fys)]
    fitted = {}

    # Baselines: same month last year, and the exponential CF-vs-TWAP curve
    fitted['Seasonal Naive'] = {'pred': df['CF_lag12'].reindex(test.index), 'model': None,
                                'feats': ['CF_lag12']}
    exp_pred, exp_params = fit_exponential(df, target_col, 'twap', 'Season', train_fys)
    fitted['Exponential'] = {'pred': exp_pred.reindex(test.index), 'model': exp_params,
                             'feats': ['twap']}

    # Weather-driven models: drop rows with a missing feature before fitting
    for name, fn in MODEL_FNS.items():
        feats = [c for c in features[name] if c in df.columns]
        tr = train.dropna(subset=feats + [target_col])
        te = test.dropna(subset=feats)
        if tr.empty or te.empty:
            fitted[name] = {'pred': pd.Series(np.nan, index=test.index), 'model': None, 'feats': feats}
            continue
        pred, mdl = fn(tr, te, feats, target_col, cfg)
        fitted[name] = {'pred': pred.reindex(test.index), 'model': mdl, 'feats': feats}

    # Ensemble: simple average of RF and XGBoost
    fitted['Ensemble'] = {'pred': (fitted['RF']['pred'] + fitted['XGBoost']['pred']) / 2,
                          'model': None, 'feats': fitted['RF']['feats']}

    # Metrics on the test months
    actual = df.loc[test.index, target_col]
    rows = []
    for name, info in fitted.items():
        pred = info['pred'].reindex(test.index)
        rows.append({'method': name, 'n': int((actual.notna() & pred.notna()).sum()),
                     **calculate_metrics(actual, pred)})
    return rows, fitted


def predict_cf(model_name, info, feat_df):
    """Predict CF for many feature rows with one fitted model (NaN where a feature is missing).

    The Ridge scaler fitted in `run_ridge` is applied automatically.
    """
    X = feat_df[info['feats']]
    ok = X.notna().all(axis=1).values
    out = np.full(len(X), np.nan)
    if ok.any():
        if model_name == 'Ridge':
            out[ok] = info['model'].predict(info['model'].scaler_.transform(X[ok].values))
        else:
            out[ok] = info['model'].predict(X[ok])
    return out


def predict_method(fitted, method, feat_df):
    """Predict CF with a method name; the Ensemble averages the RF and XGBoost predictions."""
    if method == 'Ensemble':
        return (predict_cf('RF', fitted['RF'], feat_df) + predict_cf('XGBoost', fitted['XGBoost'], feat_df)) / 2
    return predict_cf(method, fitted[method], feat_df)
