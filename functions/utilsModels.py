"""
Model utilities.

One fit/predict function per weather-driven model (`run_*`, listed in MODEL_FNS), the Exponential
baseline, the runner `run_all_models`, forecast metrics and the prediction helpers of the scenario cells.
Model settings are read from the `cfg` dict, so nothing here depends on notebook globals.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from xgboost import XGBRegressor          # on macOS XGBoost also needs OpenMP: brew install libomp

from utilsContractFactors import expfunctionCF


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


def run_rf(train, test, feats, target_col, cfg):
    """Random forest regressor."""
    m = RandomForestRegressor(**cfg['rf_params']).fit(train[feats], train[target_col])
    return pd.Series(m.predict(test[feats]), index=test.index), m


def run_xgb(train, test, feats, target_col, cfg):
    """XGBoost regressor."""
    m = XGBRegressor(**cfg['xgb_params']).fit(train[feats], train[target_col])
    return pd.Series(m.predict(test[feats]), index=test.index), m


# Weather-driven models fitted by run_all_models (name -> function)
MODEL_FNS = {'Linear': run_linear, 'Ridge': run_ridge, 'RF': run_rf, 'XGBoost': run_xgb}


def fit_exponential(df, target_col, twap_col, season_col, train_fys):
    """Fit CF = f(TWAP) with `expfunctionCF` separately per season on the training years.

    Returns the prediction for every row of df and the fitted parameters per season
    ({'a', 'b', 'n'}; n = training months behind the curve, after dropping CF <= 0, CF >= 3 and TWAP <= 0).
    """
    params = {}
    for season in sorted(df[season_col].dropna().unique()):
        train = df[df['FY'].isin(train_fys) & (df[season_col] == season)]
        sub = train[[target_col, twap_col]].dropna()
        sub = sub[(sub[target_col] > 0) & (sub[target_col] < 3) & (sub[twap_col] > 0)]
        try:
            popt, _ = curve_fit(expfunctionCF, sub[twap_col], sub[target_col], p0=[1.0, 1.0], maxfev=20000)
            params[season] = {'a': popt[0], 'b': popt[1], 'n': len(sub)}
        except Exception:
            params[season] = {'a': np.nan, 'b': np.nan, 'n': len(sub)}

    return pd.Series(predict_exponential(params, df[season_col], df[twap_col]), index=df.index), params


def predict_exponential(params, seasons, twap):
    """CF from the per-season curves of fit_exponential, evaluated at `twap` (NaN where a season has no fit)."""
    seasons, twap = np.asarray(seasons), np.asarray(twap, dtype=float)
    out = np.full(len(twap), np.nan)
    for season, p in params.items():
        mask = seasons == season
        if mask.any() and not np.isnan(p['a']):
            out[mask] = expfunctionCF(twap[mask], p['a'], p['b'])
    return out


def run_all_models(df, cfg, features, target_col='target'):
    """Fit all models on cfg['train_fys'] and evaluate them on cfg['test_fys'].

    Models: Seasonal Naive, Exponential, every weather-driven model named in `features` (a key of
    MODEL_FNS), and the Ensemble (mean of RF and XGBoost) when both of them are fitted.

    Args:
        df: Monthly feature table (needs FY, Season, twap, CF_lag12 and the target).
        cfg: Config dict (train/test FYs and model settings).
        features: Dict model name -> feature list; only these models are fitted.
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
    for name in features:
        fn = MODEL_FNS[name]
        feats = [c for c in features[name] if c in df.columns]
        tr = train.dropna(subset=feats + [target_col])
        te = test.dropna(subset=feats)
        if tr.empty or te.empty:
            fitted[name] = {'pred': pd.Series(np.nan, index=test.index), 'model': None, 'feats': feats}
            continue
        pred, mdl = fn(tr, te, feats, target_col, cfg)
        fitted[name] = {'pred': pred.reindex(test.index), 'model': mdl, 'feats': feats}

    # Ensemble: simple average of RF and XGBoost
    if 'RF' in fitted and 'XGBoost' in fitted:
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


def feature_importance(fitted, method, train):
    """Importance of each feature of a fitted model; returns (Series, signed), or (None, False) for a baseline.

    Linear: coefficient x feature standard deviation on `train` (signed). Ridge: coefficient on the
    standardised features (signed). RF / XGBoost: feature_importances_, normalised to sum to 1 (>= 0).
    Ensemble: mean of the RF and XGBoost shares. Seasonal Naive and Exponential use no features.
    """
    if method == 'Ensemble':
        rf, xgb = feature_importance(fitted, 'RF', train)[0], feature_importance(fitted, 'XGBoost', train)[0]
        return (rf + xgb) / 2, False
    info = fitted.get(method, {})
    mdl, feats = info.get('model'), info.get('feats')
    if method not in MODEL_FNS or mdl is None:
        return None, False
    if method == 'Linear':
        return pd.Series(mdl.coef_ * train[feats].std().values, index=feats), True
    if method == 'Ridge':
        return pd.Series(mdl.coef_, index=feats), True
    imp = pd.Series(mdl.feature_importances_, index=feats)
    return imp / imp.sum(), False


def ridge_pvalues(fitted, train, target_col='target'):
    """Approximate p-value of each Ridge coefficient (t-test for ridge regression, Cule et al. 2011).

    Ridge has no exact p-value. On the standardised training features X (the scaler of run_ridge) with penalty alpha:
    Var(beta) = sigma^2 A X'X A with A = (X'X + alpha I)^-1 and sigma^2 = RSS / (n - tr(H) - 1), H = X A X'.
    """
    m, feats = fitted['Ridge']['model'], fitted['Ridge']['feats']
    d = train.dropna(subset=feats + [target_col])
    X, y = m.scaler_.transform(d[feats].values), d[target_col].values
    A = np.linalg.inv(X.T @ X + m.alpha_ * np.eye(X.shape[1]))
    dof = len(y) - np.trace(X @ A @ X.T) - 1                 # - 1 for the intercept
    resid = y - m.predict(X)
    cov = (resid @ resid / dof) * A @ X.T @ X @ A
    t = m.coef_ / np.sqrt(np.diag(cov))
    return pd.Series(2 * stats.t.sf(np.abs(t), dof), index=feats)


def p_stars(p):
    """Significance stars: *** p < 0.001, ** p < 0.01, * p < 0.05, '' otherwise."""
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def linear_fit_checks(fitted, train, target_col='target'):
    """Coefficients and residual checks of the Linear and Ridge models on the training months.

    Coefficients are on standardised features (one sd), so Linear and Ridge can be compared: Linear by OLS with
    its p-values, Ridge with the approximate p-values of ridge_pvalues; VIF of each feature.

    Returns:
        (coefficients DataFrame indexed by feature, checks DataFrame with one column per model).
    """
    feats = fitted['Linear']['feats']
    d = train.dropna(subset=feats + [target_col])
    X = sm.add_constant(pd.DataFrame(StandardScaler().fit_transform(d[feats]), columns=feats, index=d.index))
    y = d[target_col]
    ols = sm.OLS(y, X).fit()
    ridge = fitted['Ridge']['model']
    ridge_resid = y.values - ridge.predict(ridge.scaler_.transform(d[feats].values))

    coef = pd.DataFrame({'Linear coef': ols.params[feats], 'Linear p': ols.pvalues[feats],
                         'Ridge coef': pd.Series(ridge.coef_, index=feats), 'Ridge p': ridge_pvalues(fitted, train, target_col),
                         'VIF': [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]})

    def checks(resid):
        return {'R2 (training months)': 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                'Durbin-Watson (2 = no autocorrelation)': durbin_watson(resid),
                'Breusch-Pagan p (> 0.05: constant variance)': het_breuschpagan(resid, X)[1],
                'Jarque-Bera p (> 0.05: normal residuals)': jarque_bera(resid)[1]}
    chk = pd.DataFrame({'Linear': checks(ols.resid.values), 'Ridge': checks(ridge_resid)})
    chk.loc['Ridge alpha (cross-validation)'] = [np.nan, ridge.alpha_]
    chk.loc['Training months'] = [len(d), len(d)]
    return coef, chk
