from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

TRADING_DAYS = 252

def _safe_float(value: float) -> float:
    if value is None or np.isnan(value) or np.isinf(value):
        raise ValueError("Computed metric is invalid (NaN or infinite).")
    return float(value)

def annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("Return series is empty.")

    mean_daily = returns.mean()
    result = (1 + mean_daily) ** TRADING_DAYS - 1
    return _safe_float(result)


def annualized_volatility(returns: pd.Series) -> float:
    if returns.empty or len(returns) < 2:
        raise ValueError("Not enough return observations to calculate volatility.")

    result = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return _safe_float(result)


def sharpe_ratio(returns: pd.Series, rf_annual: float) -> float:
    if returns.empty or len(returns) < 2:
        raise ValueError("Not enough return observations to calculate Sharpe ratio.")

    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    vol = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)

    if vol == 0 or np.isnan(vol) or np.isinf(vol):
        raise ValueError("Sharpe ratio cannot be calculated because volatility is zero or invalid.")

    ann_excess_return = (1 + excess.mean()) ** TRADING_DAYS - 1
    result = ann_excess_return / vol
    return _safe_float(result)

def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if asset_returns.empty or benchmark_returns.empty:
        raise ValueError("Asset or benchmark return series is empty.")

    if len(asset_returns) < 2 or len(benchmark_returns) < 2:
        raise ValueError("Not enough data points to calculate beta.")

    cov = np.cov(asset_returns.values, benchmark_returns.values, ddof=1)[0, 1]
    var = np.var(benchmark_returns.values, ddof=1)

    if var == 0 or np.isnan(var) or np.isinf(var):
        raise ValueError("Benchmark variance is zero or invalid; beta cannot be calculated.")

    result = cov / var
    return _safe_float(result)

def portfolio_returns(returns: pd.DataFrame, weights: list[float]) -> pd.Series:
    if returns.empty:
        raise ValueError("Portfolio returns cannot be calculated from empty data.")

    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        raise ValueError("Portfolio weights sum to zero.")

    w = w / w.sum()
    result = returns.dot(w)

    if result.empty:
        raise ValueError("Portfolio return series is empty.")

    return result

def correlation_matrix(returns: pd.DataFrame) -> dict[str, dict[str, float]]:
    if returns.empty:
        raise ValueError("Cannot compute correlation matrix from empty returns.")

    corr = returns.corr()

    if corr.isnull().values.any():
        raise ValueError("Correlation matrix contains invalid values.")

    return {
        row: {col: float(corr.loc[row, col]) for col in corr.columns}
        for row in corr.index
    }