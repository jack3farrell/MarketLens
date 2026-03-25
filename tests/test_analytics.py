"""Unit tests for the analytics service — no network calls required."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd

from marketlens.services.analytics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    beta,
    portfolio_returns,
    correlation_matrix,
    TRADING_DAYS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_returns():
    """A series of identical small positive returns."""
    return pd.Series([0.001] * 100)


@pytest.fixture
def mixed_returns():
    """A realistic-ish mix of daily returns."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.012, 252))


@pytest.fixture
def two_series(mixed_returns):
    asset = mixed_returns
    rng = np.random.default_rng(7)
    bench = pd.Series(rng.normal(0.0003, 0.010, 252))
    return asset, bench


@pytest.fixture
def returns_df():
    rng = np.random.default_rng(0)
    data = {
        "AAPL": rng.normal(0.0006, 0.013, 252),
        "MSFT": rng.normal(0.0005, 0.011, 252),
        "GOOG": rng.normal(0.0004, 0.012, 252),
    }
    return pd.DataFrame(data)


# ── annualized_return ─────────────────────────────────────────────────────────

def test_annualized_return_positive(mixed_returns):
    result = annualized_return(mixed_returns)
    assert isinstance(result, float)


def test_annualized_return_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        annualized_return(pd.Series([], dtype=float))


def test_annualized_return_formula(flat_returns):
    # flat_returns all = 0.001 → (1.001)^252 - 1
    expected = (1 + 0.001) ** TRADING_DAYS - 1
    assert abs(annualized_return(flat_returns) - expected) < 1e-9


# ── annualized_volatility ─────────────────────────────────────────────────────

def test_annualized_volatility_non_negative(mixed_returns):
    assert annualized_volatility(mixed_returns) >= 0


def test_annualized_volatility_empty_raises():
    with pytest.raises(ValueError):
        annualized_volatility(pd.Series([], dtype=float))


def test_annualized_volatility_single_value_raises():
    with pytest.raises(ValueError):
        annualized_volatility(pd.Series([0.01]))


def test_annualized_volatility_zero_for_constant():
    # Constant returns → zero variance → zero vol
    s = pd.Series([0.005] * 50)
    assert annualized_volatility(s) == pytest.approx(0.0, abs=1e-9)


# ── sharpe_ratio ──────────────────────────────────────────────────────────────

def test_sharpe_ratio_is_float(mixed_returns):
    result = sharpe_ratio(mixed_returns, rf_annual=0.045)
    assert isinstance(result, float)


def test_sharpe_ratio_empty_raises():
    with pytest.raises(ValueError):
        sharpe_ratio(pd.Series([], dtype=float), rf_annual=0.045)


def test_sharpe_ratio_constant_raises():
    # Zero volatility → division by zero → should raise
    with pytest.raises(ValueError, match="volatility"):
        sharpe_ratio(pd.Series([0.001] * 50), rf_annual=0.045)


# ── beta ──────────────────────────────────────────────────────────────────────

def test_beta_perfect_tracking():
    s = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01] * 10)
    result = beta(s, s)
    assert result == pytest.approx(1.0, abs=1e-9)


def test_beta_empty_raises():
    with pytest.raises(ValueError):
        beta(pd.Series([], dtype=float), pd.Series([0.01, 0.02]))


def test_beta_insufficient_data_raises():
    with pytest.raises(ValueError):
        beta(pd.Series([0.01]), pd.Series([0.01]))


def test_beta_returns_float(two_series):
    asset, bench = two_series
    result = beta(asset, bench)
    assert isinstance(result, float)


# ── portfolio_returns ─────────────────────────────────────────────────────────

def test_portfolio_returns_shape(returns_df):
    weights = [1 / 3, 1 / 3, 1 / 3]
    pr = portfolio_returns(returns_df, weights)
    assert len(pr) == len(returns_df)


def test_portfolio_returns_equal_weight_bounded(returns_df):
    weights = [1 / 3, 1 / 3, 1 / 3]
    pr = portfolio_returns(returns_df, weights)
    col_min = returns_df.min(axis=1)
    col_max = returns_df.max(axis=1)
    assert (pr >= col_min - 1e-9).all()
    assert (pr <= col_max + 1e-9).all()


def test_portfolio_returns_single_asset_equals_asset(returns_df):
    df = returns_df[["AAPL"]]
    pr = portfolio_returns(df, [1.0])
    pd.testing.assert_series_equal(pr, df["AAPL"], check_names=False)


def test_portfolio_returns_empty_raises():
    with pytest.raises(ValueError):
        portfolio_returns(pd.DataFrame(), [1.0])


def test_portfolio_returns_zero_weights_raises(returns_df):
    with pytest.raises(ValueError, match="zero"):
        portfolio_returns(returns_df, [0.0, 0.0, 0.0])


# ── correlation_matrix ────────────────────────────────────────────────────────

def test_correlation_matrix_diagonal_is_one(returns_df):
    corr = correlation_matrix(returns_df)
    for col in returns_df.columns:
        assert corr[col][col] == pytest.approx(1.0, abs=1e-9)


def test_correlation_matrix_symmetric(returns_df):
    corr = correlation_matrix(returns_df)
    cols = list(returns_df.columns)
    for i in cols:
        for j in cols:
            assert corr[i][j] == pytest.approx(corr[j][i], abs=1e-9)


def test_correlation_matrix_values_in_range(returns_df):
    corr = correlation_matrix(returns_df)
    for row in corr.values():
        for v in row.values():
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9


def test_correlation_matrix_empty_raises():
    with pytest.raises(ValueError):
        correlation_matrix(pd.DataFrame())