"""
Integration-style tests for the API endpoints.
These use TestClient (no real network) and mock yfinance so they run instantly.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient

from marketlens.main import app

client = TestClient(app)

# ── Shared mock data ──────────────────────────────────────────────────────────

def _mock_prices(tickers, n=252):
    """Return a deterministic DataFrame of fake adjusted-close prices."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-02", periods=n)
    data = {}
    for t in tickers:
        prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
        data[t] = prices
    return pd.DataFrame(data, index=dates)


def make_fetch_mock(*tickers_list):
    """Factory: returns a mock for fetch_adj_close that ignores real args."""
    def _fetch(tickers, start, end):
        return _mock_prices(tickers)
    return _fetch


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── / (root) ──────────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


# ── /stocks/{ticker}/metrics ──────────────────────────────────────────────────

@patch("marketlens.api.stocks.fetch_adj_close", side_effect=make_fetch_mock("AAPL"))
def test_stock_metrics_success(mock_fetch):
    r = client.get("/stocks/AAPL/metrics?start=2020-01-01&end=2024-01-01")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert isinstance(body["annualized_return"], float)
    assert isinstance(body["annualized_volatility"], float)
    assert isinstance(body["sharpe_ratio"], float)


def test_stock_metrics_bad_dates():
    r = client.get("/stocks/AAPL/metrics?start=2024-01-01&end=2020-01-01")
    assert r.status_code == 400
    assert "start date" in r.json()["detail"].lower()


def test_stock_metrics_malformed_date():
    r = client.get("/stocks/AAPL/metrics?start=01-01-2020&end=2024-01-01")
    assert r.status_code == 400


def test_stock_metrics_missing_start():
    r = client.get("/stocks/AAPL/metrics?end=2024-01-01")
    assert r.status_code == 422  # FastAPI validation error


# ── /stocks/{ticker}/beta ─────────────────────────────────────────────────────

@patch("marketlens.api.stocks.fetch_adj_close", side_effect=make_fetch_mock("AAPL", "SPY"))
def test_stock_beta_success(mock_fetch):
    r = client.get("/stocks/AAPL/beta?benchmark=SPY&start=2020-01-01&end=2024-01-01")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["benchmark"] == "SPY"
    assert isinstance(body["beta"], float)


def test_stock_beta_bad_dates():
    r = client.get("/stocks/AAPL/beta?start=2024-01-01&end=2020-01-01&benchmark=SPY")
    assert r.status_code == 400


# ── /portfolios/analyze ───────────────────────────────────────────────────────

@patch("marketlens.api.portfolios.fetch_adj_close", side_effect=make_fetch_mock("AAPL", "MSFT", "NVDA"))
def test_portfolio_analyze_success(mock_fetch):
    payload = {
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "weights": [0.4, 0.4, 0.2],
        "start": "2020-01-01",
        "end": "2024-01-01",
    }
    r = client.post("/portfolios/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["tickers"] == ["AAPL", "MSFT", "NVDA"]
    assert len(body["weights"]) == 3
    assert abs(sum(body["weights"]) - 1.0) < 1e-9
    assert isinstance(body["annualized_return"], float)
    assert isinstance(body["sharpe_ratio"], float)
    assert "AAPL" in body["correlation"]


@patch("marketlens.api.portfolios.fetch_adj_close", side_effect=make_fetch_mock("AAPL", "MSFT"))
def test_portfolio_equal_weights(mock_fetch):
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "start": "2020-01-01",
        "end": "2024-01-01",
    }
    r = client.post("/portfolios/analyze", json=payload)
    assert r.status_code == 200
    weights = r.json()["weights"]
    assert weights[0] == pytest.approx(0.5, abs=1e-9)
    assert weights[1] == pytest.approx(0.5, abs=1e-9)


def test_portfolio_weight_mismatch():
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "weights": [0.5, 0.3, 0.2],  # 3 weights for 2 tickers
        "start": "2020-01-01",
        "end": "2024-01-01",
    }
    r = client.post("/portfolios/analyze", json=payload)
    assert r.status_code == 400
    assert "weights" in r.json()["detail"].lower()


def test_portfolio_empty_tickers():
    payload = {
        "tickers": [],
        "start": "2020-01-01",
        "end": "2024-01-01",
    }
    r = client.post("/portfolios/analyze", json=payload)
    assert r.status_code == 422  # Pydantic min_length validation


def test_portfolio_bad_dates():
    payload = {
        "tickers": ["AAPL"],
        "start": "2024-01-01",
        "end": "2020-01-01",
    }
    r = client.post("/portfolios/analyze", json=payload)
    assert r.status_code == 400