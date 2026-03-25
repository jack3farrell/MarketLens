from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import List

def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for one or more tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    Raises ValueError if no usable data is returned.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        raise ValueError("No market data returned. Check ticker symbols or date range.")

    if len(tickers) == 1:
        ticker = tickers[0]
        if "Adj Close" not in data.columns:
            raise ValueError(f"No adjusted close data returned for {ticker}.")
        adj = data["Adj Close"].to_frame(name=ticker)
    else:
        adj_dict = {}
        for t in tickers:
            try:
                adj_dict[t] = data[t]["Adj Close"]
            except Exception:
                raise ValueError(f"No adjusted close data returned for {t}.")
        adj = pd.DataFrame(adj_dict)

    adj = adj.dropna(how="all").ffill().dropna()

    if adj.empty:
        raise ValueError("Price data is empty after cleaning.")

    return adj


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()

    if returns.empty:
        raise ValueError("Not enough price history to calculate returns.")

    return returns