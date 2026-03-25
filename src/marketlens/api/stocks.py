from fastapi import APIRouter, Query, HTTPException

from marketlens.core.config import settings
from marketlens.core.cache import TTLCache
from marketlens.models.schemas import StockMetrics, BetaResponse
from marketlens.services.market_data import fetch_adj_close, to_returns
from marketlens.services.analytics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    beta as beta_fn,
)
from marketlens.utils.dates import validate_date_range

router = APIRouter(prefix="/stocks", tags=["stocks"])
cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds)


@router.get("/{ticker}/metrics", response_model=StockMetrics)
def stock_metrics(
    ticker: str,
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
):
    """
    Return annualized return, volatility, and Sharpe ratio for a single stock.
    """
    key = f"metrics:{ticker}:{start}:{end}"
    cached = cache.get(key)
    if cached:
        return cached

    try:
        validate_date_range(start, end)
        t = ticker.upper()
        prices = fetch_adj_close([t], start, end)
        rets = to_returns(prices)[t]

        result = StockMetrics(
            ticker=t,
            start=start,
            end=end,
            annualized_volatility=annualized_volatility(rets),
            annualized_return=annualized_return(rets),
            sharpe_ratio=sharpe_ratio(rets, settings.risk_free_rate_annual),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    cache.set(key, result)
    return result


@router.get("/{ticker}/beta", response_model=BetaResponse)
def stock_beta(
    ticker: str,
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
    benchmark: str = Query("SPY", description="Benchmark ticker (default: SPY)"),
):
    """
    Return the beta of a stock relative to a benchmark (default SPY).
    """
    key = f"beta:{ticker}:{benchmark}:{start}:{end}"
    cached = cache.get(key)
    if cached:
        return cached

    t = ticker.upper()
    b = benchmark.upper()

    try:
        validate_date_range(start, end)
        prices = fetch_adj_close([t, b], start, end)
        rets = to_returns(prices).dropna()

        if rets.empty:
            raise ValueError("No overlapping return data available for asset and benchmark.")

        result = BetaResponse(
            ticker=t,
            benchmark=b,
            beta=beta_fn(rets[t], rets[b]),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    cache.set(key, result)
    return result