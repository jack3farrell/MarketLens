import numpy as np
from fastapi import APIRouter, HTTPException

from marketlens.core.config import settings
from marketlens.models.schemas import PortfolioRequest, PortfolioResponse
from marketlens.services.market_data import fetch_adj_close, to_returns
from marketlens.services.analytics import (
    portfolio_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    correlation_matrix,
)
from marketlens.utils.dates import validate_date_range

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.post("/analyze", response_model=PortfolioResponse)
def analyze_portfolio(req: PortfolioRequest):
    """
    Analyze a multi-asset portfolio.

    Returns annualized return, volatility, Sharpe ratio, and full correlation matrix.
    If weights are omitted the portfolio is equally weighted.
    """
    tickers = [t.upper() for t in req.tickers]
    n = len(tickers)

    # Resolve weights
    if req.weights is None:
        weights = [1.0 / n] * n
    else:
        if len(req.weights) != n:
            raise HTTPException(
                status_code=400,
                detail="weights length must match tickers length",
            )
        if sum(req.weights) == 0:
            raise HTTPException(
                status_code=400,
                detail="weights must sum to a non-zero value",
            )
        weights = req.weights

    try:
        validate_date_range(req.start, req.end)
        prices = fetch_adj_close(tickers, req.start, req.end)
        rets = to_returns(prices)
        prets = portfolio_returns(rets, weights)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze portfolio: {e}")

    # Normalize weights
    w = np.array(weights, dtype=float)
    w = (w / w.sum()).tolist()

    try:
        resp = PortfolioResponse(
            tickers=tickers,
            weights=[float(x) for x in w],
            start=req.start,
            end=req.end,
            annualized_return=annualized_return(prets),
            annualized_volatility=annualized_volatility(prets),
            sharpe_ratio=sharpe_ratio(prets, settings.risk_free_rate_annual),
            correlation=correlation_matrix(rets),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return resp