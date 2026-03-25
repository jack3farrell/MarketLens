from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class StockMetrics(BaseModel):
    ticker: str
    start: str
    end: str
    annualized_volatility: float
    annualized_return: float
    sharpe_ratio: float

class BetaResponse(BaseModel):
    ticker: str
    benchmark: str
    beta: float

class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(min_length=1)
    weights: Optional[List[float]] = None  # optional, equal-weight if None
    start: str
    end: str
    benchmark: str = "SPY"

class PortfolioResponse(BaseModel):
    tickers: List[str]
    weights: List[float]
    start: str
    end: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    correlation: Dict[str, Dict[str, float]]