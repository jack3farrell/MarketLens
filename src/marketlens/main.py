from fastapi import FastAPI
from marketlens.core.config import settings
from marketlens.api.health import router as health_router
from marketlens.api.stocks import router as stocks_router
from marketlens.api.portfolios import router as portfolios_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "A financial analytics API for stock metrics and portfolio risk analysis. "
            "Provides annualized return, volatility, Sharpe ratio, beta, and correlation data."
        ),
        version="1.0.0",
    )

    @app.get("/", tags=["root"])
    def root():
        return {
            "message": f"{settings.app_name} API is running",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "stock_metrics": "/stocks/{ticker}/metrics?start=YYYY-MM-DD&end=YYYY-MM-DD",
                "stock_beta": "/stocks/{ticker}/beta?benchmark=SPY&start=YYYY-MM-DD&end=YYYY-MM-DD",
                "portfolio_analyze": "POST /portfolios/analyze",
            },
        }

    app.include_router(health_router)
    app.include_router(stocks_router)
    app.include_router(portfolios_router)
    return app


app = create_app()