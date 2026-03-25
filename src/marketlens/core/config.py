from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MarketLens"
    environment: str = "dev"
    cache_ttl_seconds: int = 900  # 15 minutes default
    risk_free_rate_annual: float = 0.045  # used for Sharpe, this is adjustable

settings = Settings()