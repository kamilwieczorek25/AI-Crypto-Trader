"""Application configuration — loaded from environment / .env file."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Walk up from backend/app/ → backend/ → project root to find .env
_HERE = Path(__file__).parent          # backend/app/
_ENV_FILE = next(
    (p / ".env" for p in [_HERE.parent.parent, _HERE.parent, _HERE] if (p / ".env").exists()),
    _HERE.parent.parent / ".env",      # fallback: project root
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    # Admin API key (sk-ant-admin01-...) — needed for credit balance only (optional)
    ANTHROPIC_ADMIN_KEY: str = ""

    # Binance
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""

    # Trading mode
    MODE: str = "demo"           # "demo" | "real"
    REAL_TRADING: bool = False   # second safety gate

    # Risk profile — controls Claude's aggressiveness
    # conservative | balanced | aggressive | fast_profit
    RISK_PROFILE: str = "balanced"

    # Bot behaviour
    CYCLE_INTERVAL_SECONDS: int = 300
    DEMO_INITIAL_BALANCE: float = 10_000.0
    MAX_POSITION_PCT: float = 5.0
    MAX_TOTAL_EXPOSURE_PCT: float = 70.0

    # Symbol universe
    # TOP_N_SYMBOLS: 0 = all pairs passing the volume filter, >0 = hard cap
    TOP_N_SYMBOLS: int = 0
    # Minimum 24h quote volume (USDT) to be included in the universe
    MIN_VOLUME_USDT: float = 10_000_000.0
    # Max concurrent OHLCV/orderbook fetches (avoid rate-limit bans)
    FETCH_CONCURRENCY: int = 10

    # News
    CRYPTOCOMPARE_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:////data/trader.db"

    # Logging
    LOG_LEVEL: str = "INFO"

    @field_validator("CYCLE_INTERVAL_SECONDS")
    @classmethod
    def _cycle_interval_positive(cls, v: int) -> int:
        if v < 10:
            raise ValueError("CYCLE_INTERVAL_SECONDS must be >= 10")
        return v

    @field_validator("DEMO_INITIAL_BALANCE")
    @classmethod
    def _initial_balance_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("DEMO_INITIAL_BALANCE must be > 0")
        return v

    @field_validator("MIN_VOLUME_USDT")
    @classmethod
    def _min_volume_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("MIN_VOLUME_USDT must be >= 0")
        return v

    @property
    def is_demo(self) -> bool:
        return self.MODE.lower() != "real"

    @property
    def real_trading_allowed(self) -> bool:
        return self.MODE.lower() == "real" and self.REAL_TRADING


settings = Settings()
