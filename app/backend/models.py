from pydantic import BaseModel
from typing import List, Optional, Dict


class StatFilter(BaseModel):
    stat: str
    operator: str  # '>', '<', '==', '>=', '<='
    value: float


class SizingConfig(BaseModel):
    method: str = "flat"  # 'flat', 'kelly', 'martingale', 'dalembert', 'units'
    base_unit: float = 100.0
    bankroll: float = 10000.0
    fraction: Optional[float] = 0.5  # for fractional kelly
    max_units: Optional[int] = 5
    confidence_thresholds: Optional[Dict[str, int]] = None


class SizedBacktestRequest(BaseModel):
    name: str = "Custom backtest"
    filters: List[StatFilter] = []
    sizing: SizingConfig = SizingConfig()
    use_opening_line: bool = False
    juice: float = -110.0


class SizedBacktestResult(BaseModel):
    n_games: int
    wins: int
    losses: int
    pushes: int
    win_pct: float
    roi: float
    profit: float
    equity_curve: List[float]
    drawdown: float
    p_value: float
