from typing import Dict, List
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Injury:
    player: str
    type: Dict[str, str]
    timestamp: pd.Index


@dataclass(frozen=True)
class Player:
    name: str
    date: pd.Series
    daily_load: pd.Series
    atl: pd.Series
    weekly_load: pd.Series
    monotony: pd.Series
    strain: pd.Series
    acwr: pd.Series
    ctl28: pd.Series
    ctl42: pd.Series
    fatigue: pd.Series
    mood: pd.Series
    readiness: pd.Series
    sleep_duration: pd.Series
    sleep_quality: pd.Series
    soreness: pd.Series
    stress: pd.Series
    injury_ts: pd.Series
    injuries: List[Injury]
    # injuries: Dict[str, Injury]


@dataclass(frozen=True)
class Team:
    name: str
    players: Dict[str, Player]