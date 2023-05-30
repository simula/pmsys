import glob
import os
import pandas as pd

from typing import List, Dict, Any
from classes import Injury, Player, Team


def read_csv_files(path_to_folder: str) -> pd.DataFrame:
    """
        Reading all CSV files in a folder given a path,
        creates Pandas DataFrame of all the files.

    :param path_to_folder:
    :return df:
    """
    files = glob.glob(os.path.join(path_to_folder, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df


def get_player_names(daily_f_df: pd.DataFrame) -> List[Any]:
    """
        Finds all player names from DataFrame.

    :param daily_f_df:
    :return List:
    """
    group = daily_f_df.groupby("player_name")
    return list(group.groups)


def initialize_injuries(injury_df: pd.DataFrame, names: List[str]) -> Dict[str, List[Injury]]:
    """
        Creates a dictionary with player names and an array with their Injury objects.

    :param injury_df:
    :param names:
    :return injuries:
    """
    injuries: Dict = {name: [] for name in names}
    for name, injuries_by_player in injury_df.groupby("player_name"):
        injuries[name] = []
        for _, row in injuries_by_player.iterrows():
            injuries[name].append(Injury(row["player_name"], row["type"], row["timestamp"]))
    return injuries


def initialize_player(name: str, entries: pd.DataFrame, player_injuries: List[Injury], multivariate: bool) -> Player:
    """
        Creates Player object with name (player ID), timestamps, training load and
        wellness values and injuries in the form of Pandas Series.

        Cuts off beginning and end that contains NaN values.

    :param multivariate:
    :param name:
    :param entries:
    :param player_injuries:
    :return Player:
    """

    # If multivariate analysis:
    if multivariate:
        # Exclude NaN values from beginning and end:
        first_valid = entries["acwr"].first_valid_index()
        last_valid = entries["acwr"].last_valid_index()
        entries = entries.loc[first_valid:last_valid]

    time_index = pd.to_datetime(
        entries["date"].values, format="%d.%m.%Y"
    )

    return Player(
        name,
        time_index,
        entries["daily_load"].set_axis(time_index),
        entries["atl"].set_axis(time_index),
        entries["weekly_load"].set_axis(time_index),
        entries["monotony"].set_axis(time_index),
        entries["strain"].set_axis(time_index),
        entries["acwr"].set_axis(time_index),
        entries["ctl28"].set_axis(time_index),
        entries["ctl42"].set_axis(time_index),
        entries["fatigue"].set_axis(time_index),
        entries["mood"].set_axis(time_index),
        entries["readiness"].set_axis(time_index),
        entries["sleep-duration"].set_axis(time_index),
        entries["sleep-quality"].set_axis(time_index),
        entries["soreness"].set_axis(time_index),
        entries["stress"].set_axis(time_index),
        entries["injury_ts"].set_axis(time_index),
        player_injuries
    )


def initialize_players(daily_f_df: pd.DataFrame, injuries: Dict[str, List[Injury]], multivariate: bool) -> List[Player]:
    """
        Creates a list of Player objects by initializing each player.

    :param multivariate:
    :param daily_f_df:
    :param injuries:
    :return players:
    """
    players = []
    for name, group in daily_f_df.groupby("player_name"):
        player_injuries = injuries[str(name)]
        players.append(initialize_player(str(name), group, player_injuries, multivariate))
    return players


def initialize_team(name: str, players: List[Player]) -> Team:
    """
        Creates Team object using a name and a list of Player objects.

    :param name:
    :param players:
    :return Team:
    """
    players_dict: Dict = {player.name: player for player in players}
    return Team(name, players_dict)


def generate_teams(path: str, multivariate: bool) -> List[Team]:
    """
        Creates a list of Team objects.

    :param multivariate:
    :param path:
    :return teams:
    """
    teams_path = f"{path}/Subjective/per player/"
    team_names = [team_name for team_name in os.listdir(teams_path) if team_name.startswith("Team")]

    teams = []
    for team_name in team_names:
        daily_f_path = f"{teams_path}/{team_name}/daily-features"
        injury_path = f"{teams_path}/{team_name}/injuries"

        # Import daily features and injury data:
        daily_f_df = read_csv_files(daily_f_path)
        injury_df = read_csv_files(injury_path)

        # Rename date column:
        daily_f_df = daily_f_df.rename(columns={"Unnamed: 0": "date"})

        # Change ACWR values (incorrect in the dataset):
        daily_f_df['acwr'] = (daily_f_df['atl'] / daily_f_df['ctl42'])

        # Initialize injuries, players and teams:
        player_names = get_player_names(daily_f_df)
        injuries = initialize_injuries(injury_df, player_names)
        players = initialize_players(daily_f_df, injuries, multivariate)
        team = initialize_team(team_name, players)
        teams.append(team)

    return teams
