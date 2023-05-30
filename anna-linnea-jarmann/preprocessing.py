import pandas as pd
import numpy as np

from typing import Dict, List
from lifelines.utils import to_long_format, add_covariate_to_timeline
from classes import Team


def generate_durations_univariate(team: Team) -> pd.DataFrame:
    """
        Creates dataframe with durations from first observation to the first injury.
        Includes censoring for non-injured players, duration is then from first to last observation.
        Can be used in Kaplan-Meier and Nelson-Aalen.

        Created using this tutorial:
        https://www.crosstab.io/articles/events-to-durations/

    :param team:
    :return durations:
    """
    players = team.players
    durations = pd.DataFrame(columns=["entry_date",
                                      "event",
                                      "event_date",
                                      "final_obs_date",
                                      "duration"])

    for name, player in players.items():
        # Find the earliest entry:
        entry_date = player.date.min()

        # Find event (injury occurred) and it's date:
        event_date = player.injury_ts.where(player.injury_ts == 1).first_valid_index()
        event = 1 if event_date else 0

        # Censoring if no event was found (using the last date):
        final_obs_date = event_date if event_date else player.date.max()

        # Calculate duration between entry date and injury (or final date if no injury):
        duration = (final_obs_date - entry_date).total_seconds() / (60 * 60 * 24)

        # Excluding events that occur on the first day:
        if duration > 0.0:
            new_row = pd.DataFrame({
                "player_name": [name],
                "entry_date": [entry_date],
                "event": [event],
                "event_date": [event_date],
                "final_obs_date": [final_obs_date],
                "duration": [duration],
            })
            durations = pd.concat([durations, new_row])

    durations = durations.set_index("player_name")
    durations["event"] = durations["event"].astype(bool)
    # print(durations)
    # print(durations.event.value_counts())
    return durations


def generate_durations_univariate_recurrent(team: Team) -> pd.DataFrame:
    """
        Generates dataframe with durations and events for multiple injuries for each player.

        Does not include all consecutive injuries, but min. 5 days apart
        Look at location to differentiate between injuries? (implement later)

        Includes non-injury players.

    :param team:
    :return durations_multiple:
    """

    players = team.players
    durations_recurrent = pd.DataFrame(columns=["player_name", "duration", "event", "day"])

    # For all players get durations between injuries or up to last observation date if no injuries:
    for name, player in players.items():

        entry_date = player.injury_ts.keys().min()
        last_injury_date = entry_date
        last_obs_date = player.injury_ts.keys().max()

        # Check if non-injury player:
        if 1 not in player.injury_ts.values:
            date = last_obs_date
            duration = date - entry_date

            # Create new row with duration, event:
            new_dict = {
                "player_name": [name],
                "duration": [duration.total_seconds() / (60 * 60 * 24)],
                "event": [0],
                "day": [duration.total_seconds() / (60 * 60 * 24)]
            }
            new_row = pd.DataFrame(new_dict)
            durations_recurrent = pd.concat([durations_recurrent, new_row])
            continue

        # If player has injuries:
        else:
            # Get all durations between injuries:
            for date, injury_event in player.injury_ts.items():
                date = pd.to_datetime(date)

                # If it's an injury occurring after 5 days or more:
                if injury_event == 1 and (date - last_injury_date).days > 5:

                    # Set duration from entry date or last injury:
                    duration = date - last_injury_date

                    # Set day nr. x of the event:
                    day = date - entry_date

                    # Create new row with duration, training load and wellness values in that period:
                    new_dict = {
                        "player_name": [name],
                        "duration": [duration.total_seconds() / (60 * 60 * 24)],
                        "event": [1],
                        "day": [day.total_seconds() / (60 * 60 * 24)]
                    }

                    new_row = pd.DataFrame(new_dict)
                    durations_recurrent = pd.concat([durations_recurrent, new_row])

                    # Update last injury date:
                    last_injury_date = date

    durations_recurrent.event = durations_recurrent.event.astype(bool)
    # print(durations_recurrent)
    return durations_recurrent


def generate_durations_multivariate(team: Team, covariates: List[str]) -> pd.DataFrame:
    """
        Generates dataframe with durations from first observation to first injury,
        including their corresponding wellness and training load values.
        (Values are from the same date as the injury date).
        Can be used in regular Cox Proportional Hazard Regression
        (Important to be aware of convergence issues with this many variables).

    :param team:
    :param covariates:
    :return durations_and_features:
    """
    players = team.players
    durations_and_features = pd.DataFrame(columns=["player_name",
                                                   "duration",
                                                   "event"] + covariates)

    for name, player in players.items():
        # Find the earliest entry:
        entry_date = player.date.min()

        # Find event (injury occurred) and it's date:
        event_date = player.injury_ts.where(player.injury_ts == 1).first_valid_index()
        event = 1 if event_date else 0

        # Censoring if no event was found (using the last date):
        final_obs_date = event_date if event_date else player.date.max()

        # Calculate duration between entry date and injury (or final date if no injury):
        duration = (final_obs_date - entry_date).total_seconds() / (60 * 60 * 24)

        # Excluding events that occur on the first day:
        if duration > 0.0:
            new_dict = {
                "player_name": [name],
                "duration": [duration],
                "event": [event],
            }
            for cov in covariates:
                new_dict[cov] = [player.__getattribute__(cov)[final_obs_date]]

            new_row = pd.DataFrame(new_dict)
            durations_and_features = pd.concat([durations_and_features, new_row])

    durations_and_features["event"] = durations_and_features["event"].astype(bool)
    durations_and_features = durations_and_features.set_index("player_name")

    # print(durations_and_features.to_string())
    # nans = durations_and_features[durations_and_features.isna().any(axis=1)]
    # print(nans)
    # print(len(nans), " / ", len(durations_and_features))
    # print(durations_and_features.isnull().sum().sum())

    # Filling NaN values with zeros:
    durations_and_features = durations_and_features.fillna(0)

    # Options that does not work:
    # durations_and_features = durations_and_features.interpolate()
    # durations_and_features = durations_and_features.dropna()
    return durations_and_features


def generate_durations_multivariate_recurrent(team: Team, covariates: List[str]) -> pd.DataFrame:
    """

    :param team:
    :param covariates:
    :return:
    """

    players = team.players
    durations_recurrent = pd.DataFrame(columns=["player_name", "duration", "event", "prior_injury"] + covariates)

    # For all players get durations between injuries or up to last observation date if no injuries:
    for name, player in players.items():

        entry_date = player.injury_ts.keys().min()
        last_injury_date = entry_date
        last_obs_date = player.injury_ts.keys().max()

        # Check if non-injury player:
        if 1 not in player.injury_ts.values:
            date = last_obs_date
            duration = date - entry_date

            # Create new row with duration, event, prior injury, training load and wellness values in that period:
            new_dict = {
                "player_name": [name],
                "duration": [duration.total_seconds() / (60 * 60 * 24)],
                "event": [0],
                "prior_injury": [0.0]
            }
            for cov in covariates:
                # Get value from last observation date:
                new_dict[cov] = [player.__getattribute__(cov).loc[date]]

            new_row = pd.DataFrame(new_dict)
            durations_recurrent = pd.concat([durations_recurrent, new_row])
            continue

        # If player has injuries:
        else:
            # Get all durations between injuries:
            for date, injury_event in player.injury_ts.items():
                date = pd.to_datetime(date)

                # If it's an injury occurring after 5 days or more:
                if injury_event == 1 and (date - last_injury_date).days > 5:

                    # Set duration from entry date or last injury:
                    duration = date - last_injury_date

                    # Check prior injury:
                    prior_injury = 1.0 if str(name) in durations_recurrent.player_name.values else 0.0

                    # If counting prior injuries instead:
                    # prior_injury = (durations_recurrent.player_name.values == str(name)).sum()

                    # Create new row with duration, training load and wellness values in that period:
                    new_dict = {
                        "player_name": [name],
                        "duration": [duration.total_seconds() / (60 * 60 * 24)],
                        "event": [1],
                        "prior_injury": [prior_injury]
                    }
                    for cov in covariates:
                        new_dict[cov] = [player.__getattribute__(cov).loc[date]]

                    new_row = pd.DataFrame(new_dict)
                    durations_recurrent = pd.concat([durations_recurrent, new_row])

                    # Update last injury date:
                    last_injury_date = date

    durations_recurrent.event = durations_recurrent.event.astype(bool)
    durations_recurrent.prior_injury = durations_recurrent.prior_injury.astype(bool)
    # print(durations_recurrent.to_string())
    # durations_recurrent = durations_recurrent.dropna()
    # print(durations_recurrent.to_string())
    durations_recurrent = durations_recurrent.fillna(0)
    # durations_recurrent = durations_recurrent.drop(columns=["player_name"])
    # counting = durations_recurrent.groupby(['player_name'])['player_name'].count()
    # print(counting)
    return durations_recurrent


def generate_durations_time_varying(team: Team, covariates: List[str]) -> pd.DataFrame:
    """
        Generates a dataframe with durations and events,
        including wellness and training load values over time.
        (In this case ACWR, readiness, fatigue and sleep duration)
        Can be used in the Cox Time Varying model.

    :param team:
    :param covariates:
    :return DataFrame:
    """
    # Create base:
    durations_with_features = generate_durations_multivariate(team, covariates)
    base = durations_with_features[['duration',
                                    'event']].copy()
    base = base.reset_index()
    base = to_long_format(base, duration_col="duration")

    # Add covariates:
    cvs = []
    for cov in covariates:
        cv = pd.DataFrame()
        for name, player in team.players.items():
            time_diff = (pd.Series(player.__getattribute__(cov).index - player.__getattribute__(cov).index.min())) \
                                 .dt.total_seconds() / (60 * 60 * 24)
            player_cv = pd.DataFrame({
                "player_name": name,
                "time": time_diff,
                cov: player.__getattribute__(cov).values
            })
            cv = pd.concat([cv, player_cv])
        cvs.append(cv)

    # print(base.isnull().sum().sum())
    # cov: player.__getattribute__(cov).fillna(method="ffill").values
    for cv in cvs:
        base = base.pipe(add_covariate_to_timeline, cv, duration_col="time", id_col="player_name", event_col="event")
    # nans = base[base.isna().any(axis=1)]
    # print(nans)
    # print(len(nans), " / ", len(base))
    # print(base.isnull().sum().sum())
    base = base.fillna(0)
    # base = base.fillna(method="ffill")
    # base = base.interpolate()
    return base


def generate_durations_time_varying_recurrent(team: Team, covariates: List[str]) -> pd.DataFrame:
    """
        Lifelines does not allow for duplicate IDs, meaning it removes recurrent events.

    :param team:
    :param covariates:
    :return:
    """
    # Create base multiple injuries:
    durations_recurrent = generate_durations_multivariate_recurrent(team, covariates)

    base_recurrent = durations_recurrent[['player_name',
                                          'duration',
                                          'event',
                                          'prior_injury'].copy()]
    base_recurrent = to_long_format(base_recurrent, duration_col="duration")
    # print(base_recurrent[base_recurrent["player_name"] == "TeamA-c4ccf1a6-48c3-4a17-8d6c-eedd12e8680e"])

    # Add covariates:
    cvs = []
    for cov in covariates:
        cv = pd.DataFrame()
        for name, player in team.players.items():
            time_diff = (pd.Series(player.__getattribute__(cov).index - player.__getattribute__(cov).index.min())) \
                                 .dt.total_seconds() / (60 * 60 * 24)
            player_cv = pd.DataFrame({
                "player_name": name,
                "time": time_diff,
                cov: player.__getattribute__(cov).values
            })
            cv = pd.concat([cv, player_cv])
        cvs.append(cv)

    for cv in cvs:
        base_recurrent = base_recurrent.pipe(add_covariate_to_timeline, cv, duration_col="time", id_col="player_name", event_col="event").fillna(0)
        # print(base_recurrent[base_recurrent["player_name"] == "TeamA-c4ccf1a6-48c3-4a17-8d6c-eedd12e8680e"])
    base_recurrent = base_recurrent.reindex(columns=["player_name",
                                                     "start",
                                                     "stop",
                                                     "event",
                                                     "prior_injury"] + covariates)
    base_recurrent = base_recurrent.loc[~((base_recurrent["start"] == base_recurrent["stop"]) & (base_recurrent["start"] == 0))]
    # print(base_recurrent[base_recurrent["player_name"] == "TeamA-c4ccf1a6-48c3-4a17-8d6c-eedd12e8680e"])

    # nulls = base_recurrent.loc[(base_recurrent["start"] == base_recurrent["stop"]) & (base_recurrent["start"] == 0)]
    # print(nulls)
    #print(base_recurrent.to_string())
    #nans = base_recurrent[base_recurrent.isna().any(axis=1)]
    #print(nans)
    #print(len(nans), " / ", len(base_recurrent))
    # print(base_recurrent.isnull().sum().sum())
    # print(base_recurrent.to_string())
    # counting = base_recurrent.groupby("player_name").event.value_counts()
    # counting = base_recurrent.groupby(['player_name'])['event'].value_counts()
    # print(counting)
    # print(base_recurrent[base_recurrent["player_name"]=="TeamA-c4ccf1a6-48c3-4a17-8d6c-eedd12e8680e"])
    return base_recurrent


def generate_durations_multivariate_all_values(team: Team, covariates: List[str]) -> pd.DataFrame:
    """
    
    :param team: 
    :param covariates: 
    :return: 
    """
    players = team.players
    durations = pd.DataFrame(columns=["player_name",
                                      "duration",
                                      "event"] + covariates)

    for name, player in players.items():
        # Find the earliest entry:
        entry_date = player.date.min()

        # Find event (injury occurred) and it's date:
        event_date = player.injury_ts.where(player.injury_ts == 1).first_valid_index()
        event = 1 if event_date else 0

        # Censoring if no event was found (using the last date):
        final_obs_date = event_date if event_date else player.date.max()

        # Calculate duration between entry date and injury (or final date if no injury):
        duration = (final_obs_date - entry_date).total_seconds() / (60 * 60 * 24)

        # Excluding events that occur on the first day:
        if duration > 0.0:
            new_dict = {
                "player_name": [name],
                "duration": [duration],
                "event": [event],
            }
            for cov in covariates:
                new_dict[cov] = [player.__getattribute__(cov).loc[entry_date:final_obs_date].values]

            new_row = pd.DataFrame(new_dict)
            durations = pd.concat([durations, new_row])

    durations["event"] = durations["event"].astype(bool)
    # durations = durations.set_index("player_name")
    return durations


def generate_durations_multivariate_averaged(team: Team, covariates: List[str]) -> pd.DataFrame:
    """

    :param team:
    :param covariates:
    :return:
    """
    durations = generate_durations_multivariate_all_values(team, covariates)
    # durations = durations.drop("player_name", axis=1)
    durations = durations.set_index("player_name")

    functions = {}
    for cov in covariates:
        functions[cov] = "np.nanmean"

    durations_manipulated = manipulate_features(durations, functions)
    # durations_manipulated.index = list(range(len(durations_manipulated)))
    # print(durations_manipulated)
    return durations_manipulated


def generate_durations_multivariate_recurrent_all_values(team: Team, covariates: List[str]) -> pd.DataFrame:
    """
        Generates dataframe with durations and events for multiple injuries for each player,
        including wellness and training load variables.

        Does not include all consecutive injuries, but min. 5 days apart
        Look at location to differentiate between injuries? (implement later)

        Includes non-injury players.

    :param team:
    :param covariates:
    :return durations_multiple:
    """

    players = team.players
    durations_multiple = pd.DataFrame(columns=["player_name", "duration", "event", "prior_injury"] + covariates)

    # For all players get durations between injuries or up to last observation date if no injuries:
    for name, player in players.items():

        entry_date = player.injury_ts.keys().min()
        last_injury_date = entry_date
        last_obs_date = player.injury_ts.keys().max()

        # Check if non-injury player:
        if 1 not in player.injury_ts.values:
            date = last_obs_date
            duration = date - entry_date

            # Create new row with duration, event, prior injury, training load and wellness values in that period:
            new_dict = {
                "player_name": [name],
                "duration": [duration.total_seconds() / (60 * 60 * 24)],
                "event": [0],
                "prior_injury": [0.0]
            }
            for cov in covariates:
                new_dict[cov] = [player.__getattribute__(cov).loc[last_injury_date:date].values]

            new_row = pd.DataFrame(new_dict)
            durations_multiple = pd.concat([durations_multiple, new_row])
            continue

        # If player has injuries:
        else:
            # Get all durations between injuries:
            for date, injury_event in player.injury_ts.items():
                date = pd.to_datetime(date)

                # If it's an injury occurring after 5 days or more:
                if injury_event == 1 and (date - last_injury_date).days > 5:

                    # Set duration from entry date or last injury:
                    duration = date - last_injury_date

                    # Check prior injury:
                    prior_injury = 1.0 if str(name) in durations_multiple.player_name.values else 0.0

                    # If counting prior injuries instead:
                    # prior_injury = (durations_multiple.player_name.values == str(name)).sum()

                    # Create new row with duration, training load and wellness values in that period:
                    new_dict = {
                        "player_name": [name],
                        "duration": [duration.total_seconds() / (60 * 60 * 24)],
                        "event": [1],
                        "prior_injury": [prior_injury]
                    }
                    for cov in covariates:
                        new_dict[cov] = [player.__getattribute__(cov).loc[last_injury_date:date].values]

                    new_row = pd.DataFrame(new_dict)
                    durations_multiple = pd.concat([durations_multiple, new_row])

                    # Update last injury date:
                    last_injury_date = date

    durations_multiple.event = durations_multiple.event.astype(bool)
    durations_multiple.prior_injury = durations_multiple.prior_injury.astype(bool)
    return durations_multiple


def generate_durations_multivariate_recurrent_averaged(team: Team, covariates: List[str]) -> pd.DataFrame:
    """
        Creates data frame with durations and events for multiple injuries,
        but calculates the mean of the covariate values.
        Can this way use data frame in a Lifelines regression model.

    :param team:
    :param covariates:
    :return durations_manipulated:
    """
    durations_multiple = generate_durations_multivariate_recurrent_all_values(team, covariates)
    durations_multiple = durations_multiple.drop("player_name", axis=1)

    functions = {}
    for cov in covariates:
        functions[cov] = "np.nanmean"

    durations_manipulated = manipulate_features(durations_multiple, functions)
    durations_manipulated.index = list(range(len(durations_manipulated)))
    return durations_manipulated


def manipulate_features(df: pd.DataFrame, feature_generator: Dict[str, str]) -> pd.DataFrame:
    """
        Manipulates a data frame based on functions given in the feature_generator.
        The feature_generator contains variable names and a function of what to do with the values.
        So for example {"acwr": "mean"} would calulate the mean of the ACWR values.

    :param df:
    :param feature_generator:
    :return manipulated_df:
    """
    manipulated_df = df.copy()
    for feature, function in feature_generator.items():

        # Calculate (nr of nans/interval length) ratio for each feature interval:
        ratios = [np.count_nonzero(np.isnan(interval_values))/interval_values.size
                  for _, interval_values in manipulated_df[feature].items()]
        # manipulated_df[feature + "_ratio"] = ratios
        # manipulated_df[feature + "_mean"] = manipulated_df[feature].apply(eval("mean"))
        manipulated_df[feature] = manipulated_df[feature].apply(eval(function))

    return manipulated_df


