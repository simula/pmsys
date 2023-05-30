import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines.plotting import plot_lifetimes
from statsmodels.stats.outliers_influence import variance_inflation_factor

from typing import List

from read_in import generate_teams, read_csv_files
from classes import Team
from preprocessing import generate_durations_multivariate_recurrent_all_values, generate_durations_univariate, \
    generate_durations_multivariate, generate_durations_univariate_recurrent, \
    generate_durations_time_varying, generate_durations_multivariate_recurrent_averaged, \
    generate_durations_multivariate_recurrent
from constants import REDUCED_COVARIATES, REG_COVARIATES, ALL_COVARIATES, DURATIONS_EXAMPLE, DURATIONS_RECURRENT_EXAMPLE


def generate_csv_files(team: Team):
    """
        Creates CSV files with the different types of dataframes with durations, events and covariates.

    :param team:
    :return:
    """
    # Generate path:
    team_path = f"../tables/team_{team.name[4:].lower()}"

    # Regular durations:
    d = generate_durations_univariate(team)
    d.to_csv(f'{team_path}/durations.csv')

    # Durations with features:
    df = generate_durations_multivariate(team, ALL_COVARIATES)
    df.to_csv(f'{team_path}/durations_with_features.csv')

    # Durations time varying:
    dtv = generate_durations_time_varying(team, REDUCED_COVARIATES)
    dtv.to_csv(f'{team_path}/durations_time_varying.csv')

    # Durations multiple injuries:
    dmi = generate_durations_multivariate_recurrent_all_values(team, ALL_COVARIATES)
    dmi.to_csv(f'{team_path}/durations_multiple_injuries.csv')

    # Durations multiple injuries manipulated:
    dm = generate_durations_multivariate_recurrent_averaged(team, ALL_COVARIATES)
    dm.to_csv(f'{team_path}/durations_manipulated.csv')


def compute_vif(df, considered_features):
    X = df[considered_features].copy()

    # the calculation of variance inflation requires a constant
    X["Intercept"] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Covariate"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif["Covariate"] != "Intercept"]
    return vif


def compute_vif_both_teams(path: str, team_names: List[str]):
    """
        Compute the Variance Inflation Factor (VIF) for each feature for each team.

    :return:
    """

    teams_path = f"{path}/SoccerMon/Subjective/per player/"

    for team_name in team_names:
        daily_f_path = f"{teams_path}/{team_name}/daily-features"
        daily_f_df = read_csv_files(daily_f_path)

        # Remove columns:
        daily_f_df = daily_f_df.drop(columns=["Unnamed: 0", "player_name", "injury_ts"])

        # Rename columns:
        daily_f_df = daily_f_df.rename(columns={"sleep-duration": "sleep_duration", "sleep-quality": "sleep_quality"})

        # Change ACWR values (incorrect in the dataset):
        daily_f_df["acwr"] = (daily_f_df["atl"] / daily_f_df["ctl42"])

        # Remove NaNs:
        daily_f_df = daily_f_df.dropna()

        # Compute VIF:
        considered_features = ["acwr",
                               "fatigue",
                               "mood",
                               "readiness",
                               "sleep_duration",
                               "sleep_quality",
                               "soreness",
                               "stress"]

        vif = compute_vif(daily_f_df, ALL_COVARIATES).sort_values("VIF", ascending=False)
        vif["VIF"] = vif["VIF"].apply(lambda x: '%.3f' % x)
        # print(vif)
        team = team_name[-1].lower()
        vif.to_csv(f'{path}/SurvivalAnalysis/survival_analysis/tables/team_{team}/vif_team_{team}.csv')


def plot_censoring(team: Team):
    """

    :param team:
    :return:
    """
    durations = generate_durations_univariate(team)
    durations["individual"] = list(range(0, len(durations)))
    durations = durations.set_index("individual")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_lifetimes(durations=durations.duration, event_observed=durations.event, sort_by_duration=False, ax=ax)
    ax.set_xlim(0, 800)
    ax.vlines(730, -1, 28, linestyles='--')
    ax.set_xlabel("time", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/censoring_team_{team.name[-1].lower()}.pdf")


def plot_censoring_multivariate(team: Team):
    """

    :param team:
    :return:
    """
    durations = generate_durations_univariate(team)
    durations["individual"] = list(range(0, len(durations)))
    durations = durations.set_index("individual")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_lifetimes(durations=durations.duration, event_observed=durations.event, sort_by_duration=False, ax=ax)
    ax.set_xlim(0, 800)
    ax.vlines(730, -1, 28, linestyles='--')
    ax.set_xlabel("time", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/censoring_multivariate_team_{team.name[-1].lower()}.pdf")


def plot_censoring_multivariate_recurrent(team: Team):
    """

    :param team:
    :return:
    """

    durations = generate_durations_multivariate_recurrent(team, REG_COVARIATES)
    durations_2 = generate_durations_multivariate(team, REG_COVARIATES)
    durations["individual"] = list(range(0, len(durations)))
    durations = durations.set_index("individual")

    for row_index, row in durations.iterrows():
        if not row["player_name"] in list(durations_2["player_name"]):
            durations = durations.drop(index=row_index)

    durations["individual"] = list(range(0, len(durations)))
    durations = durations.set_index("individual")

    durations = durations[["duration", "event", "player_name"]]
    players = durations.groupby(["player_name"])
    nr_of_players = durations["player_name"].nunique()
    event_observed = durations.event
    event_observed_color = "#A60628"
    event_censored_color = "#348ABD"

    fig, ax = plt.subplots(figsize=(7, 5))

    # If durations is pd.Series with non-default index, then use index values as y-axis labels.
    # label_plot_bars = type(durations) is pd.Series and type(durations.index) is not pd.RangeIndex

    N = durations.shape[0]
    entry = np.zeros(N)

    for player_name, group_df in players:
        # print(durations)
        # plt.hlines(y=2, xmin=2, xmax=5)
        #print(f"{player_name}:")
        c = event_censored_color
        last_duration = 0
        for row_index, row in group_df.iterrows():
            i = row_index
            c = event_observed_color if event_observed.iloc[i] else event_censored_color
            #print(f"{row_index}. {last_duration} + {row.duration}\t{event_observed.iloc[i]}")
            m = " " if not event_observed.iloc[i] else "o"
            ax.scatter(last_duration + row.duration, player_name, color=c, marker=m, s=13)
            last_duration = last_duration + row.duration

        ax.hlines(player_name, 0, last_duration, color=c, lw=1.5)

    ax.set_yticks(range(0, nr_of_players))
    ax.set_yticklabels(range(0, nr_of_players))
    ax.set_ylim(-0.5, nr_of_players)
    ax.set_xlim(0, 800)
    ax.vlines(730, -1, nr_of_players, linestyles='--')
    ax.set_xlabel("days", fontsize=12)
    ax.set_ylabel("individuals", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/censoring_multivariate_recurrent_team_{team.name[-1].lower()}.pdf")


def plot_example_censoring():
    """

    :return:
    """

    durations = DURATIONS_EXAMPLE.duration
    event_observed = DURATIONS_EXAMPLE.event
    event_observed_color = "#A60628"
    event_censored_color = "#348ABD"

    fig, ax = plt.subplots(figsize=(5, 5))

    # If durations is pd.Series with non-default index, then use index values as y-axis labels.
    label_plot_bars = type(durations) is pd.Series and type(durations.index) is not pd.RangeIndex

    N = durations.shape[0]
    entry = np.zeros(N)

    assert durations.shape[0] == N
    assert event_observed.shape[0] == N

    for i in range(N):
        c = event_observed_color if event_observed.iloc[i] else event_censored_color
        ax.hlines(i, entry[i], durations.iloc[i], color=c, lw=1.5)
        m = "o" if not event_observed.iloc[i] else "o"
        ax.scatter(durations.iloc[i], i, color=c, marker=m, s=13)

    if label_plot_bars:
        ax.set_yticks(range(0, N))
        ax.set_yticklabels(durations.index)
    else:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylim(-0.5, N)
    ax.set_xlim(0, 40)
    ax.vlines(20, -1, len(durations), linestyles='--')
    ax.set_xlabel("days", fontsize=12)
    ax.set_ylabel("individuals", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"../figures/censoring_example.pdf")


def plot_example_censoring_recurrent():
    """

    :return:
    """
    print(DURATIONS_RECURRENT_EXAMPLE)

    durations = DURATIONS_RECURRENT_EXAMPLE.duration
    event_observed = DURATIONS_RECURRENT_EXAMPLE.event
    event_observed_color = "#A60628"
    event_censored_color = "#348ABD"

    fig, ax = plt.subplots(figsize=(5, 5))

    # If durations is pd.Series with non-default index, then use index values as y-axis labels.
    label_plot_bars = type(durations) is pd.Series and type(durations.index) is not pd.RangeIndex

    N = durations.shape[0]
    entry = np.zeros(N)

    assert durations.shape[0] == N
    assert event_observed.shape[0] == N

    for i in range(N):
        c = event_observed_color if event_observed.iloc[i] else event_censored_color
        ax.hlines(i, entry[i], durations.iloc[i], color=c, lw=1.5)
        # plt.hlines(y=2, xmin=2, xmax=5)
        m = "o" if not event_observed.iloc[i] else "o"
        ax.scatter(durations.iloc[i], i, color=c, marker=m, s=13)

    if label_plot_bars:
        ax.set_yticks(range(0, N))
        ax.set_yticklabels(durations.index)
    else:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylim(-0.5, N)
    ax.set_xlim(0, 40)
    ax.vlines(20, -1, len(durations), linestyles='--')
    ax.set_xlabel("days", fontsize=15)
    ax.set_ylabel("individuals", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f"../figures/censoring_recurrent_example.pdf")


def team_statistics_univariate(team_a: Team, team_b: Team):

    durations_a = generate_durations_univariate(team_a)
    durations_b = generate_durations_univariate(team_b)

    durations_recurrent_a = generate_durations_univariate_recurrent(team_a)
    durations_recurrent_b = generate_durations_univariate_recurrent(team_b)

    # Counting injuries:
    count_a = durations_a.event.value_counts()
    count_b = durations_b.event.value_counts()
    count_recurrent_a = durations_recurrent_a.event.value_counts()
    count_recurrent_b = durations_recurrent_b.event.value_counts()

    injury_statistics = pd.DataFrame({
        "Team A": [len(durations_a), count_a[False], count_a[True], count_recurrent_a[True]],
        "Team B": [len(durations_b), count_b[False], count_b[True], count_recurrent_b[True]]
    }, index=["Players", "Non-injured", "Injured", "All injuries"])

    injury_statistics.to_csv(f"../tables/injury_statistics_univariate.csv")


def team_statistics_multivariate(team_a: Team, team_b: Team):

    durations_a = generate_durations_multivariate(team_a, REG_COVARIATES)
    durations_b = generate_durations_multivariate(team_b, REG_COVARIATES)
    count_a = durations_a.event.value_counts()
    count_b = durations_b.event.value_counts()

    durations_recurrent_a = generate_durations_multivariate_recurrent(team_a, REG_COVARIATES)
    durations_recurrent_b = generate_durations_multivariate_recurrent(team_b, REG_COVARIATES)
    count_recurrent_a = durations_recurrent_a.event.value_counts()
    count_recurrent_b = durations_recurrent_b.event.value_counts()

    statistics = pd.DataFrame({
        "Team A": [len(durations_a), count_a[False], count_a[True], count_recurrent_a[True]],
        "Team B": [len(durations_b), count_b[False], count_b[True], count_recurrent_b[True]]
    }, index=["Nr. of players", "Nr. of non-injured players", "Nr. of injured players", "Total nr. of injuries"])

    statistics.to_csv(f"../tables/injury_statistics_multivariate.csv")

    # Counting missing data:
    missing_instances_a = durations_a[durations_a.isna().any(axis=1)]
    missing_instances_b = durations_b[durations_b.isna().any(axis=1)]

    missing_points_a = durations_a.isnull().sum().sum()
    missing_points_b = durations_b.isnull().sum().sum()

    points_a = sum([durations_a[cov].value_counts(dropna=False).sum() for cov in REG_COVARIATES])
    points_b = sum([durations_b[cov].value_counts(dropna=False).sum() for cov in REG_COVARIATES])

    missing_data_statistics = pd.DataFrame({
        "Players": [len(durations_a), len(durations_b)],
        "Players missing data": [len(missing_instances_a), len(missing_instances_b)],
        "Missing points": [missing_points_a, missing_points_b],
        "All points": [points_a, points_b]
    }, index=["Team A", "Team B"])

    missing_data_statistics.to_csv(f"../tables/missing_data_statistics_multivariate.csv")


def team_statistics_time_varying(team_a: Team, team_b: Team):

    durations_a = generate_durations_time_varying(team_a, REG_COVARIATES)
    durations_b = generate_durations_time_varying(team_b, REG_COVARIATES)

    players_a = durations_a["player_name"].nunique()
    players_b = durations_b["player_name"].nunique()

    # Counting missing data:
    missing_instances_a = durations_a[durations_a.isna().any(axis=1)]
    missing_instances_b = durations_b[durations_b.isna().any(axis=1)]

    missing_points_a = durations_a.isnull().sum().sum()
    missing_points_b = durations_b.isnull().sum().sum()

    points_a = sum([durations_a[cov].value_counts(dropna=False).sum() for cov in REG_COVARIATES])
    points_b = sum([durations_b[cov].value_counts(dropna=False).sum() for cov in REG_COVARIATES])

    missing_data_statistics = pd.DataFrame({
        "Entries": [len(durations_a), len(durations_b)],
        "Entries missing data": [len(missing_instances_a), len(missing_instances_b)],
        "Missing points": [missing_points_a, missing_points_b],
        "All points": [points_a, points_b]
    }, index=["Team A", "Team B"])

    missing_data_statistics.to_csv(f"../tables/missing_data_statistics_time_varying.csv")


# Using path from local machine:
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

# Create Team, Player and Injury objects:
teams = generate_teams(parent)
team_a = teams[1]
team_b = teams[0]

# generate_csv_files(teamA)
# generate_csv_files(teamB)
# plot_example_censoring()
# plot_example_censoring_recurrent()
plot_censoring(team_a)
plot_censoring(team_b)
plot_censoring_multivariate(team_a)
plot_censoring_multivariate(team_b)
plot_censoring_multivariate_recurrent(team_a)
plot_censoring_multivariate_recurrent(team_b)
# team_statistics_univariate(team_a, team_b)
# team_statistics_multivariate(team_a, team_b)
# team_statistics_time_varying(team_a, team_b)