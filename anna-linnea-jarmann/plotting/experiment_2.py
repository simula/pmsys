import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from read_in import generate_teams, read_csv_files
from classes import Team
from preprocessing import generate_durations_multivariate, generate_durations_multivariate_recurrent
from cox import cox_ph_regression
from constants import REG_COVARIATES, REDUCED_COVARIATES

"""
    Experiment 2 - Multivariate Models - Cox PH
    * using day-of-the-event values
    * testing for both first injuries and recurrent injuries
    * adding a covariate prior_injury for recurrent events

"""


def correlation_matrix_both_teams(path: str, team_names: List[str]):
    """
        Generates and plots correlation matrix for the variables in the dataset.

    :param path:
    :param team_names:
    :return:
    """
    teams_path = f"{path}/Subjective/per player/"

    for team_name in team_names:
        daily_f_path = f"{teams_path}/{team_name}/daily-features"
        daily_f_df = read_csv_files(daily_f_path)

        # Remove columns:
        daily_f_df = daily_f_df.drop(columns=["Unnamed: 0", "player_name", "injury_ts"])

        # Remove NaNs:
        # daily_f_df = daily_f_df.dropna()

        # Change ACWR values (incorrect in the dataset):
        daily_f_df['acwr'] = (daily_f_df['atl'] / daily_f_df['ctl42'])

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(daily_f_df.corr(), annot=True, cmap="Blues", fmt=".1f")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=17)
        plt.tick_params(axis='both', which='major', labelsize=17)

        plt.tight_layout()
        team = team_name[-1].lower()
        plt.savefig(f"../figures/team_{team}/correlation_matrix_team_{team}.pdf")


def test_cox(team: Team):
    """
        Fits a Cox PH model using a dataframe with durations, events and specific covariates.
        Prints summary and plots the results.

    :param team:
    :return:
    """

    durations_multivariate = generate_durations_multivariate(team, REG_COVARIATES)

    cph = cox_ph_regression(durations_multivariate)
    cph.print_summary()

    fig, ax = plt.subplots(figsize=(10, 10))

    cph.plot(ax=ax)
    # ax.set_title("Cox PH - First Injuries", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.label.set_size(15)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_ph_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_cox(team_a: Team, team_b: Team, covariates_a: List[str], covariates_b: List[str]):
    """

    :param covariates_b:
    :param covariates_a:
    :param team_a:
    :param team_b:
    :return:
    """

    durations_a = generate_durations_multivariate(team_a, covariates_a)
    durations_b = generate_durations_multivariate(team_b, covariates_b)

    cph_a = cox_ph_regression(durations_a)
    cph_a.print_summary()

    cph_b = cox_ph_regression(durations_b)
    cph_b.print_summary()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    cph_a.plot(ax=ax1)
    ax1.set_title("Team A", fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.label.set_size(15)

    cph_b.plot(ax=ax2)
    ax2.set_title("Team B", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig("../figures/both_teams/cox_ph_both_teams.pdf")
    plt.close()


def test_cox_recurrent(team: Team):
    durations = generate_durations_multivariate_recurrent(team, REG_COVARIATES)
    durations = durations.drop(columns=["player_name"])
    cph = cox_ph_regression(durations)

    fig, ax = plt.subplots(figsize=(10, 10))

    cph.plot(ax=ax)
    # ax.set_title("Cox PH - Recurrent Injuries", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_ph_recurrent_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_cox_recurrent(team_a: Team, team_b: Team, covariates_a: List[str], covariates_b: List[str]):
    """

    :param covariates_b:
    :param covariates_a:
    :param team_a:
    :param team_b:
    :return:
    """
    durations_a = generate_durations_multivariate_recurrent(team_a, covariates_a)
    durations_a = durations_a.drop(columns=["player_name"])
    durations_b = generate_durations_multivariate_recurrent(team_b, covariates_b)
    durations_b = durations_b.drop(columns=["player_name"])

    cph_a = cox_ph_regression(durations_a)
    cph_a.print_summary()
    cph_b = cox_ph_regression(durations_b)
    cph_b.print_summary()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    cph_a.plot(ax=ax1)
    ax1.set_title("Team A", fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.label.set_size(15)

    cph_b.plot(ax=ax2)
    ax2.set_title("Team B", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig("../figures/both_teams/cox_ph_both_teams_recurrent.pdf")
    plt.close()


def compare_structures_cox(team: Team, covariates: List[str]):
    """
        First injuries vs. Recurrent injuries for a team using Cox PH.

    :param covariates:
    :param team:
    :return:
    """
    durations_first = generate_durations_multivariate(team, ["fatigue", "acwr"])
    durations_recurrent = generate_durations_multivariate_recurrent(team, covariates)
    durations_recurrent = durations_recurrent.drop(columns=["player_name"])

    cph_first = cox_ph_regression(durations_first)
    cph_recurrent = cox_ph_regression(durations_recurrent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    cph_first.plot(ax=ax1)
    ax1.set_title("First Injuries", fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.label.set_size(15)

    cph_recurrent.plot(ax=ax2)
    ax2.set_title("Recurrent Injuries", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_ph_both_structures_team_{team.name[-1].lower()}.pdf")
    plt.close()


# Create Team, Player and Injury objects:
teams = generate_teams(path=parentdir, multivariate=True)
team_a = teams[1]
team_b = teams[0]

correlation_matrix_both_teams(parentdir, team_names=["TeamA", "TeamB"])
compare_teams_cox(team_a, team_b, REDUCED_COVARIATES, REDUCED_COVARIATES)
compare_teams_cox_recurrent(team_a, team_b, REG_COVARIATES, REG_COVARIATES)