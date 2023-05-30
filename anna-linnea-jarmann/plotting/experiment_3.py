import matplotlib.pyplot as plt

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from read_in import generate_teams
from classes import Team
from preprocessing import generate_durations_time_varying, generate_durations_time_varying_recurrent
from cox import cox_time_varying
from constants import REDUCED_COVARIATES, REG_COVARIATES

"""
    Experiment 3 - Multivariate Models - Cox Time-Varying
    * using time-dependent variables
    * testing only for first injuries because Lifelines does not allow for recurrent events
    (could try to add it if i have the time)

"""


def test_cox_time_varying(team: Team):
    """
       Fits a Cox Time Varying model using a dataframe with durations, events and time varying covariates.
       Prints summary and plots the results.

    :param team:
    :return:
    """
    durations_time_varying = generate_durations_time_varying(team, REG_COVARIATES)
    ctv = cox_time_varying(durations_time_varying)
    # ctv.print_summary()
    fig, ax = plt.subplots(figsize=(10, 6))
    ctv.plot(ax=ax)
    # ax.set_title("Cox Time-Varying", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.label.set_size(15)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_tv_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_cox_time_varying(team_a: Team, team_b: Team):
    """

    :param team_a:
    :param team_b:
    :return:
    """
    durations_a = generate_durations_time_varying(team_a, REDUCED_COVARIATES)
    durations_b = generate_durations_time_varying(team_b, REDUCED_COVARIATES)

    cph_tv_a = cox_time_varying(durations_a)
    cph_tv_b = cox_time_varying(durations_b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    cph_tv_a.plot(ax=ax1)
    ax1.set_title("Team A", fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.label.set_size(15)

    cph_tv_b.plot(ax=ax2)
    ax2.set_title("Team B", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig("../figures/both_teams/cox_tv_both_teams_small.pdf")
    plt.close()

    # cph_tv_a.check_assumptions(durations_a)
    # cph_tv_b.check_assumptions(durations_b)


def test_cox_time_varying_recurrent(team: Team):
    """
        Does not work because Lifelines removes recurrent injuries.
        Can probably add them in a different way - but no time right now.

    :param team:
    :return:
    """
    durations = generate_durations_time_varying_recurrent(team, REG_COVARIATES)
    ctv = cox_time_varying(durations)
    # ctv.print_summary()
    fig, ax = plt.subplots(figsize=(10, 6))
    ctv.plot(ax=ax)
    ax.set_title("Cox Time-Varying", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.label.set_size(15)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_tv_recurrent_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_structures_cox_time_varying(team: Team):
    """
        First injuries vs. Recurrent injuries for a team using Cox Time-Varying.

    :param team:
    :return:
    """

    covariates = ["acwr"]
    durations_first = generate_durations_time_varying(team, covariates)
    durations_recurrent = generate_durations_time_varying_recurrent(team, covariates)

    ctv_first = cox_time_varying(durations_first)
    ctv_recurrent = cox_time_varying(durations_recurrent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ctv_first.plot(ax=ax1)
    ax1.set_title("First Injuries", fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.label.set_size(15)

    ctv_recurrent.plot(ax=ax2)
    ax2.set_title("Recurrent Injuries", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_tv_both_structures_team_{team.name[-1].lower()}.pdf")
    plt.close()


# Create Team, Player and Injury objects:
teams = generate_teams(path=parentdir, multivariate=True)
team_a = teams[1]
team_b = teams[0]

compare_teams_cox_time_varying(team_a, team_b)
