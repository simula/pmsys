import pandas as pd
import matplotlib.pyplot as plt
import re

from typing import Any

from lifelines.utils import find_best_parametric_model

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from read_in import generate_teams
from classes import Team
from preprocessing import generate_durations_univariate, generate_durations_univariate_recurrent
from kaplan_meier import kaplan_meier, plot_kaplan_meier
from weibull import plot_weibull, weibull

"""
    Experiment 1 - Univariate Models:
    * comparing survival functions of univariate models with injury distribution
    
"""


def test_kaplan_meier(team: Team):
    """
        Fits a Kaplan-Meier model with durations and events.
        Fits a Kaplan-Meier model with multiple injuries for each player.
        Plots the results.

    :param team:
    :return:
    """
    durations = generate_durations_univariate(team)
    km = kaplan_meier(durations)
    plot_kaplan_meier(km)

    # Trying with multiple injuries:
    durations_multiple = generate_durations_univariate_recurrent(team)
    km_m = kaplan_meier(durations_multiple)
    plot_kaplan_meier(km_m)


def test_weibull(team: Team):
    """
        Fits a Weibull model with durations and events with multiple injuries.
        Prints summary, AIC and plots the results.

    :param team:
    :return:
    """
    print("\n\nWeibull Model:")
    durations = generate_durations_univariate_recurrent(team)
    wb = weibull(durations)
    print(wb.summary)

    lam = wb.lambda_
    rho = wb.rho_
    aic = wb.AIC_
    print("\nWeibull AIC: ", aic)

    plot_weibull(wb)


def find_best_model(team: Team) -> (Any, float):
    """
        Finds the best parametric model using the durations dataframe for multiple injuries.
        Plots the hazard rate of the best model.

    :param team:
    :return best_model, best_aic:
    """
    df = generate_durations_univariate_recurrent(team)
    best_model, best_aic = find_best_parametric_model(event_times=df.duration,
                                                      event_observed=df.event,
                                                      scoring_method="AIC")
    print(best_model)
    best_model.plot_hazard()
    return best_model, best_aic


def test_univariate_models(team: Team):
    """
        Fits 3 univariate models; Kaplan-Meier, Weibull and best parametric model.
        Plots the survival functions.

    :param team:
    :return:
    """
    durations = generate_durations_univariate(team)

    km = kaplan_meier(durations)
    wb = weibull(durations)
    best, best_aic = find_best_parametric_model(event_times=durations.duration,
                                                event_observed=durations.event,
                                                scoring_method="BIC")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    team_hist = durations.duration.loc[durations["event"]]
    ax1.hist(team_hist, bins=len(team_hist), color="lightblue")
    ax1.set_xlabel("days", fontsize=18)
    ax1.set_ylabel("injury frequency", fontsize=18)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2 = ax1.twinx()
    km.plot_survival_function(ax=ax2, label="Kaplan-Meier", ci_show=False, color="green")
    wb.plot_survival_function(ax=ax2, label="Weibull", ci_show=False, color="orange")
    label = re.split(r'[_:]', best._label)[0].replace("Fitter", "")
    best.plot_survival_function(ax=ax2, label=f"{label}", ci_show=False, color="purple")
    ax2.set_ylabel("survival", fontsize=18)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.legend(fontsize=18)

    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/univariate_models_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_univariate_models(team_a: Team, team_b: Team):
    """
        Compares the results of univariate models from two different teams.

    :param team_a:
    :param team_b:
    :return:
    """

    team_a_durations = generate_durations_univariate(team_a)
    team_b_durations = generate_durations_univariate(team_b)

    km_a = kaplan_meier(team_a_durations)
    km_b = kaplan_meier(team_b_durations)

    wb_a = weibull(team_a_durations)
    wb_b = weibull(team_b_durations)

    best_a, best_a_aic = find_best_parametric_model(event_times=team_a_durations.duration,
                                                    event_observed=team_a_durations.event,
                                                    scoring_method="BIC")
    best_b, best_b_aic = find_best_parametric_model(event_times=team_b_durations.duration,
                                                    event_observed=team_b_durations.event,
                                                    scoring_method="BIC")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Team A:
    team_a_hist = team_a_durations.duration.loc[team_a_durations["event"]]
    ax1.hist(team_a_hist, bins=len(team_a_hist), color="lightblue")
    # ax1.set_title("Team A", fontsize=20)
    ax1.set_xlabel("days", fontsize=15)
    ax1.set_ylabel("injury frequency", fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax1_2 = ax1.twinx()
    km_a.plot_survival_function(ax=ax1_2, label="Kaplan-Meier", ci_show=False, color="green")
    # na_a.plot_hazard(ax=ax1_2, label="Nelson-Aalen", ci_show=False)
    wb_a.plot_survival_function(ax=ax1_2, label="Weibull", ci_show=False, color="orange")
    # a_label = best_a._label.split()[0].replace(":", "")
    a_label = re.split(r'[_:]', best_a._label)[0]
    best_a.plot_survival_function(ax=ax1_2, label=f"{a_label}", ci_show=False, color="purple")
    ax1_2.set_ylabel("survival", fontsize=15)
    ax1_2.tick_params(axis='y', labelsize=15)
    ax1_2.legend(fontsize=15)

    # Team B:
    team_b_hist = team_b_durations.duration.loc[team_b_durations["event"]]
    ax2.hist(team_b_hist, bins=len(team_b_hist), color="lightblue")
    # ax2.set_title("Team B", fontsize=20)
    ax2.set_xlabel("days", fontsize=15)
    ax2.set_ylabel("injury frequency", fontsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax2_2 = ax2.twinx()
    km_b.plot_survival_function(ax=ax2_2, label="Kaplan-Meier", ci_show=False, color="green")
    # na_b.plot_hazard(ax=ax2_2, label="Nelson-Aalen", ci_show=False)
    wb_b.plot_survival_function(ax=ax2_2, label="Weibull", ci_show=False, color="orange")
    # b_label = best_b._label.split()[0].replace(":", "")
    b_label = re.split(r'[_:]', best_b._label)[0]
    best_b.plot_survival_function(ax=ax2_2, label=f"{b_label}", ci_show=False, color="purple")
    ax2_2.set_ylabel("survival", fontsize=15)
    ax2_2.tick_params(axis='y', labelsize=15)
    ax2_2.legend(fontsize=15)

    # plt.show()
    plt.tight_layout()
    plt.savefig("../figures/both_teams/univariate_models_hist_both_teams.pdf")
    plt.close()


def test_univariate_models_recurrent(team: Team):
    """

    :param team:
    :return:
    """
    durations = generate_durations_univariate_recurrent(team)

    km = kaplan_meier(durations)
    wb = weibull(durations)
    best, best_aic = find_best_parametric_model(event_times=durations.duration,
                                                event_observed=durations.event,
                                                scoring_method="AIC")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    team_a_hist = durations.day.loc[durations["event"]]
    ax1.hist(team_a_hist, bins=len(team_a_hist), color="lightblue")
    ax1.set_xlabel("days", fontsize=15)
    ax1.set_ylabel("injury frequency", fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax1_2 = ax1.twinx()
    km.plot_survival_function(ax=ax1_2, label="Kaplan-Meier", ci_show=False, color="green")
    wb.plot_survival_function(ax=ax1_2, label="Weibull", ci_show=False, color="orange")
    # a_label = best_a._label.split()[0].replace(":", "")
    label = re.split(r'[_:]', best._label)[0]
    best.plot_survival_function(ax=ax1_2, label=f"{label}", ci_show=False, color="purple")
    ax1_2.set_ylabel("survival", fontsize=15)
    ax1_2.tick_params(axis='y', labelsize=15)
    ax1_2.legend(fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/univariate_models_recurrent_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_univariate_models_recurrent(team_a: Team, team_b: Team):
    """
        Compares the results of univariate models from two different teams.

    :param team_a:
    :param team_b:
    :return:
    """

    team_a_durations = generate_durations_univariate_recurrent(team_a)
    team_b_durations = generate_durations_univariate_recurrent(team_b)

    km_a = kaplan_meier(team_a_durations)
    km_b = kaplan_meier(team_b_durations)

    wb_a = weibull(team_a_durations)
    wb_b = weibull(team_b_durations)

    best_a, best_a_aic = find_best_parametric_model(event_times=team_a_durations.duration,
                                                    event_observed=team_a_durations.event,
                                                    scoring_method="AIC")
    best_b, best_b_aic = find_best_parametric_model(event_times=team_b_durations.duration,
                                                    event_observed=team_b_durations.event,
                                                    scoring_method="AIC")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Team A:
    team_a_hist = team_a_durations.day.loc[team_a_durations["event"]]
    ax1.hist(team_a_hist, bins=len(team_a_hist), color="lightblue")
    ax1.set_title("Team A", fontsize=20)
    ax1.set_xlabel("days", fontsize=15)
    ax1.set_ylabel("injury frequency", fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax1_2 = ax1.twinx()
    km_a.plot_survival_function(ax=ax1_2, label="Kaplan-Meier", ci_show=False, color="green")
    # na_a.plot_hazard(ax=ax1_2, label="Nelson-Aalen", ci_show=False)
    wb_a.plot_survival_function(ax=ax1_2, label="Weibull", ci_show=False, color="orange")
    # a_label = best_a._label.split()[0].replace(":", "")
    a_label = re.split(r'[_:]', best_a._label)[0]
    best_a.plot_survival_function(ax=ax1_2, label=f"{a_label}", ci_show=False, color="purple")
    ax1_2.set_ylabel("survival", fontsize=15)
    ax1_2.tick_params(axis='y', labelsize=15)
    ax1_2.legend(fontsize=15)

    # Team B:
    team_b_hist = team_b_durations.day.loc[team_b_durations["event"]]
    ax2.hist(team_b_hist, bins=len(team_b_hist), color="lightblue")
    ax2.set_title("Team B", fontsize=20)
    ax2.set_xlabel("days", fontsize=15)
    ax2.set_ylabel("injury frequency", fontsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax2_2 = ax2.twinx()
    km_b.plot_survival_function(ax=ax2_2, label="Kaplan-Meier", ci_show=False, color="green")
    # na_b.plot_hazard(ax=ax2_2, label="Nelson-Aalen", ci_show=False)
    wb_b.plot_survival_function(ax=ax2_2, label="Weibull", ci_show=False, color="orange")
    # b_label = best_b._label.split()[0].replace(":", "")
    b_label = re.split(r'[_:]', best_b._label)[0]
    best_b.plot_survival_function(ax=ax2_2, label=f"{b_label}", ci_show=False, color="purple")
    ax2_2.set_ylabel("survival", fontsize=15)
    ax2_2.tick_params(axis='y', labelsize=15)
    ax2_2.legend(fontsize=15)

    # plt.show()
    plt.tight_layout()
    plt.savefig("figures/both_teams/univariate_models_hist_both_teams_recurrent.pdf")
    plt.close()


def compare_univariate_models_two_teams(team_a: Team, team_b: Team):
    """
        Compares the results of univariate models from two different teams.

    :param team_a:
    :param team_b:
    :return:
    """

    team_a_durations = generate_durations_univariate(team_a)
    team_b_durations = generate_durations_univariate(team_b)

    km_a = kaplan_meier(team_a_durations)
    km_b = kaplan_meier(team_b_durations)

    wb_a = weibull(team_a_durations)
    wb_b = weibull(team_b_durations)

    best_a, best_a_aic = find_best_parametric_model(event_times=team_a_durations.duration,
                                                    event_observed=team_a_durations.event,
                                                    scoring_method="AIC")
    best_b, best_b_aic = find_best_parametric_model(event_times=team_b_durations.duration,
                                                    event_observed=team_b_durations.event,
                                                    scoring_method="AIC")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    km_a.plot_survival_function(ax=ax1, label="Team A", ci_show=False)
    km_b.plot_survival_function(ax=ax1, label="Team B", ci_show=False)
    ax1.set_title("Kaplan-Meier", fontsize=20)
    ax1.set_xlabel(None)
    ax1.set_ylabel("survival", fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.legend(fontsize=15)

    ax2.set_title("Nelson-Aalen", fontsize=20)
    ax2.set_xlabel(None)
    ax2.set_ylabel("cumulative hazard", fontsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.legend(fontsize=15)

    wb_a.plot_cumulative_hazard(ax=ax3, label="Team A", ci_show=False)
    wb_b.plot_cumulative_hazard(ax=ax3, label="Team B", ci_show=False)
    ax3.set_title("Weibull", fontsize=20)
    ax3.set_xlabel("days", fontsize=15)
    ax3.set_ylabel("cumulative hazard", fontsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.legend(fontsize=15)

    best_a.plot_cumulative_hazard(ax=ax4, label=f"Team A - {best_a._label}", ci_show=False)
    best_b.plot_cumulative_hazard(ax=ax4, label=f"Team A - {best_b._label}", ci_show=False)
    ax4.set_title("Best Parametric Model", fontsize=20)
    ax4.set_xlabel("days", fontsize=15)
    ax4.set_ylabel("cumulative hazard", fontsize=15)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax4.legend(fontsize=15)

    # plt.show()
    plt.tight_layout()
    plt.savefig("figures/both_teams/univariate_models_both_teams.pdf")
    plt.close()


def plot_injury_histogram(df: pd.DataFrame):
    """
        Plots a histogram based on durations.
        Visualizes the distribution of durations for occured injuries.

    :param df:
    :return:
    """
    injury_df = df[df["event"] == 1]
    plt.hist(injury_df.duration, bins=len(injury_df))
    plt.xlabel("Days")
    plt.ylabel("Injury Frequency")
    plt.show()


# Create Team, Player and Injury objects:
teams = generate_teams(path=parentdir, multivariate=False)
team_a = teams[1]
team_b = teams[0]

compare_teams_univariate_models(team_a, team_b)
