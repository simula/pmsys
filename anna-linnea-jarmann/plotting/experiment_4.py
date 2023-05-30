import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from typing import List

from lifelines import CoxPHFitter

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from read_in import generate_teams
from classes import Team
from preprocessing import generate_durations_multivariate_recurrent_averaged, generate_durations_multivariate_averaged
from cox import cox_ph_regression, cox_penalizing, test_prop_hazard, get_optimal_values, cox_optimal_penalizing
from constants import REG_COVARIATES, REDUCED_COVARIATES

"""
    Experiment 4 - Multivariate Models - Cox PH with Regularization
    * using averaged covariates
    * testing for first injuries and recurrent injuries
    * adding regularization for feature selection improvement

"""


def test_cox_averaged(team: Team):
    """

    :param team:
    :return:
    """

    durations = generate_durations_multivariate_averaged(team, ["acwr", "sleep_duration"])
    print(durations)
    cph = cox_ph_regression(durations)

    fig, ax = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax)
    ax.set_title("Cox PH - First Injuries - Averaged Values", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.label.set_size(15)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_ph_averaged_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_cox_averaged(team_a: Team, team_b: Team, covariates_a: List[str], covariates_b: List[str]):
    """

    :param team_a:
    :param team_b:
    :return:
    """
    durations_a = generate_durations_multivariate_averaged(team_a, covariates_a)
    durations_b = generate_durations_multivariate_averaged(team_b, covariates_b)

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
    plt.savefig("../figures/both_teams/cox_ph_both_teams_averaged.pdf")
    plt.close()


def test_cox_averaged_recurrent(team: Team):
    """

    :param team:
    :return:
    """

    durations = generate_durations_multivariate_recurrent_averaged(team, REG_COVARIATES)
    cph = cox_ph_regression(durations)

    fig, ax = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax)
    ax.set_title("Cox PH - Recurrent Injuries - Averaged Values", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.label.set_size(15)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_ph_recurrent_averaged_team_{team.name[-1].lower()}.pdf")
    plt.close()


def compare_teams_cox_averaged_recurrent(team_a: Team, team_b: Team, covariates_a: List[str], covariates_b: List[str]):
    """

    :param team_a:
    :param team_b:
    :return:
    """
    durations_a = generate_durations_multivariate_recurrent_averaged(team_a, covariates_a)
    durations_b = generate_durations_multivariate_recurrent_averaged(team_b, covariates_b)

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
    plt.savefig("../figures/both_teams/cox_ph_both_teams_averaged_recurrent.pdf")
    plt.close()


def compare_structures_cox_averaged(team: Team, covariates: List[str]):
    """
        First injuries vs. Recurrent injuries for a team using Cox PH.

    :param covariates:
    :param team:
    :return:
    """

    durations_first = generate_durations_multivariate_averaged(team, ["acwr", "sleep_quality", "fatigue"])
    durations_recurrent = generate_durations_multivariate_recurrent_averaged(team, covariates)

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
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_ph_averaged_both_structures_team_{team.name[-1].lower()}.pdf")
    plt.close()


def test_cox_penalizing(team: Team, start: float, stop: float, steps: int, nr_of_folds: int,
                        threshold: float, covariates: List[str]):
    """
        Fits several Cox PH models with penalizing using different values.
        Cross validates each model to find the best penalizing value.
        Fits another Cox PH model using the optimal penalizing value for covariate selection.
        Covariates are selected with a threshold determining if it's coefficient is far enough from zero.

        Plots the coefficients for different penalizing values.
        Plots c index and BIC values for the different penalizing values.
        Plots the risk of each covariate.

        Checks the proportional hazard assumption.

    :param team:
    :param start:
    :param stop:
    :param steps:
    :param nr_of_folds:
    :param threshold:
    :return:
    """

    durations = generate_durations_multivariate_recurrent_averaged(team, covariates)

    # First test Cox models with different penalizing values and cross validation:
    cph_params, cv_results = cox_penalizing(durations, start, stop, steps, nr_of_folds)

    # Plot coefficients and penalizing values:
    plot_coefficients_and_penalizer(team, cph_params, steps)

    # Plot C index and BIC values:
    plot_c_and_bic(team, cv_results, steps)

    # Run Cox regression with the optimal penalizing value and the top variables:
    optimal_p, optimal_c, optimal_bic = get_optimal_values(cv_results)
    cph_optimal, df_optimal, optimal_covariates = cox_optimal_penalizing(durations, optimal_p, threshold)
    cph_optimal.print_summary()

    # Plotting optimal Cox models:
    plot_cox_optimal(team, cph_optimal, optimal_p, steps)

    # Plotting covariates risk:
    plot_covariates_risk(team, cph_optimal, optimal_p, steps)

    # Plotting covariates impact on survival rate:
    plot_covariates_survival_rate(team, cph_optimal, steps)

    # Test the proportional hazard assumption:
    print("\nTesting the proportional hazard assumption:")
    test_prop_hazard(cph_optimal, df_optimal)

    print("\nOptimal lambda:\t\t", optimal_p)
    print("Optimal BIC:\t\t", optimal_bic)
    print("Optimal C-index:\t", optimal_c)
    print()


def plot_coefficients_and_penalizer(team: Team, cph_params: dict, steps: int):
    """

    :param team:
    :param cph_params:
    :param steps:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    pd.DataFrame(cph_params).T.plot(ax=ax)
    # ax.set_title("Cox Penalizing", fontsize=20)
    ax.set_xlabel("λ", fontsize=18)
    ax.set_ylabel("coefficients", fontsize=18)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_penalizing_team_{team.name[-1].lower()}_{steps}.pdf")
    plt.close()


def plot_c_and_bic(team: Team, cv_results: dict, steps: int):
    """
        Plots the concordance index and BIC values for different penalizing values.
        Using the results from cross validation of a Cox model with penalizing.

    :param team:
    :param cv_results:
    :param steps:
    :return:
    """
    pen_values = [p for p, result in cv_results.items()]
    mean_c = [mean(result["c_scores"]) for p, result in cv_results.items()]
    mean_bic = [mean(result["bic_scores"]) for p, result in cv_results.items()]

    # Create axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(pen_values, mean_c, color="orange", label="mean c")
    ax1.set_xlabel("λ", fontsize=18)
    ax1.set_ylabel("average c index", fontsize=18)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)
    ax2.plot(pen_values, mean_bic, color="green", label="mean bic")
    ax2.set_ylabel("average bic values", fontsize=18)
    ax2.tick_params(axis="y", labelsize=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0, fontsize=18)
    # ax.set_title("C and BIC", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_c_bic_team_{team.name[-1].lower()}_{steps}.pdf")
    plt.close()


def plot_cox_optimal(team: Team, cph: CoxPHFitter, optimal_p: float, steps: int):
    """

    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax)
    ax.set_title(f"λ = {optimal_p:.3f}", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=18)
    ax.xaxis.label.set_size(18)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_optimal_team_{team.name[-1].lower()}_{steps}.pdf")
    plt.close()


def plot_covariates_risk(team: Team, cph_optimal: CoxPHFitter, optimal_p: float, steps: int):
    """
        Plots the risk of each covariate from a Cox PH model.

    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, value in cph_optimal.params_.items():
        if key == "prior_injury":
            ax.plot(cph_optimal.params_[key] * np.linspace(start=0, stop=1, num=5), label=key)

        if key == "acwr":
            ax.plot(cph_optimal.params_[key] * np.linspace(start=0, stop=5, num=5), label=key)

        if key == "readiness":
            ax.plot(cph_optimal.params_[key] * np.linspace(start=1, stop=10, num=5), label=key)

        if key == "sleep_duration":
            ax.plot(cph_optimal.params_[key] * np.linspace(start=0, stop=12, num=5), label=key)

        if key in ["fatigue", "mood", "sleep_quality", "soreness", "stress"]:
            ax.plot(cph_optimal.params_[key] * np.linspace(start=1, stop=5, num=5), label=key)

    ax.set_title(f"λ = {optimal_p:.3f}", fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    plt.savefig(f"../figures/team_{team.name[-1].lower()}/cox_risk_factors_team_{team.name[-1].lower()}_{steps}.pdf")
    plt.close()


def plot_covariates_survival_rate(team: Team, cph: CoxPHFitter, steps: int):
    """
        Plots the survival function using different values for each covariate.
        Includes the baseline hazard.

    :return:
    """
    for key, value in cph.params_.items():
        if key == "prior_injury":
            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111)
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
            ax = cph.plot_partial_effects_on_outcome(key, values=range(0, 2), cmap="coolwarm")
            # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=15)
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc=0, fontsize=18)
            # plt.title("Prior Injury", fontsize=20)
            plt.xlabel("days", fontsize=15)
            plt.ylabel("survival probability", fontsize=15)
            plt.tick_params(axis="x", labelsize=15)
            plt.tick_params(axis="y", labelsize=15)
            plt.legend(fontsize=18, labels=range(0, 2))
            plt.tight_layout()
            plt.savefig(f"../figures/team_{team.name[-1].lower()}/team_{team.name[-1].lower()}_prior_injury_{steps}.pdf")
            plt.close()

        if key == "acwr":
            ax = cph.plot_partial_effects_on_outcome(key, values=np.arange(0, 4, 0.5), cmap="coolwarm")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc=0, fontsize=18)
            # plt.title("ACWR", fontsize=20)
            plt.xlabel("days", fontsize=15)
            plt.ylabel("survival probability", fontsize=15)
            plt.tick_params(axis="x", labelsize=15)
            plt.tick_params(axis="y", labelsize=15)
            plt.legend(fontsize=18, labels=np.arange(0, 4, 0.5))
            plt.tight_layout()
            plt.savefig(f"../figures/team_{team.name[-1].lower()}/team_{team.name[-1].lower()}_acwr_{steps}.pdf")
            plt.close()

        if key == "readiness":
            ax = cph.plot_partial_effects_on_outcome(key, values=np.arange(1, 11, 2), cmap="coolwarm")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc=0, fontsize=18)
            # plt.title("Readiness", fontsize=20)
            plt.xlabel("days", fontsize=15)
            plt.ylabel("survival probability", fontsize=15)
            plt.tick_params(axis="x", labelsize=15)
            plt.tick_params(axis="y", labelsize=15)
            plt.legend(fontsize=18, labels=np.arange(1, 11, 2))
            plt.tight_layout()
            plt.savefig(f"../figures/team_{team.name[-1].lower()}/team_{team.name[-1].lower()}_readiness_{steps}.pdf")
            plt.close()

        if key == "sleep_duration":
            ax = cph.plot_partial_effects_on_outcome(key, values=np.arange(0, 12, 2), cmap="coolwarm")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc=0, fontsize=18)
            # plt.title("Sleep Duration", fontsize=20)
            plt.xlabel("days", fontsize=15)
            plt.ylabel("survival probability", fontsize=15)
            plt.tick_params(axis="x", labelsize=15)
            plt.tick_params(axis="y", labelsize=15)
            plt.legend(fontsize=18, labels=np.arange(0, 12, 2))
            plt.tight_layout()
            plt.savefig(f"../figures/team_{team.name[-1].lower()}/team_{team.name[-1].lower()}_sleep_duration_{steps}.pdf")
            plt.close()

        if key in ["fatigue", "mood", "sleep_quality", "soreness", "stress"]:
            pretty_key = (key.replace("_", " ")).title()
            ax = cph.plot_partial_effects_on_outcome(key, values=range(1, 6), cmap="coolwarm")
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc=0, fontsize=18)
            # plt.title(f"{pretty_key}", fontsize=20)
            plt.xlabel("days", fontsize=15)
            plt.ylabel("survival probability", fontsize=15)
            plt.tick_params(axis="x", labelsize=15)
            plt.tick_params(axis="y", labelsize=15)
            plt.legend(fontsize=18, labels=range(1, 6))
            plt.tight_layout()
            plt.savefig(f"../figures/team_{team.name[-1].lower()}/team_{team.name[-1].lower()}_{key}_{steps}.pdf")
            plt.close()


def compare_teams_cox_penalizing(team_a: Team, team_b: Team, start: float, stop: float, steps: int, nr_of_folds: int,
                      threshold: float):
    """
         Comparing results from Cox regression from two teams.

    :param team_a:
    :param team_b:
    :param start:
    :param stop:
    :param steps:
    :param nr_of_folds:
    :param threshold:
    :return:
    """
    team_a_durations = generate_durations_multivariate_recurrent_averaged(team_a, REG_COVARIATES)
    team_b_durations = generate_durations_multivariate_recurrent_averaged(team_b, REG_COVARIATES)

    # First test Cox models with different penalizing values and cross validation:
    cph_params_a, cv_results_a = cox_penalizing(team_a_durations, start, stop, steps, nr_of_folds)
    cph_params_b, cv_results_b = cox_penalizing(team_b_durations, start, stop, steps, nr_of_folds)

    # Plot coefficients and penalizing values:
    plot_coefficients_and_penalizer_two_teams(cph_params_a, cph_params_b, steps)

    # Plot C index and BIC values:
    plot_c_and_bic_two_teams(cv_results_a, cv_results_b, start, stop, steps, nr_of_folds)

    # Run Cox regression with the optimal penalizing value and the top variables:
    optimal_p_a, optimal_c_a, optimal_bic_a = get_optimal_values(cv_results_a)
    cph_optimal_a, df_optimal_a, optimal_variables_a = cox_optimal_penalizing(team_a_durations, optimal_p_a, threshold)

    optimal_p_b, optimal_c_b, optimal_bic_b = get_optimal_values(cv_results_b)
    cph_optimal_b, df_optimal_b, optimal_variables_b = cox_optimal_penalizing(team_b_durations, optimal_p_b, threshold)

    # Plotting optimal Cox models:
    plot_cox_optimal_two_teams(cph_optimal_a, cph_optimal_b, optimal_p_a, optimal_p_b, steps)

    # Plotting covariates risk:
    plot_covariates_risk_two_teams(cph_optimal_a, cph_optimal_b, steps, optimal_p_a, optimal_p_b)

    # Plotting covariates impact on survival rate for both teams:
    plot_covariates_survival_rate_two_teams(team_a, team_b, cph_optimal_a, cph_optimal_b, steps)

    # Test the proportional hazard assumption:
    print("\nTesting the proportional hazard assumption:")
    test_prop_hazard(cph_optimal_a, df_optimal_a)
    test_prop_hazard(cph_optimal_b, df_optimal_b)
    print()

    # scaled_schoenfeld_residuals = cph_optimal_a.compute_residuals(df_optimal_a, kind="scaled_schoenfeld")
    # print(scaled_schoenfeld_residuals)
    # scaled_schoenfeld_residuals = scaled_schoenfeld_residuals.sort_index()
    # plt.plot(scaled_schoenfeld_residuals.index, scaled_schoenfeld_residuals["acwr"])
    # plt.show()


def plot_coefficients_and_penalizer_two_teams(cph_params_a, cph_params_b, steps):
    """

    :param cph_params_a:
    :param cph_params_b:
    :param steps:
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 15))
    pd.DataFrame(cph_params_a).T.plot(ax=ax1)
    ax1.set_title("Team A", fontsize=20)
    ax1.set_xlabel("λ", fontsize=15)
    ax1.set_ylabel("coefficients", fontsize=15)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)
    # box1 = ax1.get_position()
    # ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax1.legend(fontsize=15)

    pd.DataFrame(cph_params_b).T.plot(ax=ax2)
    ax2.set_title("Team B", fontsize=20)
    ax2.set_xlabel("λ", fontsize=15)
    ax2.set_ylabel("coefficients", fontsize=15)
    ax2.tick_params(axis="x", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    # box2 = ax2.get_position()
    # ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax2.legend(fontsize=15)

    # fig.suptitle(f"Cox PH Regression with Penalizing λ: {start} - {stop} ({steps} steps)")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/both_teams/cox_penalizing_both_teams_{steps}.pdf")
    plt.close()


def two_scales(ax1, x, y1, y2):
    """
        Generates two y axes in the same plot.
        Makes it possible to plot two different plots next to each other.

    :param ax1:
    :param x:
    :param y1:
    :param y2:
    :return:
    """
    ax2 = ax1.twinx()
    ax1.plot(x, y1, color="orange", label="mean c")
    ax1.set_xlabel("λ", fontsize=15)
    ax1.set_ylabel("average c index", fontsize=15)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)
    ax2.plot(x, y2, color="green", label="mean bic")
    ax2.set_ylabel("average bic values", fontsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0, fontsize=15)
    return ax1, ax2


def plot_c_and_bic_two_teams(cv_results_a, cv_results_b, start, stop, steps, nr_of_folds):
    """
        Plots the concordance index and BIC values for different penalizing values.
        Using the results from cross validation of two Cox models with penalizing from two teams.

    :param cv_results_a:
    :param cv_results_b:
    :param start:
    :param stop:
    :param steps:
    :param nr_of_folds:
    :return:
    """
    # Team A:
    pen_values_a = [p for p, result in cv_results_a.items()]
    mean_c_a = [mean(result["c_scores"]) for p, result in cv_results_a.items()]
    mean_bic_a = [mean(result["bic_scores"]) for p, result in cv_results_a.items()]

    # Team B:
    pen_values_b = [p for p, result in cv_results_b.items()]
    mean_c_b = [mean(result["c_scores"]) for p, result in cv_results_b.items()]
    mean_bic_b = [mean(result["bic_scores"]) for p, result in cv_results_b.items()]

    # Create axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1, ax1a = two_scales(ax1, pen_values_a, mean_c_a, mean_bic_a)
    ax1.set_title("Team A", fontsize=20)

    ax2, ax2a = two_scales(ax2, pen_values_b, mean_c_b, mean_bic_b)
    ax2.set_title("Team B", fontsize=20)

    # fig.suptitle(f"Average C Index and BIC Values from Cox PH Regression using {nr_of_folds}-fold Cross Validation"
    #             f" - λ: {start} - {stop} ({steps} steps)")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/both_teams/c_and_bic_both_teams_{steps}.pdf")
    plt.close()


def plot_cox_optimal_two_teams(cph_a: CoxPHFitter, cph_b: CoxPHFitter, optimal_p_a: float, optimal_p_b: float, steps: int):
    """
    :param cph_a:
    :param cph_b:
    :param optimal_p_a:
    :param optimal_p_b:
    :param steps:
    :return:
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    cph_a.plot(ax=ax1)
    ax1.set_title(f"Team A - λ = {optimal_p_a:.3f}", fontsize=20)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.xaxis.label.set_size(15)

    cph_b.plot(ax=ax2)
    ax2.set_title(f"Team B - λ = {optimal_p_b:.3f}", fontsize=20)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.xaxis.label.set_size(15)

    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/both_teams/cox_optimal_both_teams_{steps}.pdf")
    plt.close()


def plot_covariates_risk_two_teams(cph_optimal_a: CoxPHFitter, cph_optimal_b: CoxPHFitter, steps: int,
                                   optimal_p_a: float, optimal_p_b: float):
    """
        Plots the risk of each covariate from two Cox PH models from two teams.

    :param cph_optimal_a:
    :param cph_optimal_b:
    :param steps:
    :param optimal_p_a:
    :param optimal_p_b:
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    for key, value in cph_optimal_a.params_.items():
        if key == "acwr":
            ax1.plot(cph_optimal_a.params_[key] * np.linspace(start=0, stop=5, num=5), label=key)

        if key == "readiness":
            ax1.plot(cph_optimal_a.params_[key] * np.linspace(start=1, stop=10, num=5), label=key)

        if key == "sleep_duration":
            ax1.plot(cph_optimal_a.params_[key] * np.linspace(start=0, stop=12, num=5), label=key)

        if key in ["fatigue", "mood", "sleep_quality", "soreness", "stress"]:
            ax1.plot(cph_optimal_a.params_[key] * np.linspace(start=1, stop=5, num=5), label=key)

    ax1.set_title(f"Team A - λ = {optimal_p_a:.3f}", fontsize=20)
    ax1.legend(fontsize=15)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)

    for key, value in cph_optimal_b.params_.items():
        if key == "acwr":
            ax2.plot(cph_optimal_b.params_[key] * np.linspace(start=0, stop=5, num=5), label=key)

        if key == "readiness":
            ax2.plot(cph_optimal_b.params_[key] * np.linspace(start=1, stop=10, num=5), label=key)

        if key == "sleep_duration":
            ax2.plot(cph_optimal_b.params_[key] * np.linspace(start=0, stop=12, num=5), label=key)

        if key in ["fatigue", "mood", "sleep_quality", "soreness", "stress"]:
            ax2.plot(cph_optimal_b.params_[key] * np.linspace(start=1, stop=5, num=5), label=key)

    ax2.set_title(f"Team B - λ = {optimal_p_b:.3f}", fontsize=20)
    ax2.legend(fontsize=15)
    ax2.tick_params(axis="x", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)

    # fig.suptitle(f"Injury Risk Variables from Cox PH Regression with Threshold: {threshold}")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"../figures/both_teams/risk_factors_both_teams_{steps}.pdf")
    plt.close()


def plot_covariates_survival_rate_two_teams(team_a: Team, team_b: Team, cph_a: CoxPHFitter, cph_b: CoxPHFitter, steps: int):
    """

    :param steps:
    :param team_b:
    :param team_a:
    :param cph_a:
    :param cph_b:
    :return:
    """
    plot_covariates_survival_rate(team=team_a, cph=cph_a, steps=steps)
    plot_covariates_survival_rate(team=team_b, cph=cph_b, steps=steps)


# Create Team, Player and Injury objects:
teams = generate_teams(path=parentdir, multivariate=True)
team_a = teams[1]
team_b = teams[0]

# test_cox_averaged(team_a)
# test_cox_averaged_recurrent(team_a)
# compare_structures_cox_averaged(team_a, REG_COVARIATES)
# compare_structures_cox_averaged(team_b, REDUCED_COVARIATES)
compare_teams_cox_averaged(team_a, team_b, REDUCED_COVARIATES, REDUCED_COVARIATES)
compare_teams_cox_averaged_recurrent(team_a, team_b, REG_COVARIATES, REG_COVARIATES)
test_cox_penalizing(team=team_a, start=0.01, stop=0.9, steps=20, nr_of_folds=5, threshold=0.1, covariates=REG_COVARIATES)
test_cox_penalizing(team=team_b, start=0.01, stop=0.9, steps=20, nr_of_folds=5, threshold=0.1, covariates=REG_COVARIATES)
# compare_teams_cox_penalizing(team_a, team_b, start=0.01, stop=0.9, steps=20, nr_of_folds=5, threshold=0.01)
