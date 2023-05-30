import numpy as np
import pandas as pd

from statistics import mean
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
from lifelines.utils import k_fold_cross_validation


def cox_ph_regression(durations_with_features: pd.DataFrame) -> CoxPHFitter:
    """
        Fitting a Cox Proportional Hazard model using durations, events and selected covariates.

    :param durations_with_features:
    :return cph:
    """
    cph = CoxPHFitter()
    cph.fit(durations_with_features, duration_col='duration', event_col='event')
    return cph


def cox_time_varying(durations_time_varying: pd.DataFrame) -> CoxTimeVaryingFitter:
    """
        Fitting a Cox Time Varying model for time series data.

    :param durations_time_varying:
    :return ctv:
    """
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(durations_time_varying,
            id_col="player_name",
            event_col="event",
            start_col="start",
            stop_col="stop",
            show_progress=False)
    return ctv


def cox_penalizing(durations: pd.DataFrame, start: float, stop: float, steps: int, folds: int) -> (dict, dict):
    """
        Fitting Cox PH models with different penalizing values between start and stop.
        Using k-fold cross validation to compare the goodness of the models.

    :param durations:
    :param stop:
    :param start:
    :param steps:
    :param folds:
    :return cph_params, cv_results:
    """

    # Checking variance of prior injury:
    # events = df['event'].astype(bool)
    # print(df.loc[events, 'prior_injury'].var())
    # print(df.loc[~events, 'prior_injury'].var())

    cph_params = {}
    cv_results = {}
    for ind, p in enumerate(np.linspace(start, stop, steps)):
        print(f"{ind+1} / {steps}")
        cph = CoxPHFitter(l1_ratio=1., penalizer=p).fit(durations, duration_col="duration", event_col="event")
        cph_params[p] = cph.params_

        # Cross validation:
        c_scores = k_fold_cross_validation(cph, durations, 'duration', event_col='event', k=folds, scoring_method="concordance_index")
        log_likelihoods = k_fold_cross_validation(cph, durations, 'duration', event_col='event', k=folds, scoring_method="log_likelihood")
        bic_scores = [len(cph.params_) * np.log(len(durations)) - 2 * log_likelihood for log_likelihood in log_likelihoods]
        cv_results[p] = {"c_scores": c_scores, "bic_scores": bic_scores}

    return cph_params, cv_results


def get_optimal_values(cv_results: dict) -> (float, float, float):
    """
        Finds the optimal penalizing value based on the lowest BIC and highest c index combination.

    :param cv_results:
    :return optimal_pen, optimal_c, optimal_bic:
    """

    # Calculating the mean BIC from the k nr. of folds:
    mean_bic = [mean(result["bic_scores"]) for p, result in cv_results.items()]

    optimal_pen = 0.0
    optimal_c = 0.0
    optimal_bic = max(mean_bic)
    for p, result in cv_results.items():
        c = mean(result["c_scores"])
        bic = mean(result["bic_scores"])
        if bic < optimal_bic and c > optimal_c:
            optimal_pen = p
            optimal_c = c
            optimal_bic = bic
    return optimal_pen, optimal_c, optimal_bic


def cox_optimal_penalizing(durations: pd.DataFrame, optimal_p: float, threshold: float) -> (CoxPHFitter, pd.DataFrame):
    """
        Fits a Cox PH model using the optimal penalizing value.
        Finds the optimal covariates by using the threshold.
        The threshold determines the minimum difference between zero and a covariant's coeffeicient
        for the covariant to be considered optimal.
        Fits a new Cox PH model with only the optimal covariants.

    :param durations:
    :param optimal_p:
    :param threshold:
    :return optimal_cph, optimal_df, optimal_values:
    """
    cph = CoxPHFitter(l1_ratio=1., penalizer=optimal_p).fit(durations, duration_col="duration", event_col="event")
    optimal_variables = ['duration', 'event']
    optimal_values = {}
    for key, value in cph.params_.items():
        if abs(value) > threshold:
            optimal_variables.append(str(key))
            optimal_values[key] = value
    optimal_df = durations[optimal_variables].copy()
    optimal_cph = CoxPHFitter().fit(optimal_df, duration_col="duration", event_col="event")
    return optimal_cph, optimal_df, optimal_values


def test_prop_hazard(cph: CoxPHFitter, df: pd.DataFrame):
    """
        Checks the propotional hazard assumption.

    :param cph:
    :param df:
    :return:
    """
    cph.check_assumptions(df, show_plots=True)