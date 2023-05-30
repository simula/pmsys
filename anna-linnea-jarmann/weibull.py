import numpy as np
import pandas as pd

from typing import List
from lifelines import WeibullFitter, WeibullAFTFitter
from matplotlib import pyplot as plt


def weibull(durations: pd.DataFrame) -> WeibullFitter:
    """
        Fits a Weibull model using the durations data frame.
    :param durations:
    :return wb:
    """
    wb = WeibullFitter()
    wb.fit(durations.duration, durations.event)
    return wb


def weibull_aft(df: pd.DataFrame, covariates: List[str]) -> WeibullAFTFitter:
    cov_df: pd.DataFrame = df[["duration", "event"]].copy()
    cov_df.event = cov_df.event.astype(bool)
    for cov in covariates:
        cov_df[cov] = df[cov]
    wb_aft = WeibullAFTFitter()
    wb_aft.fit(df=cov_df, duration_col=cov_df.duration, event_col=cov_df.event)
    return wb_aft


def weibull_distribution(x, lam, rho):
    return np.exp(-(x/lam)**rho)


def plot_weibull(wb: WeibullFitter):
    wb.survival_function_.plot()
    plt.xlabel("Days")
    plt.ylabel("")
    plt.show()
