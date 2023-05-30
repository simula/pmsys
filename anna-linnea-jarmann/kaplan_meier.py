import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def kaplan_meier(durations: pd.DataFrame) -> KaplanMeierFitter:
    km = KaplanMeierFitter()
    km.fit(durations.duration, durations.event)
    return km


def plot_kaplan_meier(km: KaplanMeierFitter):
    km.plot_survival_function()
    plt.show()

