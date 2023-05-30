import pandas as pd
import numpy as np
from ml_util import *
from baseline_framework import *
#from LSTM_benchmark import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set_theme()
sns.set_style("dark")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
sns.set_palette("deep")
import warnings
warnings.filterwarnings("ignore")


def createBarplots(df):
    """
	creates barplots from dataframe

	Arguments:
        df: dataset
	
    Returns:
		-
	"""
    figsize = (9, 6)
    plt.figure(figsize=figsize)
    print(df)

    ax = sns.barplot(x = "window_out", y = "values", hue = "models", data = df, width=0.7)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig("experiment_plots/multistep_barplot_sessions")


def main():
    """
    Creates barplots showing the difference between using direct and one-step-ahead forecasting
	"""

    df = createDataset()
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "month", "day"]
    list_of_df = []

    """"""""""""""""""""""""""""""
    configNr = 200
    runOnce = True
    only_real_days = False

    n_in = 7
    n_out = 3
    multistep = True
    ml_rmse1 = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr, only_real_days)
    data_df = pd.DataFrame(data=ml_rmse1.loc[:].values.flatten().tolist(), columns=["values"])
    data_df["models"] = ["xgb", "lin", "tree"]
    list_of_df.append(data_df)
    """"""""""""""""""""""""""""""
    configNr = 201
    runOnce = True

    n_in = 7
    n_out = 3
    multistep = False
    ml_rmse2 = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr, only_real_days)
    data_df = pd.DataFrame(data=ml_rmse2.loc[:].values.flatten().tolist(), columns=["values"])
    data_df["models"] = ["xgb", "lin", "tree"]
    list_of_df.append(data_df)
    """"""""""""""""""""""""""""""
    configNr = 202
    runOnce = True

    n_in = 7
    n_out = 7
    multistep = True
    ml_rmse3 = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr, only_real_days)
    data_df = pd.DataFrame(data=ml_rmse3.loc[:].values.flatten().tolist(), columns=["values"])
    data_df["models"] = ["xgb", "lin", "tree"]
    list_of_df.append(data_df)
    """"""""""""""""""""""""""""""
    configNr = 203
    runOnce = True

    n_in = 7
    n_out = 7
    multistep = False
    ml_rmse4 = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr, only_real_days)
    data_df = pd.DataFrame(data=ml_rmse4.loc[:].values.flatten().tolist(), columns=["values"])
    data_df["models"] = ["xgb", "lin", "tree"]
    list_of_df.append(data_df)
    """"""""""""""""""""""""""""""


    merged = pd.concat(list_of_df)
    merged["values"] = merged["values"].round(decimals=2)
    merged["window_out"] = ["Direct- 3 out"]*3+["Multistep- 3 out"]*3+["Direct- 7 out"]*3+["Multistep- 7 out"]*3

    createBarplots(merged)

    titleText = "direct-versus-one-step-ahead"
    df_name = "experiment_data/"+titleText
    merged.to_pickle(df_name)


if __name__ == "__main__":
    main()
