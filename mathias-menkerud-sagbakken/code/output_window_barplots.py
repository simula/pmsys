import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
sns.set_theme()
sns.set_style("dark")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
sns.set_palette("deep")

from baseline_framework import *
from TFT_benchmark import *
from LSTM_benchmark import *

def createLineplots(df):
    """
	creates lineplots showing how RMSE moves with output window size

	Arguments:
        df: dataset
	
    Returns:
		-
	"""
    print(df)
    df = df.reset_index(drop=True)
    df.to_pickle("experiment_plots/data/output_windowDF2")
    ax = sns.lineplot(data=df[list(df.columns)])
    ax.set(title="RMSE-score when increasing forecasting horizon", xlabel='Output window size', ylabel='RMSE-score')
    plt.savefig("experiment_plots/output_window_lineplot_sessions")


def createBarplots(df):
    """
	creates barplots showing how RMSE moves with output window size

	Arguments:
        df: dataset
	
    Returns:
		-
	"""
    figsize = (9, 6)
    plt.figure(figsize=figsize)

    ax = sns.barplot(x = "window_out", y = "values", hue = "models", data = df, width=0.7)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig("experiment_plots/output_window_barplot_sessions")



def main():
    """
    visualizes how RMSE of readiness moves when increasing output window size
	"""

    df = createDataset()
    df = to_sessions(df)
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "month", "day"]
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month", "daysSinceLastSession", "Metabolic_power", "Duration", "match"]
    #columnNames = ["readiness"]
    configNr = 101
    runOnce = True
    out_windows = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    list_of_df = []
    Lineplot_ML = []

    for i in range(len(out_windows)):

        n_in = 7
        n_out = out_windows[i]
        multistep = True
        ml_rmse = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr)

        nr_players = len(list(df['player_name_x'].unique()))
        batch_size = 16
        epoch = 50
        sequence_length = 7
        n_out = out_windows[i]
        learning_rate = 1e-3
        num_hidden_units = 32 
        lstm_rmse = run_benchmark_lstm(df, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr)

        input_window = 7
        forecast_horizon = out_windows[i]
        tft_rmse = run_benchmark_tft(df, columnNames, forecast_horizon, input_window, players, runOnce, configNr)

        ml_rmse["lstm"] = lstm_rmse
        ml_rmse["tft"] = tft_rmse

        ml_rmse = ml_rmse.round(decimals=2)
        Lineplot_ML.append(ml_rmse)
        data_df = pd.DataFrame(data=ml_rmse.loc[:].values.flatten().tolist(), columns=["values"])
        data_df["models"] = ["xgb", "lin", "tree", "lstm", "tft"]

        list_of_df.append(data_df)

    createLineplots(pd.concat(Lineplot_ML))

    merged = pd.concat(list_of_df)
    merged["window_out"] = [1]*5+[2]*5+[3]*5+[4]*5+[5]*5+[6]*5+[7]*5+[8]*5+[9]*5+[10]*5+[11]*5+[12]*5+[13]*5+[14]*5
    print(merged)
    createBarplots(merged)

    titleText = "output-window-sizes"
    df_name = "experiment_data/"+titleText
    merged.to_pickle(df_name)

    """"""""""""""""""""""""""""""

if __name__ == "__main__":
    main()
