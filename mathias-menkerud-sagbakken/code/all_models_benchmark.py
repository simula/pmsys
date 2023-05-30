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



def createBoxplots(ml, lstm, tft, configNr, titleText):
    """
    creates boxplot for each model for rmse from predictions

    Arguments:
        ml: df of rmse values for ml models 
        lstm: df of rmse values for lstm models 
        tft: df of rmse values for tft models 
        configNr: configNr
        titleText: part of filename

    
    Returns:
       -
    """

    ml["lstm"] = lstm
    ml["tft"] = tft

    melted_df = pd.melt(ml)
    hist = sns.boxplot(x='variable', y='value', data=melted_df)
    hist.set(title="", xlabel=None, ylabel="RMSE-Scores")
    hist.set_xticks(range(5), labels=["XGB", "LIN", "TREE", "LSTM", "TFT"])
    ml.loc[len(ml.index)] = [ml['xgb'].mean(), ml['lin'].mean(), ml['tree'].mean(), ml['lstm'].mean(), ml['tft'].mean()]
    ml.loc[len(ml.index)] = [ml['xgb'].head(-1).median(), ml['lin'].head(-1).median(), ml['tree'].head(-1).median(), ml['lstm'].head(-1).median(), ml['tft'].head(-1).median()]
    ml.loc[len(ml.index)] = [ml['xgb'].min(), ml['lin'].min(), ml['tree'].min(), ml['lstm'].min(), ml['tft'].min()]
    ml.loc[len(ml.index)] = [ml['xgb'].max(), ml['lin'].max(), ml['tree'].max(), ml['lstm'].max(), ml['tft'].max()]
    ml.loc[len(ml.index)] = [ml['xgb'].head(-4).std(), ml['lin'].head(-4).std(), ml['tree'].head(-4).std(), ml['lstm'].head(-4).std(), ml['tft'].head(-4).std()]
    ml.columns = ['XGB: RMSE', 'Lin: RMSE', 'Tree: RMSE', "LSTM: RMSE", "TFT: RMSE"] 
    ml = ml.round(decimals=3)
    
    index_labels=[]
    for i in range(len(ml)-5):
        index_labels.append("PLayer"+str(i+1))
    index_labels = index_labels+["MEAN","MEDIAN", "MIN", "MAX", "SD"]
    ml.index = index_labels
    
    df_name = "experiment_plots/data/boxplots_config"+str(configNr)+titleText
    ml.to_pickle(df_name)

    plots_and_table(ml.tail(5))
    figname = "experiment_plots/boxplots_config"+str(configNr)
    plt.savefig(figname, bbox_inches="tight", pad_inches=1)
    plt.close()

def main():
    """
    call on all models to make predictions, store results, and create boxplots
    """

    df = createDataset()
    #df = to_sessions(df)

    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())

    """"""""""""""""""""""""""""""
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    #columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month", "daysSinceLastSession", "Metabolic_power", "Duration", "match"]
    #columnNames = ["readiness"]
    configNr = 61
    runOnce = False
    multistep = True
    only_real_days = True
    titleText = 'RMSE-SCORES-FOR-CONFIG51-feature_selection--1-out--team A-only-actual-days'

    n_in = 7
    n_out = 1
    ml_rmse = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr, only_real_days)
    #print(ml_rmse)

    nr_players = len(list(df['player_name_x'].unique()))
    batch_size = 16
    epoch = 50
    sequence_length = 7
    n_out = 1
    learning_rate = 1e-3
    num_hidden_units = 32 
    lstm_rmse = run_benchmark_lstm(df, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr, only_real_days)

    input_window = 7
    forecast_horizon = 1
    tft_rmse = run_benchmark_tft(df, columnNames, forecast_horizon, input_window, players, runOnce, configNr, only_real_days)

    createBoxplots(ml_rmse, lstm_rmse, tft_rmse, configNr, titleText)

    df_name = "experiment_data/"+titleText+"tft"
    tft_rmse.to_pickle(df_name)
    df_name = "experiment_data/"+titleText+"lstm"
    lstm_rmse.to_pickle(df_name)
    df_name = "experiment_data/"+titleText+"ml"
    ml_rmse.to_pickle(df_name)

    """"""""""""""""""""""""""""""
    columnNames = ["readiness"]
    configNr = 43
    runOnce = False
    titleText = 'RMSE-SCORES-FOR-CONFIG-Only-Readiness--1-out--team A and B-- for B'

    n_in = 7
    n_out = 1
    multistep = True
    ml_rmse = runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr)

    nr_players = len(list(df['player_name_x'].unique()))
    batch_size = 16
    epoch = 50
    sequence_length = 7
    n_out = 1
    learning_rate = 1e-3
    num_hidden_units = 32 
    lstm_rmse = run_benchmark_lstm(df, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr)

    input_window = 7
    forecast_horizon = 1
    tft_rmse = run_benchmark_tft(df, columnNames, forecast_horizon, input_window, players, runOnce, configNr)

    createBoxplots(ml_rmse, lstm_rmse, tft_rmse, configNr, titleText)
    """"""""""""""""""""""""""""""

if __name__ == "__main__":
    main()
