import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ml_util import *
from dataloader import *
from lstmM1 import *
from LSTM_benchmark import *

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
sns.set_theme()
sns.set_style("dark")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
sns.set_palette("deep")



def getInputWindows_LSTM(df, n_in, n_out):
    """
    creates plot showing RMSE of different input and output window sizes

    Arguments:
        df: entire dataset
        n_out: output window
        n_in: input_window
    
    Returns:
        dataframe of results
    """
    nr_players = len(list(df['player_name_x'].unique()))
    batch_size = 16
    epoch = 15
    learning_rate = 1e-3
    num_hidden_units = 32
    configNr = 102
    runOnce = True
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month", "daysSinceLastSession", "Metabolic_power", "Duration", "match"]
    
    players = list(df['player_name_x'].unique())

    results_lstm = []

    for i in range(len(n_out)):
        print(i,"/",len(n_out))
        lstm_temp = []
        for j in range(len(n_in)):
            print(j,"/",len(n_in))

            lstm_rmse = run_benchmark_lstm(df, n_in[j], n_out[i], columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr)

            lstm_temp.append(lstm_rmse.iloc[0]['lstm_rmse'])

        results_lstm.append(lstm_temp)


    results_lstm = pd.DataFrame(
    {'Output window = 1': results_lstm[0],
     'Output window = 3': results_lstm[1],
     'Output window = 7': results_lstm[2],
     'Output window = 14': results_lstm[3]
    })



    return results_lstm


def main():

    """
    Plots the input and output window RMSE
    """
    
    """CONSTANT VALUES"""
    df = createDataset()
    df = to_sessions(df)
    df = df.loc[df['Team_name'] == "TeamA"]
    n_in = list(range(1, 21))
    n_out = [1,3,7,14]
    range_tx = [x for x in n_in]
    print(range_tx)
    lstm_rmse = getInputWindows_LSTM(df,n_in, n_out)
    """---------------"""

    sns.lineplot(data=lstm_rmse[['Output window = 1', 'Output window = 3', 'Output window = 7', 'Output window = 14']]).set(title='', xlabel="Input window value", ylabel="RMSE-Scores")
    plt.xticks(range_tx)
    plt.legend(loc='upper right')
    plt.savefig("experiment_plots/input_window_lineplot_lstm_sessions", pad_inches=1)
    plt.close()

    titleText = "Inputwindow-LSTM-Team-A-sessions"
    df_name = "experiment_data/"+titleText
    lstm_rmse.to_pickle(df_name)

    """--------------------------"""


if __name__ == "__main__":
    main()