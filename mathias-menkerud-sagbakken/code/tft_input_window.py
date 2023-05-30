import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from baseline_framework import *
from ml_util import *
from dataloader import *
from lstmM1 import *
from LSTM_benchmark import *
from TFT_benchmark import *

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
sns.set_theme()
sns.set_style("dark")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
sns.set_palette("deep")



def getInputWindows_TFT(df, n_in, n_out):
    """
	creates lineplots showing how RMSE moves with output window size

	Arguments:
        df: dataset
        n_in: input window
        n_out: output_window
	
    Returns:
		-
	"""
    configNr = 102
    runOnce = True
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month", "daysSinceLastSession", "Metabolic_power", "Duration", "match"]
    
    players = list(df['player_name_x'].unique())

    results_tft = []

    for i in range(len(n_out)):
        print(i,"/",len(n_out))
        tft_temp = []
        for j in range(len(n_in)):
            print(j,"/",len(n_in))

            tft_rmse = run_benchmark_tft(df, columnNames, n_out[i], n_in[j], players, runOnce, configNr)

            tft_temp.append(tft_rmse.iloc[0]['tft_rmse'])

        results_tft.append(tft_temp)


    results_tft = pd.DataFrame(
    {'Output window = 1': results_tft[0],
     'Output window = 3': results_tft[1],
     'Output window = 7': results_tft[2],
     'Output window = 14': results_tft[3]
    })



    return results_tft


def main():
    """
    plots tft rmse for different input window sizes
	"""
    
    """CONSTANT VALUES"""
    df = createDataset()
    df = to_sessions(df)
    df = df.loc[df['Team_name'] == "TeamA"]
    n_in = list(range(1, 21))
    n_out = [1,3,7,14]
    range_tx = [x for x in n_in]
    print(range_tx)
    tft_rmse = getInputWindows_TFT(df,n_in, n_out)
    """---------------"""

    sns.lineplot(data=tft_rmse[['Output window = 1', 'Output window = 3', 'Output window = 7', 'Output window = 14']]).set(title='', xlabel="Input window value", ylabel="RMSE-Scores")
    plt.xticks(range_tx)
    plt.savefig("experiment_plots/input_window_lineplot_tft_sessions", pad_inches=1)
    plt.close()

    titleText = "Inputwindow-TFT-Team-A-sessions"
    df_name = "experiment_data/"+titleText
    tft_rmse.to_pickle(df_name)


    """--------------------------"""


if __name__ == "__main__":
    main()