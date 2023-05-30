from baseline_framework import *
from ml_util import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
sns.set_theme()
sns.set_style("dark")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
sns.set_palette("deep")

def getInputWindows_RMSE(df, n_in, n_out, columnNames):
    """
    calculates rmse when input window size increases

    Arguments:
        df: dataset in the form of a pandas dataframe 
        n_in: input window
        n_out: output window
        columnNames: list of features

    
    Returns:
        RMSE of predictions
    """
    results_xgb = []
    results_LIN = []
    results_TREE = []
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    train = train[columnNames]
    test = test[columnNames]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    num_features = len(train.columns.tolist())
    train_scalar = StandardScaler()
    train = train_scalar.fit_transform(train)
    test = train_scalar.transform(test)

    for i in range(len(n_out)):
        print(i,"/",len(n_out))
        xgb_temp = []
        lin_temp = []
        tree_temp = []
        for j in range(len(n_in)):
            print(j,"/",len(n_in))
            train_multistep = series_to_supervised(train.copy(), n_in[j])
            test_multistep = series_to_supervised(test.copy(), n_in[j])
            features = train_multistep.columns.tolist()[:-num_features]
            targets = test_multistep.columns.tolist()[-num_features:]

            X_train = train_multistep[features]
            y_train = train_multistep[targets]
            X_test = test_multistep[features]
            y_test = test_multistep[targets]

            y_test = renameColumns(y_test, columnNames)

            modelXGB = fitXGBoost(X_train, y_train, X_test, y_test)
            modelLin = fitLinearReg(X_train, y_train)
            modelTree = fitTree(X_train, y_train)

            y_predictXGB = recursive_multistep_prediction(modelXGB, X_test, n_out[i], num_features, columnNames)
            xgb_temp.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))

            y_predictLIN = recursive_multistep_prediction(modelLin, X_test, n_out[i], num_features, columnNames)
            lin_temp.append(calculate_RMSE(y_test, y_predictLIN, train_scalar, columnNames))

            y_predictTREE = recursive_multistep_prediction(modelTree, X_test, n_out[i], num_features, columnNames)
            tree_temp.append(calculate_RMSE(y_test, y_predictTREE, train_scalar, columnNames))
        results_xgb.append(xgb_temp)
        results_LIN.append(lin_temp)
        results_TREE.append(tree_temp)

    results_xgb = pd.DataFrame(
    {'Output window = 1': results_xgb[0],
     'Output window = 3': results_xgb[1],
     'Output window = 7': results_xgb[2],
     'Output window = 14': results_xgb[3]
    })

    results_lin = pd.DataFrame(
    {'Output window = 1': results_LIN[0],
     'Output window = 3': results_LIN[1],
     'Output window = 7': results_LIN[2],
     'Output window = 14': results_LIN[3]
    })

    results_tree = pd.DataFrame(
    {'Output window = 1': results_TREE[0],
     'Output window = 3': results_TREE[1],
     'Output window = 7': results_TREE[2],
     'Output window = 14': results_TREE[3]
    })

    return results_xgb, results_lin, results_tree


def main():
    """
    plots rmse when input window size increases
    """
    
    """CONSTANT VALUES"""
    path = "../data/mysql_dataset/complete_dataset"
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0", axis=1)
    df['date'] =  pd.to_datetime(df['date'])
    df['day'] = df.date.dt.dayofweek.astype(str).astype("category").astype(int)
    df["month"] = df.date.dt.month.astype(str).astype("category").astype(int)
    df = to_sessions(df)
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month", "daysSinceLastSession", "Metabolic_power", "Duration", "match"]
    """---------------"""

    n_in = list(range(1, 21))
    n_out = [1,3,7,14]
    range_tx = [x for x in n_in]
    print(range_tx)
    
    results_xgb, results_lin, results_tree = getInputWindows_RMSE(df, n_in, n_out, columnNames)
    sns.lineplot(data=results_xgb[['Output window = 1', 'Output window = 3', 'Output window = 7', 'Output window = 14']]).set(title='', xlabel="Input window value", ylabel="RMSE-Scores")
    plt.xticks(range_tx)
    plt.savefig("experiment_plots/input_window_lineplot_xgb_sessions", pad_inches=1)
    plt.close()

    sns.lineplot(data=results_lin[['Output window = 1', 'Output window = 3', 'Output window = 7', 'Output window = 14']]).set(title='', xlabel="Input window value", ylabel="RMSE-Scores")
    plt.xticks(range_tx)
    plt.savefig("experiment_plots/input_window_lineplot_lin_sessions", pad_inches=1)
    plt.close()

    sns.lineplot(data=results_tree[['Output window = 1', 'Output window = 3', 'Output window = 7', 'Output window = 14']]).set(title='', xlabel="Input window value", ylabel="RMSE-Scores")
    plt.xticks(range_tx)
    plt.savefig("experiment_plots/input_window_lineplot_tree_sessions", pad_inches=1)
    plt.close()

    print(results_xgb)
    print(results_lin)
    print(results_tree)

    titleText = "Input-Window-plots-"
    df_name = "experiment_data/"+titleText+"xgb"
    results_xgb.to_pickle(df_name)
    df_name = "experiment_data/"+titleText+"lin"
    results_lin.to_pickle(df_name)
    df_name = "experiment_data/"+titleText+"tree"
    results_tree.to_pickle(df_name)
    """--------------------------"""


if __name__ == "__main__":
    main()