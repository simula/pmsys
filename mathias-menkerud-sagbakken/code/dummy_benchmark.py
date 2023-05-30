import pandas as pd
import numpy as np
from ml_util import *
from baseline_framework import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.dummy import DummyRegressor
sns.set_theme()
sns.set_style("dark")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
sns.set_palette("deep")
import warnings
warnings.filterwarnings("ignore")


def createLinePlots(actual ,predicted, i, name):
    """
    creates lineplot for test set

    Arguments:
        actual: actual y
        predicted: predicted y
        i: confignr
        name: string file name
        configNr: configNr
    
    Returns:
        -
    """
    RMSE = mean_squared_error(actual["readiness"], predicted["readiness"], squared=False)
    MSE = mean_squared_error(actual["readiness"], predicted["readiness"], squared=True)
    print(f'MSE Score on Test set: {MSE:0.4f}')
    print(f'RMSE Score on Test set: {RMSE:0.4f}')
    fig, ax = plt.subplots(figsize=(15, 5))
    actual["readiness"].plot(ax=ax, label='Actual', title='RMSE: '+str(RMSE), linewidth=1)
    predicted["readiness"].plot(ax=ax, label='Predicted', linewidth=1)
    plt.xlabel('Timesteps in days', fontsize=18)
    plt.ylabel('Readiness value', fontsize=16)
    ax.axvline(0, color='black', ls='--')
    ax.legend(['Actual', 'Predicted'])
    plt.savefig(name+str(i))
    plt.close()


def createBoxplots(ml, configNr, titleText):
    """
    creates boxplot for test set

    Arguments:
        ml: pandas dataframe results
        configNr: confignr
        titleText: string file name
    
    Returns:
        -
    """

    melted_df = pd.melt(ml)
    hist = sns.boxplot(x='variable', y='value', data=melted_df)
    hist.set(title=titleText, xlabel=None, ylabel="RMSE-Scores")
    hist.set_xticks(range(1), labels=["Dummy-model"])
    ml.loc[len(ml.index)] = [ml['dummy'].mean()]
    ml.loc[len(ml.index)] = [ml['dummy'].head(-1).median()]
    ml.loc[len(ml.index)] = [ml['dummy'].min()]
    ml.loc[len(ml.index)] = [ml['dummy'].max()]
    ml.loc[len(ml.index)] = [ml['dummy'].head(-4).std()]
    ml.columns = ['dummy: RMSE'] 
    ml = ml.round(decimals=3)
    
    index_labels=[]
    for i in range(len(ml)-5):
        index_labels.append("PLayer"+str(i+1))
    index_labels = index_labels+["MEAN","MEDIAN", "MIN", "MAX", "SD"]
    ml.index = index_labels
    plots_and_table(ml.tail(5))
    figname = "experiment_plots+/boxplots_dummy_team_b_config"+str(configNr)
    plt.savefig(figname, bbox_inches="tight", pad_inches=1)
    plt.close()


def runBenchmarkDummy(df, columnNames, n_in, n_out, players, runOnce, configNr):
    """
    Runs dummy model on all players on a given team

    Arguments:
        df: entire series.
        columnNames: features 
        n_out: output window
        n_in: input_window
        players: players
        runOnce: runOnce
        configNr: configNr
    
    Returns:
        dataframe of results
    """

    results_dummy = []

    #players = list(df['player_name_x'].unique())
    print("amount of players to train each config: ",len(players))

    for i in range(len(players)):

        print(i+1,"/",len(players))
        all_but_one = players[:i] + players[i+1:]
        df_train = df[df['player_name_x'].isin(all_but_one)]
        df_test = df.loc[df['player_name_x'] == players[i]]

        train = df_train[columnNames]
        test = df_test[columnNames]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        num_features = len(train.columns.tolist())

        train_scalar = StandardScaler()
        train = train_scalar.fit_transform(train)
        test = train_scalar.transform(test)

        train_multistep = series_to_supervised(train.copy(), n_in)
        test_multistep = series_to_supervised(test.copy(), n_in)
        features = train_multistep.columns.tolist()[:-num_features]
        targets = test_multistep.columns.tolist()[-num_features:]

        X_train = train_multistep[features]
        y_train = train_multistep[targets]
        X_test = test_multistep[features]
        y_test = test_multistep[targets]

        y_test = renameColumns(y_test, columnNames)

        dummy_regr = DummyRegressor(strategy="median")
        dummy_regr.fit(X_train, y_train)
        pred = dummy_regr.predict(X_test)
        pred = pd.DataFrame(data = np.array(pred), columns = columnNames)

        pred = pd.DataFrame(train_scalar.inverse_transform(pred), columns=columnNames)
        y_test_ = pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames)

        RMSE = mean_squared_error(y_test_["readiness"], pred["readiness"], squared=False)

        results_dummy.append(RMSE)

        if (i == 0):
            if configNr < 100:
                createLinePlots(y_test_, pred, configNr, "experiment_plots/dummy_lineplot_mean")

        if runOnce:
            break

    results_df = pd.DataFrame(
    {'dummy': results_dummy
    })


    return results_df



def main():

    df = createDataset()
    #df = to_sessions(df)

    df = df.loc[df['Team_name'] == "TeamB"]
    players = list(df['player_name_x'].unique())
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    #columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month", "daysSinceLastSession", "Metabolic_power", "Duration"]
    n_in = 7
    n_out = 1
    configNr = 3
    runOnce = False
    """"""""""""""""""""""""""""""
    
    dummy_rmse = runBenchmarkDummy(df, columnNames, n_in, n_out, players, runOnce, configNr)

    print(dummy_rmse)
    
    createBoxplots(dummy_rmse, configNr, "")


if __name__ == "__main__":
    main()
