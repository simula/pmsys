import pandas as pd
import numpy as np
from ml_util import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings("ignore")

def feature_selection(columnNames, test):
    """
	feature selection

	Arguments:
        columnNames: features
        test: df pandas dataframe

    Returns:
		selected features
	"""
    n_in = 1
    n_out = 1
    test = test[columnNames]
    test_direct = series_to_supervised(test.copy(), n_in, n_out)

    num_features = len(test.columns.tolist())
    features = test_direct.columns.tolist()[:-num_features]
    target = test_direct.columns.tolist()[-num_features:][0]

    X_test = test_direct[features]
    y_test = test_direct[target]

    features_selected = f_classif(X_test, y_test)

    results = {'f-statistics': list(features_selected[0]),
            'p-value': list(np.round(features_selected[1], 3)),
            'feature': columnNames}

    results = pd.DataFrame(results)
    print(results)

    results = results.loc[results['p-value'] < 0.05]

    columns = list(results["feature"])
    if "readiness" not in columns:
        columns.append("readiness")

    return columns

def createLinePlots(actual ,predicted, i, name):
    """
    creates lineplot for test set

    Arguments:
        actual: actual y
        predicted: predicted y
        i: confignr
        name: string file name
    
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
    
def to_actual_dates(train_scalar, columnNames, actual_days_with_imputed, actual_days, y_test, pred):
    """
    Only choose the actual days

    Arguments:
        train_scalar: scaled train set
        columnNames: list of features
        actual_days_with_imputed: list of all dates.
        actual_days: List of only actual days 
        y_test: actual y-values
        pred: prediction
    
    Returns:
        Pandas DataFrame with only actual dates
    """
    y_test = pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames)
    pred = pd.DataFrame(train_scalar.inverse_transform(pred), columns=columnNames)
    pred["date"] = actual_days_with_imputed
    y_test["date"] = actual_days_with_imputed
    pred = pred[pred['date'].isin(actual_days)]
    y_test = y_test[y_test['date'].isin(actual_days)]
    pred = pred.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return y_test, pred

def runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, runOnce, configNr, only_real_days):
    """
    Runs tft on all players on a given team

    Arguments:
        df: pandas dataframe
        columnNames: features 
        n_out: output window
        n_in: input_window
        multistep: forecasting method
        players: players
        runOnce: runOnce
        only_real_days: only_real_days
        configNr: configNr
    
    Returns:
        RMSE of predictions
    """

    results_xgb = []
    results_LIN = []
    results_TREE = []

    print("amount of players to train each config: ",len(players))

    df_pre = to_sessions(df)

    for i in range(len(players)):

        print(i+1,"/",len(players))
        all_but_one = players[:i] + players[i+1:]
        df_train = df[df['player_name_x'].isin(all_but_one)]
        df_test = df.loc[df['player_name_x'] == players[i]]
        test_pre = df_pre.loc[df_pre['player_name_x'] == players[i]]
        actual_days = list(test_pre["date"])[n_in:]
        actual_days_with_imputed = list(df_test["date"])[n_in:]

        #columnNames_selected = feature_selection(columnNames, df_test)
        #print(columnNames_selected)

        columnNames_selected = columnNames
        train = df_train[columnNames_selected]
        test = df_test[columnNames_selected]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        num_features = len(train.columns.tolist())

        train_scalar = StandardScaler()
        train = train_scalar.fit_transform(train)
        test = train_scalar.transform(test)

        if multistep:
            train_multistep = series_to_supervised(train.copy(), n_in)
            test_multistep = series_to_supervised(test.copy(), n_in)
            features = train_multistep.columns.tolist()[:-num_features]
            targets = test_multistep.columns.tolist()[-num_features:]

            X_train = train_multistep[features]
            y_train = train_multistep[targets]
            X_test = test_multistep[features]
            y_test = test_multistep[targets]

            y_test = renameColumns(y_test, columnNames_selected)

            modelXGB = fitXGBoost(X_train, y_train, X_test, y_test)
            modelLin = fitLinearReg(X_train, y_train)
            modelTree = fitTree(X_train, y_train)

            if only_real_days:
                y_predictXGB = recursive_multistep_prediction(modelXGB, X_test, n_out, num_features, columnNames_selected)
                y_test_, y_predictXGB = to_actual_dates(train_scalar, columnNames_selected, actual_days_with_imputed, actual_days, y_test, y_predictXGB)
                results_xgb.append(calculate_RMSE_actual_days(y_test_, y_predictXGB))

                y_predictLIN = recursive_multistep_prediction(modelLin, X_test, n_out, num_features, columnNames_selected)
                y_test_, y_predictLIN = to_actual_dates(train_scalar, columnNames_selected, actual_days_with_imputed, actual_days, y_test, y_predictLIN)
                results_LIN.append(calculate_RMSE_actual_days(y_test_, y_predictLIN))

                y_predictTREE = recursive_multistep_prediction(modelTree, X_test, n_out, num_features, columnNames_selected)
                y_test_, y_predictTREE = to_actual_dates(train_scalar, columnNames_selected, actual_days_with_imputed, actual_days, y_test, y_predictTREE)
                results_TREE.append(calculate_RMSE_actual_days(y_test_, y_predictTREE))

            else:
                y_predictXGB = recursive_multistep_prediction(modelXGB, X_test, n_out, num_features, columnNames_selected)
                results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames_selected))

                y_predictLIN = recursive_multistep_prediction(modelLin, X_test, n_out, num_features, columnNames_selected)
                results_LIN.append(calculate_RMSE(y_test, y_predictLIN, train_scalar, columnNames_selected))

                y_predictTREE = recursive_multistep_prediction(modelTree, X_test, n_out, num_features, columnNames_selected)
                results_TREE.append(calculate_RMSE(y_test, y_predictTREE, train_scalar, columnNames_selected))


        if not multistep:
            train_direct = series_to_supervised(train.copy(), n_in, n_out)
            test_direct = series_to_supervised(test.copy(), n_in, n_out)
            features = train_direct.columns.tolist()[:-num_features*n_out]
            targets = test_direct.columns.tolist()[-num_features:]
            X_train = train_direct[features]
            y_train = train_direct[targets]
            X_test = test_direct[features]
            y_test = test_direct[targets]

            y_test = renameColumns(y_test, columnNames)

            modelXGB = fitXGBoost(X_train, y_train, X_test, y_test)
            modelLin = fitLinearReg(X_train, y_train)
            modelTree = fitTree(X_train, y_train)

            y_predictXGB = modelXGB.predict(X_test)
            y_predictXGB = pd.DataFrame(data = np.array(y_predictXGB), columns = columnNames)
            results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))

            y_predictLIN = modelLin.predict(X_test)
            y_predictLIN = pd.DataFrame(data = np.array(y_predictLIN), columns = columnNames)
            results_LIN.append(calculate_RMSE(y_test, y_predictLIN, train_scalar, columnNames))

            y_predictTREE = modelTree.predict(X_test)
            y_predictTREE = pd.DataFrame(data = np.array(y_predictTREE), columns = columnNames)
            results_TREE.append(calculate_RMSE(y_test, y_predictTREE, train_scalar, columnNames))

        if (i == 0):
            if configNr < 100:
                if only_real_days:
                    createLinePlots(y_test_, y_predictXGB, configNr, "experiment_plots/XGB_lineplot")
                    createLinePlots(y_test_, y_predictLIN, configNr, "experiment_plots/LIN_lineplot")
                    createLinePlots(y_test_, y_predictTREE, configNr, "experiment_plots/TREE_lineplot")

                else:
                    createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictXGB), columns=columnNames), configNr, "experiment_plots/XGB_lineplot")
                    createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictLIN), columns=columnNames), configNr, "experiment_plots/LIN_lineplot")
                    createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictTREE), columns=columnNames), configNr, "experiment_plots/TREE_lineplot")

        if runOnce:
            break

    results_df = pd.DataFrame(
    {'xgb': results_xgb,
     'lin': results_LIN,
     'tree': results_TREE
    })

    return results_df


def calculate_RMSE(actual, predicted, train_scalar, columnNames):
    """
    calculates rmse of predictions

    Arguments:
        actual: actual values 
        predicted: predicted values 
        train_scalar: scaled train set
        columnNames: list of features

    
    Returns:
        RMSE of predictions
    """
    actual = actual.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)
    actual = pd.DataFrame(train_scalar.inverse_transform(actual), columns=columnNames)
    predicted = pd.DataFrame(train_scalar.inverse_transform(predicted), columns=columnNames)
    results = pd.DataFrame(data={'actual': actual["readiness"], 'predicted': predicted["readiness"]})
    RMSE = mean_squared_error(results['actual'], results['predicted'], squared=False)

    return RMSE

def calculate_RMSE_actual_days(actual, predicted):
    """
    calculates rmse of predictions for actual days

    Arguments:
        actual: actual values 
        predicted: predicted values 

    
    Returns:
        RMSE of predictions
    """
    results = pd.DataFrame(data={'actual': actual["readiness"], 'predicted': predicted["readiness"]})
    RMSE = mean_squared_error(results['actual'], results['predicted'], squared=False)

    return RMSE

def plots_and_table(df):
    """
    creates table for numeric values

    Arguments:
        df: pandas dataframe 

    
    Returns:
        -
    """
    
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=df.values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        bbox=[0, -0.4, 1, 0.3],
                        loc='bottom')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)