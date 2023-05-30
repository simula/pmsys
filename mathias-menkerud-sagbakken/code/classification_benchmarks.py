from sklearn.linear_model import RidgeClassifierCV
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.transformations.panel.rocket import (
    Rocket,
    MiniRocket,
    MiniRocketMultivariate,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_util import *
from baseline_framework import *
from LSTM_benchmark import *
from LSTM_classifier import *

def to_actual_dates(actual_days_with_imputed, actual_days, results):
    """
    Only choose the actual days

    Arguments:
        actual_days_with_imputed: list of all dates.
        actual_days: List of only actual days 
        results: actual and prediction y values
    
    Returns:
        Pandas DataFrame with only actual dates
    """
    results["date"] = actual_days_with_imputed
    results = results[results['date'].isin(actual_days)]
    results = results.reset_index(drop=True)

    return results

def readiness_to_peaks(df):
    """
    converts readiness feature to peaks betweeen 1 and 3

    Arguments:
        df: list of all dates.
    
    Returns:
        transformed dataset 
    """
    df = df.replace({'readiness' : { 1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 1, 6 : 1, 7 : 1, 8 : 2, 9 : 2, 10: 2}})

    return df

def toChange(df, n_out):
    """
    converts readiness to change in readiness

    Arguments:
        df: pandas dataframe
        n_out: output window
    
    Returns:
        transformed dataset
    """
    players = list(df['player_name_x'].unique())
    pdList = []

    for i in range(len(players)):
        df_ = df.loc[df['player_name_x'] == players[i]]
        df_['readiness'] = df_['readiness'] - df_['readiness'].shift(-n_out)
        df_.loc[df_['readiness'] > 0, 'readiness'] = 2
        df_.loc[df_['readiness'] == 0, 'readiness'] = 1
        df_.loc[df_['readiness'] < 0, 'readiness'] = 0
        df_.drop(df_.tail(n_out).index,inplace=True)
        pdList.append(df_)
    df = pd.concat(pdList)
    return df

def ridgeClassifier(X_train_transform, y_train):
    """
    fit a ridge classifier

    Arguments:
        X_train_transform: transformed train set
        y_train: y train
    
    Returns:
        fitted ridge classifier
    """
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True, cv = None).fit(X_train_transform, y_train.values.ravel())
    return classifier

def generate_auc_scores(y_test, y_pred):
    """
    generate area under curve scores

    Arguments:
        y_test: actual
        y_pred: predicted
    
    Returns:
        auc score
    """
    y_pred = abs(y_pred)
    metric_score = roc_auc_score(y_test, y_pred, average='macro', sample_weight=None, multi_class='ovr')
    return metric_score

def generateAndSaveCM(classifier, X_test, y_test, plot_name):
    """
    plots confusion matrix

    Arguments:
        classifier: classifier
        X_test: x test
        y_test: y test
        plot_name: filename for plot
    
    Returns:
        -
    """
    df = createDataset()
    df_pre = to_sessions(df)
    players = list(df['player_name_x'].unique())
    test_after = df.loc[df['player_name_x'] == players[0]]
    test_pre = df_pre.loc[df_pre['player_name_x'] == players[0]]
    test_pre = test_pre.reset_index(drop=True)
    actual_days = list(test_pre["date"])[8:]
    actual_days_with_imputed = list(test_after["date"])[8:]

    print("Accuracy",classifier.score(X_test, y_test.values.ravel()))
    y_pred = classifier.predict(X_test)
    results_df = pd.DataFrame(
    {'actual': list(y_test),
     'pred': y_pred
    })
    #results_df = to_actual_dates(actual_days_with_imputed, actual_days, results_df)
    #print(results_df)
    #plot_confusion_matrix(classifier, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    #cm = confusion_matrix(results_df["actual"], results_df["pred"], labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
    disp.plot() 
    acc = classifier.score(X_test, y_test.values.ravel())
    title = "Accuracy: "+str(round(acc,2))
    disp.ax_.set_title(title)
    #plt.title("Rocket- accuracy: ",acc)
    plt.savefig("experiment_plots/"+plot_name)

def generateAndSaveCM_LSTM(results, plot_name):
    """
    plots confusion matrix

    Arguments:
        results: predicted and actual values stored in pandas dataframe
        plot_name: filename for plot

    Returns:
        -
    """
    df = createDataset()
    df_pre = to_sessions(df)
    players = list(df['player_name_x'].unique())
    test_after = df.loc[df['player_name_x'] == players[0]]
    test_pre = df_pre.loc[df_pre['player_name_x'] == players[0]]
    test_pre = test_pre.reset_index(drop=True)
    actual_days = list(test_pre["date"])[2:]
    actual_days_with_imputed = list(test_after["date"])[2:]
    results_df = pd.DataFrame(
    {'actual': results["actual"],
     'predicted': results["predicted"]
    })
    #results = to_actual_dates(actual_days_with_imputed, actual_days, results_df)
    #print(results)

    classes = [list(range(0, int(results['actual'].nunique())))]
    cm = confusion_matrix(results["actual"], results["predicted"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot() 
    acc = accuracy_score(results['actual'], results['predicted'])
    title = "Accuracy: "+str(round(acc,2))
    disp.ax_.set_title(title)
    #plt.title("Rocket- accuracy: ",acc)
    plt.savefig("experiment_plots/"+plot_name)


def rocket_config(train, test, val, n_in, n_out, columnNames, plot_name):
    """
    Generate results for rocket model

    Arguments:
        train: train set
        test: test set
        val: val set
        n_in: input window
        n_out: output window
        columnNames: list of columns
        plot_name: filename for plot

    Returns:
        actual and predictions
    """
    train = train[columnNames]
    test = test[columnNames]
    val = val[columnNames]

    train_scalar = StandardScaler()
    train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames)
    test = pd.DataFrame(train_scalar.transform(test), columns=columnNames)
    
    train_copy = pd.DataFrame(train_scalar.inverse_transform(train), columns=columnNames)
    test_copy = pd.DataFrame(train_scalar.inverse_transform(test), columns=columnNames)
    train[["readiness"]] = train_copy[["readiness"]].astype(int)
    test[["readiness"]] = test_copy[["readiness"]].astype(int)

    train_direct = series_to_supervised(train.copy(), n_in, n_out)
    test_direct = series_to_supervised(test.copy(), n_in, n_out)
    val_direct = series_to_supervised(val.copy(), n_in, n_out)

    num_features = len(train.columns.tolist())
    features = train_direct.columns.tolist()[:-num_features]
    target = train_direct.columns.tolist()[-num_features:][0]

    X_train = train_direct[features]
    y_train = train_direct[target]
    X_test = test_direct[features]
    y_test = test_direct[target]
    X_val = val_direct[features]
    y_val = val_direct[target]

    X_train = from_2d_array_to_nested(X_train)
    X_test = from_2d_array_to_nested(X_test)
    X_val = from_2d_array_to_nested(X_val)

    rocket = MiniRocketMultivariate(num_kernels=10000, n_jobs=-1, random_state=42)  
    rocket.fit(X_train)
    X_train_transform = rocket.transform(X_train)
    X_test_transform = rocket.transform(X_test)
    X_val_transform = rocket.transform(X_val)

    classifier = ridgeClassifier(X_train_transform, y_train)

    #X_test_transform = rocket.transform(X_test)

    generateAndSaveCM(classifier, X_test_transform, y_test, plot_name)

    y_pred = classifier.predict(X_test_transform)

    return list(y_test), y_pred


def ridge_config(train, test, val, n_in, n_out, columnNames, plot_name):
    """
    Generate results for ridge model

    Arguments:
        train: train set
        test: test set
        val: val set
        n_in: input window
        n_out: output window
        columnNames: list of columns
        plot_name: filename for plot

    Returns:
        actual and predictions
    """

    train = train[columnNames]
    test = test[columnNames]
    val = val[columnNames]

    train_scalar = StandardScaler()
    train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames)
    test = pd.DataFrame(train_scalar.transform(test), columns=columnNames)
    
    train_copy = pd.DataFrame(train_scalar.inverse_transform(train), columns=columnNames)
    test_copy = pd.DataFrame(train_scalar.inverse_transform(test), columns=columnNames)
    train[["readiness"]] = train_copy[["readiness"]].astype(int)
    test[["readiness"]] = test_copy[["readiness"]].astype(int)

    train_direct = series_to_supervised(train.copy(), n_in, n_out)
    test_direct = series_to_supervised(test.copy(), n_in, n_out)
    val_direct = series_to_supervised(val.copy(), n_in, n_out)

    num_features = len(train.columns.tolist())
    features = train_direct.columns.tolist()[:-num_features]
    target = train_direct.columns.tolist()[-num_features:][0]

    X_train = train_direct[features]
    y_train = train_direct[target]
    X_test = test_direct[features]
    y_test = test_direct[target]
    X_val = val_direct[features]
    y_val = val_direct[target]
    
    classifier = ridgeClassifier(X_train, y_train)

    generateAndSaveCM(classifier, X_test, y_test, plot_name)

    y_pred = classifier.predict(X_test)

    return list(y_test), y_pred


def dummy_config(train, test, val, n_in, n_out, columnNames, plot_name):
    """
    Generate results for dummy model

    Arguments:
        train: train set
        test: test set
        val: val set
        n_in: input window
        n_out: output window
        columnNames: list of columns
        plot_name: filename for plot

    Returns:
        actual and predictions
    """

    train = train[columnNames]
    test = test[columnNames]
    val = val[columnNames]

    train_scalar = StandardScaler()
    train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames)
    test = pd.DataFrame(train_scalar.transform(test), columns=columnNames)
    
    train_copy = pd.DataFrame(train_scalar.inverse_transform(train), columns=columnNames)
    test_copy = pd.DataFrame(train_scalar.inverse_transform(test), columns=columnNames)
    train[["readiness"]] = train_copy[["readiness"]].astype(int)
    test[["readiness"]] = test_copy[["readiness"]].astype(int)

    train_direct = series_to_supervised(train.copy(), n_in, n_out)
    test_direct = series_to_supervised(test.copy(), n_in, n_out)
    val_direct = series_to_supervised(val.copy(), n_in, n_out)

    num_features = len(train.columns.tolist())
    features = train_direct.columns.tolist()[:-num_features]
    target = train_direct.columns.tolist()[-num_features:][0]

    X_train = train_direct[features]
    y_train = train_direct[target]
    X_test = test_direct[features]
    y_test = test_direct[target]
    X_val = val_direct[features]
    y_val = val_direct[target]
    
    classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(X_train, y_train)

    generateAndSaveCM(classifier, X_test, y_test, plot_name)

    y_pred = classifier.predict(X_test)

    return list(y_test), y_pred



def xgb_config(train, test, val, n_in, n_out, columnNames, plot_name):
    """
    Generate results for xgboost model

    Arguments:
        train: train set
        test: test set
        val: val set
        n_in: input window
        n_out: output window
        columnNames: list of columns
        plot_name: filename for plot

    Returns:
        actual and predictions
    """
    train = train[columnNames]
    test = test[columnNames]
    val = val[columnNames]

    train_scalar = StandardScaler()
    train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames)
    test = pd.DataFrame(train_scalar.transform(test), columns=columnNames)
    
    train_copy = pd.DataFrame(train_scalar.inverse_transform(train), columns=columnNames)
    test_copy = pd.DataFrame(train_scalar.inverse_transform(test), columns=columnNames)
    train[["readiness"]] = train_copy[["readiness"]].astype(int)
    test[["readiness"]] = test_copy[["readiness"]].astype(int)

    train_direct = series_to_supervised(train.copy(), n_in, n_out)
    test_direct = series_to_supervised(test.copy(), n_in, n_out)
    val_direct = series_to_supervised(val.copy(), n_in, n_out)

    num_features = len(train.columns.tolist())
    features = train_direct.columns.tolist()[:-num_features]
    target = train_direct.columns.tolist()[-num_features:][0]

    X_train = train_direct[features]
    y_train = train_direct[target]
    X_test = test_direct[features]
    y_test = test_direct[target]
    X_val = val_direct[features]
    y_val = val_direct[target]

    xgbc = XGBClassifier(learning_rate=0.06)
    xgbc.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=100)

    generateAndSaveCM(xgbc, X_test, y_test, plot_name)

    y_pred = xgbc.predict(X_test)

    return list(y_test), y_pred


def lstm_config(df, columnNames, n_in, n_out, plot_name):
    """
    Generate results for LSTM model

    Arguments:
        df: dataset
        n_in: input window
        n_out: output window
        columnNames: list of columns
        plot_name: filename for plot

    Returns:
        actual and predictions
    """

    nr_players = len(list(df['player_name_x'].unique()))
    batch_size = 16
    epoch = 40
    sequence_length = n_in
    n_out = n_out
    learning_rate = 1e-3
    num_hidden_units = 32
    configNr = 5
    runOnce = True
    players = list(df['player_name_x'].unique())

    lstm_data = run_lstm_classifier(df, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr)

    generateAndSaveCM_LSTM(lstm_data, plot_name)

    results_df = pd.DataFrame(
    {'actual': lstm_data["actual"],
     'pred': lstm_data["predicted"]
    })

    return lstm_data["actual"], lstm_data["predicted"]


def main():
    """
    Runs classification experiment for all models and stores plots and results
    """

    n_out = 1
    n_in = 7
    #readiness needs to be first index
    columnNames = ["readiness", "daily_load", "fatigue", "mood", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    """"""""""""""""""""""""""""""
    df = createDataset()
    df = toChange(df, n_out)
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    val = df.loc[df['player_name_x'] == players[1]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    ridge = []
    ridge_test, ridge_pred = ridge_config(train, test, val, n_in, n_out, columnNames, "RIDGE_config1")
    print("f1",f1_score(ridge_test, ridge_pred, average='weighted'))
    ridge.append(accuracy_score(ridge_test, ridge_pred))
    ridge.append(f1_score(ridge_test, ridge_pred, average='weighted'))

    rocket = []
    rocket_test, rocket_pred = rocket_config(train, test, val, n_in, n_out, columnNames, "ROCKET_config1")
    rocket.append(accuracy_score(rocket_test, rocket_pred))
    rocket.append(f1_score(rocket_test, rocket_pred, average='weighted'))

    xgboost = []
    xgb_test, xgb_pred = xgb_config(train, test, val, n_in, n_out, columnNames, "XGB_config1")
    xgboost.append(accuracy_score(xgb_test, xgb_pred))
    xgboost.append(f1_score(xgb_test, xgb_pred, average='weighted'))

    dummy = []
    dummy_test, dummy_pred = dummy_config(train, test, val, n_in, n_out, columnNames, "DUMMY_config1")
    dummy.append(accuracy_score(dummy_test, dummy_pred))
    dummy.append(f1_score(dummy_test, dummy_pred, average='weighted'))

    lstm = []
    lstm_test, lstm_pred = lstm_config(df, columnNames, n_in, n_out, "LSTM_config1")
    lstm.append(accuracy_score(lstm_test, lstm_pred))
    lstm.append(f1_score(lstm_test, lstm_pred, average='weighted'))

    config1 = pd.DataFrame(
    {'ridge': ridge,
     'rocket': rocket,
     'xgboost': xgboost,
     'dummy': dummy,
     'lstm': lstm
    })
    print(config1)
    df_name = "experiment_data/"+"classification-confgi1"
    config1.to_pickle(df_name)

    """"""""""""""""""""""""""""""
    df = createDataset()
    df["readiness"] -= 1
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    val = df.loc[df['player_name_x'] == players[1]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    ridge = []
    ridge_test, ridge_pred = ridge_config(train, test, val, n_in, n_out, columnNames, "RIDGE_config2")
    ridge.append(accuracy_score(ridge_test, ridge_pred))
    ridge.append(f1_score(ridge_test, ridge_pred, average='weighted'))

    rocket = []
    rocket_test, rocket_pred = rocket_config(train, test, val, n_in, n_out, columnNames, "ROCKET_config2")
    rocket.append(accuracy_score(rocket_test, rocket_pred))
    rocket.append(f1_score(rocket_test, rocket_pred, average='weighted'))

    xgboost = []
    xgb_test, xgb_pred = xgb_config(train, test, val, n_in, n_out, columnNames, "XGB_config2")
    xgboost.append(accuracy_score(xgb_test, xgb_pred))
    xgboost.append(f1_score(xgb_test, xgb_pred, average='weighted'))

    dummy = []
    dummy_test, dummy_pred = dummy_config(train, test, val, n_in, n_out, columnNames, "DUMMY_config2")
    dummy.append(accuracy_score(dummy_test, dummy_pred))
    dummy.append(f1_score(dummy_test, dummy_pred, average='weighted'))

    lstm = []
    lstm_test, lstm_pred = lstm_config(df, columnNames, n_in, n_out, "LSTM_config2")
    lstm.append(accuracy_score(lstm_test, lstm_pred))
    lstm.append(f1_score(lstm_test, lstm_pred, average='weighted'))

    config2 = pd.DataFrame(
    {'ridge': ridge,
     'rocket': rocket,
     'xgboost': xgboost,
     'dummy': dummy,
     'lstm': lstm
    })
    print(config2)
    df_name = "experiment_data/"+"classification-confgi2"
    config2.to_pickle(df_name)


    """"""""""""""""""""""""""""""
    df = createDataset()
    df = readiness_to_peaks(df)
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    val = df.loc[df['player_name_x'] == players[1]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    ridge = []
    ridge_test, ridge_pred = ridge_config(train, test, val, n_in, n_out, columnNames, "RIDGE_config3")
    ridge.append(accuracy_score(ridge_test, ridge_pred))
    ridge.append(f1_score(ridge_test, ridge_pred, average='weighted'))

    rocket = []
    rocket_test, rocket_pred = rocket_config(train, test, val, n_in, n_out, columnNames, "ROCKET_config3")
    rocket.append(accuracy_score(rocket_test, rocket_pred))
    rocket.append(f1_score(rocket_test, rocket_pred, average='weighted'))

    xgboost = []
    xgb_test, xgb_pred = xgb_config(train, test, val, n_in, n_out, columnNames, "XGB_config3")
    xgboost.append(accuracy_score(xgb_test, xgb_pred))
    xgboost.append(f1_score(xgb_test, xgb_pred, average='weighted'))

    dummy = []
    dummy_test, dummy_pred = dummy_config(train, test, val, n_in, n_out, columnNames, "DUMMY_config3")
    dummy.append(accuracy_score(dummy_test, dummy_pred))
    dummy.append(f1_score(dummy_test, dummy_pred, average='weighted'))

    lstm = []
    lstm_test, lstm_pred = lstm_config(df, columnNames, n_in, n_out, "LSTM_config3")
    lstm.append(accuracy_score(lstm_test, lstm_pred))
    lstm.append(f1_score(lstm_test, lstm_pred, average='weighted'))

    config3 = pd.DataFrame(
    {'ridge': ridge,
     'rocket': rocket,
     'xgboost': xgboost,
     'dummy': dummy,
     'lstm': lstm
    })
    print(config3)
    df_name = "experiment_data/"+"classification-confgi3"
    config3.to_pickle(df_name)
    
    """"""""""""""""""""""""""""""
    n_out = 1
    n_in = 5
    columnNames = ["readiness", "fatigue", "mood", "daily_load", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day"]
    df = createDataset()
    df = to_sessions(df)
    df = toChange(df, n_out)
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    val = df.loc[df['player_name_x'] == players[1]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    ridge = []
    ridge_test, ridge_pred = ridge_config(train, test, val, n_in, n_out, columnNames, "RIDGE_Sessions_config1")
    ridge.append(accuracy_score(ridge_test, ridge_pred))
    ridge.append(f1_score(ridge_test, ridge_pred, average='weighted'))

    rocket = []
    rocket_test, rocket_pred = rocket_config(train, test, val, n_in, n_out, columnNames, "ROCKET_Sessions_config1")
    rocket.append(accuracy_score(rocket_test, rocket_pred))
    rocket.append(f1_score(rocket_test, rocket_pred, average='weighted'))

    xgboost = []
    xgb_test, xgb_pred = xgb_config(train, test, val, n_in, n_out, columnNames, "XGB_Sessions_config1")
    xgboost.append(accuracy_score(xgb_test, xgb_pred))
    xgboost.append(f1_score(xgb_test, xgb_pred, average='weighted'))

    dummy = []
    dummy_test, dummy_pred = dummy_config(train, test, val, n_in, n_out, columnNames, "DUMMY_Sessions_config1")
    dummy.append(accuracy_score(dummy_test, dummy_pred))
    dummy.append(f1_score(dummy_test, dummy_pred, average='weighted'))

    lstm = []
    lstm_test, lstm_pred = lstm_config(df, columnNames, n_in, n_out, "LSTM_Sessions_config1")
    lstm.append(accuracy_score(lstm_test, lstm_pred))
    lstm.append(f1_score(lstm_test, lstm_pred, average='weighted'))

    config1 = pd.DataFrame(
    {'ridge': ridge,
     'rocket': rocket,
     'xgboost': xgboost,
     'dummy': dummy,
     'lstm': lstm
    })
    print(config1)
    df_name = "experiment_data/"+"classification-sessions-confgi1"
    config1.to_pickle(df_name)

    """"""""""""""""""""""""""""""
    
    df = createDataset()
    df = to_sessions(df)
    df["readiness"] -= 2
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    val = df.loc[df['player_name_x'] == players[1]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    ridge = []
    ridge_test, ridge_pred = ridge_config(train, test, val, n_in, n_out, columnNames, "RIDGE_Sessions_config2")
    ridge.append(accuracy_score(ridge_test, ridge_pred))
    ridge.append(f1_score(ridge_test, ridge_pred, average='weighted'))

    rocket = []
    rocket_test, rocket_pred = rocket_config(train, test, val, n_in, n_out, columnNames, "ROCKET_Sessions_config2")
    rocket.append(accuracy_score(rocket_test, rocket_pred))
    rocket.append(f1_score(rocket_test, rocket_pred, average='weighted'))

    xgboost = []
    xgb_test, xgb_pred = xgb_config(train, test, val, n_in, n_out, columnNames, "XGB_Sessions_config2")
    xgboost.append(accuracy_score(xgb_test, xgb_pred))
    xgboost.append(f1_score(xgb_test, xgb_pred, average='weighted'))

    dummy = []
    dummy_test, dummy_pred = dummy_config(train, test, val, n_in, n_out, columnNames, "DUMMY_Sessions_config2")
    dummy.append(accuracy_score(dummy_test, dummy_pred))
    dummy.append(f1_score(dummy_test, dummy_pred, average='weighted'))

    lstm = []
    lstm_test, lstm_pred = lstm_config(df, columnNames, n_in, n_out, "LSTM_Sessions_config2")
    lstm.append(accuracy_score(lstm_test, lstm_pred))
    lstm.append(f1_score(lstm_test, lstm_pred, average='weighted'))

    config2 = pd.DataFrame(
    {'ridge': ridge,
     'rocket': rocket,
     'xgboost': xgboost,
     'dummy': dummy,
     'lstm': lstm
    })
    print(config2)
    df_name = "experiment_data/"+"classification-sessions-confgi2"
    config2.to_pickle(df_name)

    """"""""""""""""""""""""""""""
    
    df = createDataset()
    df = to_sessions(df)
    df = readiness_to_peaks(df)
    df = df.loc[df['Team_name'] == "TeamA"]
    players = list(df['player_name_x'].unique())
    
    players = list(df['player_name_x'].unique())
    train = df[df['player_name_x'].isin(players[2:])]
    test = df.loc[df['player_name_x'] == players[0]]
    val = df.loc[df['player_name_x'] == players[1]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    ridge = []
    ridge_test, ridge_pred = ridge_config(train, test, val, n_in, n_out, columnNames, "RIDGE_Sessions_config3")
    ridge.append(accuracy_score(ridge_test, ridge_pred))
    ridge.append(f1_score(ridge_test, ridge_pred, average='weighted'))

    rocket = []
    rocket_test, rocket_pred = rocket_config(train, test, val, n_in, n_out, columnNames, "ROCKET_Sessions_config3")
    rocket.append(accuracy_score(rocket_test, rocket_pred))
    rocket.append(f1_score(rocket_test, rocket_pred, average='weighted'))

    xgboost = []
    xgb_test, xgb_pred = xgb_config(train, test, val, n_in, n_out, columnNames, "XGB_Sessions_config3")
    xgboost.append(accuracy_score(xgb_test, xgb_pred))
    xgboost.append(f1_score(xgb_test, xgb_pred, average='weighted'))

    dummy = []
    dummy_test, dummy_pred = dummy_config(train, test, val, n_in, n_out, columnNames, "DUMMY_Sessions_config3")
    dummy.append(accuracy_score(dummy_test, dummy_pred))
    dummy.append(f1_score(dummy_test, dummy_pred, average='weighted'))

    lstm = []
    lstm_test, lstm_pred = lstm_config(df, columnNames, n_in, n_out, "LSTM_Sessions_config3")
    lstm.append(accuracy_score(lstm_test, lstm_pred))
    lstm.append(f1_score(lstm_test, lstm_pred, average='weighted'))

    config3 = pd.DataFrame(
    {'ridge': ridge,
     'rocket': rocket,
     'xgboost': xgboost,
     'dummy': dummy,
     'lstm': lstm
    })
    print(config3)
    df_name = "experiment_data/"+"classification-sessions-confgi3"
    config3.to_pickle(df_name)

if __name__ == "__main__":
    main()
