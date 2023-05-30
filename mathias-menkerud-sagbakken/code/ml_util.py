import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sktime.datatypes._panel._convert import from_2d_array_to_nested

def createDataset():
    """
    creates the dataset

    Arguments:
        -

    Returns:
        Pandas DataFrame of the dataset
    """
    path = "../data/mysql_dataset/complete_dataset"
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0", axis=1)
    df['date'] =  pd.to_datetime(df['date'])
    df['day'] = df.date.dt.dayofweek.astype(str).astype("category").astype(int)
    df["month"] = df.date.dt.month.astype(str).astype("category").astype(int)

    return df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    add feature lags to pd dataframe 

    Arguments:
        data: pandas dataframe.
        n_in: Number of lag observations as input 
        n_out: Number of observations as output
        dropnan: Boolean whether or not to drop rows with NaN values.
    
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('ft%d_t-%d' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('ft%d_t' % (j+1)) for j in range(n_vars)]
        else:
            names += [('ft%d_t+%d' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def fitXGBoost(X_train, y_train, X_test, y_test):
    """
    fits data to XGBOOST model

    Arguments:
        X_train: feature training data
        y_train: target training data
        X_test: feature validation data
        y_test: target validation data
    
    Returns:
        fitted XGBOOST model
    """
    reg = xgb.XGBRegressor(booster='gbtree',    
                       n_estimators=200, #200
                       objective='reg:squarederror',
                       learning_rate=0.07, 
                       colsample_bytree = 0.9704161741146843, 
                       gamma = 3.472716930386355, 
                       max_depth = 9, 
                       min_child_weight = 9, 
                       reg_alpha = 44, 
                       reg_lambda = 0.454959775303947,
                       seed=0)

    reg.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100, early_stopping_rounds=20)

    return reg


def fitLinearReg(X_train, y_train):
    """
    fits data to Linear regression model

    Arguments:
        X_train: feature training data
        y_train: target training data
    
    Returns:
        fitted Linear regression model
    """
    reg = LinearRegression()

    reg = reg.fit(X_train, y_train)

    return reg


def fitTree(X_train, y_train):
    """
    fits data to tree model

    Arguments:
        X_train: feature training data
        y_train: target training data
    
    Returns:
        fitted tree model
    """

    reg = tree.DecisionTreeRegressor(random_state=42)
    #reg = TimeSeriesForestRegressor(min_interval=7, n_estimators=300, n_jobs=-1, random_state=42)
    reg = reg.fit(X_train, y_train)

    return reg


def plotReadiness(df):
    """
    plots prediction vs actual values

    Arguments:
        df: pandas dataframe containing acutal and predicted columns
    
    Returns:
        plot
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    df["actual"].plot(ax=ax, label='Actual', title='Predicted vs. Actual', linewidth=1)
    df["predicted"].plot(ax=ax, label='Predicted', linewidth=1)
    plt.xlabel('Timesteps in days', fontsize=18)
    plt.ylabel('Readiness value', fontsize=16)
    ax.axvline(0, color='black', ls='--')
    ax.legend(['Actual', 'Predicted'])
    plt.show()


def recursive_multistep_prediction(model, X_test, timestepsOut, num_features, columnNames):
    """
    make recursive multistep prediction reusing output data to make prediction for next timestep

    Arguments:
        model: fitted model
        X_test: feature validation data
        timestepsOut: number of timesteps in the future to predict
        num_features: number of features in dataset
        columnNames: names of all features
    
    Returns:
        predictions n timesteps in the future 
    """

    y_predict = []

    for day in range(len(X_test)):
        current_day = X_test.iloc[[day]]
        for steps in range(timestepsOut):
            pred = model.predict(current_day)
            current_day = current_day.T.shift(-num_features).T
            current_day.iloc[:, -num_features:] = pred
        y_predict.append(pred.flatten())

    y_predict = pd.DataFrame(data = np.array(y_predict), columns = columnNames)

    return y_predict


def renameColumns(df, columnNames):
    """
    rename the columns

    Arguments:
        df: pandas dataframe
        columnNames: column names to replace current
    
    Returns:
        plot
    """
    
    df.columns.values[-len(columnNames):] = columnNames
    return df


def addReadinessLag(df, lag):
    """
    adds readiness feature lag for one timestep (creates the target value)

    Arguments:
        df: dataframe with all data
        index: player index
        player_range: days the player has data

    Returns:
        train and test set
    """
    target = 'readiness_t+1'

    df[target] = df['readiness'].shift(-lag)

    df.dropna(inplace=True)

    return df



def to_sessions(df):
    """
    Converts the dataset to a session to session dataset avoiding
    imputed data 

    Arguments:
        data: pandas dataframe.
        n_in: Number of lag observations as input 
        n_out: Number of observations as output
        dropnan: Boolean whether or not to drop rows with NaN values.
    
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    col_list =  list(df["Duration"])
    daysSinceLastSession = [1]

    counter = 1
    for i in range(len(col_list)):
        if col_list[i] != '0':
            counter = 1
        else:
            counter += 1
        daysSinceLastSession.append(counter)

    df["daysSinceLastSession"] = daysSinceLastSession[:-1]

    df = df.loc[df['Duration'] != '0']

    df['Duration'] = pd.to_timedelta(df['Duration']).dt.total_seconds() 

    return df
