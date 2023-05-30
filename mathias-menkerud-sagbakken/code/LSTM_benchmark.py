import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from ml_util import *
from dataloader import *
from lstmM1 import *

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plots_and_table(df):
    """
    creates table for numeric values

    Arguments:
        df: dataset

    
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
    
    #plt.ylabel("RMSE-Scores".format())
    #plt.xticks([])


def to_actual_dates(actual_days_with_imputed, actual_days, results):
    """
    convert dataset to only include dates that are not a result of imputation

    Arguments:
        actual_days_with_imputed: list of all dates
        actual_days: list of only real days
        results: pandas dataframe with actual and predicted

    
    Returns:
        dataframe with only actual days
    """
    results["date"] = actual_days_with_imputed
    results = results[results['date'].isin(actual_days)]
    results = results.reset_index(drop=True)

    return results


def predict(data_loader, model):
    """
    predicts a time-step

    Arguments:
        data_loader: data
        model: lstm model

    
    Returns:
        prediction
    """

    output = torch.tensor([]).to("cpu")
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output


def createLinePlots(df_test, ystar_col, i):
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
    RMSE = mean_squared_error(df_test["readiness_t+1"], df_test[ystar_col], squared=False)
    MSE = mean_squared_error(df_test["readiness_t+1"], df_test[ystar_col], squared=True)
    print(f'MSE Score on Test set: {MSE:0.4f}')
    print(f'RMSE Score on Test set: {RMSE:0.4f}')
    fig, ax = plt.subplots(figsize=(15, 5))
    df_test["readiness_t+1"].plot(ax=ax, label='Actual', title='RMSE: '+str(RMSE), linewidth=1)
    df_test[ystar_col].plot(ax=ax, label='Predicted', linewidth=1)
    plt.xlabel('Timesteps in days', fontsize=18)
    plt.ylabel('Readiness value', fontsize=16)
    ax.axvline(0, color='black', ls='--')
    ax.legend(['Actual', 'Predicted'])
    plt.savefig("experiment_plots/LSTM_lineplot"+str(i))
    plt.close()


def run_benchmark_lstm(df_, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr, only_real_days):
    """
    Runs lstm on all players on a given team using leave one out cross validation

    Arguments:
        df_: entire dataset
        forecast_horizon: output window
        sequence_length: input window
        columnNames: features 
        learning_rate: learning rate
        num_hidden_units = number of hidden units
        batch_size: batch size
        epoch: epoch
        nr_players: number of players
        players: players
        runOnce: runOnce
        only_real_days: only_real_days
        configNr: configNr
    
    Returns:
        rmse of predictions
    """

    df = df_.copy()
    #players = list(df['player_name_x'].unique())
    lstm_rmse = []
    df_pre = to_sessions(df)
    n_in = sequence_length

    for i in range(nr_players):

        print(str(i)+"/"+str(len(players)))
        all_but_one = players[:i] + players[i+1:]
        train = df[df['player_name_x'].isin(all_but_one)]
        test = df.loc[df['player_name_x'] == players[i]]
        cutoff = int(len(test)*0.7)
        test_fine_tune = test[:cutoff]
        test = test[cutoff:]
        train = train.append(test_fine_tune, ignore_index=True)
        test_pre = df_pre.loc[df_pre['player_name_x'] == players[i]]
        actual_days = list(test_pre["date"])[1:]
        actual_days_with_imputed = list(test["date"])[1:]
        #train = df[df['player_name_x'].isin(players[1:-1])]
        #test = df.loc[df['player_name_x'] == players[0]]

        train = train[columnNames]
        test = test[columnNames]

        train = addReadinessLag(train, n_out)
        test = addReadinessLag(test, n_out)
        features = train.columns.tolist()[:-1]
        target = train.columns.tolist()[-1]

        columnNames_ = columnNames+["readiness_t+1"]

        train_scalar = StandardScaler()
        train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames_)
        test = pd.DataFrame(train_scalar.transform(test), columns=columnNames_)

        train_fine_tune = train.tail(cutoff)
        train = train.iloc[:-cutoff]

        print(train)

        fine_tune_loader, test_loader, fine_tune_Dataset, testDataset = createDataloader(train_fine_tune, test, batch_size, sequence_length, features, target)

        train_loader, test_loader, trainDataset, testDataset = createDataloader(train, test, batch_size, sequence_length, features, target)

        model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        early_stopper = EarlyStopper(patience=3, min_delta=0.005)
        for ix_epoch in range(epoch):
            print(f"Epoch {ix_epoch}\n---------")
            train_model(train_loader, model, loss_function, optimizer=optimizer)
            validation_loss = test_model(test_loader, model, loss_function)
            print()
            print(validation_loss)
            if early_stopper.early_stop(validation_loss):
                print("Training stopping at epoch: ",ix_epoch)             
                break
        
        early_stopper = EarlyStopper(patience=1, min_delta=0.005)
        for ix_epoch in range(epoch):
            print(f"Epoch {ix_epoch}\n---------")
            train_model(fine_tune_loader, model, loss_function, optimizer=optimizer)
            validation_loss = test_model(test_loader, model, loss_function)
            print()
            print(validation_loss)
            if early_stopper.early_stop(validation_loss):
                print("Training stopping at epoch: ",ix_epoch)             
                break
        
        train_eval_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False)

        df_train = train.copy()
        df_test = test.copy()
        target_mean = df_train["readiness_t+1"].mean()
        target_stdev = df_train["readiness_t+1"].std()

        ystar_col = "Model forecast"
        df_train[ystar_col] = predict(train_eval_loader, model).cpu().numpy()
        df_test["readiness"] = predict(test_loader, model).cpu().numpy()

        df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

        for c in df_out.columns:
            df_out[c] = df_out[c] * target_stdev + target_mean

        df_test = df_test.reset_index(drop=True)

        df_test = pd.DataFrame(train_scalar.inverse_transform(df_test), columns=columnNames_)
        if only_real_days:
            df_test = to_actual_dates(actual_days_with_imputed, actual_days, df_test)
            lstm_rmse.append(mean_squared_error(df_test['readiness_t+1'], df_test['readiness'], squared=False))
        else:
            lstm_rmse.append(mean_squared_error(df_test['readiness_t+1'], df_test['readiness'], squared=False))

        if (i == 0):
            if configNr < 100:
                createLinePlots(df_test, "readiness", configNr)
        
        if runOnce:
            break

    lstm_rmse = pd.DataFrame(
    {'lstm_rmse': lstm_rmse
    })

    return lstm_rmse


def main():
    """
    predictions using lstm on an entire team. results are stored and visualized through boxplots and linplot
    """

    df = createDataset()
    df = df.loc[df['Team_name'] == "TeamA"]

    nr_players = len(list(df['player_name_x'].unique()))
    batch_size = 16
    epoch = 20
    sequence_length = 7
    n_out = 1
    learning_rate = 1e-3
    num_hidden_units = 32
    configNr = 61
    runOnce = False
    only_real_days = True
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    players = list(df['player_name_x'].unique())

    lstm_rmse = run_benchmark_lstm(df, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr, only_real_days)

    melted_df = pd.melt(lstm_rmse)
    sns.boxplot(x='variable', y='value', data=melted_df).set(title='RMSE-values')
    lstm_rmse.loc[len(lstm_rmse.index)] = [lstm_rmse['lstm_rmse'].mean()]
    lstm_rmse.loc[len(lstm_rmse.index)] = [lstm_rmse['lstm_rmse'].min()]
    lstm_rmse.loc[len(lstm_rmse.index)] = [lstm_rmse['lstm_rmse'].max()]
    lstm_rmse = lstm_rmse.round(decimals=3)
    index_labels=[]
    for i in range(len(lstm_rmse)-3):
        index_labels.append("PLayer"+str(i+1))
    index_labels = index_labels+["mean", "min", "max"]
    lstm_rmse.index = index_labels
    plots_and_table(lstm_rmse.tail(3))
    plt.savefig("experiment_plots/boxplots_lstm", bbox_inches="tight", pad_inches=1)


if __name__ == "__main__":
    main()

