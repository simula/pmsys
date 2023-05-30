from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class TimeseriesDataset(Dataset):
    """
	A dataset made for timeseries data

	Arguments:
        dataframe: dataframe of all data
        features: feature names
        target: target name(s)
        sequence_length: length of sequence/input window
	
    Returns:
		A timeseriesDataset
	"""
    def __init__(self, dataframe, features, target, sequence_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.X = torch.tensor(dataframe[features].values, device = ('cpu')).float()
        self.y = torch.tensor(dataframe[target].values, device = ('cpu')).long() #long/float

        #a = dataframe[target].astype(int).values
        #b = np.zeros((a.size, a.max()+1))
        #b[np.arange(a.size)-1, a] = 1
        #self.b = torch.tensor(b)
        #print(b[0].shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def createDataloader(train, test, batch_size, sequence_length, features, target):
    """
	Creates a dataloader for time series data

	Arguments:
        train: train data
        test: validation data
        batch_size: size of batches
        sequence_length: length of sequence/input window
        features: feature names
        target: target name(s)
	
    Returns:
		A timeseriesDataset
	"""

    torch.manual_seed(101)

    trainDataset = TimeseriesDataset(
        train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    testDataset = TimeseriesDataset(
        test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, trainDataset, testDataset

class ShallowRegressionLSTM(nn.Module):
    """
	LSTM model

	Arguments:
        num_sensors: number of features
        hidden_units: number of hidden units
	
    Returns:
		out- predicted values
	"""
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 2
        self.dropout = nn.Dropout(0)

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=10) #1/10

    def forward(self, x):
        x = self.dropout(x)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device = ('cpu')).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device = ('cpu')).requires_grad_()
        
        lstm_out, (hn, c) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]) # First dim of Hn is num_layers, which is set to 1 above. #.flatten()

        return out #torch.argmax(output, dim=1)
        

        

def train_model(data_loader, model, loss_function, optimizer):
    """
	train lstm model

	Arguments:
        data_loader: data loader
        model: lstm model
        loss_function: entropy loss
        optimizer: adam
	
    Returns:
		fitted model
	"""
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        criterion = loss_function
        #loss = torch.sqrt(criterion(output, y))
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return model

def test_model(data_loader, model, loss_function):
    """
	predict using lstm model

	Arguments:
        data_loader: data loader
        model: lstm model
        loss_function: entropy loss
	
    Returns:
		model loss
	"""
    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss


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
	table for numeric values

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



def createDataset():
    """
	creates dataset

	Arguments:
        -
	
    Returns:
		dataset
	"""
    path = "../data/mysql_dataset/complete_dataset"
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0", axis=1)
    df['date'] =  pd.to_datetime(df['date'])
    df['day'] = df.date.dt.dayofweek.astype(str).astype("category").astype(int)
    df["month"] = df.date.dt.month.astype(str).astype("category").astype(int)

    return df


def predict(data_loader, model):
    """
	use lstm model to predict

	Arguments:
        data_loader: data loader
        model: lstm model
	
    Returns:
		predictions
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


def run_lstm_classifier(df_, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr):
    """
    Runs lstm on all players on a given team using leave one out cross validation

    Arguments:
        df_: entire dataset
        n_out: output window
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

    for i in range(nr_players):
        print(str(i)+"/"+str(len(players)))
        all_but_one = players[:i] + players[i+1:]
        train = df[df['player_name_x'].isin(all_but_one)]
        test = df.loc[df['player_name_x'] == players[i]]
        #train = df[df['player_name_x'].isin(players[1:-1])]
        #test = df.loc[df['player_name_x'] == players[0]]

        train = train[columnNames]
        test = test[columnNames]

        train = addReadinessLag(train, n_out)
        test = addReadinessLag(test, n_out)
        #train["readiness_t+1"] = train["readiness_t+1"]-1 
        #test["readiness_t+1"] = test["readiness_t+1"]-1 
        #train["readiness"] = train["readiness"]-1 
        #test["readiness"] = test["readiness"]-1 
        features = train.columns.tolist()[:-1]
        target = train.columns.tolist()[-1]

        columnNames_ = columnNames+["readiness_t+1"]

        train_scalar = StandardScaler()
        train = pd.DataFrame(train_scalar.fit_transform(train), columns=columnNames_)
        test = pd.DataFrame(train_scalar.transform(test), columns=columnNames_)

        train_copy = pd.DataFrame(train_scalar.inverse_transform(train), columns=columnNames_)
        test_copy = pd.DataFrame(train_scalar.inverse_transform(test), columns=columnNames_)
        train[["readiness_t+1", "readiness"]] = train_copy[["readiness_t+1", "readiness"]].astype(int)
        test[["readiness_t+1", "readiness"]] = test_copy[["readiness_t+1", "readiness"]].astype(int)

        train_loader, test_loader, trainDataset, testDataset = createDataloader(train, test, batch_size, sequence_length, features, target)

        model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
        loss_function = nn.CrossEntropyLoss()
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
        
        train_eval_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False)

        x = predict(test_loader, model)
        results_classes = []
        for i in range(len(x.numpy())):
            results_classes.append(np.argmax(x.numpy()[i]))

        df_train = train.copy()
        df_test = test.copy()

        ystar_col = "Model forecast"
        df_test[ystar_col] = results_classes

        y = predict(train_eval_loader, model)
        results_classes_y = []
        for i in range(len(y.numpy())):
            results_classes_y.append(np.argmax(y.numpy()[i]))

        fig, ax = plt.subplots(figsize=(15, 5))
        df_test["readiness_t+1"].plot(ax=ax, label='Actual', title='Predicted vs. Actual', linewidth=1)
        df_test[ystar_col].plot(ax=ax, label='Predicted', linewidth=1)
        plt.xlabel('Timesteps in days', fontsize=18)
        plt.ylabel('Readiness value', fontsize=16)
        ax.axvline(0, color='black', ls='--')
        ax.legend(['Actual', 'Predicted'])
        plt.close()

        lstm_rmse.append(mean_squared_error(df_test['readiness_t+1'], df_test['readiness'], squared=False))

        if (i == 0):
            if configNr < 100:
                createLinePlots(df_test, "readiness", configNr)
        
        if runOnce:
            break

    lstm_rmse = pd.DataFrame(
    {'lstm_accuracy': lstm_rmse
    })

    results = {'actual': df_test["readiness_t+1"],
        'predicted': df_test["Model forecast"]}

    results = pd.DataFrame(results)

    return results


def main():
    """
    Run classification experiment on the lstm model classifying readiness    
    """

    df = createDataset()
    df = df.loc[df['Team_name'] == "TeamA"]

    nr_players = len(list(df['player_name_x'].unique()))
    batch_size = 16
    epoch = 10
    sequence_length = 7
    n_out = 1
    learning_rate = 1e-3
    num_hidden_units = 32
    configNr = 101
    runOnce = True
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "day", "month"]
    players = list(df['player_name_x'].unique())

    lstm_data = run_lstm_classifier(df, sequence_length, n_out, columnNames, learning_rate, num_hidden_units, batch_size, epoch, nr_players, players, runOnce, configNr)

    print(lstm_data)


if __name__ == "__main__":
    main()

