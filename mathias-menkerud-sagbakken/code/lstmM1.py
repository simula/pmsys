from torch import nn
import torch

class ShallowRegressionLSTM(nn.Module):
    """
	LSTM regression model

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

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1) #1/10

    def forward(self, x):
        x = self.dropout(x)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device = ('cpu')).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device = ('cpu')).requires_grad_()
        
        lstm_out, (hn, c) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten() # First dim of Hn is num_layers, which is set to 1 above. #.flatten()

        #return self.act(out) 
        return out #torch.argmax(output, dim=1)

        

def train_model(data_loader, model, loss_function, optimizer):
    """
	Trains the lstm model

	Arguments:
        data_loader: data loader
        model: lstm model
        loss_function: loss function (mse loss)
        optimizer: adam
	
    Returns:
		trained model
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
	Trains the lstm model

	Arguments:
        data_loader: data loader
        model: lstm model
        loss_function: loss function (mse loss)
	
    Returns:
		test model loss
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
